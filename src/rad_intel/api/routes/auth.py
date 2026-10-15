"""
Logging in to the closed review workspace.

There is no registration endpoint, by design. Accounts are created by an
administrator, either from the console in the workspace or with
scripts/manage_clinicians.py. The only thing an outsider can do here is submit
an access request, which records their details for the administrator and
grants nothing.

Logging in takes two steps and two separate requests:

1. `POST /code` (clinician: email + Rad-Intel ID) or `POST /admin/code`
   (administrator: email + password) names an account and sends a one-time
   code to the phone number on its record.
2. `POST /login` (email + code) exchanges the code for a session.

Neither step alone opens a session, so neither a Rad-Intel ID read off a
notepad nor an administrator's password is enough on its own. If the code
cannot be delivered the attempt is refused outright: the code is cancelled and
never returned in the response, because a system that shows the code on screen
when its message provider is down has a second factor in name only.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from rad_intel.api.ratelimit import client_address, limit_by_address
from rad_intel.api.schemas import (
    AccessRequestSubmission,
    AdminCodeRequest,
    AuthMeResponse,
    ChangePasswordRequest,
    ClinicianCodeRequest,
    CodeRequestResponse,
    VerifyCodeRequest,
)
from rad_intel.auth import accounts, activity, notify, otp, session
from rad_intel.auth.dependencies import current_clinician, require_clinician, session_token

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

# One message for a wrong email, a wrong Rad-Intel ID, a disabled account and
# an account that does not exist, so this endpoint cannot be used to find out
# which addresses are enrolled.
_NO_MATCH = (
    "Those details do not match an active account. Check the email address and "
    "Rad-Intel ID exactly as the study administrator gave them to you. If they "
    "still do not work, contact the administrator — accounts are issued by hand."
)
_NO_MATCH_ADMIN = "Incorrect email or password."


def _cookie_secure() -> bool:
    """Set RAD_INTEL_COOKIE_SECURE=1 behind HTTPS so the cookie is never sent in clear."""
    return os.environ.get("RAD_INTEL_COOKIE_SECURE", "").strip() in ("1", "true", "yes")


def _me(user: dict) -> AuthMeResponse:
    return AuthMeResponse(
        authenticated=True,
        id=user["id"],
        email=user["email"],
        full_name=user["full_name"],
        institution=user["institution"],
        role=user["role"],
        radintel_id=user.get("radintel_id"),
        phone=user.get("phone"),
        must_change_password=bool(user.get("must_change_password")),
    )


async def _issue_and_send(user: dict, request: Request) -> CodeRequestResponse:
    """Mint a code for an identified account and deliver it, or refuse."""
    code, message = await run_in_threadpool(
        otp.issue_code, user["id"], notify.channel_name(), user.get("phone_tail"),
    )
    if not code:
        activity.record("login.code_throttled", user=user, ip=client_address(request),
                        outcome="refused")
        raise HTTPException(status_code=429, detail=message)

    delivered, channel, detail = await run_in_threadpool(
        notify.send_verification_code, user.get("phone"), code, user["full_name"]
    )
    if not delivered:
        # Do not leave a live code behind that nobody received.
        otp.invalidate_outstanding(user["id"])
        activity.record("login.code_undelivered", user=user, detail=channel,
                        ip=client_address(request), outcome="error")
        raise HTTPException(status_code=503, detail=detail)

    activity.record("login.code_sent", user=user, detail=f"channel={channel}",
                    ip=client_address(request))
    return CodeRequestResponse(sent=True, channel=channel, detail=detail,
                               expires_in_seconds=otp.CODE_TTL_SECONDS)


@router.post("/code", response_model=CodeRequestResponse,
             dependencies=[Depends(limit_by_address("auth-code", 12, 900,
                                                    "verification code requests"))])
async def request_clinician_code(req: ClinicianCodeRequest, request: Request):
    """Send a one-time code to the phone registered against a clinician account."""
    user = await run_in_threadpool(accounts.identify_clinician,
                                   req.email, req.radintel_id)
    if not user:
        activity.record("login.identify_failed", actor=req.email.strip().lower(),
                        ip=client_address(request), outcome="refused",
                        detail="clinician email + Rad-Intel ID")
        raise HTTPException(status_code=401, detail=_NO_MATCH)
    return await _issue_and_send(user, request)


@router.post("/admin/code", response_model=CodeRequestResponse,
             dependencies=[Depends(limit_by_address("auth-admin-code", 10, 900,
                                                    "administrator sign-in attempts"))])
async def request_admin_code(req: AdminCodeRequest, request: Request):
    """Send a one-time code to the phone registered against the administrator account."""
    user = await run_in_threadpool(accounts.identify_admin, req.email, req.password)
    if not user:
        activity.record("login.admin_identify_failed", actor=req.email.strip().lower(),
                        ip=client_address(request), outcome="refused")
        raise HTTPException(status_code=401, detail=_NO_MATCH_ADMIN)
    return await _issue_and_send(user, request)


@router.post("/login", response_model=AuthMeResponse,
             dependencies=[Depends(limit_by_address("auth-login", 20, 900,
                                                    "sign-in attempts"))])
async def login(req: VerifyCodeRequest, request: Request, response: Response):
    """Exchange a verification code for a session."""
    user = await run_in_threadpool(accounts.get_by_email, req.email)
    if not user or user["status"] != "active":
        # Still spend a comparable amount of work, so a non-existent address
        # cannot be told apart from a wrong code by timing.
        await run_in_threadpool(otp.verify_code, "no-such-account", req.code)
        activity.record("login.failed", actor=req.email.strip().lower(),
                        ip=client_address(request), outcome="refused")
        raise HTTPException(status_code=401, detail=_NO_MATCH)

    ok, message = await run_in_threadpool(otp.verify_code, user["id"], req.code)
    if not ok:
        activity.record("login.code_rejected", user=user, detail=message,
                        ip=client_address(request), outcome="refused")
        raise HTTPException(status_code=401, detail=message)

    def _open_session() -> str:
        accounts.note_successful_login(user["id"])
        return session.issue(user["id"], ip=client_address(request),
                             user_agent=request.headers.get("user-agent"))

    token = await run_in_threadpool(_open_session)
    response.set_cookie(
        key=session.COOKIE_NAME,
        value=token,
        # No max_age and no expires: a session cookie, which the browser
        # discards when it closes. The server-side idle timeout is the
        # backstop for browsers that restore their previous tabs.
        httponly=True,       # not readable from JavaScript
        samesite="lax",      # not sent on cross-site requests
        secure=_cookie_secure(),
        path="/",
    )
    activity.record("login.success", user=user, ip=client_address(request))
    return _me(user)


@router.post("/logout")
async def logout(response: Response, request: Request,
                 token: str | None = Depends(session_token),
                 user: dict | None = Depends(current_clinician)):
    """End this session only. Other devices stay signed in."""
    await run_in_threadpool(session.destroy, token)
    response.delete_cookie(session.COOKIE_NAME, path="/")
    if user:
        activity.record("logout", user=user, ip=client_address(request))
    return {"signed_out": True}


@router.get("/me", response_model=AuthMeResponse)
async def me(user: dict | None = Depends(current_clinician)):
    """
    The signed-in account's own details.

    These are shown to the account holder alone. No endpoint returns another
    clinician's email, phone number or Rad-Intel ID to anyone but an
    administrator.
    """
    if not user:
        return AuthMeResponse(authenticated=False)
    return _me(user)


@router.post("/password")
async def change_password(req: ChangePasswordRequest, request: Request,
                          user: dict = Depends(require_clinician)):
    """
    Change the signed-in account's password.

    Only administrator accounts have a password. A clinician's first factor is
    the Rad-Intel ID their administrator issued, which only the administrator
    can rotate, so there is nothing here for them to change.
    """
    if user["role"] != "admin":
        raise HTTPException(
            status_code=400,
            detail=("Clinician accounts have no password. Your first factor is your "
                    "Rad-Intel ID, which the study administrator issues and can "
                    "replace if it is lost."),
        )
    ok, message = await run_in_threadpool(accounts.change_own_password, user["id"],
                                          req.current_password, req.new_password)
    if not ok:
        activity.record("admin.password_change_failed", user=user,
                        ip=client_address(request), outcome="refused")
        raise HTTPException(status_code=400, detail=message)
    activity.record("admin.password_changed", user=user, ip=client_address(request))
    return {"changed": True, "detail": message}


@router.post("/request-access",
             dependencies=[Depends(limit_by_address("auth-request-access", 4, 3600,
                                                    "access requests"))])
async def request_access(req: AccessRequestSubmission, request: Request):
    """
    Record a request to join the study. Creates no account and grants no access.

    The details go to the administrator, who verifies the clinician's
    registration independently before issuing anything. The phone number is
    required because it is where sign-in codes would be sent, and it is stored
    encrypted.
    """
    missing = [
        label for label, value in (
            ("full name", req.full_name), ("email", req.email),
            ("institution", req.institution), ("phone number", req.phone),
        ) if not (value or "").strip()
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Please give your {', '.join(missing)}.",
        )
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="That email address looks incomplete.")
    digits = [c for c in req.phone if c.isdigit()]
    if len(digits) < 8:
        raise HTTPException(
            status_code=400,
            detail=("Give the phone number in full, with its country code — for example "
                    "+91 98765 43210. Sign-in codes are sent to it."),
        )

    request_id = await run_in_threadpool(
        accounts.create_access_request, req.full_name, req.email, req.institution,
        req.phone, req.registration, req.message,
    )
    reference = request_id[:8]

    emailed, email_detail = await run_in_threadpool(
        notify.send_access_request_email,
        req.full_name.strip(), req.email.strip().lower(), req.institution.strip(),
        req.phone.strip(), (req.registration or "").strip() or None,
        (req.message or "").strip() or None, reference,
    )
    if emailed:
        accounts.mark_request_notified(request_id)
    activity.record("access_request.submitted", actor=req.email.strip().lower(),
                    detail=f"ref={reference} emailed={emailed}",
                    ip=client_address(request))

    return {
        "submitted": True,
        "reference": reference,
        # Whether the notification email went out is the administrator's
        # problem, not the applicant's: the request is recorded either way and
        # is waiting in the console.
        "notified": emailed,
        "notify_detail": email_detail,
        "detail": (
            "Request recorded. The study administrator verifies every applicant's "
            "registration personally and will contact you on the number or email you "
            "gave. No account exists, and no access has been granted, until they "
            "create one."
        ),
    }
