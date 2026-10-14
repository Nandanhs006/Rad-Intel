"""
Sign-in for the closed review workspace.

There is no registration endpoint on purpose. Accounts are created by an
administrator with scripts/manage_clinicians.py. The only thing an outsider
can do here is submit an access request, which records their details for the
administrator and grants nothing.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from rad_intel.api.schemas import (
    AccessRequestSubmission,
    AuthMeResponse,
    ChangePasswordRequest,
    LoginRequest,
)
from rad_intel.auth import accounts, session
from rad_intel.auth.dependencies import current_clinician, require_clinician

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post("/login", response_model=AuthMeResponse)
async def login(req: LoginRequest, response: Response):
    user = accounts.authenticate(req.email, req.password)
    if not user:
        # One message for every failure mode, so this cannot be used to find
        # out which addresses have accounts.
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    response.set_cookie(
        key=session.COOKIE_NAME,
        value=session.issue(user["id"]),
        max_age=session.SESSION_TTL_SECONDS,
        httponly=True,       # not readable from JavaScript
        samesite="lax",      # not sent on cross-site requests
        secure=False,        # set True behind HTTPS
        path="/",
    )
    return AuthMeResponse(
        authenticated=True,
        email=user["email"],
        full_name=user["full_name"],
        institution=user["institution"],
        role=user["role"],
        must_change_password=bool(user["must_change_password"]),
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(session.COOKIE_NAME, path="/")
    return {"signed_out": True}


@router.get("/me", response_model=AuthMeResponse)
async def me(user: dict | None = Depends(current_clinician)):
    if not user:
        return AuthMeResponse(authenticated=False)
    return AuthMeResponse(
        authenticated=True,
        email=user["email"],
        full_name=user["full_name"],
        institution=user["institution"],
        role=user["role"],
        must_change_password=bool(user["must_change_password"]),
    )


@router.post("/password")
async def change_password(req: ChangePasswordRequest,
                          user: dict = Depends(require_clinician)):
    ok, message = accounts.change_own_password(user["id"], req.current_password,
                                               req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"changed": True, "detail": message}


@router.post("/request-access")
async def request_access(req: AccessRequestSubmission, request: Request):
    """Record a request to join. Creates no account and grants no access."""
    if not req.full_name.strip() or not req.email.strip() or not req.institution.strip():
        raise HTTPException(status_code=400, detail="Name, email and institution are required.")

    request_id = accounts.create_access_request(
        full_name=req.full_name,
        email=req.email,
        institution=req.institution,
        registration=req.registration,
        message=req.message,
    )
    return {
        "submitted": True,
        "reference": request_id[:8],
        "detail": (
            "Request recorded. The study administrator reviews each one and will "
            "contact you at the address you gave. No account exists until they "
            "create one."
        ),
    }
