"""
The administrator's console.

Everything here is behind `require_admin`, and administrator is a property of
an account rather than of a request, so there is no header, token or query
parameter that can turn a clinician into one.

The console is what makes the closed group workable in practice: accounts are
issued by hand, so somebody has to be able to see who has one, search for a
person by name or Rad-Intel ID when they phone up having lost it, issue a
replacement, and revoke access. It also exposes the activity log and the open
sessions, which are the two things that make "every clinician can see every
case" accountable rather than merely permitted.

Two guards are enforced rather than left to care:

* The administrator cannot delete or disable their own account, which would
  lock the only person who can issue accounts out of the system.
* The last active administrator cannot be removed or demoted, for the same
  reason.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from rad_intel.api.ratelimit import client_address, limit
from rad_intel.api.dependencies import ModelManager, get_model_manager
from rad_intel.api.schemas import (
    AdminProfileUpdate,
    AdminUserCreate,
    AdminUserUpdate,
    ModelReleaseRequest,
)
from rad_intel.models import registry
from rad_intel.auth import accounts, activity, notify, session
from rad_intel.auth.dependencies import require_admin

router = APIRouter(prefix="/api/v1/admin", tags=["Administration"])

CONTACT_NOTE = (
    "Accounts, Rad-Intel IDs and anything that needs correcting are handled by the "
    "study administrator personally. If a clinician has a problem — a lost ID, a "
    "changed phone number, a case uploaded by mistake, a question about how their "
    "data is used — they should contact the administrator directly using the details "
    "they were given when their account was issued."
)


def _admin_view(account: dict) -> dict:
    """Everything the administrator may see about an account, hash material aside."""
    return {
        "id": account["id"],
        "full_name": account["full_name"],
        "email": account["email"],
        "institution": account["institution"],
        "phone": account.get("phone"),
        "radintel_id": account.get("radintel_id"),
        "role": account["role"],
        "status": account["status"],
        "admin_note": account.get("admin_note"),
        "created_at": account["created_at"],
        "last_login_at": account.get("last_login_at"),
    }


def _active_admin_count() -> int:
    return len(accounts.admins())


@router.get("/overview")
async def overview(admin: dict = Depends(require_admin)):
    """Headline numbers plus whether code and email delivery are actually working."""
    everyone = accounts.list_clinicians()
    every_request = accounts.list_access_requests()
    requests_waiting = [r for r in every_request if r["status"] == "new"]
    # Requests that never reached an inbox. A misconfigured or expired SMTP
    # password fails quietly from the applicant's side -- they are told the
    # administrator will be in touch either way, which is true -- so the only
    # place it can surface is here.
    unemailed = [r for r in every_request if not r.get("notified")]
    return {
        "contact_note": CONTACT_NOTE,
        "users": {
            "total": len(everyone),
            "active": sum(1 for u in everyone if u["status"] == "active"),
            "disabled": sum(1 for u in everyone if u["status"] != "active"),
            "admins": sum(1 for u in everyone if u["role"] == "admin"),
        },
        "access_requests_waiting": len(requests_waiting),
        "access_requests_unemailed": len(unemailed),
        "open_sessions": len(session.list_active()),
        "activity_entries": activity.count(),
        "delivery": notify.delivery_status(),
    }


# --- accounts --------------------------------------------------------------

@router.get("/users")
async def list_users(q: str | None = None, admin: dict = Depends(require_admin)):
    """
    Every account, or the ones matching `q`.

    A Rad-Intel ID has to be given in full because IDs are stored encrypted
    and found through a keyed index; names, emails and institutions are
    substring matches.
    """
    found = accounts.search_clinicians(q)
    return {"query": q or "", "count": len(found),
            "users": [_admin_view(u) for u in found]}


@router.get("/new-id")
async def new_id(admin: dict = Depends(require_admin)):
    """
    A fresh, unused Rad-Intel ID for the add-user form.

    Generated here rather than in the browser so it is checked against the
    ones already issued, and so the alphabet stays in one place.
    """
    return {"radintel_id": accounts.generate_radintel_id()}


@router.post("/users", status_code=201,
             dependencies=[Depends(limit("admin-create-user", 40, 3600, "new accounts"))])
async def create_user(req: AdminUserCreate, request: Request,
                      admin: dict = Depends(require_admin)):
    """
    Issue an account.

    The response carries the Rad-Intel ID in the clear. It is the only time it
    is returned outside the administrator's own view of that account, and it
    has to be passed to the clinician by hand — not emailed alongside a link
    to the workspace, which would put both factors in one place.
    """
    if req.role not in ("clinician", "admin"):
        raise HTTPException(status_code=400, detail="Role must be clinician or admin.")
    digits = [c for c in (req.phone or "") if c.isdigit()]
    if len(digits) < 8:
        raise HTTPException(
            status_code=400,
            detail=("A phone number with its country code is required — sign-in codes "
                    "are sent to it. For example +91 98765 43210."),
        )
    try:
        created = accounts.create_clinician(
            email=req.email, full_name=req.full_name, institution=req.institution,
            password=req.password, role=req.role, phone=req.phone,
            radintel_id=req.radintel_id, admin_note=req.admin_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    activity.record("admin.user_created", user=admin, ip=client_address(request),
                    detail=f"{created['email']} ({created['role']})")
    return {
        "user": _admin_view(created),
        "radintel_id": created["radintel_id"],
        "detail": (
            f"Account created. Give {created['full_name']} their Rad-Intel ID "
            f"({created['radintel_id']}) in person or by a channel they already trust. "
            f"They log in with their email address, that ID, and a one-time code sent "
            f"to the registered number."
        ),
    }


@router.patch("/users/{user_id}")
async def update_user(user_id: str, req: AdminUserUpdate, request: Request,
                      admin: dict = Depends(require_admin)):
    """Correct an account's details, change its role, or disable it."""
    target = accounts.get_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="No such account.")

    if req.role and req.role not in ("clinician", "admin"):
        raise HTTPException(status_code=400, detail="Role must be clinician or admin.")
    if req.status and req.status not in ("active", "disabled"):
        raise HTTPException(status_code=400, detail="Status must be active or disabled.")

    losing_admin = (target["role"] == "admin" and target["status"] == "active"
                    and (req.role == "clinician" or req.status == "disabled"))
    if losing_admin:
        if target["id"] == admin["id"]:
            raise HTTPException(
                status_code=400,
                detail="You cannot disable or demote your own administrator account.",
            )
        if _active_admin_count() <= 1:
            raise HTTPException(
                status_code=400,
                detail="This is the only active administrator. Promote another first.",
            )

    try:
        updated = accounts.update_clinician(user_id, **req.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail="No such account.")

    # Revoking access has to take effect now, not when a session happens to
    # expire, so any session the account holds is closed immediately.
    if updated["status"] != "active":
        session.destroy_all_for_user(user_id)

    activity.record("admin.user_updated", user=admin, ip=client_address(request),
                    detail=f"{updated['email']}: "
                           f"{', '.join(req.model_dump(exclude_unset=True).keys())}")
    return {"user": _admin_view(updated)}


@router.post("/users/{user_id}/rotate-id")
async def rotate_id(user_id: str, request: Request, admin: dict = Depends(require_admin)):
    """Issue a replacement Rad-Intel ID, e.g. for a clinician who has lost theirs."""
    target = accounts.get_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="No such account.")
    new = accounts.rotate_radintel_id(user_id)
    if not new:
        raise HTTPException(status_code=500, detail="Could not issue a new ID.")

    # The old ID must stop working everywhere at once, which means ending any
    # session opened with it.
    session.destroy_all_for_user(user_id)
    activity.record("admin.radintel_id_rotated", user=admin, ip=client_address(request),
                    detail=target["email"])
    return {
        "radintel_id": new,
        "detail": (f"New Rad-Intel ID issued for {target['full_name']}. The previous one "
                   f"no longer works and their open sessions have been closed. Pass the "
                   f"new ID to them yourself."),
    }


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, request: Request,
                      admin: dict = Depends(require_admin)):
    """
    Delete an account.

    The reviews that account contributed are deliberately *not* deleted with
    it. They are part of the shared material other participants have already
    reviewed and trained on, and silently removing them would change the
    provenance of a model that has already been built. Delete the records
    explicitly from the review queue if that is what is wanted.
    """
    target = accounts.get_by_id(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="No such account.")
    if target["id"] == admin["id"]:
        raise HTTPException(status_code=400,
                            detail="You cannot delete your own administrator account.")
    if target["role"] == "admin" and target["status"] == "active" and _active_admin_count() <= 1:
        raise HTTPException(status_code=400,
                            detail="This is the only active administrator.")

    session.destroy_all_for_user(user_id)
    accounts.delete_clinician(user_id)
    activity.record("admin.user_deleted", user=admin, ip=client_address(request),
                    detail=target["email"])
    return {
        "deleted": True,
        "detail": (f"{target['full_name']} can no longer log in. Their reviews remain in "
                   f"the queue, attributed to their name, because other participants "
                   f"have already seen and trained on them."),
    }


# --- own account -----------------------------------------------------------

@router.patch("/profile")
async def update_own_profile(req: AdminProfileUpdate, request: Request,
                             admin: dict = Depends(require_admin)):
    """Change the administrator's own name, email, institution or phone number."""
    fields = req.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="Nothing to change.")
    if "phone" in fields:
        digits = [c for c in (fields["phone"] or "") if c.isdigit()]
        if len(digits) < 8:
            raise HTTPException(
                status_code=400,
                detail=("Keep a reachable phone number with its country code: your "
                        "own sign-in codes are sent to it."),
            )
    try:
        updated = accounts.update_clinician(admin["id"], **fields)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    activity.record("admin.profile_updated", user=admin, ip=client_address(request),
                    detail=", ".join(fields.keys()))
    return {"user": _admin_view(updated)}


# --- access requests -------------------------------------------------------

@router.get("/requests")
async def list_requests(status: str | None = None, admin: dict = Depends(require_admin)):
    """
    People who have asked for access.

    Shown in full, including the phone number, because the administrator's job
    is to verify the applicant against their medical registration before
    issuing anything.
    """
    rows = accounts.list_access_requests(status=status)
    return {
        "count": len(rows),
        "requests": [{
            "id": r["id"],
            "reference": r["id"][:8],
            "created_at": r["created_at"],
            "full_name": r["full_name"],
            "email": r["email"],
            "institution": r["institution"],
            "phone": r.get("phone"),
            "registration": r.get("registration"),
            "message": r.get("message"),
            "status": r["status"],
            "emailed_to_admin": bool(r.get("notified")),
        } for r in rows],
    }


# --- releasing a model ------------------------------------------------------

@router.get("/models/releasable")
async def releasable_models(admin: dict = Depends(require_admin)):
    """
    Fine-tuned checkpoints on disk, their scores, and whether each is public.

    The base checkpoints are not here: they are built in, always offered, and
    cannot be released or withdrawn.
    """
    from rad_intel.api.routes.feedback import WEIGHTS_DIR, _checkpoint_meta

    rows = []
    for item in registry.candidates():
        meta = await run_in_threadpool(_checkpoint_meta, WEIGHTS_DIR / item["file"])
        after = meta.get("metrics_after") or {}
        provenance = meta.get("provenance") or {}
        rows.append({
            **item,
            "metrics": {k: v for k, v in after.items() if isinstance(v, (int, float))},
            "reviewed_cases": provenance.get("reviewed_cases"),
            "created_at": provenance.get("created_at"),
            "base_checkpoint": provenance.get("base_checkpoint"),
        })
    return {"candidates": rows, "released": registry.released()}


@router.post("/models/release",
             dependencies=[Depends(limit("admin-release", 20, 3600, "releases"))])
async def release_model(req: ModelReleaseRequest, request: Request,
                        manager: ModelManager = Depends(get_model_manager),
                        admin: dict = Depends(require_admin)):
    """
    Put a fine-tuned version on the public model list.

    The password is asked for every time, not once per session. Everything
    else an administrator does affects the study; this one puts a model in
    front of whoever visits the site, so it should take a deliberate act
    rather than a click on a page that happened to be left open.
    """
    if not await run_in_threadpool(accounts.identify_admin, admin["email"], req.password):
        activity.record("model.release_refused", user=admin, ip=client_address(request),
                        detail="wrong password", outcome="refused")
        raise HTTPException(status_code=403, detail="Password incorrect. Nothing was released.")

    from rad_intel.api.routes.feedback import WEIGHTS_DIR, _checkpoint_meta

    meta = await run_in_threadpool(_checkpoint_meta, WEIGHTS_DIR / Path(req.file).name)
    metrics = {k: v for k, v in (meta.get("metrics_after") or {}).items()
               if isinstance(v, (int, float))}
    if not metrics:
        raise HTTPException(
            status_code=400,
            detail=("That checkpoint carries no scores, so there is nothing to show "
                    "people about it. Releasing a model nobody can judge is worse "
                    "than not releasing one."),
        )

    try:
        entry = registry.release(req.file, released_by=admin["email"],
                                 metrics=metrics, note=req.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    manager.forget(entry["name"])
    activity.record("model.released", user=admin, ip=client_address(request),
                    detail=f"{entry['file']} as {entry['name']}")
    return {
        "released": entry["name"],
        "detail": (f"{entry['name']} is now on the public model list on the Analyse "
                   f"tab, shown with the scores it was measured at. The model the "
                   f"site uses by default has not changed."),
        "entry": entry,
    }


@router.post("/models/withdraw",
             dependencies=[Depends(limit("admin-release", 20, 3600, "releases"))])
async def withdraw_model(req: ModelReleaseRequest, request: Request,
                         manager: ModelManager = Depends(get_model_manager),
                         admin: dict = Depends(require_admin)):
    """Take a released version off the public list. The checkpoint is kept."""
    if not await run_in_threadpool(accounts.identify_admin, admin["email"], req.password):
        raise HTTPException(status_code=403, detail="Password incorrect. Nothing changed.")

    name = req.file
    if not registry.withdraw(name):
        raise HTTPException(status_code=404, detail=f"{name} is not on the public list.")
    manager.forget(name)
    activity.record("model.withdrawn", user=admin, ip=client_address(request), detail=name)
    return {"withdrawn": name,
            "detail": f"{name} is no longer offered. Its checkpoint is still on disk."}


@router.post("/test-email",
             dependencies=[Depends(limit("admin-test-email", 10, 600, "test emails"))])
async def send_test_email(request: Request, admin: dict = Depends(require_admin)):
    """
    Send a test message to the administrator's address.

    So SMTP can be checked after configuring it, rather than by submitting a
    fake access request and waiting to see whether anything arrives.
    """
    if not notify.email_configured():
        raise HTTPException(
            status_code=503,
            detail=("No SMTP server is configured, so there is nothing to test. Set "
                    "SMTP_HOST, SMTP_FROM and the rest in .env, restart, and try again."),
        )
    sent, detail = await run_in_threadpool(
        notify.send_email,
        "Rad-Intel: email delivery is working",
        ("This is a test from the Rad-Intel administrator console.\n\n"
         "If you are reading it, access requests will reach you at this address.\n"),
    )
    activity.record("admin.test_email", user=admin, ip=client_address(request),
                    detail=detail, outcome="ok" if sent else "error")
    if not sent:
        raise HTTPException(status_code=502, detail=detail)
    return {"sent": True, "detail": detail}


@router.post("/requests/{request_id}/resend")
async def resend_request_email(request_id: str, request: Request,
                               admin: dict = Depends(require_admin)):
    """Email a request that was recorded while mail delivery was not working."""
    match = next((r for r in accounts.list_access_requests() if r["id"] == request_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="No such request.")

    sent, detail = await run_in_threadpool(
        notify.send_access_request_email,
        match["full_name"], match["email"], match["institution"], match.get("phone"),
        match.get("registration"), match.get("message"), request_id[:8],
    )
    if sent:
        accounts.mark_request_notified(request_id)
    activity.record("admin.request_resent", user=admin, ip=client_address(request),
                    detail=f"{request_id[:8]} {detail}", outcome="ok" if sent else "error")
    if not sent:
        raise HTTPException(status_code=502, detail=detail)
    return {"sent": True, "detail": detail}


@router.patch("/requests/{request_id}")
async def set_request_status(request_id: str, status: str, request: Request,
                             admin: dict = Depends(require_admin)):
    """Mark a request contacted, approved or declined. Grants nothing by itself."""
    if status not in ("new", "contacted", "approved", "declined"):
        raise HTTPException(
            status_code=400,
            detail="Status must be new, contacted, approved or declined.",
        )
    if not accounts.set_request_status(request_id, status):
        raise HTTPException(status_code=404, detail="No such request.")
    activity.record("admin.request_status", user=admin, ip=client_address(request),
                    detail=f"{request_id[:8]} -> {status}")
    return {"updated": True, "status": status}


@router.delete("/requests/{request_id}")
async def delete_request(request_id: str, request: Request,
                         admin: dict = Depends(require_admin)):
    """Remove a request once it has been dealt with."""
    if not accounts.delete_access_request(request_id):
        raise HTTPException(status_code=404, detail="No such request.")
    activity.record("admin.request_deleted", user=admin, ip=client_address(request),
                    detail=request_id[:8])
    return {"deleted": True}


# --- oversight -------------------------------------------------------------

@router.get("/activity")
async def read_activity(limit_rows: int = 150, offset: int = 0,
                        user_id: str | None = None, action: str | None = None,
                        admin: dict = Depends(require_admin)):
    """The activity log, newest first."""
    rows = activity.recent(limit=limit_rows, offset=offset, user_id=user_id, action=action)
    return {"total": activity.count(), "entries": rows}


@router.get("/sessions")
async def open_sessions(admin: dict = Depends(require_admin)):
    """Who currently has the workspace open."""
    names = {u["id"]: u for u in accounts.list_clinicians()}
    rows = []
    for row in session.list_active():
        who = names.get(row["user_id"])
        rows.append({
            **row,
            "full_name": who["full_name"] if who else "(deleted account)",
            "email": who["email"] if who else None,
            "institution": who["institution"] if who else None,
            "is_you": row["user_id"] == admin["id"],
        })
    return {"count": len(rows), "sessions": rows}


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str, request: Request,
                        admin: dict = Depends(require_admin)):
    """Force one session to end, without disabling the account."""
    if not session.destroy_by_id(session_id):
        raise HTTPException(status_code=404, detail="No such session.")
    activity.record("admin.session_closed", user=admin, ip=client_address(request),
                    detail=session_id)
    return {"closed": True}
