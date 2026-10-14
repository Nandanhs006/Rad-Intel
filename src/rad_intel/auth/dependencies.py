"""FastAPI dependencies that gate the review workspace."""

from fastapi import Depends, HTTPException, Request

from rad_intel.auth import accounts, session


def current_clinician(request: Request) -> dict | None:
    """The signed-in account, or None. Never raises."""
    user_id = session.verify(request.cookies.get(session.COOKIE_NAME))
    if not user_id:
        return None
    user = accounts.get_by_id(user_id)
    # A disabled account must lose access immediately, not at token expiry.
    if not user or user["status"] != "active":
        return None
    return user


def require_clinician(user: dict | None = Depends(current_clinician)) -> dict:
    """Reject anyone who is not a signed-in, active clinician."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Sign in with a clinician account to use the review workspace.",
        )
    return user


def require_admin(user: dict = Depends(require_clinician)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    return user
