"""FastAPI dependencies that gate the review workspace."""

from fastapi import Depends, HTTPException, Request

from rad_intel.auth import accounts, session


def session_token(request: Request) -> str | None:
    """The raw cookie value, for signing out one session specifically."""
    return request.cookies.get(session.COOKIE_NAME)


def current_clinician(request: Request) -> dict | None:
    """
    The signed-in account, or None. Never raises.

    The returned dict carries `session_mode`: which door this session was
    opened by. It is read from the session row, never from the request, so
    there is nothing a caller can set to award themselves a different one.
    """
    found = session.lookup(request.cookies.get(session.COOKIE_NAME))
    if not found:
        return None
    user = accounts.get_by_id(found["user_id"])
    # A disabled or deleted account must lose access immediately, not at
    # session expiry. The session row is dropped too, so the stale cookie
    # stops costing a lookup on every subsequent request.
    if not user or user["status"] != "active":
        session.destroy_all_for_user(found["user_id"])
        return None
    return {**user, "session_mode": found["mode"]}


def require_clinician(user: dict | None = Depends(current_clinician)) -> dict:
    """Reject anyone who is not a signed-in, active clinician."""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Log in with a clinician account to use the review workspace.",
        )
    return user


def is_admin_session(user: dict | None) -> bool:
    """
    An administrator *and* signed in as one.

    Both halves matter. The role says the account is allowed the console; the
    mode says they proved it with the administrator password rather than with
    a Rad-Intel ID, which is the thing an administrator writes down and hands
    to people. Without the second half, an ID read off a desk would reach the
    controls that can read and delete every clinician's data.
    """
    return bool(user and user.get("role") == "admin"
                and user.get("session_mode") == "admin")


def require_admin(user: dict = Depends(require_clinician)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required.")
    if user.get("session_mode") != "admin":
        raise HTTPException(
            status_code=403,
            detail=("You are logged in as a clinician. Log out and log back in through "
                    "Administrator to reach the console — your Rad-Intel ID gets you a "
                    "clinician session, and administrator controls need the password."),
        )
    return user
