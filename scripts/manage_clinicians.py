"""
Administer the closed clinician workspace from the command line.

There is no sign-up page. Accounts exist only because they were created here
or in the administrator console, which is what keeps the review workspace a
named group rather than an open service.

A clinician logs in with their email address, the Rad-Intel ID printed by
`add`, and a one-time code sent to their registered phone. Hand the ID over in
person or by a channel they already trust -- not in the same message as the
link to the workspace, which would put both factors in one place.

    python scripts/manage_clinicians.py add --email a@b.org --name "Dr A" \
        --institution "X Hospital" --phone "+91 98765 43210"
    python scripts/manage_clinicians.py add --email me@x.org --name "Me" \
        --institution "MCE" --phone "+91 ..." --role admin --password "..."
    python scripts/manage_clinicians.py list
    python scripts/manage_clinicians.py show     --email a@b.org
    python scripts/manage_clinicians.py requests
    python scripts/manage_clinicians.py new-id   --email a@b.org
    python scripts/manage_clinicians.py phone    --email a@b.org --phone "+91 ..."
    python scripts/manage_clinicians.py password --email me@x.org
    python scripts/manage_clinicians.py disable  --email a@b.org
    python scripts/manage_clinicians.py enable   --email a@b.org
    python scripts/manage_clinicians.py delete   --email a@b.org
    python scripts/manage_clinicians.py migrate
    python scripts/manage_clinicians.py delivery
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rad_intel.auth import accounts, notify, session  # noqa: E402


def _temp_password() -> str:
    """A readable one-off password for an administrator account."""
    words = ("harbour", "lantern", "meadow", "pilot", "quartz", "ridge",
             "saffron", "timber", "violet", "willow", "cobalt", "ember")
    return f"{secrets.choice(words)}-{secrets.choice(words)}-{secrets.randbelow(9000) + 1000}"


def _require_phone(phone: str | None) -> str:
    digits = [c for c in (phone or "") if c.isdigit()]
    if len(digits) < 8:
        sys.exit("A phone number with its country code is required: sign-in codes go to it.\n"
                 "  e.g. --phone \"+91 98765 43210\"")
    return phone  # type: ignore[return-value]


def cmd_add(args: argparse.Namespace) -> None:
    phone = _require_phone(args.phone)
    password = None
    if args.role == "admin":
        password = args.password or _temp_password()
    try:
        user = accounts.create_clinician(
            args.email, args.name, args.institution, password,
            role=args.role, phone=phone, admin_note=args.note,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"\nAccount created for {user['full_name']} ({user['email']})")
    print(f"  institution  : {user['institution']}")
    print(f"  role         : {user['role']}")
    print(f"  phone        : {user['phone']}")
    print(f"  Rad-Intel ID : {user['radintel_id']}")
    if password:
        print(f"  password     : {password}  (administrator account; change it after signing in)")
    print("\nThey log in with their email address, that Rad-Intel ID, and a one-time")
    print("code sent to the phone number above. Give them the ID by a channel they")
    print("already trust, and not in the same message as the workspace link.")
    if not notify.sms_configured():
        print("\n  Note: no message provider is configured, so codes cannot be sent to a")
        print("  phone yet. Run `manage_clinicians.py delivery` to see what is missing.")


def cmd_list(_args: argparse.Namespace) -> None:
    rows = accounts.list_clinicians()
    if not rows:
        print("No accounts yet. Create one with: manage_clinicians.py add --help")
        return
    print(f"\n{'EMAIL':<30}{'NAME':<20}{'INSTITUTION':<22}{'RAD-INTEL ID':<15}"
          f"{'ROLE':<10}{'STATUS':<10}LAST LOGIN")
    print("-" * 132)
    for r in rows:
        last = (r["last_login_at"] or "never")[:16].replace("T", " ")
        print(f"{r['email'][:28]:<30}{r['full_name'][:18]:<20}{r['institution'][:20]:<22}"
              f"{(r.get('radintel_id') or '— none —'):<15}{r['role']:<10}"
              f"{r['status']:<10}{last}")
    print()


def cmd_show(args: argparse.Namespace) -> None:
    user = accounts.get_by_email(args.email) if args.email else accounts.get_by_radintel_id(args.id)
    if not user:
        sys.exit("No matching account.")
    for key, value in accounts.self_view(user).items():
        print(f"  {key:<14}: {value}")
    print(f"  {'admin note':<14}: {user.get('admin_note') or '—'}")


def cmd_requests(args: argparse.Namespace) -> None:
    rows = accounts.list_access_requests(status=args.status)
    if not rows:
        print("No access requests.")
        return
    for r in rows:
        print(f"\n[{r['status']}] {r['created_at'][:16].replace('T', ' ')}  ref {r['id'][:8]}")
        print(f"  {r['full_name']} <{r['email']}>")
        print(f"  {r['institution']}" + (f"  ·  reg {r['registration']}" if r["registration"] else ""))
        print(f"  phone: {r.get('phone') or 'not given'}"
              f"   (emailed to admin: {'yes' if r.get('notified') else 'no'})")
        if r["message"]:
            print(f"  \"{r['message']}\"")
        print("  verify their registration independently, then:")
        print(f"    python scripts/manage_clinicians.py add --email {r['email']} "
              f"--name \"{r['full_name']}\" --institution \"{r['institution']}\" "
              f"--phone \"{r.get('phone') or ''}\"")
    print()


def cmd_new_id(args: argparse.Namespace) -> None:
    user = accounts.get_by_email(args.email)
    if not user:
        sys.exit(f"No account for {args.email}")
    new = accounts.rotate_radintel_id(user["id"])
    closed = session.destroy_all_for_user(user["id"])
    print(f"New Rad-Intel ID for {user['full_name']}: {new}")
    print(f"The previous ID no longer works" +
          (f" and {closed} open session(s) were closed." if closed else "."))
    print("Pass the new ID to them yourself.")


def cmd_phone(args: argparse.Namespace) -> None:
    phone = _require_phone(args.phone)
    user = accounts.get_by_email(args.email)
    if not user:
        sys.exit(f"No account for {args.email}")
    accounts.update_clinician(user["id"], phone=phone)
    print(f"Phone number updated for {user['email']}. Codes now go to {phone}.")


def cmd_password(args: argparse.Namespace) -> None:
    user = accounts.get_by_email(args.email)
    if not user:
        sys.exit(f"No account for {args.email}")
    if user["role"] != "admin":
        sys.exit("Only administrator accounts have a password. A clinician's first "
                 "factor is their Rad-Intel ID; reissue it with `new-id`.")
    password = args.password or _temp_password()
    accounts.set_password(args.email, password)
    print(f"Password reset for {args.email}\n  new password: {password}")


def cmd_status(args: argparse.Namespace, status: str) -> None:
    user = accounts.get_by_email(args.email)
    if not user or not accounts.set_status(args.email, status):
        sys.exit(f"No account for {args.email}")
    if status == "disabled":
        closed = session.destroy_all_for_user(user["id"])
        print(f"{args.email} is now disabled. {closed} open session(s) closed; they "
              f"cannot log in again.")
    else:
        print(f"{args.email} is active again.")


def cmd_delete(args: argparse.Namespace) -> None:
    user = accounts.get_by_email(args.email)
    if not user:
        sys.exit(f"No account for {args.email}")
    if user["role"] == "admin" and len(accounts.admins()) <= 1:
        sys.exit("That is the only active administrator. Create another one first.")
    if not args.yes:
        confirm = input(f"Delete the account for {user['full_name']} <{user['email']}>? "
                        f"Their reviews stay in the queue. [y/N] ")
        if confirm.strip().lower() not in ("y", "yes"):
            sys.exit("Cancelled.")
    session.destroy_all_for_user(user["id"])
    accounts.delete_clinician(user["id"])
    print(f"Deleted {user['email']}. Their reviews remain in the queue, attributed to "
          f"their name, because other participants have already seen them.")


def cmd_migrate(_args: argparse.Namespace) -> None:
    """Give a Rad-Intel ID to accounts created before IDs existed."""
    issued = []
    for user in accounts.list_clinicians():
        if not user.get("radintel_id"):
            new = accounts.rotate_radintel_id(user["id"])
            issued.append((user, new))
    if not issued:
        print("Every account already has a Rad-Intel ID.")
    else:
        print("Issued Rad-Intel IDs:\n")
        for user, new in issued:
            print(f"  {user['email']:<34}{new}")
        print("\nPass each one to its owner. They cannot log in without it.")

    missing_phone = [u for u in accounts.list_clinicians() if not u.get("phone")]
    if missing_phone:
        print("\nThese accounts have no phone number, so no sign-in code can reach them:")
        for user in missing_phone:
            print(f"  {user['email']}")
            print(f"    python scripts/manage_clinicians.py phone --email {user['email']} "
                  f"--phone \"+91 ...\"")


def cmd_delivery(_args: argparse.Namespace) -> None:
    status = notify.delivery_status()
    print("\nDelivery configuration\n" + "-" * 60)
    for key, value in status.items():
        print(f"  {key:<22}: {value}")
    print()
    if not status["sms_configured"]:
        print("One-time codes cannot be sent to a phone. Set, in .env:")
        print("    TWILIO_ACCOUNT_SID=...")
        print("    TWILIO_AUTH_TOKEN=...")
        print("    TWILIO_FROM=+1...            # SMS sender")
        print("    TWILIO_WHATSAPP_FROM=whatsapp:+14155238886   # for WhatsApp instead")
        print("\nOr, to try the workspace on this machine with no provider at all:")
        print("    RAD_INTEL_OTP_ECHO=1        # writes codes to the server log only")
        print("  That is a local test mode. Remove it before anyone else uses the site.\n")
    if not status["email_configured"]:
        print(f"Access requests cannot be emailed to {status['admin_email']}. Set, in .env:")
        print("    SMTP_HOST=smtp.gmail.com")
        print("    SMTP_PORT=587")
        print("    SMTP_USER=you@gmail.com")
        print("    SMTP_PASSWORD=...           # a Gmail app password, not your login")
        print("    SMTP_FROM=you@gmail.com")
        print("\nRequests are recorded either way and are listed by the `requests` "
              "command and in the console.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="Create an account and print its Rad-Intel ID")
    a.add_argument("--email", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--institution", required=True)
    a.add_argument("--phone", required=True, help="With country code, e.g. \"+91 98765 43210\"")
    a.add_argument("--role", default="clinician", choices=["clinician", "admin"])
    a.add_argument("--password", help="Administrator accounts only; omit to generate one")
    a.add_argument("--note", help="Private note for the administrator")
    a.set_defaults(func=cmd_add)

    sub.add_parser("list", help="List accounts").set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="Full details of one account")
    sh.add_argument("--email")
    sh.add_argument("--id", help="Rad-Intel ID, in full")
    sh.set_defaults(func=cmd_show)

    r = sub.add_parser("requests", help="Show access requests")
    r.add_argument("--status", default=None, help="new | contacted | approved | declined")
    r.set_defaults(func=cmd_requests)

    n = sub.add_parser("new-id", help="Issue a replacement Rad-Intel ID")
    n.add_argument("--email", required=True)
    n.set_defaults(func=cmd_new_id)

    ph = sub.add_parser("phone", help="Change the number codes are sent to")
    ph.add_argument("--email", required=True)
    ph.add_argument("--phone", required=True)
    ph.set_defaults(func=cmd_phone)

    p = sub.add_parser("password", help="Reset an administrator's password")
    p.add_argument("--email", required=True)
    p.add_argument("--password")
    p.set_defaults(func=cmd_password)

    d = sub.add_parser("disable", help="Revoke access immediately")
    d.add_argument("--email", required=True)
    d.set_defaults(func=lambda args: cmd_status(args, "disabled"))

    e = sub.add_parser("enable", help="Restore access")
    e.add_argument("--email", required=True)
    e.set_defaults(func=lambda args: cmd_status(args, "active"))

    rm = sub.add_parser("delete", help="Delete an account")
    rm.add_argument("--email", required=True)
    rm.add_argument("--yes", action="store_true", help="Skip the confirmation")
    rm.set_defaults(func=cmd_delete)

    sub.add_parser("migrate", help="Issue IDs to accounts that predate them"
                   ).set_defaults(func=cmd_migrate)
    sub.add_parser("delivery", help="Show how codes and emails are delivered"
                   ).set_defaults(func=cmd_delivery)

    args = ap.parse_args()
    accounts.init()
    args.func(args)


if __name__ == "__main__":
    main()
