"""
Administer the closed clinician workspace.

There is no sign-up page. Accounts exist only because they were created here,
which is what keeps the review workspace a named group rather than an open
service.

    python scripts/manage_clinicians.py add    --email a@b.org --name "Dr A" --institution "X Hospital"
    python scripts/manage_clinicians.py list
    python scripts/manage_clinicians.py requests
    python scripts/manage_clinicians.py reset   --email a@b.org
    python scripts/manage_clinicians.py disable --email a@b.org
    python scripts/manage_clinicians.py enable  --email a@b.org
"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rad_intel.auth import accounts  # noqa: E402


def _temp_password() -> str:
    """A readable one-off password the clinician changes on first sign-in."""
    words = ("harbour", "lantern", "meadow", "pilot", "quartz", "ridge",
             "saffron", "timber", "violet", "willow", "cobalt", "ember")
    return f"{secrets.choice(words)}-{secrets.choice(words)}-{secrets.randbelow(9000) + 1000}"


def cmd_add(args: argparse.Namespace) -> None:
    password = args.password or _temp_password()
    try:
        user = accounts.create_clinician(args.email, args.name, args.institution,
                                         password, role=args.role)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"\nAccount created for {user['full_name']} ({user['email']})")
    print(f"  institution : {user['institution']}")
    print(f"  role        : {user['role']}")
    print(f"  password    : {password}")
    print("\nSend these by a channel the clinician already trusts, not in the same")
    print("email as the link. They are prompted to change it on first sign-in.")


def cmd_list(_args: argparse.Namespace) -> None:
    rows = accounts.list_clinicians()
    if not rows:
        print("No accounts yet. Create one with: manage_clinicians.py add --help")
        return
    print(f"\n{'EMAIL':<34}{'NAME':<22}{'INSTITUTION':<26}{'ROLE':<11}{'STATUS':<10}LAST SIGN-IN")
    print("-" * 125)
    for r in rows:
        last = (r["last_login_at"] or "never")[:16].replace("T", " ")
        print(f"{r['email']:<34}{r['full_name']:<22}{r['institution'][:24]:<26}"
              f"{r['role']:<11}{r['status']:<10}{last}")
    print()


def cmd_requests(args: argparse.Namespace) -> None:
    rows = accounts.list_access_requests(status=args.status)
    if not rows:
        print("No access requests.")
        return
    for r in rows:
        print(f"\n[{r['status']}] {r['created_at'][:16].replace('T', ' ')}  ref {r['id'][:8]}")
        print(f"  {r['full_name']} <{r['email']}>")
        print(f"  {r['institution']}" + (f"  ·  reg {r['registration']}" if r["registration"] else ""))
        if r["message"]:
            print(f"  \"{r['message']}\"")
        print(f"  approve with: python scripts/manage_clinicians.py add --email {r['email']} "
              f"--name \"{r['full_name']}\" --institution \"{r['institution']}\"")
    print()


def cmd_reset(args: argparse.Namespace) -> None:
    password = args.password or _temp_password()
    if not accounts.set_password(args.email, password):
        sys.exit(f"No account for {args.email}")
    print(f"Password reset for {args.email}\n  new password: {password}")
    print("They are prompted to change it on next sign-in.")


def cmd_status(args: argparse.Namespace, status: str) -> None:
    if not accounts.set_status(args.email, status):
        sys.exit(f"No account for {args.email}")
    print(f"{args.email} is now {status}."
          + (" Any open session stops working on the next request." if status == "disabled" else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="Create a clinician account")
    a.add_argument("--email", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--institution", required=True)
    a.add_argument("--password", help="Omit to generate a one-off password")
    a.add_argument("--role", default="clinician", choices=["clinician", "admin"])
    a.set_defaults(func=cmd_add)

    sub.add_parser("list", help="List accounts").set_defaults(func=cmd_list)

    r = sub.add_parser("requests", help="Show access requests")
    r.add_argument("--status", default=None, help="new | contacted | declined")
    r.set_defaults(func=cmd_requests)

    p = sub.add_parser("reset", help="Reset a password")
    p.add_argument("--email", required=True)
    p.add_argument("--password")
    p.set_defaults(func=cmd_reset)

    d = sub.add_parser("disable", help="Revoke access")
    d.add_argument("--email", required=True)
    d.set_defaults(func=lambda args: cmd_status(args, "disabled"))

    e = sub.add_parser("enable", help="Restore access")
    e.add_argument("--email", required=True)
    e.set_defaults(func=lambda args: cmd_status(args, "active"))

    args = ap.parse_args()
    accounts.init()
    args.func(args)


if __name__ == "__main__":
    main()
