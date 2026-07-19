#!/usr/bin/env python3
"""Ultrabull user management.

    python manage.py adduser <username>          # prompts for password
    python manage.py adduser <username> -p PASS  # non-interactive
    python manage.py passwd  <username>          # reset password
    python manage.py deluser <username>
    python manage.py listusers

On the server, run it inside the container so it hits the mounted DB:
    docker exec -it ultrabull python manage.py adduser abhishek
"""
import argparse
import getpass
import sys

from api.auth import create_user, delete_user, list_users, set_password


def _password(args) -> str:
    if getattr(args, "password", None):
        return args.password
    pw = getpass.getpass("Password: ")
    if getpass.getpass("Repeat  : ") != pw:
        sys.exit("Passwords do not match.")
    if len(pw) < 8:
        sys.exit("Use at least 8 characters.")
    return pw


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Ultrabull UI/API users")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("adduser", help="create a user")
    p_add.add_argument("username")
    p_add.add_argument("-p", "--password", help="password (omit to be prompted)")

    p_pw = sub.add_parser("passwd", help="reset a user's password")
    p_pw.add_argument("username")
    p_pw.add_argument("-p", "--password")

    p_del = sub.add_parser("deluser", help="delete a user")
    p_del.add_argument("username")

    sub.add_parser("listusers", help="list users")

    args = parser.parse_args()
    try:
        if args.cmd == "adduser":
            create_user(args.username, _password(args))
            print(f"created '{args.username.strip().lower()}'")
        elif args.cmd == "passwd":
            set_password(args.username, _password(args))
            print(f"password updated for '{args.username.strip().lower()}'")
        elif args.cmd == "deluser":
            delete_user(args.username)
            print(f"deleted '{args.username.strip().lower()}'")
        elif args.cmd == "listusers":
            users = list_users()
            if not users:
                print("no users yet — add one with: python manage.py adduser <name>")
            for name, created in users:
                print(f"{name}\t(created {created})")
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
