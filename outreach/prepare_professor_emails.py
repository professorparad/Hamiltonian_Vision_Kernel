#!/usr/bin/env python3
"""Create one .eml draft per professor from a CSV contact list.

Usage:
  1. Fill the Email column in outreach/professors_for_outreach.csv.
  2. Edit outreach/professor_outreach_template.md if needed.
  3. Run: python3 outreach/prepare_professor_emails.py --sender "Your Name <you@example.com>"

The script writes individual .eml files into outreach/generated_emails/.
"""

from __future__ import annotations

import argparse
import csv
import email.message
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONTACTS = ROOT / "professors_for_outreach.csv"
DEFAULT_TEMPLATE = ROOT / "professor_outreach_template.md"
DEFAULT_OUTPUT = ROOT / "generated_emails"


def clean_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "professor"


def last_name(full_name: str) -> str:
    parts = full_name.replace("(", " ").replace(")", " ").split()
    return parts[-1] if parts else "Professor"


def shorten(value: str, max_len: int = 120) -> str:
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip(" ,;") + "..."


def render(template: str, row: dict[str, str], sender_name: str) -> tuple[str, str]:
    body = template
    replacements = {
        "Name": row.get("Name", ""),
        "LastName": last_name(row.get("Name", "")),
        "Institution": row.get("Institution", ""),
        "Department": row.get("Department", ""),
        "Focus": row.get("Focus", ""),
        "FocusShort": shorten(row.get("Focus", ""), 95),
        "Relevance": row.get("Relevance", ""),
        "RelevanceShort": shorten(row.get("Relevance", ""), 110),
        "YourName": sender_name,
    }
    for key, value in replacements.items():
        body = body.replace("{{" + key + "}}", value)

    subject = "Collaboration inquiry on Hamiltonian Vision Kernel"
    if body.startswith("Subject:"):
        first, _, rest = body.partition("\n")
        subject = first.replace("Subject:", "", 1).strip()
        body = rest.lstrip()
    return subject, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contacts", type=Path, default=DEFAULT_CONTACTS)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sender", required=True, help='Example: "Your Name <you@example.com>"')
    parser.add_argument("--sender-name", help="Name used in the email signature")
    args = parser.parse_args()

    sender_name = args.sender_name or args.sender.split("<", 1)[0].strip()
    template = args.template.read_text(encoding="utf-8")
    args.output.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    with args.contacts.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recipient = row.get("Email", "").strip()
            if not recipient:
                skipped += 1
                continue

            subject, body = render(template, row, sender_name)
            msg = email.message.EmailMessage()
            msg["From"] = args.sender
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.set_content(body)

            number = row.get("No.", str(created + 1)).strip()
            name = clean_filename(row.get("Name", "professor"))
            path = args.output / f"{number.zfill(2)}_{name}.eml"
            path.write_text(msg.as_string(), encoding="utf-8")
            created += 1

    print(f"Created {created} individual email draft(s).")
    print(f"Skipped {skipped} row(s) without an email address.")
    print(f"Output directory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

