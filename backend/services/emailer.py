"""SMTP email delivery with a demo-friendly dry-run mode.

Configuration is read at call time (not import time) so environment changes and
tests are reflected without re-importing the module. When SMTP is not configured
or ``EMAIL_DRY_RUN`` is truthy, emails are logged instead of sent, which lets the
whole delivery loop be demonstrated end-to-end without a real mail server.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional


def _truthy(value: Optional[str]) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


def _dry_run_active(smtp_host: Optional[str]) -> bool:
    """Dry-run when explicitly forced via EMAIL_DRY_RUN or when no SMTP host is set."""
    return _truthy(os.getenv("EMAIL_DRY_RUN")) or not smtp_host


def is_dry_run() -> bool:
    """Public: True when emails are logged instead of sent (single source of
    truth, reused by the API's /info endpoint so it can't disagree with reality)."""
    return _dry_run_active(os.getenv("SMTP_HOST"))


def try_send(
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> Optional[str]:
    """Attempt delivery. Returns ``None`` on success (or on a logged dry-run),
    otherwise a short error string (``ExceptionType: message``) describing why it
    failed.

    Kept separate from :func:`send_email` so the worker can surface the *real*
    reason a send failed (e.g. a bad Gmail App Password) in its result and logs
    instead of silently swallowing it behind a bare ``False``.
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    from_email = os.getenv("FROM_EMAIL") or smtp_user or "facts@localhost"

    if _dry_run_active(smtp_host):
        print(
            f"[EMAIL DRY-RUN] To: {to_email} | Subject: {subject}\n"
            f"{body_text}\n{'-' * 40}"
        )
        return None

    try:
        # Build the message inside the try: a subject/recipient carrying stray
        # CR/LF (header injection) raises here, and we must not let that escape
        # and tear down the worker's delivery loop.
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        msg.set_content(body_text)
        if body_html:
            msg.add_alternative(body_html, subtype="html")

        # Verify the server certificate and hostname so credentials/content
        # can't be intercepted by a MITM on the SMTP connection.
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.starttls(context=context)
            if smtp_user and smtp_pass:
                smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        return None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print("Error sending email:", error)
        return error


def send_email(
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> bool:
    """Send an email. Returns True on success (or on a logged dry-run).

    Thin boolean wrapper around :func:`try_send` for callers that only care
    whether delivery succeeded, not why it didn't.
    """
    return try_send(to_email, subject, body_text, body_html) is None
