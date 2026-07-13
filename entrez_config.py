"""Shared NCBI E-utilities identity.

NCBI's E-utilities Terms of Service require an email address on every request so
they can contact you before throttling/blocking. Rather than hard-code an
address, RIBOSCOPE reads it from the ``NCBI_EMAIL`` environment variable so each
user identifies with their own address.

Usage:
    export NCBI_EMAIL="you@example.com"    # bash / zsh
    $env:NCBI_EMAIL = "you@example.com"    # PowerShell

Ref: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
import os


def get_entrez_email() -> str:
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        raise SystemExit(
            "NCBI_EMAIL is not set. NCBI E-utilities require an email address.\n"
            "Set it before running fetch scripts, e.g.:\n"
            '  export NCBI_EMAIL="you@example.com"   (bash/zsh)\n'
            '  $env:NCBI_EMAIL = "you@example.com"   (PowerShell)'
        )
    return email
