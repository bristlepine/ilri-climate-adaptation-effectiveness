"""
gdocs_auth.py

One-time OAuth2 authentication for Google Docs + Drive API.
Run this once — it opens a browser for consent and saves token.json.

Usage:
    conda run -n ilri01 python deliverables/gdocs_auth.py
"""

from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

CREDS_DIR   = Path(__file__).resolve().parent / ".credentials"
CLIENT_FILE = CREDS_DIR / "client_secret.json"
TOKEN_FILE  = CREDS_DIR / "token.json"


def authenticate() -> Credentials:
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            print("Token refreshed.")
        else:
            if not CLIENT_FILE.exists():
                raise FileNotFoundError(
                    f"client_secret.json not found at {CLIENT_FILE}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
            print("Authentication successful.")

        TOKEN_FILE.write_text(creds.to_json())
        print(f"Token saved to {TOKEN_FILE}")

    return creds


if __name__ == "__main__":
    creds = authenticate()
    print("\nGoogle API connection: OK")
    print(f"Token valid: {creds.valid}")
    print(f"Scopes:      {creds.scopes}")
