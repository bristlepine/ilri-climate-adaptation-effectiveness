"""
gdocs_review_comments.py

Review, respond to, and resolve comments on a Google Doc.
Intended to be called interactively — not as standalone automation.

Capabilities:
  - List all open comments on a document (with author, location, text)
  - Reply to a comment
  - Resolve a comment
  - Make a text edit in the document in response to a comment
  - Export a full comment report as JSON or Markdown

Usage:
  from deliverables.gdocs_review_comments import CommentReviewer
  reviewer = CommentReviewer(doc_id="...")
  comments = reviewer.list_open()
  reviewer.reply(comment_id, "Reply text.")
  reviewer.resolve(comment_id)
  reviewer.edit_text(old_text, new_text)

Or from the command line (report only):
  conda run -n ilri01 python deliverables/gdocs_review_comments.py --doc-id DOC_ID
  conda run -n ilri01 python deliverables/gdocs_review_comments.py --doc-id DOC_ID --md
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import List, Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── auth ──────────────────────────────────────────────────────────────────────
CREDS_DIR  = Path(__file__).resolve().parent / ".credentials"
TOKEN_FILE = CREDS_DIR / "token.json"
SCOPES     = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]


def _get_creds() -> Credentials:
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


# =============================================================================
# CommentReviewer
# =============================================================================

class CommentReviewer:
    """
    Wraps the Google Drive and Docs APIs for comment review workflows.

    Parameters
    ----------
    doc_id : str
        The Google Doc ID (from the URL).
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        creds       = _get_creds()
        self.drive  = build("drive", "v3", credentials=creds)
        self.docs   = build("docs",  "v1", credentials=creds)

    # ── read ──────────────────────────────────────────────────────────────────

    def list_open(self) -> List[dict]:
        """Return all unresolved comments, ordered by creation time."""
        result = self.drive.comments().list(
            fileId=self.doc_id,
            fields="comments(id,author,content,createdTime,quotedFileContent,replies,resolved)",
            includeDeleted=False,
        ).execute()
        comments = result.get("comments", [])
        return [c for c in comments if not c.get("resolved", False)]

    def list_all(self) -> List[dict]:
        """Return all comments including resolved ones."""
        result = self.drive.comments().list(
            fileId=self.doc_id,
            fields="comments(id,author,content,createdTime,quotedFileContent,replies,resolved)",
            includeDeleted=False,
        ).execute()
        return result.get("comments", [])

    def get_doc_text(self) -> str:
        """Return the plain text of the document body."""
        doc  = self.docs.documents().get(documentId=self.doc_id).execute()
        body = doc.get("body", {}).get("content", [])
        lines = []
        for elem in body:
            para = elem.get("paragraph")
            if not para:
                continue
            text = "".join(
                r.get("textRun", {}).get("content", "")
                for r in para.get("elements", [])
            )
            if text.strip():
                lines.append(text.rstrip("\n"))
        return "\n".join(lines)

    # ── write ─────────────────────────────────────────────────────────────────

    def reply(self, comment_id: str, text: str) -> dict:
        """Post a reply to a comment."""
        result = self.drive.replies().create(
            fileId=self.doc_id,
            commentId=comment_id,
            fields="id,content,createdTime",
            body={"content": text},
        ).execute()
        print(f"[comments] Replied to {comment_id}: {text[:60]}...")
        return result

    def resolve(self, comment_id: str, reply_text: Optional[str] = None) -> dict:
        """
        Mark a comment as resolved.
        Optionally post a final reply before resolving.
        """
        if reply_text:
            self.reply(comment_id, reply_text)
        result = self.drive.comments().update(
            fileId=self.doc_id,
            commentId=comment_id,
            fields="id,resolved",
            body={"resolved": True},
        ).execute()
        print(f"[comments] Resolved: {comment_id}")
        return result

    def edit_text(self, old_text: str, new_text: str) -> dict:
        """
        Replace the first occurrence of old_text with new_text in the document.
        Uses replaceAllText — exact match, case-sensitive.
        """
        result = self.docs.documents().batchUpdate(
            documentId=self.doc_id,
            body={"requests": [{
                "replaceAllText": {
                    "containsText": {"text": old_text, "matchCase": True},
                    "replaceText":  new_text,
                }
            }]},
        ).execute()
        replacements = result.get("replies", [{}])[0].get("replaceAllText", {}).get("occurrencesChanged", 0)
        print(f"[comments] edit_text: {replacements} replacement(s) made.")
        return result

    def insert_after(self, anchor_text: str, new_text: str) -> dict:
        """
        Insert new_text immediately after the first occurrence of anchor_text.
        Useful for inserting a new paragraph or sentence in response to a comment.
        """
        doc     = self.docs.documents().get(documentId=self.doc_id).execute()
        content = doc.get("body", {}).get("content", [])

        insert_idx = None
        for elem in content:
            para = elem.get("paragraph")
            if not para:
                continue
            text = "".join(
                r.get("textRun", {}).get("content", "")
                for r in para.get("elements", [])
            )
            if anchor_text in text:
                insert_idx = elem.get("endIndex", 0) - 1
                break

        if insert_idx is None:
            raise ValueError(f"anchor_text not found in document: {anchor_text!r}")

        result = self.docs.documents().batchUpdate(
            documentId=self.doc_id,
            body={"requests": [{
                "insertText": {
                    "location": {"index": insert_idx},
                    "text": "\n" + new_text,
                }
            }]},
        ).execute()
        print(f"[comments] Inserted text after anchor at index {insert_idx}.")
        return result

    # ── report ────────────────────────────────────────────────────────────────

    def report_markdown(self, comments: Optional[List[dict]] = None) -> str:
        """Format open comments as a readable Markdown report."""
        if comments is None:
            comments = self.list_open()
        if not comments:
            return "No open comments."

        lines = [f"## Open Comments — {self.doc_id}\n"]
        for i, c in enumerate(comments, 1):
            author  = c.get("author", {}).get("displayName", "Unknown")
            created = c.get("createdTime", "")[:10]
            content = c.get("content", "").strip()
            quoted  = c.get("quotedFileContent", {}).get("value", "").strip()
            replies = c.get("replies", [])

            lines.append(f"### Comment {i} — {author} ({created})")
            lines.append(f"**ID:** `{c['id']}`")
            if quoted:
                lines.append(f"**Quoted text:** "{textwrap.shorten(quoted, 120)}"")
            lines.append(f"**Comment:** {content}")
            if replies:
                lines.append("**Replies:**")
                for r in replies:
                    ra = r.get("author", {}).get("displayName", "Unknown")
                    lines.append(f"  - {ra}: {r.get('content', '').strip()}")
            lines.append("")

        return "\n".join(lines)

    def report_json(self, comments: Optional[List[dict]] = None) -> str:
        if comments is None:
            comments = self.list_open()
        return json.dumps(comments, indent=2, ensure_ascii=False)


# =============================================================================
# CLI entry point (report only — no edits from command line)
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="List open comments on a Google Doc")
    ap.add_argument("--doc-id", required=True, help="Google Doc ID")
    ap.add_argument("--md",     action="store_true", help="Output as Markdown (default: plain summary)")
    ap.add_argument("--json",   action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    reviewer = CommentReviewer(args.doc_id)
    comments = reviewer.list_open()
    print(f"Open comments: {len(comments)}\n")

    if args.json:
        print(reviewer.report_json(comments))
    else:
        print(reviewer.report_markdown(comments))


if __name__ == "__main__":
    main()
