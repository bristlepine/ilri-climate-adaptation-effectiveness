"""
Push the latest step16 figures into D4 v01.

Pads each PNG to the target bounding-box aspect ratio (white background)
before uploading so CENTER_CROP never clips the content.

Usage:
  conda run -n ilri01 python deliverables/_d4_refresh_images.py
"""
from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT      = Path(__file__).resolve().parent.parent
CREDS_DIR = Path(__file__).resolve().parent / ".credentials"
SCOPES    = ["https://www.googleapis.com/auth/documents",
             "https://www.googleapis.com/auth/drive"]
DOC_ID    = "1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs"
STEP16    = ROOT / "scripts" / "outputs" / "step16"

# obj_id → (png_stem, target_w_pt, target_h_pt)
IMAGE_TARGETS = {
    "kix.xh7ax16lorkv": ("roses_flow",        720, 405),
    "kix.evobad1yk7iw": ("evidence_gap_map",   468, 332),
    "kix.w324aps3ogb8": ("geographic_bar",      468, 195),
    "kix.p5raizum4giy": ("geographic_map",      468, 252),
    "kix.uywukpw8ae8z": ("methodology_bar",     468, 290),
    "kix.s8dk7eqtrxas": ("temporal_trends",     468, 218),
}


def _pad_to_ratio(png_path: Path, tw: float, th: float) -> bytes:
    """Pad image with white to match target aspect ratio, return PNG bytes."""
    img = Image.open(png_path).convert("RGB")
    iw, ih = img.size
    tr = tw / th
    ir = iw / ih
    if abs(ir - tr) < 0.01:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()
    if ir > tr:
        new_h = int(round(iw / tr))
        out = Image.new("RGB", (iw, new_h), (255, 255, 255))
        out.paste(img, (0, (new_h - ih) // 2))
    else:
        new_w = int(round(ih * tr))
        out = Image.new("RGB", (new_w, ih), (255, 255, 255))
        out.paste(img, ((new_w - iw) // 2, 0))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def main() -> None:
    creds = Credentials.from_authorized_user_file(
        str(CREDS_DIR / "token.json"), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    drive = build("drive", "v3", credentials=creds)
    docs  = build("docs",  "v1", credentials=creds)

    temp_ids: list[str] = []
    requests: list[dict] = []

    for obj_id, (stem, tw, th) in IMAGE_TARGETS.items():
        png_path = STEP16 / f"{stem}.png"
        if not png_path.exists():
            print(f"  SKIP {stem}.png — not found in {STEP16}")
            continue

        img_bytes = _pad_to_ratio(png_path, tw, th)
        tmp = Path(f"/tmp/d4_padded_{stem}.png")
        tmp.write_bytes(img_bytes)

        fid = drive.files().create(
            body={"name": f"{stem}.png"},
            media_body=MediaFileUpload(str(tmp), mimetype="image/png", resumable=False),
            fields="id",
        ).execute()["id"]
        drive.permissions().create(
            fileId=fid, body={"type": "anyone", "role": "reader"}
        ).execute()
        temp_ids.append(fid)

        requests.append({"replaceImage": {
            "imageObjectId": obj_id,
            "uri": f"https://drive.google.com/uc?export=view&id={fid}",
            "imageReplaceMethod": "CENTER_CROP",
        }})
        print(f"  {stem}.png → padded {tw}:{th}, uploaded")
        time.sleep(0.3)

    if requests:
        docs.documents().batchUpdate(
            documentId=DOC_ID, body={"requests": requests}
        ).execute()
        print(f"Replaced {len(requests)} images in D4.")

    for fid in temp_ids:
        drive.files().delete(fileId=fid).execute()
    print("Done — https://docs.google.com/document/d/1i2dQUoXuNPwjvV_fT5CMiy0fvADy8oWCGK3sdHGApCs/edit")


if __name__ == "__main__":
    main()
