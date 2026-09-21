"""Download MikanQuiz question crops and coarse labels for offline evaluation."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.parse

from mikan_font_probe.paths import ROOT

BASE = "https://mikanquiz.xyz"
DATA = ROOT / "data"
IMAGES = DATA / "images"


def curl(url: str, cookie_file: str | None = None, body: bytes | None = None) -> bytes:
    command = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45"]
    if cookie_file:
        command.extend(["--cookie", cookie_file, "--cookie-jar", cookie_file])
    if body is not None:
        command.extend(["--header", "Content-Type: application/json", "--data-binary", "@-"])
    command.append(url)
    return subprocess.run(command, input=body, check=True, capture_output=True).stdout


def main() -> None:
    user = os.environ.get("MK_ADMIN_USER")
    password = os.environ.get("MK_ADMIN_PASS")
    if not user or not password:
        raise SystemExit("Set MK_ADMIN_USER and MK_ADMIN_PASS in the environment")

    with tempfile.TemporaryDirectory(prefix="mikanfont-auth-") as auth_dir:
        cookie_file = str(pathlib.Path(auth_dir) / "cookies.txt")
        session = json.loads(
            curl(BASE + "/api/v1/admin/web/login", cookie_file, json.dumps({"username": user, "password": password}).encode())
        )
        if session.get("username") != user:
            raise SystemExit("Login response did not confirm the requested account")

        IMAGES.mkdir(parents=True, exist_ok=True)
        records = []
        page = 1
        total = None
        while total is None or len(records) < total:
            result = json.loads(curl(BASE + f"/api/v1/admin/questions?page={page}", cookie_file))
            total = int(result["total"])
            records.extend(result["items"])
            if not result["items"]:
                raise RuntimeError(f"No items on page {page}, expected {total}")
            page += 1

    entries = []
    for question in records:
        image_url = question.get("image_url")
        if not image_url:
            continue
        suffix = pathlib.PurePosixPath(urllib.parse.urlsplit(image_url).path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            suffix = ".img"
        filename = f"q{question['id']}{suffix}"
        target = IMAGES / filename
        status = "cached" if target.is_file() and target.stat().st_size else "downloaded"
        if status == "downloaded":
            for attempt in range(3):
                try:
                    payload = curl(image_url)
                    if not payload or payload[:1] == b"<":
                        raise ValueError("Unexpected image payload")
                    target.write_bytes(payload)
                    break
                except (OSError, subprocess.CalledProcessError, ValueError):
                    if attempt == 2:
                        status = "failed"
                    else:
                        time.sleep(attempt + 1)
        entries.append(
            {
                "question_id": question["id"],
                "coarse_label": question.get("answer") or question.get("correct_option", {}).get("name"),
                "difficulty": question.get("difficulty"),
                "image_url": image_url,
                "image_file": f"images/{filename}" if status != "failed" else None,
                "status": status,
            }
        )

    manifest = {"source": BASE, "total_questions": total, "entries": entries}
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    labels: dict[str, int] = {}
    for entry in entries:
        labels[entry["coarse_label"]] = labels.get(entry["coarse_label"], 0) + 1
    print(json.dumps({"questions": total, "images": len(entries), "status": {s: sum(e["status"] == s for e in entries) for s in {e["status"] for e in entries}}, "labels": labels}, ensure_ascii=False))


if __name__ == "__main__":
    main()
