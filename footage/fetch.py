"""Download the clips listed in manifest.json into footage/clips/ and verify their hashes."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

FOOTAGE = Path(__file__).resolve().parent
CLIPS = FOOTAGE / "clips"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    clips = json.loads((FOOTAGE / "manifest.json").read_text())["clips"]
    if not clips:
        print("manifest.json has no clips yet.")
        return 0

    CLIPS.mkdir(exist_ok=True)
    failed = 0
    for clip in clips:
        dest = CLIPS / clip["filename"]
        if not clip.get("download_url"):
            if not dest.exists():
                print(f"MISSING: {clip['id']} is self-recorded; place it at {dest}")
                failed += 1
                continue
        elif not dest.exists():
            print(f"downloading {clip['id']} ...")
            req = urllib.request.Request(clip["download_url"], headers={"User-Agent": "laNCA-lamp"})
            with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)

        expected = clip.get("sha256")
        if expected and sha256(dest) != expected:
            print(f"HASH MISMATCH: {clip['id']} ({dest.name})")
            failed += 1
        else:
            print(f"ok: {clip['id']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
