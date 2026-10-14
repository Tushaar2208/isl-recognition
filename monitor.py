#!/usr/bin/env python3
"""
Clean Live Monitor for ISL Video Download
Only prints a new line when a video finishes downloading.
Checks every 5 seconds.
"""
import time
from pathlib import Path

raw_dir = Path("data/raw_videos")
total_target = 958

print("=" * 65, flush=True)
print("  INCLUDE-50 Video Download Monitor", flush=True)
print("  (Updates automatically every 5s when a new video lands)", flush=True)
print("  Press Ctrl+C to exit monitor anytime", flush=True)
print("=" * 65, flush=True)

prev_count = -1


def safe_stat(f):
    try:
        return f.stat()
    except (FileNotFoundError, PermissionError):
        return None


while True:
    try:
        # Ignore any temporary extraction directories
        vids = []
        for f in raw_dir.rglob("*"):
            if any(part.startswith("_temp") for part in f.parts):
                continue
            if f.suffix.lower() in [".mov", ".mp4"]:
                st = safe_stat(f)
                if st and st.st_size > 10000:
                    vids.append((f, st))

        count = len(vids)

        if count != prev_count and count > 0:
            pct = (count / total_target) * 100
            total_mb = sum(st.st_size for _, st in vids) / (1024 * 1024)

            latest_file, _ = max(vids, key=lambda item: item[1].st_mtime)
            latest_str = f"{latest_file.parent.name}/{latest_file.name}"

            print(
                f"[{count:3d}/{total_target}] ({pct:5.1f}%) | "
                f"{total_mb:6.1f} MB | "
                f"New: {latest_str}",
                flush=True
            )
            prev_count = count

            if count >= total_target:
                print("\n🎉 ALL VIDEOS DOWNLOADED!", flush=True)
                break

        time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        break
    except Exception:
        # Ignore transient filesystem races
        time.sleep(5)
