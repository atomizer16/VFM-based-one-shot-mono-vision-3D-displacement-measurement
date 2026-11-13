from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    tracker_dir = Path(__file__).resolve().parent / "RoMaV2-main"
    sys.path.insert(0, str(tracker_dir))
    runpy.run_path(str(tracker_dir / "long_time_track.py"), run_name="__main__")
