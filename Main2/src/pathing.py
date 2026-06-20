from __future__ import annotations

import sys
from pathlib import Path


def add_main_package_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    main_dir = repo_root / "Main"
    main_path = str(main_dir)
    if main_path in sys.path:
        sys.path.remove(main_path)
    sys.path.insert(0, main_path)
