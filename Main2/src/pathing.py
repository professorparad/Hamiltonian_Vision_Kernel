from __future__ import annotations

import sys
from pathlib import Path


def add_main_package_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    main_dir = repo_root / "Main"
    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))
