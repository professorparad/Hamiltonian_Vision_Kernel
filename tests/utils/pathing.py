import sys
from pathlib import Path


def add_main_to_path():
    project_root = Path(__file__).resolve().parents[2]
    main_root = project_root / "Main"
    if str(main_root) not in sys.path:
        sys.path.insert(0, str(main_root))
