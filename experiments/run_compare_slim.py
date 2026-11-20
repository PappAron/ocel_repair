"""Run ocel_repair's main.py with attribute loading switched off.

No model uses event/object attributes, and loading them for BPI 2017 costs ~1 GB of RAM.
Run from inside ocel_repair, with the same arguments as main.py, e.g.
  python experiments\\run_compare_slim.py compare data\\bpi2017-ocel2-e2o.sqlite --loader sqlite --epochs 2000
"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
import sys

sys.path.insert(0, os.getcwd())
from adapters.loaders import SqliteLoader  # noqa: E402

SqliteLoader._extension_attributes = staticmethod(lambda connection, table, identifier: {})
import main  # noqa: E402

main.main()
