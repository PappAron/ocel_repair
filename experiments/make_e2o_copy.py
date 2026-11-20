"""Copy an OCEL 2.0 SQLite file and remove all object-to-object links (E2O-only version).
Usage: python make_e2o_copy.py IN.sqlite OUT.sqlite"""
import os as _os, sys as _sys  # run from the repository root; makes adapters/ and core/ importable
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import shutil
import sqlite3
import sys

src, dst = sys.argv[1], sys.argv[2]
shutil.copyfile(src, dst)
con = sqlite3.connect(dst)
con.execute("DELETE FROM object_object")
con.commit()
con.execute("VACUUM")
print("object_object rows:", con.execute("SELECT count(*) FROM object_object").fetchone()[0],
      "| event_object rows:", con.execute("SELECT count(*) FROM event_object").fetchone()[0])
con.close()
