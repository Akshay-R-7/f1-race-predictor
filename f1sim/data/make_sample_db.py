"""Build f1_sample.sqlite from the synthetic generator: python -m f1sim.data.make_sample_db"""
import sys

from .db import connect, write
from .sample import generate

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "f1_sample.sqlite"
    races, results = generate()
    print(write(connect(path), races, results, reject_log="rejected_rows.jsonl"))
