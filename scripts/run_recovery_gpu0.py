"""MR-only GPU 0 entry point; reuses the migration supervisor and frozen runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_recovery_gpu2 import main


if __name__ == '__main__':
    main(gpu='0', amendment='prereg_recovery_gpu0_amendment.md',
         additional_sources=['scripts/run_recovery_gpu0.py', 'paper/tools/test_gpu0_job.py'])
