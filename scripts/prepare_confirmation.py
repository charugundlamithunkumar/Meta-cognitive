"""CPU-only sample allocation, without reading model outcomes or launching inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from paper.tools.audit_allocations import scan
from scripts.confirmation_common import settings, build_sample_manifest
from src.lhe_data import verify


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path)
    a=ap.parse_args();cfg=settings();out=a.out or ROOT/cfg['manifest']
    audit_path=out.with_name(out.stem+'_allocation.json')
    if out.exists() or audit_path.exists():ap.error('Allocation outputs exist; refusing overwrite')
    verify()
    audit=scan([ROOT/'runs',ROOT/'previous work/runs',ROOT/'previous work/results'])
    manifest=build_sample_manifest(cfg,audit)
    out.parent.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    out.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(manifest=str(out),ids=manifest['ids'],sha256=manifest['sha256'],
                         audit_records=len(audit['records']),note='CPU allocation only; no model answers generated')))


if __name__=='__main__':main()
