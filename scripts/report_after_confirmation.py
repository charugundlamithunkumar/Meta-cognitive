"""Wait for real C completion, then build the requested results-only PDF."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.meta_runner import atomic_text


def main():
    job=ROOT/'runs/confirmation_C_main_20261118'
    status=job/'final_report_status.json'
    final=ROOT/'results/all_experiments_20261118'
    def save(state,**extra):
        body=dict(status=state,updated=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),**extra)
        atomic_text(status,json.dumps(body,indent=2)+'\n');print(json.dumps(body),flush=True)
    if final.exists():
        if (final/'all_experimental_results.pdf').is_file() and (final/'report_inputs.json').is_file():
            save('complete',pdf=str(final/'all_experimental_results.pdf'));return
        raise FileExistsError('Report destination exists but is incomplete; preserve and inspect')
    save('waiting_for_confirmation')
    while True:
        current=json.loads((job/'status.json').read_text())
        if current['signature']['mock']:raise ValueError('Mock results are not an experimental report')
        if current['status']=='failed':
            save('blocked_by_experiment_failure',error=current.get('error'));return
        if current['status']=='complete':break
        time.sleep(30)
    stage=ROOT/'results'/('all_experiments_build_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        save('building',staging=str(stage))
        subprocess.run([sys.executable,str(ROOT/'scripts/build_results_pdf.py'),'--out',str(stage)],
                       cwd=ROOT,check=True)
        info=subprocess.check_output(['pdfinfo',str(stage/'all_experimental_results.pdf')],text=True)
        (stage/'pdfinfo.txt').write_text(info)
        stage.rename(final)
        save('complete',pdf=str(final/'all_experimental_results.pdf'))
    except BaseException as e:
        save('failed',error=str(e),preserved_staging=str(stage));raise


if __name__=='__main__':main()
