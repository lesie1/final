"""Run a trusted worker and record the actual exception, separately from its log."""
import json
import os
from pathlib import Path
import runpy
import sys
import traceback
from worker_errors import classify, RESOURCE_EXIT_CODE


if __name__=='__main__':
    report=Path(os.environ.pop('EXPERIMENT_WORKER_FAILURE_PATH'))
    script=sys.argv[1];sys.argv=sys.argv[1:]
    try:
        runpy.run_path(script,run_name='__main__')
    except Exception as exc:
        value=dict(category=classify(exc),exception_type=type(exc).__name__,message=str(exc))
        try:
            report.write_text(json.dumps(value,ensure_ascii=False)+'\n',encoding='utf-8')
        except OSError:
            pass  # A full disk can prevent a receipt; the exit remains a failure.
        if value['category']=='resource':
            traceback.print_exc()
            raise SystemExit(RESOURCE_EXIT_CODE)
        raise
