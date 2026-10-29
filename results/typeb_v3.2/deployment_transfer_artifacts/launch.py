"""Read the authorized API credential from stdin; keep it only in process memory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

R = Path(__file__).resolve().parent


def main():
    assert json.loads((R / 'ready.json').read_text())['screen_margin'] == 1.0
    checks = json.loads((R / 'screening_check/checks.json').read_text())
    assert checks['screen1_runs_cv_at_gap_minus1'] and checks['test_cache_read_blocked']
    pidfile = R / 'supervisor_pid.json'
    if pidfile.exists():
        old = json.loads(pidfile.read_text())
        proc = Path('/proc') / str(old['pid'])
        assert not proc.exists() or str(R / 'supervisor.py').encode() not in (proc / 'cmdline').read_bytes(), 'Already running'
    key = sys.stdin.readline().strip()
    assert key and not any(c.isspace() for c in key), 'Missing credential'
    try:
        req = urllib.request.Request('https://api.deepseek.com/models', headers={'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.load(response)
        assert 'deepseek-v4-pro' in [x['id'] for x in data['data']], 'Configured model unavailable'
    except urllib.error.HTTPError as error:
        raise SystemExit('API_CREDENTIAL_CHECK_HTTP_' + str(error.code)) from None
    env = dict(os.environ, DEEPSEEK_API_KEY=key, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8')
    with (R / 'supervisor.log').open('ab') as log:
        worker = subprocess.Popen([sys.executable, '-u', str(R / 'supervisor.py')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=env, start_new_session=True)
    del key, env
    pidfile.write_text(json.dumps(dict(pid=worker.pid, started=time.time())))
    print('STARTED', worker.pid, flush=True)


if __name__ == '__main__':
    main()
