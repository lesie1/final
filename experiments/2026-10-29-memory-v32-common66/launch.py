"""Read an existing API credential from stdin; retain only in supervisor memory."""
import json, os, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
R=Path(__file__).resolve().parent
key=sys.stdin.readline().strip()
assert key and not any(c.isspace() for c in key), 'Missing credential'
try:
    req=urllib.request.Request('https://api.deepseek.com/models',headers={'Authorization':'Bearer '+key})
    with urllib.request.urlopen(req,timeout=30) as response:d=json.load(response)
    assert 'deepseek-v4-pro' in [x['id'] for x in d['data']], 'Configured model not available'
except urllib.error.HTTPError as e:
    raise SystemExit('API_CREDENTIAL_CHECK_HTTP_'+str(e.code)) from None
pidfile=R/'supervisor_pid.json'
if pidfile.exists():
    old=json.loads(pidfile.read_text());p=Path('/proc')/str(old['pid'])
    assert not p.exists() or str(R/'supervisor.py').encode() not in (p/'cmdline').read_bytes(), 'Already running'
env=dict(os.environ,DEEPSEEK_API_KEY=key,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8')
with (R/'supervisor.log').open('ab') as log:
    p=subprocess.Popen([sys.executable,'-u',str(R/'supervisor.py')],stdin=subprocess.DEVNULL,stdout=log,
                       stderr=subprocess.STDOUT,env=env,start_new_session=True)
del key,env
pidfile.write_text(json.dumps(dict(pid=p.pid,started=time.time())))
print('STARTED',p.pid,flush=True)
