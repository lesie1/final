"""Prepare a quarantined HDD TEST cache; never load a model or compute scores."""
import os
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

import copy
import datetime
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import resource
import shutil
import subprocess
import time
import traceback
import zipfile

ROOT = Path('/root/autodl-tmp/hdd_final_test_20261029')
SOURCE = Path('/root/autodl-tmp/final_v31_20261028')
SESSION = SOURCE / 'hdd/session'
RUNTIME = SOURCE / 'runtime'
CACHE = ROOT / 'cache_test'
RAW = CACHE / 'raw_sources'
sys.path.insert(0, str(RUNTIME))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


START = time.time()
PROGRESS = {}


def status(stage, **fields):
    PROGRESS.update(fields)
    PROGRESS.update(stage=stage, updated_at=now(), pid=os.getpid(),
                    elapsed_seconds=round(time.time() - START, 1),
                    disk_free_gib=round(shutil.disk_usage(ROOT).free / 2**30, 2),
                    scoring_started=False, model_loaded=False)
    save(ROOT / 'status.json', PROGRESS)


def event(kind, **fields):
    row = dict(at=now(), event=kind, **fields)
    with (ROOT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def guard():
    if shutil.disk_usage(ROOT).free < 40 * 2**30:
        raise RuntimeError('Stopped preprocessing to preserve 40 GiB free scratch space for training')
    maximum = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists() and maximum.read_text().strip() != 'max':
        def headroom():
            # Clean inactive filesystem cache is reclaimable; counting it as occupied
            # RAM would deadlock this I/O job after staging the raw files.
            stat = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            reclaimable = max(0, int(stat.get('inactive_file', 0))
                             - int(stat.get('file_dirty', 0))
                             - int(stat.get('file_writeback', 0)))
            return int(maximum.read_text()) - int(current.read_text()) + reclaimable
        paused = False
        while headroom() < 10 * 2**30:
            if not paused:
                event('resource_pause', reason='Less than 10 GiB cgroup memory headroom')
            paused = True
            status('waiting_for_memory')
            time.sleep(10)
        if paused:
            event('resource_resume')


def assert_unchanged(seals):
    for name, expected in seals.items():
        if sha(name) != expected:
            raise RuntimeError('Protected source changed: ' + name)


def verify_isolation(trainval):
    # Run the same read block used by search in a separate process.
    probe = CACHE / 'isolation_probe.txt'
    probe.write_text('quarantined test material\n', encoding='utf-8')
    isolation = ROOT / 'isolation_check.json'
    save(isolation, dict(forbidden_reads=[str(Path('/root/autodl-tmp/backblaze_2024/zips'))],
                         readonly_roots=[str(trainval), str(RUNTIME)]))
    code = (
        "import os,sys;sys.dont_write_bytecode=True;sys.path.insert(0,sys.argv[1]);"
        "os.environ['TYPEA_SEARCH_ISOLATION']=sys.argv[2];"
        "import search_isolation;search_isolation.install();"
        "open(sys.argv[3]).read(1)\n"
        "try: open(sys.argv[4]).read()\n"
        "except PermissionError: print('TEST_READ_BLOCKED')\n"
        "else: raise RuntimeError('TEST cache was visible to search')\n"
    )
    result = subprocess.run([sys.executable, '-c', code, str(RUNTIME), str(isolation),
                             str(trainval / 'manifest.json'), str(probe)],
                            capture_output=True, text=True, timeout=20, check=True)
    if result.stdout.strip() != 'TEST_READ_BLOCKED':
        raise RuntimeError('Isolation check did not pass')
    event('isolation_verified', test_read_blocked=True, trainval_read_allowed=True)


def extract_verified(c, m, original):
    expected = {x['path']: x['sha256'] for x in original['source_receipts']}
    if len(expected) != len(original['source_receipts']):
        raise RuntimeError('Duplicate source receipt paths')
    entries = []
    archives = []
    for archive in sorted(Path(c['data_dir']).glob('*.zip')):
        archives.append(dict(path=str(archive), size=archive.stat().st_size,
                             mtime_ns=archive.stat().st_mtime_ns))
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                p = PurePosixPath(item.filename)
                if item.is_dir() or not any(p.match(g['glob']) for g in m['observations']):
                    continue
                if p.is_absolute() or '..' in p.parts or item.filename.startswith('__MACOSX/'):
                    continue
                if item.filename not in expected:
                    raise RuntimeError('Unexpected raw source member: ' + item.filename)
                entries.append((archive, item.filename, item.file_size))
    entries.sort(key=lambda x: x[1])
    if len(entries) != len(expected) or {x[1] for x in entries} != set(expected):
        raise RuntimeError('ZIP members do not exactly match the sealed source receipts')
    total = sum(x[2] for x in entries)
    # Full-year row count is estimated from the TRAIN/VAL period only for capacity planning.
    win = original['windows']
    factor = (win['test'][1] - win['train'][0]) / (win['val'][1] - win['train'][0])
    projected_rows = original['rows_before_failure_filter'] * factor * 1.1
    estimated_db = projected_rows * (len(original['features']) * 4 + 120) * 1.2
    extra_required = total + estimated_db + 16 * 2**30 + 40 * 2**30
    if not list(RAW.glob('data_Q*/*.csv')) and extra_required > shutil.disk_usage(ROOT).free:
        raise RuntimeError('Insufficient disk headroom for raw extraction plus conservative SQLite estimate')
    save(ROOT / 'capacity_plan.json', dict(raw_bytes=total, source_files=len(entries),
         projected_rows=round(projected_rows), estimated_sqlite_bytes=round(estimated_db),
         reserve_bytes=40 * 2**30, available_bytes=shutil.disk_usage(ROOT).free,
         note='Capacity estimate only; disk and memory guards remain active.'))
    event('source_extraction_started', files=len(entries), raw_gib=round(total / 2**30, 2))
    verified = []
    for index, (archive, member, size) in enumerate(entries, 1):
        guard()
        destination = RAW.joinpath(*PurePosixPath(member).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size != size or sha(destination) != expected[member]:
                raise RuntimeError('Existing staged source does not match receipt: ' + member)
        else:
            temporary = destination.with_suffix('.csv.partial')
            h = hashlib.sha256()
            with zipfile.ZipFile(archive) as z, z.open(member) as source, temporary.open('wb') as target:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                    target.write(chunk)
                    h.update(chunk)
            if temporary.stat().st_size != size or h.hexdigest() != expected[member]:
                raise RuntimeError('Extracted source hash differs: ' + member)
            os.replace(temporary, destination)
        verified.append(dict(path=member, sha256=expected[member], bytes=size))
        status('extracting_and_verifying_sources', files_done=index, files_total=len(entries))
        if index % 30 == 0 or index == len(entries):
            event('source_extraction_progress', done=index, total=len(entries))
    for item in archives:
        st = Path(item['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
            raise RuntimeError('Raw archive changed during staging')
    save(ROOT / 'source_verification.json', dict(files=verified, archives=archives,
         all_source_sha256_match=True, checked_at=now()))


def main():
    ROOT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    lock = (ROOT / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'ready.json').exists():
        from history_math import verify
        verify(CACHE)
        print('ALREADY_READY', flush=True)
        return
    os.nice(15)
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:4])
    resource.setrlimit(resource.RLIMIT_AS, (8 * 2**30, 8 * 2**30))
    subprocess.run(['ionice', '-c', '2', '-n', '7', '-p', str(os.getpid())], check=True)
    temporary = CACHE / 'temporary'
    temporary.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(temporary)
    os.environ['SQLITE_TMPDIR'] = str(temporary)

    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    import data_ingest
    from history_math import verify

    c = read(SESSION / 'config.json')
    m = read(SESSION / 'mapping.json')
    trainval = Path(c['active_cache'])
    original = read(trainval / 'manifest.json')
    if m['failure']['mode'] != 'indicator' or m['followup']['mode'] != 'last_observation':
        raise RuntimeError('Unexpected HDD source protocol')
    protected = [SESSION / n for n in ['config.json', 'mapping.json', 'code_seal.json']]
    protected += [trainval / 'manifest.json', trainval / 'devices.json']
    protected += sorted(RUNTIME.rglob('*.py'))
    seals = {str(p): sha(p) for p in protected}
    existing_seal = ROOT / 'source_preservation.json'
    if existing_seal.exists():
        assert_unchanged(read(existing_seal)['files'])
    else:
        save(existing_seal, dict(files=seals, captured_at=now(),
             note='Active search state and events continue to change normally; sealed code, configuration and TRAIN/VAL identity must not change.'))
    event('preparation_started', user_authorized_parallel_preparation=True,
          scoring_authorized_only_after_model_and_threshold_freeze=True,
          cpu_affinity=sorted(os.sched_getaffinity(0)), address_space_limit_gib=8,
          io_priority='best-effort 7', nice=15,
          computation='Unmodified sealed data_ingest.normalize(final_test=True)')
    verify_isolation(trainval)
    status('verifying_trainval_cache')
    verify(trainval)
    extract_verified(c, m, original)
    (ROOT / 'cache').mkdir(exist_ok=True)
    shutil.copy2(trainval / 'devices.json', ROOT / 'cache/devices.json')
    nc = copy.deepcopy(c)
    nc.update(output_root=str(ROOT), data_dir=str(RAW), min_free_disk_gb=40,
              num_threads=1, score_workers=1, memory_limit_gb=8)
    save(ROOT / 'normalization_config.json', nc)
    save(ROOT / 'mapping.json', m)

    # Observe the unmodified normalizer; wrappers only report progress or wait for resources.
    original_frames = data_ingest.frames
    visits = {}
    counters = {'metadata_files': 0, 'ingest_files': 0,
                'metadata_rows': 0, 'ingest_rows': 0, 'written_devices': 0}
    def observed_frames(path, *args, **kwargs):
        key = str(path)
        visits[key] = visits.get(key, 0) + 1
        phase = 'metadata' if visits[key] == 1 else 'ingest'
        for frame in original_frames(path, *args, **kwargs):
            guard()
            counters[phase + '_rows'] += len(frame)
            status('normalizing_' + phase, **counters,
                   current_source=Path(path).name)
            yield frame
        counters[phase + '_files'] += 1
        status('normalizing_' + phase, **counters)
        if counters[phase + '_files'] % 30 == 0:
            event('normalization_progress', phase=phase, **counters)
    original_writer = data_ingest.Writer
    class ObservedWriter(original_writer):
        def add(self, frame):
            super().add(frame)
            counters['written_devices'] += 1
            if counters['written_devices'] % 500 == 0:
                guard()
                status('writing_device_cache', **counters)
    data_ingest.frames = observed_frames
    data_ingest.Writer = ObservedWriter
    status('normalizing_metadata', **counters)
    manifest = data_ingest.normalize(nc, m, original['spec']['inventory'], final_test=True)
    status('verifying_finished_cache', **counters)
    verify(CACHE)
    for field in ['features', 'windows', 'coverage', 'source_receipts']:
        if manifest[field] != original[field]:
            raise RuntimeError('Prepared cache differs from TRAIN/VAL contract: ' + field)
    for field in ['mapping', 'inventory', 'coverage']:
        if manifest['spec'][field] != original['spec'][field]:
            raise RuntimeError('Prepared cache identity differs: ' + field)
    previous_ids = read(trainval / 'devices.json')
    if read(CACHE / 'devices.json')[:len(previous_ids)] != previous_ids:
        raise RuntimeError('Device identifiers changed')
    assert_unchanged(seals)
    save(ROOT / 'ready.json', dict(completed_at=now(), cache=str(CACHE),
         manifest_sha256=sha(CACHE / 'manifest.json'),
         feature_count=len(manifest['features']), test_window=manifest['windows']['test'],
         source_receipts_verified=True, source_seals_unchanged=True,
         test_preprocessing_opened=True, test_scoring_opened=False,
         model_loaded=False, agent_received_test_information=False,
         elapsed_seconds=time.time() - START,
         next_step='Freeze the search-selected model, recipe, boosting step and threshold after confirmation; only then run TEST inference without further selection.'))
    event('cache_ready', manifest_sha256=sha(CACHE / 'manifest.json'), scoring_started=False)
    status('ready', **counters)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            event('preparation_failed', error_type=type(error).__name__, message=str(error))
            status('failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise
