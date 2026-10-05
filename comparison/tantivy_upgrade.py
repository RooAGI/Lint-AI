#!/usr/bin/env python3
"""Prepare old indexes or compare two server binaries on identical old indexes."""
import argparse
import contextlib
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('phase', choices=['prepare', 'compare', 'verify'])
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path)
p.add_argument('--work-dir', type=Path, required=True)
p.add_argument('--records', type=int, default=23366)
p.add_argument('--repetitions', type=int, default=5)
p.add_argument('--requests', type=int, default=1000)
p.add_argument('--output', type=Path)
a = p.parse_args()
a.work_dir.mkdir(parents=True, exist_ok=True)
work = a.work_dir.resolve()
queries = ['deployment configuration system decision', 'configuration', 'deployment', 'decision 42', 'memory record 123', 'user memory', 'nonexistentzzword', 'system decision 23365']


def call(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/{path}', data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


@contextlib.contextmanager
def server(binary, layout, index, port, label):
    cmd = [str(binary.resolve()), '--bind', f'127.0.0.1:{port}', '--index', str(index), '--project-root', str(work), '--allow-unauthenticated']
    if layout == 'single': cmd += ['--single-index']
    with (work / f'{label}.log').open('w') as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=log, cwd=ROOT)
        try:
            for _ in range(1800):
                if process.poll() is not None: raise RuntimeError(f'server exited: {label}')
                try:
                    call(port, 'health')
                    break
                except Exception: time.sleep(.1)
            else: raise RuntimeError('health timeout')
            yield
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def rankings(port):
    return {q:call(port, 'search', {'query':q, 'user_id':'bench-user', 'top_k':20}) for q in queries}


def immutable_files(index):
    return {str(f.relative_to(index)):hashlib.sha256(f.read_bytes()).hexdigest() for f in index.rglob('*') if f.is_file() and f.suffix in ('.store', '.term', '.idx', '.pos', '.fast', '.fieldnorm', '.json') and 'lexical' in str(f)}


if a.phase == 'prepare':
    for i, layout in enumerate(['single', 'segment']):
        index = work / f'old-{layout}'
        if index.exists(): raise RuntimeError(f'refusing to overwrite {index}')
        with server(a.baseline, layout, index, 18770+i, f'prepare-{layout}'):
            subprocess.run([sys.executable, str(ROOT/'comparison/seed_lint_ai.py'), '--url', f'http://127.0.0.1:{18770+i}/add', '--count',str(a.records), '--batch-size','1024','--sessions','5','--bulk'],check=True)
            (work/f'{layout}-rankings.json').write_text(json.dumps(rankings(18770+i),indent=2))
        (work/f'{layout}-files.json').write_text(json.dumps(immutable_files(index),indent=2))
        print(f'prepared {layout}',flush=True)
elif a.phase == 'verify':
    if not a.candidate: p.error('--candidate required for verify')
    texts = [
        'Orion deployment uses blue green releases with PostgreSQL rollback.',
        'Orion deployment uses canary releases for the payment gateway.',
        'Orion configuration stores secrets in the vault with short lived tokens.',
        'Vega deployment uses rolling releases on Kubernetes.',
        'Vega configuration sets the database connection pool to forty.',
        'PostgreSQL rollback restores the transaction snapshot.',
        'Redis caching expires stale entries after thirty seconds.',
        'Python dependencies use uv and a pinned lockfile.',
        'Rust concurrency uses an owned mutex guard during persistence.',
        'The kitchen recipe includes saffron rice and roasted almonds.',
    ]
    queries = ['Orion deployment', 'PostgreSQL rollback', 'Vega configuration',
               'Redis caching', 'Python dependencies', 'Rust concurrency',
               'saffron almonds', 'nonexistentzzword']
    report = {'documents':texts, 'queries':queries, 'layouts':{}}
    for layout in ['single','segment']:
        original = work / f'varied-old-{layout}'
        if original.exists(): raise RuntimeError(f'refusing to overwrite {original}')
        with server(a.baseline,layout,original,18781,f'varied-prepare-{layout}'):
            call(18781,'add/batch',[{'request_id':f'varied-{i}', 'messages':[{'role':'user','content':t}], 'user_id':'bench-user', 'session_id':f'varied-session-{i%5}'} for i,t in enumerate(texts)])
        hashes = immutable_files(original)
        versions = {}
        for version,binary in [('0.25.0',a.baseline),('0.26.2',a.candidate)]:
            index = work/f'varied-{layout}-{version}'
            shutil.copytree(original,index)
            with server(binary,layout,index,18781,f'varied-{layout}-{version}'):
                versions[version] = rankings(18781)
            if not hashes or hashes != immutable_files(index): raise RuntimeError('lexical files changed')
            shutil.rmtree(index)
        checks = {q: [x['id'] for x in versions['0.25.0'][q]['data']] == [x['id'] for x in versions['0.26.2'][q]['data']] for q in queries}
        report['layouts'][layout] = {'ordered_ids_preserved':checks, 'responses':versions, 'lexical_files_unchanged':True}
        print(layout,checks,flush=True)
        if not all(checks.values()): raise RuntimeError('ordered search results changed')
        if any(not versions['0.26.2'][q]['data'] for q in queries if q!='nonexistentzzword'): raise RuntimeError('expected positive query returned no results')
    (a.output or work/'varied-report.json').write_text(json.dumps(report,indent=2)+'\n')

else:
    if not a.candidate: p.error('--candidate required for compare')
    report = {'binary_sha256': {v: hashlib.sha256(b.read_bytes()).hexdigest() for v,b in [('0.25.0',a.baseline),('0.26.2',a.candidate)]}, 'records':a.records,'sessions':5,'requests_per_cell':a.requests,'repetitions':a.repetitions,'warmup_requests':100,'runs':[], 'compatibility':[]}
    for layout in ['single','segment']:
        for repetition in range(a.repetitions):
            order = [('0.25.0',a.baseline),('0.26.2',a.candidate)]
            if repetition % 2: order.reverse()
            for version,binary in order:
                label = f'{layout}-{repetition}-{version}'
                index = work/label
                shutil.copytree(work/f'old-{layout}',index)
                with server(binary,layout,index,18780,label):
                    result = rankings(18780)
                    old = json.loads((work/f'{layout}-rankings.json').read_text())
                    def normalized(v):
                        if isinstance(v,dict): return {k:normalized(x) for k,x in v.items() if k not in ['latency_ms','elapsed_ms','timing','diagnostics','score_breakdown']}
                        if isinstance(v,list): return [normalized(x) for x in v]
                        return v
                    compatible = normalized(old) == normalized(result)
                    report['compatibility'].append({'layout':layout,'version':version,'run':repetition,'responses_identical':compatible, 'correct_search_results': all((not result[q]['data']) if q=='nonexistentzzword' else (result[q]['data'][0]['id']==old[q]['data'][0]['id']) for q in ['decision 42','memory record 123','system decision 23365','nonexistentzzword'])})
                    (work/f'{label}-rankings.json').write_text(json.dumps(result,indent=2))
                    completed = subprocess.run([sys.executable,str(ROOT/'comparison/http_latency.py'),'--url','http://127.0.0.1:18780/search','--payload',json.dumps({'query':queries[0],'user_id':'bench-user','top_k':20}),'--requests',str(a.requests),'--warmup-requests','100'],capture_output=True,text=True,check=True)
                    measurements = [json.loads(line) for line in completed.stdout.splitlines()]
                original = json.loads((work/f'{layout}-files.json').read_text())
                if not original: raise RuntimeError('no lexical files checked')
                unchanged = original == immutable_files(index)
                report['compatibility'][-1]['lexical_files_unchanged'] = unchanged
                report['runs'].append({'layout':layout,'version':version,'run':repetition,'measurements':measurements})
                print(label,measurements, 'responses_identical=',compatible,'lexical_unchanged=',unchanged,flush=True)
                shutil.rmtree(index)
    report['summary'] = {}
    for layout in ['single','segment']:
        report['summary'][layout] = {}
        for c in [1,10]:
            vals = {v:[m['throughput_per_s'] for r in report['runs'] if r['layout']==layout and r['version']==v for m in r['measurements'] if m['concurrency']==c] for v in ['0.25.0','0.26.2']}
            med = {v:statistics.median(x) for v,x in vals.items()}
            report['summary'][layout][str(c)] = {'median_req_s':med,'change_percent':100*(med['0.26.2']/med['0.25.0']-1)}
    (a.output or work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))
