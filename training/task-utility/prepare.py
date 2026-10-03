"""Pinned, local real-regression preparation. No model/provider calls."""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / '.pointsman-local/research/task-utility'
PYTHON = sys.executable
CASES = {
    'pointsman-stdin-unref': {
        'family': 'jadeonstudio/pointsman:hook-stdin-event-loop',
        'buggy': '62be7695b8525294838c08775db354b65024fe4d',
        'fixed': '2e75a2c267f61abebd8fc4ff50b72f8663c28f12',
        'issue': 'A hook waiting for stdin that never ends exits before its timeout can produce the expected no-output completion.',
        'test': 'tests/hooks.test.mjs',
        'oracle_source': 'src/hooks.mjs', 'oracle_function': 'withTimeout',
    },
    'sympy-13890': {
        'family': 'sympy/sympy:negative-rational-power-13890',
        'first_fix': 'dec149d4daabdd8fa327bf0ec0b6141063b4e514',
        'fixed': 'bc27c3c036265dc701f2843626ff7220b35ea54b',
        'issue_url': 'https://github.com/sympy/sympy/issues/13890',
        'issue': '(-x/4 - S(1)/12)**x - 1 simplifies to an inequivalent expression. For x=S(9)/5, N(e.subs(x,a)) differs from N(simplify(e).subs(x,a)).',
        'test': 'sympy/core/tests/test_numbers.py',
        'oracle_source': 'sympy/core/numbers.py', 'oracle_function': 'Integer._eval_power',
    },
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def save(name, value):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + '\n')

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'pointsman-local-regression-preparation'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def extract(data, destination, strip_top=False):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = archive.getmembers()
        if strip_top:
            for member in members:
                member.name = member.name.partition('/')[2]
            members = [member for member in members if member.name]
        archive.extractall(destination, members=members, filter='data')

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'pins.json').exists():
        raise SystemExit('Frozen pins already exist; use run, not preparation overwrite.')
    pin = dict(CASES['pointsman-stdin-unref'])
    for arm in ['buggy', 'fixed']:
        archive = subprocess.check_output(['git', 'archive', '--format=tar', pin[arm]], cwd=REPO)
        pin[arm + '_archive_sha256'] = digest(archive)
        extract(archive, OUT / 'pointsman-stdin-unref' / arm)
    pin['license'] = 'MIT'
    pin['license_sha256'] = digest((OUT / 'pointsman-stdin-unref/buggy/LICENSE').read_bytes())
    pin['test_only_patch_sha256'] = digest(b'')
    pins = {'version': 'real-regression-corpus-v1', 'pointsman-stdin-unref': pin}
    sym = dict(CASES['sympy-13890'])
    meta = json.loads(fetch('https://api.github.com/repos/sympy/sympy/commits/' + sym['first_fix']))
    sym['buggy'] = meta['parents'][0]['sha']
    for arm in ['buggy', 'fixed']:
        archive = fetch('https://codeload.github.com/sympy/sympy/tar.gz/' + sym[arm])
        sym[arm + '_archive_sha256'] = digest(archive)
        extract(archive, OUT / 'sympy-13890' / arm, strip_top=True)
    sym['license'] = 'BSD-3-Clause; additional notices retained in LICENSE'
    sym['license_sha256'] = digest((OUT / 'sympy-13890/buggy/LICENSE').read_bytes())
    # The validator uses identical upstream tests on both snapshots; implementation stays unchanged.
    test_bytes = (OUT / 'sympy-13890/fixed' / sym['test']).read_bytes()
    before = (OUT / 'sympy-13890/buggy' / sym['test']).read_bytes()
    (OUT / 'sympy-13890/buggy' / sym['test']).write_bytes(test_bytes)
    sym['test_before_sha256'], sym['test_after_sha256'] = digest(before), digest(test_bytes)
    diff = subprocess.run(['diff', '-u', '--label', 'a/' + sym['test'], '--label', 'b/' + sym['test'], '-', str(OUT / 'sympy-13890/fixed' / sym['test'])], input=before, stdout=subprocess.PIPE, check=False).stdout
    (OUT / 'sympy-13890/test-only.patch').write_bytes(diff)
    sym['test_only_patch_sha256'] = digest(diff)
    pins['sympy-13890'] = sym
    save('pins.json', pins)
    print(json.dumps({'prepared': list(CASES), 'pins': str(OUT / 'pins.json')}))

def runtimes():
    name = 'node-v22.18.0-darwin-arm64.tar.gz'
    base = 'https://nodejs.org/dist/v22.18.0/'
    sums = fetch(base + 'SHASUMS256.txt').decode()
    expected = next(line.split()[0] for line in sums.splitlines() if line.split()[-1] == name)
    archive = fetch(base + name)
    if digest(archive) != expected:
        raise SystemExit('Node archive checksum mismatch')
    extract(archive, OUT / 'runtimes/node22', strip_top=True)
    save('runtime-pins.json', {'node': {'version': 'v22.18.0', 'url': base + name, 'sha256': expected, 'checksum_url': base + 'SHASUMS256.txt'}, 'python': {'version': '3.9.23', 'source': 'uv-managed python-build-standalone', 'mpmath': '1.3.0'}})
    print(json.dumps({'node': str(OUT / 'runtimes/node22/bin/node')}))

def run(command, cwd, timeout=60):
    env = {'PATH': '/usr/bin:/bin:/opt/homebrew/bin', 'HOME': str(OUT / 'isolated-home'), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0'}
    start = time.perf_counter()
    try:
        p = subprocess.run(command, cwd=cwd, env=env, capture_output=True, timeout=timeout)
        return {'command': command, 'exit_code': p.returncode, 'elapsed_ms': (time.perf_counter()-start)*1000, 'stdout': p.stdout.decode(errors='replace'), 'stderr': p.stderr.decode(errors='replace')}
    except subprocess.TimeoutExpired as exc:
        return {'command': command, 'exit_code': None, 'timeout': True, 'elapsed_ms': (time.perf_counter()-start)*1000, 'stdout': (exc.stdout or b'').decode(errors='replace'), 'stderr': (exc.stderr or b'').decode(errors='replace')}

def baseline(root, issue, report):
    """One code batch: reporter paths -> source -> static imports/symbol references.

    This function receives no fix, oracle, expected path, or expected function.
    """
    start = time.perf_counter()
    inventory = [p for p in root.rglob('*') if p.is_file() and p.suffix in {'.mjs', '.py'} and '.git' not in p.parts]
    refs = set()
    # Warning locations are environment evidence, not the failing traceback's source frontier.
    diagnostic = report[report.rfind('Traceback (most recent call last):'):] if 'Traceback (most recent call last):' in report else report
    failure_symbols = set(re.findall(r'line \d+, in (\w+)', diagnostic))
    operation_terms = {'pow', 'power'} if '**' in issue else set()
    for raw in re.findall(r'(?:file://)?((?:/?[\w.@-]+/)*[\w.@-]+\.(?:mjs|py))(?::\d+)?', diagnostic):
        p = Path(raw)
        relative = p.relative_to(root).as_posix() if p.is_absolute() and p.is_relative_to(root) else raw
        if (root / relative).is_file() and not Path(relative).is_absolute() and '..' not in Path(relative).parts:
            refs.add(relative)
    selected = set(refs)
    texts = {}
    for relative in sorted(refs):
        text = (root / relative).read_text(errors='replace'); texts[relative] = text
        if relative.endswith('.py'):
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    source = node.module.replace('.', '/') + '.py'
                    if (root / source).is_file(): selected.add(source)
        else:
            for match in re.finditer(r'import\s*\{([^}]+)\}\s*from\s*[\'"]([^\'"]+)', text):
                target = (root / relative).parent / match[2]
                if target.is_file():
                    selected.add(target.resolve().relative_to(root.resolve()).as_posix())
    # Same-source call expansion, including helper callees of named imported entrypoints.
    packet = []
    for relative in sorted(selected):
        text = texts.get(relative) or (root / relative).read_text(errors='replace')
        if relative.endswith('.py'):
            tree = ast.parse(text)
            functions = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append({'symbol': node.name, 'line': node.lineno, 'end': node.end_lineno})
                if isinstance(node, ast.ClassDef):
                    functions += [{'symbol': node.name + '.' + child.name, 'line': child.lineno, 'end': child.end_lineno} for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
            # A bounded symbol index for issue operators and the observed failing functions.
            functions = [f for f in functions if f['symbol'] in failure_symbols or any(term in f['symbol'].lower() for term in operation_terms)]
        else:
            functions = [{'symbol': m[1], 'line': text[:m.start()].count('\n')+1} for m in re.finditer(r'(?:async\s+)?function\s+(\w+)\s*\(', text)]
        packet.append({'path': relative, 'sha256': digest(text.encode()), 'functions': functions})
    payload = {'version': 'competent-static-batch-v2', 'issue': issue, 'report_source_refs': sorted(refs), 'source_packet': packet, 'index_only': True, 'full_source_retained_at_snapshot': True, 'complete_cause_proven': False, 'decision_calls': 0, 'inference_calls': 0}
    size = len(json.dumps(payload).encode())
    if size > 24000:
        raise RuntimeError('BASELINE_OUTPUT_BUDGET')
    return payload, {'elapsed_ms': (time.perf_counter()-start)*1000, 'packet_bytes': size, 'inventory_files': len(inventory), 'selected_files': len(selected)}

def remeasure_baseline():
    pins = json.loads((OUT / 'pins.json').read_text())
    results = {'version': 'bounded-frontier-index-v2', 'test_runs_reused': 'attempt-2', 'cases': {}}
    for case in CASES:
        pin = pins[case]
        report = json.loads((OUT / 'attempt-2' / (case + '-buggy-run.json')).read_text())
        packet, timing = baseline(OUT / case / 'buggy', pin['issue'], report['stdout'] + '\n' + report['stderr'])
        save('baseline-v2/' + case + '-blind-packet.json', packet)
        found = [p for p in packet['source_packet'] if p['path'] == pin['oracle_source']]
        recalled = bool(found) and any(f['symbol'] == pin['oracle_function'] for f in found[0]['functions'])
        results['cases'][case] = {'baseline': timing, 'oracle_path_function_recalled': recalled, 'semantic_frontier_arm': 'NO-NEED' if recalled else 'UNPROVEN', 'index_only': True, 'task_success': 'NOT_MEASURED'}
    save('baseline-v2/evidence.json', results)
    print(json.dumps(results, indent=2))

def remaining_test():
    command = [str(OUT / '.venv/bin/python'), '-c', 'from sympy.core.tests.test_numbers import test_powers_Integer; test_powers_Integer(); print("PASS: test_powers_Integer")']
    result = run(command, OUT / 'sympy-13890/buggy')
    save('attempt-2/sympy-13890-buggy-powers-run.json', result)
    print(json.dumps(result, indent=2))

def measure():
    pins = json.loads((OUT / 'pins.json').read_text())
    (OUT / 'isolated-home').mkdir(exist_ok=True)
    legacy = len(sys.argv) > 2 and sys.argv[2] == 'legacy'
    attempt = 'attempt-2' if legacy else 'attempt-1'
    node = str(OUT / 'runtimes/node22/bin/node') if legacy else subprocess.check_output(['which', 'node'], text=True).strip()
    python = str(OUT / '.venv/bin/python') if legacy else PYTHON
    runtimes = {'node': run([node, '--version'], REPO), 'python': run([python, '-c', 'import sys,mpmath; print(sys.version); print(mpmath.__version__)'], REPO)}
    results = {'version': 'real-task-preparation-v1', 'runtimes': runtimes, 'cases': {}}
    for case in CASES:
        pin = pins[case]
        records = {}
        for arm in ['buggy', 'fixed']:
            root = OUT / case / arm
            command = [node, '--test', '--test-reporter=tap', '--test-name-pattern=stdin that never ends', pin['test']] if case.startswith('pointsman') else [python, '-c', 'from sympy.core.tests.test_numbers import test_issue_13890, test_powers_Integer; test_issue_13890(); test_powers_Integer(); print("PASS: test_issue_13890 and test_powers_Integer")']
            records[arm] = run(command, root)
            save(attempt + '/' + case + '-' + arm + '-run.json', records[arm])
        root = OUT / case / 'buggy'
        packet, timing = baseline(root, pin['issue'], records['buggy']['stdout'] + '\n' + records['buggy']['stderr'])
        save(attempt + '/' + case + '-blind-packet.json', packet)
        found = [p for p in packet['source_packet'] if p['path'] == pin['oracle_source']]
        oracle_recalled = bool(found) and any(f['symbol'] == pin['oracle_function'] for f in found[0]['functions'])
        results['cases'][case] = {'buggy_exit': records['buggy']['exit_code'], 'fixed_exit': records['fixed']['exit_code'], 'red_green': records['buggy']['exit_code'] not in (0, None) and records['fixed']['exit_code'] == 0, 'baseline': timing, 'oracle_path_function_recalled': oracle_recalled, 'semantic_arm': 'NO-NEED' if oracle_recalled else 'UNPROVEN', 'source_hashes': {arm: digest((OUT / case / arm / pin['oracle_source']).read_bytes()) for arm in ['buggy','fixed']}, 'test_sha256': digest((root / pin['test']).read_bytes())}
    save(attempt + '/evidence.json', results)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    {'prepare': prepare, 'runtimes': runtimes, 'run': measure, 'baseline': remeasure_baseline, 'remaining-test': remaining_test}[sys.argv[1]]()
