# Real task utility seeds

Both pinned regressions reproduce a real failure before the upstream fix and
pass afterward. A competent deterministic batch already recalls each repair's
source path and function from the failure report, imports and symbol index.
The semantic **source-frontier selection** segment is therefore **NO-NEED for
these two cases**. The independent oracle is kept outside the blind builder.

| Case | Runtime | Buggy → fixed | Bounded batch | Packet file |
| --- | --- | --- | --- | --- |
| pointsman pending stdin | Node 22.18.0 | exit 1 → 0 | 10.385 ms | 8,060 bytes |
| SymPy #13890/#13895 | Python 3.9.23, mpmath 1.3.0 | AssertionError → PASS | 103.553 ms | 8,536 bytes |

[evidence.json](evidence.json) contains public revisions, licenses, hashes,
test IDs, recall and measurement scope. Timings are one local observation after
test execution; OS cache was not controlled. Compact JSON sizes are separately
recorded. Packets are indexes, not complete cause proofs or generated repairs.
Task success, parent-request savings, p50/p95, quality and model qualification
remain unmeasured. Runner model/provider/GPU calls were zero; preparation-agent
cost is UNKNOWN. No runtime mode or checkpoint was enabled.

Preparation issues, preserved separately from quality outcomes:

- Node 26.5.0 passed both revisions; it did not reproduce the historical runtime
  failure. Node 22.18.0 established the regression without source changes.
- Python 3.12.8 failed while importing the old `collections.Mapping` API.
  Python 3.9.23 ran the original source without compatibility patches.
- The first SymPy index was 105,588 bytes, above the 24,000-byte envelope.
  Restricting the traceback frontier and operator/function index produced the
  bounded packet using the same captured test results; tests were not repeated.

## Reproduce

These commands are for a fresh checkout/evidence directory on macOS arm64 with
Git, Python 3 and uv already available. They fetch pinned public artifacts and
write only task-local files; no system software is installed. `prepare` refuses
to overwrite an existing pins file. Existing passing evidence need not be rerun.

```sh
python3 training/task-utility/prepare.py prepare
python3 training/task-utility/prepare.py runtimes
uv --cache-dir .pointsman-local/research/task-utility/uv-cache python install --install-dir .pointsman-local/research/task-utility/runtimes/python 3.9.23
uv --cache-dir .pointsman-local/research/task-utility/uv-cache venv --python .pointsman-local/research/task-utility/runtimes/python/cpython-3.9.23-macos-aarch64-none/bin/python3.9 .pointsman-local/research/task-utility/.venv
uv --cache-dir .pointsman-local/research/task-utility/uv-cache pip install --python .pointsman-local/research/task-utility/.venv/bin/python mpmath==1.3.0
python3 training/task-utility/prepare.py run legacy
python3 training/task-utility/prepare.py remaining-test
python3 training/task-utility/prepare.py baseline
```

The remaining-test command covers the buggy integer-power test that the first
failing SymPy assertion prevented from running. The baseline command reuses
captured reports. The two source families and all descendant views must remain
together in any later split. This public folder includes the oracle and must
not be supplied to a blind model arm; use only the separated neutral-ID packet.
Owned dependency download cache was removed after preparation. Licensed source
snapshots, isolated runtimes and evidence were retained for readback.
