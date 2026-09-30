# Demo notes

Recorded 2026-09-30 on the development machine, for reviewers who want to know
which claims were actually executed rather than argued.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU, 8.0 GiB VRAM |
| Host RAM | 63.82 GiB |
| Service | Ollama at `127.0.0.1:11434`, 44 models |
| Dependency | `wyrd-placement-core` 0.1.0 from GitHub |
| Private estate | **not required, and not consulted** |
| Models present | 44 |
| Upstream | `Ollama_Controller/src` working tree, repaired seam |

**None of these numbers transfer to your machine, and the demo is more useful
for that.** What transfers is the shape of each result.

## Claim 1 — a model that cannot fit is refused before anything is loaded

```console
$ pgate --text place qwen3.5:9b-opencode-32k --context 32768
model      : qwen3.5:9b-opencode-32k
ADMITTED   : False
REASON     : EXCEEDS_RESOURCE_BUDGET
REQUIRED   : 10.14 GiB   budget_vram: 7.0 GiB
LOAD       : not attempted (refused before any request)
```

The same model at the default 8192-token context requires 7.14 GiB against a
7.0 GiB free-VRAM budget and is still refused — the budget is real, and the
margin is thin.

Confirmed: `/api/ps` was empty before and after. Nothing was requested.

## Claim 2 — a model that fits is loaded, and the result is verified

```console
$ pgate --text place qwen3.5:2b-opencode-64k --unload-after
VERDICT    : ADMITTED
RESERVED   : 3.55 GiB VRAM
PLAN BOUND : True
RESIDENCY  : qwen3.5:2b-opencode-64k  size=2.3 GiB  size_vram=2.3 GiB  fully_gpu=True
IDENTITY   : digest match=True  tag match=True
UNLOADED   : still resident: False  (service resident count now 0)
```

Verified: `size_vram == size`, digest matched the plan, residency released on
request.

## Claim 3 — partial residency is refused, not rounded up

This is the claim the demo exists for.

Re-verified live **after** the migration to `wyrd-placement-core`:

```console
$ pgate --text place tinyllama:1.1b --keep-alive 20s --unload-after
VERDICT    : ADMITTED
RESIDENCY  : tinyllama:1.1b  size=0.65 GiB  size_vram=0.65 GiB  fully_gpu=True
IDENTITY   : digest match=True  tag match=True
UNLOADED   : tinyllama:1.1b still resident: False  (service resident count now 0)

$ pgate --text place tinyllama:1.1b --num-gpu 1 --keep-alive 20s --unload-after
VERDICT    : VERIFICATION_FAILED
DETAIL     : placement verification failed: FAILED; model loaded but evidence
             does not match requested placement; fail closed
RESIDENCY  : tinyllama:1.1b  size=0.7 GiB  size_vram=0.1 GiB  fully_gpu=False
IDENTITY   : digest match=True  tag match=True
UNLOADED   : tinyllama:1.1b still resident: False  (service resident count now 0)
$ echo $?
0
```

`size_vram` is non-zero — one layer really is on the card. The request was full
GPU residency and 0.1 of 0.7 GiB is not that, so the system refuses, and the
refusal is still exit 0 because the evaluation completed.

Measured across the service's own `--num-gpu` semantics:

| `--num-gpu` | observed `size_vram` | verdict |
|---:|---|---|
| 0 | 0 | CPU-resident, as asked |
| 1 | 0.1 GiB of 0.7 GiB | **refused** — one layer is not full residency |
| 99 | 0.65 GiB of 0.65 GiB | verified |

**A note on the development machine.** The laptop crashed partway through this
session and the model service went down with it. It was restarted and the live
claims above were re-run from scratch against the migrated code, using the
smallest local weight and a short `--keep-alive` so the GPU was under the lightest
possible load. Nothing about the crash was reproduced, and no evidence from before
the crash is relied on above.

## Claim 4 — identity is checked separately from size

A tag that is resident under different weights than the plan named must not
report `VERIFIED`. Upstream's own comment for `place_and_load_model` declares
this requirement and the original code did not implement it; the repair added the
check at the observation site.

Verified live:

- correct digest → `identity match=True`, `VERDICT: ADMITTED`
- wrong digest injected at the observation point → `identity_mismatch_after_load`,
  refused before verification, fail-closed

## Claim 5 — residency is never released implicitly

`pgate place` leaves the model loaded and says so. Two consecutive full runs
above ended with an explicit resident count of 0 only because `--unload-after`
was passed. Without that flag, `/api/ps` retains the model.

A failed verification also retains residency by design: the model may be
partially loaded, and blind release could destroy state the operator did not ask
to destroy. Placement Gate reports the residency so it is visible.

## Claim 6 — a missing capability produces no answer

```console
$ # with wyrd-placement-core uninstalled, the import fails at module load and
$ # pgate refuses to start rather than answering from a substitute
ModuleNotFoundError: No module named 'wyrd_placement_core'
```

No plan, no estimate, no fallback. Pinned by four tests, one asserting the
unavailable payload contains no `models` key at all.

## Claim 7 — import is not mutation

`import pgate_demo` imports no upstream module. Asserted by a test that inspects
`sys.modules` after import.

The reason is specific rather than aesthetic: in the estate this demo was drawn
from, 104 of 174 modules in one source tree execute behaviour at import — one
performs a governed filesystem deletion, another shells out to `cmd /c` and
rebuilds a CUDA project. A tool that exists to inspect a machine has no business
importing one that mutates it on sight.

## Claim 8 — the adapter has no placement logic of its own

46 tests, including source scans asserting the absence of:

- budget arithmetic and verdict literals (`ADMITTED`, `VERIFIED`,
  `EXCEEDS_RESOURCE_BUDGET`, …)
- tree scanning and side-effect imports (`os.walk`, `pkgutil`, `glob`,
  `scandir`, bare `subprocess`, `os.system`)
- any HTTP path outside the three upstream admits
- token generation (`send_chat`, `stream_chat`)

## What was **not** verified

Stated so the record is honest:

- **`HOST_ONLY` and `HYBRID_STATIC`** are reachable but not live-qualified on
  this machine. `HYBRID_STATIC` additionally needs an explicit capability grant
  upstream and is refused as `CAPABILITY_UNVERIFIED` without one.
- **The characteristics registry could not be read.** Its own verifier rejects
  its current state:
  `effective state digest does not match registry_digest (corruption?)`. The
  registry was not modified and no substitute source was used.
- **No concurrency claim.** Residency is read back immediately after the load
  request. Under concurrent loads, the read-back reflects whatever the service
  reported at that instant.
- **No load is digest-pinned on the wire.** The request carries a tag; the digest
  is verified at observation. There is a window between them.
- **This is a workstation.** One GPU, one service, one host. Nothing here is a
  claim about multi-node or multi-tenant behaviour.

## Reproducing

```console
$ git clone https://github.com/Wyrd-Flux/placement-gate
$ cd placement-gate
$ python -m venv .venv && .venv/bin/python -m pip install .
$ pgate doctor
$ pgate --text selftest --live
```

`selftest --live` loads and releases exactly one real model. It picks the
smallest **local** weight, skipping cloud placeholders — Ollama lists remote
models with a tag and a near-zero size, and asking one to become resident locally
would fail for reasons that have nothing to do with placement.

Expected tail:

```
  [PASS] live: model became resident after the load    resident=1
  [PASS] live: residency digest matches the planned model
  [PASS] live: full GPU residency observed, not rounded
  [PASS] live: cleanup released residency              resident after unload: 0
  selftest passed
```

## Upstream repair, for context

Placement Gate packages a seam that had never run. Its first statement imported
a symbol from the wrong module, and seven further defects were found by executing
it. Recorded in `Ollama_Controller/RESOURCE_AWARE_PLACEMENT_VERIFICATION.md`,
with 10 method-level tests added there.

`policy/inference_placement.py` — the planner whose reasoning Placement Gate
surfaces — was **not** modified. Only the runtime seam that executes a plan was
repaired.
