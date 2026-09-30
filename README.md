# Placement Gate

**This machine has a finite hardware budget. Placement Gate can inspect that budget, refuse a model that will not fit, load one that should fit, and then verify that the expected model actually became resident under the expected identity.**

Placement Gate is a small command-line application assembled almost entirely
from capability that already exists upstream. It binds to the
`ollama_controller` package at runtime and **reimplements none of it** — no
placement policy, no budget arithmetic, no verification rule. Every refusal and
every verdict printed here was decided upstream and is reported with the
upstream's own reason string.

```console
$ pgate hardware
detection status : OK   contradictory=False
memory           : 63.82 GiB total, 46.94 GiB available
gpu              : DISCRETE NVIDIA GeForce RTX 3070 Laptop GPU (OK)
vram             : 8.0 GiB total, 7.0 GiB free
profile sha256   : 9e0aba616a46fc30...
note             : contradictory or unknown evidence must fail closed; it is never rounded into a capacity

$ pgate plan qwen3.5:9b-opencode-32k
model      : qwen3.5:9b-opencode-32k  (6.14 GiB, digest 461e5a8986d9f7a0)
hardware   : detection=OK  vram_total=8.0 GiB  free=7.0 GiB
requested  : DEVICE_ONLY num_gpu=99  ctx=8192

ADMITTED   : False
REASON     : EXCEEDS_RESOURCE_BUDGET
TOPOLOGY   : PROHIBITED
REQUIRED   : 7.14 GiB   budget_vram: 7.0 GiB
DETAIL     : DEVICE_ONLY exceeds VRAM budget
note       : a plan is a request. It grants no execution authority and is not evidence that anything is loaded.
$ echo $?
0
```

That exit code is deliberate. The command ran, and the answer is "no". See
[Exit codes](#exit-codes).

---

## Contents

- [Install](#install)
- [Point it at the upstream package](#point-it-at-the-upstream-package)
- [Walkthrough](#walkthrough)
- [The two claims: planning vs. live verification](#the-two-claims-planning-vs-live-verification)
- [Partial residency is not GPU residency](#partial-residency-is-not-gpu-residency)
- [Identity](#identity)
- [Residency is never released implicitly](#residency-is-never-released-implicitly)
- [Commands](#commands)
- [Exit codes](#exit-codes)
- [Tests](#tests)
- [Known limits](#known-limits)
- [License](#license)

---

## Install

```console
$ python -m venv .venv
$ .venv/Scripts/python -m pip install -e .        # Windows
$ .venv/bin/python -m pip install -e .             # POSIX
```

Placement Gate has **no runtime dependencies of its own** (`dependencies = []`).
It does need `pydantic`, which the upstream package requires:

```console
$ python -m pip install -e . pydantic
```

If `pydantic` is absent, `pgate` says exactly that rather than failing obscurely:

```
detail     : env:PGATE_UPSTREAM_PATH: ModuleNotFoundError: No module named 'pydantic'
```

## Point it at the upstream package

The `ollama_controller` package is not on PyPI. Point Placement Gate at a copy:

```console
$ export PGATE_UPSTREAM_PATH=/path/to/Ollama_Controller/src
$ pgate doctor
upstream capability load
  [ok  ] backends_base          AVAILABLE
  [ok  ] chat_adapter           AVAILABLE
  [ok  ] controller             AVAILABLE
  [ok  ] hardware_facts         AVAILABLE
  [ok  ] hardware_memory        AVAILABLE
  [ok  ] hardware_nvidia        AVAILABLE
  [ok  ] hardware_observer      AVAILABLE
  [ok  ] ledger                 AVAILABLE
  [ok  ] model_profile          AVAILABLE
  [ok  ] placement_planner      AVAILABLE
```

If the upstream package is installed normally, `PGATE_UPSTREAM_PATH` is not
needed. Nothing is written to disk to find it: if a capability cannot be
resolved, `pgate` says so and exits non-zero.

---

## Walkthrough

The transcript below is from the development machine. **Your numbers will
differ** — that is the point. What transfers is the shape: a budget, a refusal,
a load, and an identity check.

**1. What does this machine actually have?**

```console
$ pgate hardware
detection status : OK   contradictory=False
memory           : 63.82 GiB total, 46.94 GiB available
gpu              : DISCRETE NVIDIA GeForce RTX 3070 Laptop GPU (OK)
vram             : 8.0 GiB total, 7.0 GiB free
```

**2. A model that will not fit, refused before anything is loaded**

```console
$ pgate place qwen3.5:9b-opencode-32k
model      : qwen3.5:9b-opencode-32k
ADMITTED   : False
REASON     : EXCEEDS_RESOURCE_BUDGET
DETAIL     : DEVICE_ONLY exceeds VRAM budget
LOAD       : not attempted (refused before any request)
```

Nothing was requested of the model server. The refusal happened in the
planner, against measured VRAM.

**3. A model that should fit**

```console
$ pgate plan qwen3.5:2b-opencode-64k
ADMITTED   : True
REASON     : PLANNED
TOPOLOGY   : GPU_RESIDENT
REQUIRED   : 3.55 GiB   budget_vram: 7.0 GiB
PLAN HASH  : 0000a57ebb6537d2f37793a23ad766a900db708809e3df0593a5af9eaeb7ef95
```

**4. Load it, and verify what actually happened**

```console
$ pgate --text place qwen3.5:2b-opencode-64k --unload-after
model      : qwen3.5:2b-opencode-64k  digest faf2034e63b8a04f
VERDICT    : ADMITTED
ADMITTED   : True
RESERVED   : 3.55 GiB VRAM
PLAN BOUND : True
RESIDENCY  : qwen3.5:2b-opencode-64k  size=2.3 GiB  size_vram=2.3 GiB  fully_gpu=True
IDENTITY   : digest match=True  tag match=True
UNLOADED   : qwen3.5:2b-opencode-64k still resident: False  (service resident count now 0)
```

**5. And the case that matters most — a load that does not satisfy the request**

```console
$ pgate --text place qwen3.5:2b-opencode-64k --num-gpu 1 --unload-after
model      : qwen3.5:2b-opencode-64k  digest faf2034e63b8a04f
VERDICT    : VERIFICATION_FAILED
DETAIL     : placement verification failed: FAILED; model loaded but evidence
             does not match requested placement; fail closed
RESIDENCY  : qwen3.5:2b-opencode-64k  size=2.88 GiB  size_vram=0.64 GiB  fully_gpu=False
IDENTITY   : digest match=True  tag match=True
```

`size_vram` is non-zero, so something is on the GPU. The request was full GPU
residency, and 0.64 GiB of 2.88 GiB is not that. The system refuses. It does not
round the observation up to match what was asked for.

Note the subtlety: `--num-gpu 1` asks Ollama for **one layer** on the GPU, not
"one GPU". Ollama's own semantics. Placement Gate passes the value through and
lets the verification step judge the result.

---

## The two claims: planning vs. live verification

Placement Gate reports which of these you are looking at, and never blurs them.

| | `pgate plan` | `pgate place` |
|---|---|---|
| Reads measured hardware | yes | yes |
| Computes a content-addressed plan | yes | yes |
| Refuses when the model will not fit | yes | yes |
| **Requests a load** | **no** | yes |
| **Observerves post-load residency** | **no** | yes |
| **Verifies the placement** | **no** | yes |
| Evidence produced | a request | a verified result |

A plan is a request. It grants no execution authority and is not evidence that
anything is loaded. `pgate plan` says so in its own output, every time.

`pgate place` on a refused plan does not attempt the load. To override that and
force the request anyway, pass `--execute-anyway`.

---

## Partial residency is not GPU residency

The upstream verifier derives a verdict from four independent channels — what the
server reported, GPU-attributable bytes, host-attributable bytes, and
attribution status. `size_vram > 0` is one input, not a verdict.

Placement Gate surfaces the observed numbers and never summarises them as
"satisfied":

```
RESIDENCY  : qwen3.5:2b-opencode-64k  size=2.88 GiB  size_vram=0.64 GiB  fully_gpu=False
```

`fully_gpu` is computed as `size_vram >= size`. It is a presentation of the
observed evidence, not a second placement decision — the decision is upstream's.

## Identity

A model is **not** verified because the tag matched, because the size is about
right, or because something is resident. When the server reports a digest, it
must agree with the digest the plan named:

```console
$ pgate --text place <model>
IDENTITY   : digest match=True  tag match=True
```

Upstream refuses a load whose residency entry reports a *different* digest than
the plan requested, and reports it as `identity_mismatch_after_load`. A tag can
be re-pointed at different weights between the load request and the observation;
a placement verdict is a claim about the weights that actually ran.

A model *name* is not an identity either. `pgate plan qwen3.5` on a service
holding several `qwen3.5:*` weights is refused as ambiguous rather than resolved
by preference:

```
status     : AMBIGUOUS
detail     : 'qwen3.5' names 11 distinct models. A model name is not an identity: pass an exact tag.
```

## Residency is never released implicitly

`pgate place` loads a model and **leaves it loaded**. Residency is the operator's
to manage. Pass `--unload-after` to ask for the release, or do it yourself.

This is not caution; it mirrors upstream. A failed verification deliberately
*retains* the reservation, because the model may be partially loaded and
releasing it blindly could destroy state the operator did not ask you to
destroy. `pgate` reports the residency it observes so you can see what needs
cleaning up.

---

## Commands

| Command | Does | Needs the service |
|---|---|---|
| `pgate doctor` | what upstream capabilities resolved, and how | no |
| `pgate hardware` | observed hardware facts | no |
| `pgate census` | the local service's models, minimally described | yes |
| `pgate plan <model>` | compute a plan. Loads nothing | yes, unless `--size-bytes` |
| `pgate place <model>` | execute the placement path | yes |
| `pgate selftest` | deterministic checks | no (`--live` to include a load) |
| `pgate exit-codes` | print the contract this CLI implements | no |

Useful flags:

- `--endpoint host:port` — where the model service is (default `127.0.0.1:11434`)
- `--placement DEVICE_ONLY|HOST_ONLY|HYBRID_STATIC|MANAGED_FALLBACK`
- `--num-gpu N` — passed through to the service unchanged
- `--context N` — the context window to budget for (default: the smaller of the
  model's declared capacity and 8192)
- `--unload-after` — release residency after a live placement
- `--execute-anyway` — attempt the load even when the plan was refused
- `--live` — include a real load and unload in `selftest`
- `--text` — human-readable output; JSON is the default

`census` reports only what a placement decision uses: tag, digest, size, context
length, parameter label, quantization, and whether it is currently resident. No
file paths, no modification timestamps, no model contents.

## Exit codes

Execution status and domain verdict are separate. A governed refusal is a
successful evaluation.

| Code | Meaning |
|---:|---|
| 0 | the requested evaluation completed — **including a refusal** |
| 2 | a required upstream capability was unavailable |
| 3 | the model service was unreachable |
| 4 | a `selftest` assertion did not hold |
| 64 | the invocation was malformed |

```console
$ pgate plan qwen3.5:9b-opencode-32k >/dev/null; echo $?
0                                  # refused, and that is a correct answer
$ pgate --endpoint 127.0.0.1:1 census >/dev/null; echo $?
3                                  # could not ask the question
$ pgate plan --bogus >/dev/null 2>&1; echo $?
64                                 # a typo
```

These domain outcomes are **all** exit 0 — the question was asked and answered:

| Outcome | Means |
|---|---|
| `ADMITTED` | loaded, and the placement verified |
| `REFUSED` | upstream refused at the planning stage |
| `EXCEEDS_RESOURCE_BUDGET` | the model needs more than the observed budget |
| `HARDWARE_UNKNOWN` | hardware evidence absent or contradictory |
| `VERIFICATION_FAILED` | loaded, but evidence does not match the request. Fail-closed, model left loaded |
| `NOT_FOUND` | no such tag, or the name is ambiguous across several |
| `UNAVAILABLE` | a capability was missing; nothing was substituted |
| `SERVER_UNREACHABLE` | the service could not be asked |

Run `pgate exit-codes` to print the full per-command contract from the code that
implements it.

## Tests

```console
$ python -m pytest -q
46 passed
```

The suite asserts that this package **delegates**. It checks that the adapter
contains no placement arithmetic, that it never widens the upstream operation
allowlist, that it never generates tokens, that an ambiguous model name is
reported rather than resolved, that partial residency is not rounded up, and
that a missing capability produces no substituted answer. The placement rules
themselves are tested upstream and are not duplicated here.

Without the upstream package the suite skips rather than inventing behaviour.

## Known limits

- **`--num-gpu` is the service's semantics, not this architecture's.** The
  upstream module is explicit that it "does not claim that Ollama's `num_gpu`
  input is an exact layer-placement control". Small values mean a layer count.
- **A load is not digest-pinned on the wire.** The request carries a tag; the
  digest is checked at observation time. There is a window between the two, and
  closing it would need a service-side pin.
- **`--context` defaults conservatively** to 8192 tokens when a model declares
  more. A larger window needs a larger budget, and the planner will refuse if it
  does not fit. Pass `--context` to see the real figure.
- **Failures do not unload.** Reported, not acted on.
- **`HOST_ONLY` and `HYBRID_STATIC`** are reachable but were not live-qualified
  on the development machine. `HYBRID_STATIC` additionally needs an explicit
  capability grant upstream, or it is refused as `CAPABILITY_UNVERIFIED`.

## License

**ADAPTER-ONLY. Upstream licensing unresolved.** This repository's own code is
MIT. It redistributes no upstream code, no model weights, and no internal
corpora or machine state. See [`LICENSE`](LICENSE) and
[`docs/LICENSING.md`](docs/LICENSING.md).

## See also

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — what is delegated where
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — what is copied, what is bound
- [`docs/LICENSING.md`](docs/LICENSING.md) — upstream ownership and the open question
- [`docs/DEMO-NOTES.md`](docs/DEMO-NOTES.md) — how each claim was verified
