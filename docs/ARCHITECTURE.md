# Architecture

Placement Gate is an adapter over one upstream package. This document records
exactly what is delegated and what — almost nothing — is local.

## The shape

```
                    ┌────────────────────────────────────────┐
                    │            pgate_demo.cli              │
                    │  doctor · hardware · census · plan ·   │
                    │  place · selftest                      │
                    └──────────────────┬─────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────┐
                    │         pgate_demo.core                 │
                    │  PlacementGateSession                   │
                    │  observe_hardware · census              │
                    │  model_facts · plan · place             │
                    │  observe_residency · unload             │
                    │  characteristics                        │
                    └──────────────────┬─────────────────────┘
                                       │  ProviderSet.require(name)
                    ┌──────────────────▼─────────────────────┐
                    │       pgate_demo.providers              │
                    │  named surfaces only, resolved lazily    │
                    │  named surfaces only, resolved lazily    │
                    │  1. importlib by name, from the dependency │
                    │  2. the declared dependency              │
                    └───────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────┐
                    │   pgate_demo.placement (self-contained)  │
                    │                                         │
                    │  hardware/{facts,observer,windows_      │
                    │            memory,nvidia_query}         │
                    │  policy/{model_profile,admission,       │
                    │          inference_placement}           │
                    │  backends/{backends,chat}      <- I/O   │
                    │  ledger/{ledger,models,residency}       │
                    │  controller/placement       <- the seam  │
                    │  topology                                │
                    │                                         │
                    │  no dependencies but pydantic            │
                    └─────────────────────────────────────────┘
```

## What is delegated, and to where

| Placement Gate operation | Delegates to | Decides |
|---|---|---|
| `observe_hardware` | `hardware.observer.RealHardwareObserver` + `windows_memory` + `nvidia_query` | what the machine has, and whether the evidence is contradictory |
| `census` | `backends.chat.OllamaChatAdapter` (admitted GETs) | which models the service offers |
| `plan` | `policy.inference_placement.plan_inference_placement` | whether a model fits, in which topology, with which reason |
| `place` | `placement/controller/placement.py` → `PlacementRunner.place_and_load_model` | the whole lifecycle, including whether to keep the model loaded |
| `observe_residency` | the service's `/api/ps`, decoded as raw evidence | nothing — it reports |
| `unload` | the adapter's admitted `POST /api/chat` with `keep_alive=0` | nothing — it is an explicit request |
| `characteristics` | **nothing** — an explicit unverified JSON read | nothing; the payload carries `verified: false` |

### The one thing that is not upstream

`fully_gpu_resident` in `observe_residency` is a local presentation helper:

```python
"fully_gpu_resident": bool(vram and size and vram >= size)
```

It is a rendering of observed bytes, not a second placement decision. The
decision is `verify_placement`'s, upstream, and Placement Gate reports its verdict
verbatim. This is asserted by a test that scans the adapter for budget
arithmetic and verdict literals.

## The three upstream facts this demo depends on

**1. A plan grants nothing.** `ModelAdmissionPlan` is a request. It has no
execution authority, and an admitted plan is not evidence that anything is
loaded. `pgate plan` says so in its output on every run.

**2. A verdict requires observed evidence.** `verify_placement` derives its
result from four channels — what the service reported, GPU-attributable bytes,
host-attributable bytes, and attribution status. Any required channel missing or
`UNKNOWN` yields `UNCONFIRMED`, never a pass.

**3. Identity is separate from size.** `derive_attestation` judges topology from
byte counts. It does **not** compare digests. The identity check therefore lives
at the observation site in `place_and_load_model`: a residency entry whose
reported digest contradicts the planned digest is refused as
`identity_mismatch_after_load`, before verification is even attempted.

That last point is worth stating plainly, because it is the one place where
Placement Gate's demo would be actively misleading if the check were missing. A
tag that is resident under different weights than the plan named would otherwise
report `VERIFIED`.

## Two failure modes this package is built to avoid

**Eager import.** `import pgate_demo` imports no core module. Resolution is
lazy: each capability is resolved on first use. A test asserts that importing
the package leaves `sys.modules` free of `pgate_demo.placement`, and another asserts
the provider layer contains no `os.walk`, `pkgutil`, `glob` or `scandir`.

This is not squeamishness. The estate this demo is drawn from contains a
directory where 104 of 174 modules execute behaviour at import — one of them
performs a governed filesystem deletion, another shells out to `cmd /c` and
rebuilds a CUDA project. Importing a source tree is a *mutating operation*
there. A tool whose job is to inspect a machine has no business doing that.

**Substituted answers.** A missing capability produces `UNAVAILABLE` and a
non-zero exit. It never produces a plausible-looking plan. Four tests pin this,
including one that asserts the unavailable payload contains no `models` key at
all.

## Layering rules this codebase follows

- **No placement policy.** No budget arithmetic, no verdict literals, no
  topology inference. The upstream planner is called and its result rendered.
- **No widened allowlist.** Exactly three operations are used: `GET /api/tags`,
  `GET /api/ps`, `POST /api/chat`. Upstream's chat adapter admits exactly those
  three and refuses anything else — including `POST /api/generate`, which was
  the obvious way to release residency. Placement Gate uses the admitted
  operation instead of asking for a new one, and a test enforces it.
- **No token generation.** Placement is an observation. `send_chat` and
  `stream_chat` appear nowhere in the package.
- **No implicit cleanup.** `place` leaves the model loaded. `--unload-after` is
  the only path that releases it, and it says so.
- **No machine paths in version control, and nowhere they could go.**
  `pgate.providers.json` is deleted; resolution is a named import of a module
  inside this package, so there is no file left in which a local path could be
  recorded.
- **No private data in output.** Private payload keys are stripped before
  rendering. `census` reports tag, digest, size, context, parameters,
  quantization and residency — and nothing else.
