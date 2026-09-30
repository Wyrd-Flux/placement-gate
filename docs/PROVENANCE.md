# Source provenance

Recorded 2026-09-30.

## Upstream sources

| Upstream | Location | Used for |
|---|---|---|
| `ollama_controller` | `Ollama_Controller/src` (externalised gitdir under `C:/LocalGitDirs/`) | every decision |
| `ollama_controller.registries.model_characteristics` | same package | optional characteristics read |
| Model Characteristics Registry | `C:/Projects/Model_Characteristics_Registry` | optional, operator-supplied root |

`ollama_controller` is a single package that Placement Gate consumes whole. It is
not vendored, not copied, and not modified by this repository.

## What was copied: nothing

```console
$ grep -rnE "class (HardwareProfile|ModelProfile|PlacementPolicy|ModelAdmissionPlan|OllamaChatAdapter|Controller)\b" pgate_demo/
$ # no matches
```

| File | Role |
|---|---|
| `pgate_demo/providers.py` | named-surface resolution, lazy |
| `pgate_demo/core.py` | session; builds upstream inputs, unpacks upstream results |
| `pgate_demo/cli.py` | argument parsing, rendering, exit-code mapping |
| `pgate_demo/exit_codes.py` | the documented contract |
| `pgate_demo/selftest.py` | conformance checks against the bound surfaces |
| `pgate_demo/pgate.providers.json` | empty search-path list + documentation |
| `tests/test_pgate.py` | 46 adapter tests |
| `README.md`, `LICENSE`, `docs/*`, `pyproject.toml` | packaging and documentation |

The only local computation over upstream data is `fully_gpu_resident`, a boolean
rendering of two observed byte counts. It is a presentation helper; the
placement decision is upstream's. A test scans the adapter for budget
arithmetic and verdict literals to keep it that way.

## Three adaptations, and why

Each is at the call site, documented where it occurs, and removable.

### 1. Model identity comes from the plan, not the model

`ModelProfile` carries sizing metadata — digest, family, quantization, observed
bytes. It has no model tag, no context window and no keep-alive. The plan does
carry all three. So the load profile is built from `plan.model_id` and
`plan.requested_context_tokens`, and identity therefore cannot diverge between
the request, the observation and the plan hash.

### 2. `--num-gpu` is passed through, not interpreted

Ollama reads a small `num_gpu` as a **layer** count. Live evidence on the
development machine:

| `--num-gpu` | observed `size_vram` | verdict |
|---:|---|---|
| 1 | 0.64 GiB of 2.88 GiB | refused: not GPU residency |
| 0 | 0 | CPU-resident |
| 99 | 2.30 GiB of 2.30 GiB | verified |
| -1 | 2.30 GiB of 2.30 GiB | verified, but rejected upstream by the policy validator |

Placement Gate passes the operator's value through unchanged and lets
verification judge the result. The alternative — remapping values to suit
Ollama — would put placement policy in the adapter, which is the one thing this
package must not do.

### 3. Residency is released through an admitted operation

Releasing a model needs `keep_alive=0`. The obvious call is
`POST /api/generate`, and the upstream chat adapter **refuses** it: it admits
exactly `GET /api/tags`, `GET /api/ps` and `POST /api/chat`. That refusal is the
system working.

Placement Gate uses the admitted operation instead:

```python
adapter._request("POST", "/api/chat",
                 {"model": tag, "messages": [], "stream": False, "keep_alive": 0})
```

An empty message list with `keep_alive=0` releases the model without generating.
Confirmed live: the service returns `"done_reason": "unload"` and `/api/ps`
empties. A test asserts `/api/generate` appears nowhere in the adapter's
executable code.

## Upstream code that was repaired before packaging

Placement Gate packages a repaired path. The repair is recorded here because the
demo's central claim depends on it.

`Controller.place_and_load_model()` had never executed. Its first statement
imported a symbol from the wrong module, and five further names were referenced
but never imported. No test called the method; the 14 existing placement tests
exercise the planning functions and bypass it.

Repaired in `Ollama_Controller` on 2026-09-30, in the runtime seam only:

- 8 unresolved imports/names corrected to their authoritative modules
- 7 further defects proven by live execution and fixed: a wrong ledger call
  signature, a raw connection passed where a transaction handle is required, an
  illegal visibility value, a nonexistent keyword argument, three field reads
  from a class that has none of them, and — the substantive one — **the method
  performed no load at all**, reading `/api/ps` and hoping.
- a digest-identity check added at the observation site, closing a gap the
  method's own comment declared but the code did not implement
- `tests/test_placement_runtime.py` added: 10 method-level tests

Full record: `Ollama_Controller/RESOURCE_AWARE_PLACEMENT_VERIFICATION.md`.

**No planner semantics were changed.** `policy/inference_placement.py`,
`hardware/`, `core/`, `backends/` and `ledger/` were not touched.

## What this repository does not redistribute

- any upstream source
- any model weights, GGUF files, or blobs
- any ledger, corpus, or machine-specific state
- any characteristics registry data

`pgate census` reads the local service at runtime and reports what it finds. It
does not cache, persist, or transmit it.

## Licensing status

This repository's code: MIT.

`ollama_controller`: not published on PyPI as of 2026-09-30, and no license file
in its source tree. The Model Characteristics Registry carries no license file
either.

Nothing upstream is redistributed, so publication requires no grant. But absent a
license that code is all-rights-reserved by default, so this repository has no
durable right to depend on it. See `docs/LICENSING.md`.

## Verifying these claims

```console
$ pgate doctor
```

`doctor` prints, per capability, the module name and the resolution source. That
is the receipt for the whole of this document.
