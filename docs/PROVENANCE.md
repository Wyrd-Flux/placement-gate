# Source provenance

Recorded 2026-09-30, and updated three times: for the migration to a public core,
then for folding that core back in, and now for carrying the seam's own tests
with it. Each stage is recorded, because each happened.

## What Placement Gate depends on

```
placement-gate → pydantic → Python standard library
```

One dependency, and it is not a Wyrd Flux package. `pgate.providers.json` is
deleted; resolution is a named import of a module inside this repository.

## The three stages

### Stage 1 — adapter-only (`1c42286`)

Bound to an internal `ollama_controller` at runtime via `PGATE_UPSTREAM_PATH`.
Redistributed nothing upstream, so needed no license — and consequently could not
be run by anyone who did not already have the internal source tree.

### Stage 2 — public core (`bb5e52f`)

The placement primitives were extracted into
[`Wyrd-Flux/wyrd-placement-core`](https://github.com/Wyrd-Flux/wyrd-placement-core)
(Apache-2.0) and taken as a git dependency. Runnable by anyone; a
cross-repository dependency with exactly one consumer, pinned to a branch.

### Stage 3 — folded back in (this release)

The primitives live in [`pgate_demo/placement/`](https://github.com/Wyrd-Flux/placement-gate/tree/main/pgate_demo/placement)
and the git dependency is gone. **The core repository was not deleted**; it is
archived and retained for provenance.

The operator's reasoning:

> The current core libraries each have exactly one public consumer. That does not
> yet justify a permanent public package boundary. A reusable-looking
> implementation is not automatically a reusable subsystem. Extraction should
> follow demonstrated reuse.

## Files in `pgate_demo/placement/`

### Internal source, as originally recorded

Thirteen modules carried across verbatim.

| File in `pgate_demo/placement/` | Internal source under `.../src/ollama_controller/` | SHA-256 of source | Bytes |
|---|---|---|---|
| `hardware/facts.py` | `hardware/facts.py` | `f269a78a0b387986ef4f90cc6a8b17a7a961f98982e442ee545d4a2d0c35c35e` | 6,549 |
| `hardware/observer.py` | `hardware/observer.py` | `d65c1b00871eb942b7a10565840f92a63c310ebe85fa987868672f2e15aaa3e8` | 2,144 |
| `hardware/windows_memory.py` | `hardware/windows_memory.py` | `b2d72524860d180e676fbcf525875cac63d9c52d8915a3a8b8d798ac5770ae4d` | 1,882 |
| `hardware/nvidia_query.py` | `hardware/nvidia_query.py` | `013d651765a748eadaa7bcff04f3ab5040104ecca1ffafd5e004ba87ab8c80c0` | 5,835 |
| `topology.py` | `core/topology.py` | `cfc4bbdc9bd3c895a3dccd942ed341bdd9ef42731b7b49d6ec0ac4ad8dd7e017` | 7,161 |
| `policy/model_profile.py` | `policy/model_profile.py` | `c4a5042fbb808216382bdb559785f5bb9d34fb7ca0d0008cf18219a70a997878` | 6,922 |
| `policy/admission.py` | `policy/admission.py` | `a71b636971e3dd836db3a6694b20508e4b5bd2642a01304f6d02b5d0f79b4648` | 11,590 |
| `policy/inference_placement.py` | `policy/inference_placement.py` | `351176f9b8e36de2123a7977cb77e272f7510b6e28fee463dfc089c3ddffe269` | 22,698 |
| `backends/backends.py` | `backends/backends.py` | `245462eedd5010c24aef871f75d7883d28a04dd9c5775ae6b78d22eb0a59913b` | 1,928 |
| `backends/chat.py` | `backends/chat.py` | `57b40d6d7ce093a6e820e2b970b1db6242e45a5b0ca04bdcfad8d1fc7026e32c` | 18,100 |
| `ledger/models.py` | `core/models.py` | `f3b627b0b91aaeffaefd4e7113e3c06b11df712e835b6ab94e7d582a1da4bfcd` | 7,777 |
| `ledger/residency.py` | `core/residency.py` | `2944f923cfeb2febb9d7403600580baa12115dbd2b3f456b24d55bfbae398c74` | 6,748 |
| `ledger/ledger.py` | `ledger/ledger.py` | `4e317d5c0dfd4816e7c07e0312ddc167e2427679440b60ca31a25bf8d2b46ea2` | 70,104 |

### The seam, extracted rather than copied

`controller/placement.py` contains `Controller.place_and_load_model()` and
`Controller._fail_and_reconcile()` — **230 of the source file's 5,562 lines
(4%)** — moved into a class named `PlacementRunner`.

The extracted methods are **AST-identical** to the source, verified by comparing
unparsed statement trees with docstrings, annotations and import ordering
normalised:

```
place_and_load_model     first stmt: same | remaining 24 statements: IDENTICAL
_fail_and_reconcile      first stmt: same | remaining  4 statements: IDENTICAL
```

Differences from the source are confined to: the class name; type annotations
dropped from the two signatures; `__init__(ledger, backend)` plus `set_backend`,
matching the construction idiom of the class it came from; extended docstrings; one
unused local import removed; and one `(Stage 5)` comment reference dropped,
because there is no Stage 5 here.

## The commit the code passed through

Every file was public at
[`Wyrd-Flux/wyrd-placement-core@0fadaec`](https://github.com/Wyrd-Flux/wyrd-placement-core/tree/0fadaecadd66c3a61ddb6d648490692ad0de5af8)
before being moved here. That repository is archived, not deleted.

## Changes made during extraction

| Change | Why |
|---|---|
| `from ..core.topology import` → `from ..topology import` | `topology.py` was hoisted to the package root |
| `from ..core.models import` → `from .models import` (in `ledger/`) | `models.py` and `residency.py` moved under `ledger/` |
| `from ..core.residency import` → `from .residency import` | same |
| `from ..core.models import BackendMode` → `from ..ledger.models import` (in `backends/`) | same |
| `from ..core.topology import` → `from ..topology import` (in `policy/model_profile.py`) | same |
| **`ledger.py`: 22 of 36 tables removed** | see below |
| `ledger.py`: `WORKSPACE_ROOT` override | unchanged in this fold |

### The one substantive subtraction

`ledger.py` declared **36 tables**. Twenty-two were never read or written by any
method in the class — they belonged to other components: checkpoint,
memory-lane, authorization, lease, turn, bootstrap.

Their `CREATE TABLE` blocks, their append-only trigger entries, and three
now-empty schema constants were removed. Fifteen placement-relevant tables remain.
Package size fell from 433,202 to 174,506 bytes.

This is a subtraction rather than a redesign, and the placement path is provably
untouched — all six methods the seam calls are AST-identical before and after:

```
append_hardware_profile    IDENTICAL
append_model_profile       IDENTICAL
append_admission           IDENTICAL
append_event_tx            IDENTICAL
store_content_bytes        IDENTICAL
connect                    IDENTICAL
```

## Changes made during folding

**One reference.** `wyrd_placement_core` → `pgate_demo.placement`, in one module
that named the package in a docstring. No relative import changed depth: the tree
moved from `wyrd_placement_core/<x>/m.py` to `pgate_demo/placement/<x>/m.py`, so
every `..` still resolves the same way.

**One behavioural fix, carried with the code.** The internal
`capability_inventory` was never extracted — it hardcodes `C:\G1\concepts`. So
`census --registry-root` reads the JSON document directly and says so in the
payload (`verified: false`, plus why). Placement never consults it.

## Tests carried with the code

Three suites now ship in `tests/`:

| File | What it is |
|---|---|
| `test_pgate.py` | the CLI adapter suite |
| `test_placement_seam.py` | the upstream method-level placement suite, carried across with only its import paths and its session-plumbing helper adapted. Ten tests: the load is issued, residency observed after it, partial residency fails closed, a wrong digest does not satisfy the plan, unknown hardware never attempts a load, `HOST_ONLY` requests zero GPU, no path generates tokens |
| `test_placement_primitives.py` | hardware composition, the planner closure, the transport allowlist, the ledger schema, and the properties that make this package self-contained |

95 tests, none of which skips. All run with no other Wyrd Flux package installed.

## Provenance facts

**Authorship.** The source repository had 120 commits: 134 by
`UrukuTelal <urukutelal@users.noreply.github.com>`, 4 machine-authored by
`opencode <opencode@local>` under operator direction, and 1 by
`F-2 integration <f2@canonical.invalid>`.

**No license file has ever existed** in the source repository, and none was added
at any stage of extraction, publication or folding.

`NO_VCS_HISTORY_AT_SOURCE` does **not** apply here — unlike the Veritas
primitives, `Ollama_Controller` is fully versioned. It applies only to the three
modules folded into Veritas, and is recorded in that repository's provenance.

## Third-party material

None carried across. The extracted files import only the standard library and
`pydantic`. No vendored directories, no embedded copyright headers. The source
repository declared `pydantic>=2.0` and `textual>=0.60`; only `pydantic` is needed
here, because `textual` belonged to the TUI that stayed internal.

## Licensing

- **Placement Gate:** MIT. See [`LICENSE`](../LICENSE).
- **The primitives were originally published under Apache-2.0** in
  `wyrd-placement-core`, by operator decision. They are MIT here — a strictly more
  permissive grant of the same code by the same rights holder.

No license file was added to, or modified in, `Ollama_Controller` at any stage.

## What is deliberately absent

| Not shipped | Why |
|---|---|
| the internal controller's other 58 methods | memory lanes, checkpoint generations, authorization chains, a governance transmission gate — a coherent internal control surface no external caller can use |
| the rest of `ollama_controller` (~1.4 MB) | TUI, runtime service, checkpoint operator |
| 22 unreferenced ledger tables | declared and never used |
| Model Characteristics Registry | optional, and unverifiable without its private reader |

## A note on the development machine

The laptop crashed partway through this work and the model service went down with
it. It was restarted, and every live claim in
[`docs/DEMO-NOTES.md`](DEMO-NOTES.md) was re-run afterwards against the code as it
stands now — using the smallest local weight and a short `--keep-alive` so the GPU
was under the lightest possible load. Nothing about the crash reproduced, and no
evidence from before it is relied on.

## If a second consumer appears

That is the condition for extracting a shared library again, and the condition
under which `wyrd-placement-core` is archived rather than deleted. Until then the
primitives live here, where the only consumer can reach them without a
cross-repository install.
