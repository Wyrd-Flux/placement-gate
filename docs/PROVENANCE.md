# Source provenance

Recorded 2026-09-30, and updated for the migration to a public core.

## What Placement Gate depends on now

```
placement-gate
  └─ wyrd-placement-core   (Apache-2.0, public, installed from GitHub)
       └─ pydantic → Python standard library
```

That is the complete dependency graph. No source-tree binding, no environment
variable, no private registry.

`wyrd-placement-core` was extracted from an internal repository for this
migration. Its own
[`PROVENANCE.md`](https://github.com/Wyrd-Flux/wyrd-placement-core/blob/main/docs/PROVENANCE.md)
records the exact source paths, SHA-256 hashes, the AST comparison that proves
the placement seam is unchanged, and the one schema subtraction made.

Placement Gate copies none of it.

## What this repository contains

| File | Role |
|---|---|
| `pgate_demo/cli.py` | argument parsing, output rendering, exit-code mapping |
| `pgate_demo/core.py` | session; builds inputs for the core, unpacks its results |
| `pgate_demo/providers.py` | named capability resolution over the installed dependency |
| `pgate_demo/exit_codes.py` | the documented contract |
| `pgate_demo/selftest.py` | deterministic checks, plus an opt-in live load |
| `tests/test_pgate.py` | 48 adapter tests |

Zero lines of `wyrd_placement_core` are copied here.

## The migration (2026-09-30, this release)

**Before.** Placement Gate bound to an internal `ollama_controller` package by
path, at runtime, through `PGATE_UPSTREAM_PATH`, with `pgate.providers.json`
holding an empty `search_paths` list. It shipped no upstream bytes and therefore
needed no license — and consequently **could not be run by anyone who did not
already have the internal source tree**, which is most of the audience a public
demo exists for.

**After.** One ordinary dependency on `wyrd-placement-core`.

### What changed in the code

| Before | After |
|---|---|
| `PGATE_UPSTREAM_PATH` / `PGATE_<NAME>_PATH` | declared in `pyproject.toml` |
| `pgate.providers.json` with an empty search list | deleted; resolution is a named import |
| `Controller.place_and_load_model()` | `PlacementRunner.place_and_load_model()` |
| `controller.create_session()` + `bind_identity()` | `session_id` / `identity_id` strings |
| a private module for the characteristics registry | an explicit **unverified** JSON read |

That last row deserves a note, because it is the one place where behaviour is
not a straight delegation.

### The characteristics registry

The internal registry reader lived in `ollama_controller.registries.model_characteristics`,
which was not extracted — it is an optional subsystem and its own verifier
currently rejects its state.

So `census --registry-root` now reads the JSON document directly and **says so in
the payload**:

```
UNVERIFIED. These counts are read from the document as supplied; no digest or
integrity check was performed, because the registry's own verifier is not part of
the public placement core.
```

A reader that skipped the digest check while looking like the real one would be
worse than no reader. Placement never consults the registry: a plan comes from
measured hardware and observed model facts, which is a different kind of claim.

## The eight semantics that were preserved

These are what the demo claims, and what `wyrd-placement-core` had to carry
across intact. Each was re-verified live after the migration (see
[`docs/DEMO-NOTES.md`](DEMO-NOTES.md)):

1. an over-budget model is refused **before** any load request
2. unknown hardware fails closed
3. the load actually happens, through the admitted backend operation
4. residency is observed **after** the load
5. the required residency is checked against what was requested
6. partial residency is rejected when full residency was required
7. identity binds to the planned digest, not the tag
8. a domain refusal is exit 0, distinct from a process failure

## Third-party material

None. `pgate_demo` imports only the standard library and `wyrd_placement_core`.
No vendored directories, no embedded upstream code, no model weights, no
internal corpora.

## Licensing

- **Placement Gate:** MIT. See [`LICENSE`](../LICENSE).
- **wyrd-placement-core:** Apache-2.0, by operator decision.

No license file was added to, or modified in, any internal source repository as
part of the extraction or this migration. The internal trees remain unlicensed;
the grant covers the extracted public work.

## What is deliberately absent

| Not shipped | Why |
|---|---|
| any `ollama_controller` module | not needed by anything here |
| the internal controller's other 58 methods | memory lanes, checkpoints, authorization chains |
| model weights, ledgers, machine state | nothing local is version-controlled |
| the characteristics registry | optional, unverifiable without its private verifier |

History was not rewritten. The adapter-only release remains in the log, so the
earlier decision and its reversal are both auditable.
