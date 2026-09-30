# Licensing analysis

Recorded 2026-09-30, for the `Wyrd-Flux/placement-gate` publication decision.

## Short answer

Placement Gate's own code is MIT. It **redistributes nothing upstream**, so
publication requires no permission from anyone. Upstream licensing is
**unresolved**, and no license was added to any upstream repository.

## What is in this repository

| Path | Origin | License |
|---|---|---|
| `pgate_demo/*.py` | written for this repository | MIT (this repo's `LICENSE`) |
| `tests/test_pgate.py` | written for this repository | MIT |
| `README.md`, `docs/*`, `pyproject.toml` | written for this repository | MIT |
| `pgate_demo/pgate.providers.json` | written for this repository | MIT |
| anything from `ollama_controller` | **nothing** | — |
| anything from the characteristics registry | **nothing** | — |
| model weights, GGUF files, ledgers, corpora | **nothing** | — |

The package declares no runtime dependency on `ollama_controller` in
`pyproject.toml`, because it is not installable from any index. It is resolved
at runtime through `PGATE_UPSTREAM_PATH` or an already-installed copy. A
recipient who has no upstream copy gets `pgate doctor` reporting every capability
`UNAVAILABLE` — not a broken import, and not a silent fallback.

## The upstream position

`ollama_controller`:

- not published on PyPI (`GET /pypi/ollama-controller/json` → `404`)
- no `LICENSE`, `COPYING`, or `NOTICE` file in its source tree
- no `license` field in its `pyproject.toml`
- commit authorship spans three distinct identities

The Model Characteristics Registry likewise carries no license file.

Absent a license grant, copyright defaults to **all rights reserved**. So the
code is not available to the public on its own terms, regardless of how it was
obtained or who holds it.

## Why that does not block this publication

There is a real difference between *depending on* code and *redistributing* it,
and it cuts in a specific direction here.

1. **No upstream bytes ship.** Placement Gate imports `ollama_controller` at
   runtime from a path the recipient supplies. Nothing upstream is copied into
   this repository, its wheels, or its sdist. So this publication does not
   distribute the upstream code, and the default "all rights reserved" rule —
   which governs *distribution* — is not engaged.

2. **The operator holds the rights.** The upstream tree is the operator's own
   local work. They can point their own `pgate` at it today with complete
   confidence. The MIT grant in this repo's `LICENSE` is separate and does not
   purport to cover upstream.

3. **A recipient gains nothing they did not already have.** Someone who reads
   this repository learns *how to call* an API they must already possess the code
   for. There is no path by which `pip install placement-gate` hands them
   upstream code.

## The open question, stated plainly

The honest limitation is not a distribution problem. It is this: **this
repository claims no durable right to depend on its upstream.** If the operator
later relicenses `ollama_controller` under terms that forbid runtime binding, or
if a third party asserts rights in it, Placement Gate's central dependency breaks
without this repo having violated anything. The dependency is licensed by
courtesy, not by grant.

Two things follow, and neither has been done here:

- **Nothing upstream was modified.** No `LICENSE` was added to
  `Ollama_Controller` or to the characteristics registry. Choosing a license for
  someone else's project is not an adapter's decision.
- **No fabricated provenance.** This file does not claim upstream is MIT, Apache,
  or proprietary. It says what was checked, on what date, and what was found.

If the operator wants a durable basis, the resolution is a decision about
`ollama_controller` itself — publish it under terms that permit runtime
binding, or vendor it with a recorded grant. Both are the operator's call, not
this package's.

## Self-contained bundling is out

There is no `vendor/`, no `wheelhouse/`, no copied capability surface, and no
staged upstream tarball. `LICENSE` carries this decision in capitals:

```
Self-contained bundling is PROHIBITED until upstream licensing is explicitly
settled by the operator.
```

## Verifying the no-redistribution claim

```console
$ grep -rnE "^(class|def|from|import) " pgate_demo/ | grep -v "^\S*:.*pgate_demo\|from \.\|import \("
```

Every upstream symbol is reached through `ProviderSet.module(name)` or the
adapter's own `_request`, never by importing upstream from this package. The
package imports exactly one upstream thing anywhere: an `importlib.import_module`
call inside the resolver, for a name the operator configured.

## Applies to

`LICENSE` is Placement Gate's own. It conveys nothing in `ollama_controller` or
the Model Characteristics Registry.
