# Licensing

Recorded 2026-09-30, and updated for the migration to a public core.

## Short answer

| Component | License | Where |
|---|---|---|
| Placement Gate, including `pgate_demo/placement/` | **MIT** | [`LICENSE`](../LICENSE) |
| `wyrd-placement-core` (archived, retained) | was **Apache-2.0** | its own `LICENSE`, at its own repository |

Placement Gate declares one dependency, `pydantic`. Nothing private is
redistributed, and nothing cross-repository is required to run it.

## The change, and why it was necessary

Placement Gate 0.1.0 was published adapter-only: it bound to an internal source
tree at runtime via `PGATE_UPSTREAM_PATH` and redistributed none of it. That was
a defensible reading of "publish nothing you cannot license", and it produced a
repository that **nobody without the internal estate could run**.

The operator resolved this on 2026-09-30 by licensing the extracted public work
explicitly (D1: Apache-2.0) and directing that both demos depend on it normally
(D4: GitHub-only distribution in this phase).

## What the migration did and did not do

**Did:** extract the placement primitives into
`wyrd-placement-core`, and record their source paths and SHA-256 hashes in that
package's `docs/PROVENANCE.md` — along with an AST-level comparison proving the
placement seam is unchanged.

**Did not:** add, modify, or remove a license in any internal source repository.
`Ollama_Controller` remains unlicensed and unmodified. The Apache-2.0 grant
covers the extracted public work.

**Did not:** fabricate history. All three stages are new commits. The adapter-only
release, the separate-core release, and this fold are all present in the git log,
so each decision and its reversal are auditable.

**Did not:** delete the core repository. `wyrd-placement-core` is archived and
retained; its commit history and provenance records remain readable.

**Did not:** weaken the grant. The primitives were Apache-2.0 as a separate package
and are MIT here. Both are grants of the same code by the same rights holder, and
MIT is the more permissive of the two.

**Did not:** add a license upstream. `Ollama_Controller` had no license file at any
point in this work, and still has none.

## The one non-delegation

`census --registry-root` reads a Model Characteristics Registry document
**without verifying it**, and says so in its payload:

```
UNVERIFIED. These counts are read from the document as supplied; no digest or
integrity check was performed, because the registry's own verifier is not part
of the public placement core.
```

This is recorded here rather than only in the code because it is the one place
where Placement Gate reports on data it has not checked. A reader that skipped
the digest check while resembling the real one would be worse than none.

Placement never consults the registry. A plan is computed from measured hardware
and observed model facts, which is a different kind of claim from a record of
what is already known about a specimen.

## Third-party material

None. `pgate_demo` imports only the standard library and `wyrd_placement_core`.
The core imports the standard library and `pydantic`. No vendored subtrees, no
embedded copyright headers, no copyleft dependency.

## Why Apache-2.0 for the core, and MIT for the demo

Apache-2.0 for the core because it is intended as reusable infrastructure:
permissive reuse, plus an explicit patent grant and a contribution-licence
clause, which matter for a library others will build on.

Placement Gate is a demo rather than a library, and remains MIT. The asymmetry
is deliberate — the code meant to be depended upon gets the stronger grant.

## Not published

| Not shipped | Licensing consequence |
|---|---|
| any `ollama_controller` module | not needed by anything here |
| the internal controller's other 58 methods | memory lanes, checkpoints, authorization chains |
| model weights, ledgers, machine state | nothing local is version-controlled |
| the characteristics registry | optional, unverifiable without its private verifier |
| 22 unreferenced ledger tables | declared but never used by any method |

Nothing in either repository is licensed that was not extracted deliberately, and
nothing extracted is left unlicensed.

## Verifying

```console
$ pip show wyrd-placement-core     # Apache-2.0, per its own LICENSE
$ git log --oneline               # history preserved; the migration is a new commit
```
