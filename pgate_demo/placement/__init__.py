"""Placement primitives: observe the hardware, plan, load, and verify.

These modules are the implementation Placement Gate reports on. They are not a
wrapper -- they are the hardware probes, the placement policy, the Ollama
adapter, the durable ledger and the load seam themselves, folded in so this
repository is self-contained.

=================================  =========================================
Module                              Question
=================================  =========================================
``hardware/``                      what is this machine, as measured?
``policy/``                         will this model fit, and why not if not?
``backends/``                       how do I ask the model service?
``ledger/``                         what happened, durably?
``controller/placement.py``         run all four for one plan, and verify it
``topology.py``                     the evidence types the planner and the
                                    verifier share
=================================  =========================================

Provenance for every file is in ``docs/PROVENANCE.md``: the internal source
path, the source SHA-256, and the AST comparison that proves the placement seam
is unchanged from the method it was extracted from.
"""
