"""Thin adapters around the authors' TopoDockQ code.

The authors' scripts run unchanged: this package stages their repository into a
per-run working directory, runs them, and records provenance. Two pieces mirror
the authors' code, and both are checked against the originals:
  * ``preprocessing.py`` copies the feature selection / scaling block of
    ``04_inference_from_generated_csv.py`` so it can be saved once and reused;
  * ``topo_features.py`` is a Python 3.13 port of the authors' feature generator,
    whose original ships only as Python 3.8 bytecode (verified identical on the
    published features; the original also runs unchanged in ``.venv-feature38``).

This package does not import or depend on the agent code in ``topodockq_agents``.
"""
