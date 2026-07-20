"""
retrieval-failure-mechanism.

When a model FAILS at long-context retrieval, what is the mechanistic signature
at the retrieval-head level, and is it causal? This package builds the
experiment suite on top of the prior library (``src/``, the RoPE/retrieval-head paper) and
the prior detector (``rhp/``, the retrieval-head-profile paper). Those two are imported
only, never edited (task §0, §12).

Measurement only: no module in this package interprets a result (§12). Every
number written to ``results/`` carries a provenance record (§1.6) and is gated
on the researcher-authored ``configs/preregistration.yaml`` (§1.2).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
