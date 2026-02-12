"""Backward-compatible FTTA shim.

This module preserves the historic import path:
`tableshift.FTTA_src.FTTA.FTTA`.
"""

from ftta.tta import FTTA, FTTAMethod, TtaMethod, TtaRegistry

__all__ = ["FTTA", "FTTAMethod", "TtaMethod", "TtaRegistry"]
