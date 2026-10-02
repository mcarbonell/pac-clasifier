"""
Retrocompatibility wrapper for pac_v2.
Delegates to the unified pac.PurifyingArchetypeClassifier.
"""

from pac.classifier import PurifyingArchetypeClassifier


class PurifyingArchetypeClassifierV2(PurifyingArchetypeClassifier):
    """
    Deprecated retrocompatibility wrapper for PurifyingArchetypeClassifierV2.
    Defaults to bifurcation_mode='confusion'.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('bifurcation_mode', 'confusion')
        super().__init__(*args, **kwargs)


__all__ = ["PurifyingArchetypeClassifierV2"]
