"""
PAC - Purifying Archetype Classifier
"""

from pac.classifier import PurifyingArchetypeClassifier

# Aliases for convenience and backward compatibility
PAC = PurifyingArchetypeClassifier
PurifyingArchetypeClassifierV2 = PurifyingArchetypeClassifier

__all__ = ["PurifyingArchetypeClassifier", "PAC", "PurifyingArchetypeClassifierV2"]
__version__ = "0.2.0"
