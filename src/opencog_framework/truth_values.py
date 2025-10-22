"""
Truth values for probabilistic reasoning
"""

import math
from typing import Tuple


class TruthValue:
    """Represents truth values with strength and confidence"""
    
    def __init__(self, strength: float = 1.0, confidence: float = 1.0):
        """
        Initialize truth value
        
        Args:
            strength: Probability/strength of truth (0.0 to 1.0)
            confidence: Confidence in the strength value (0.0 to 1.0)
        """
        self.strength = max(0.0, min(1.0, strength))
        self.confidence = max(0.0, min(1.0, confidence))
    
    def get_strength(self) -> float:
        """Get the strength component"""
        return self.strength
    
    def get_confidence(self) -> float:
        """Get the confidence component"""
        return self.confidence
    
    def get_mean(self) -> float:
        """Get the mean of the probability distribution"""
        return self.strength
    
    def get_variance(self) -> float:
        """Calculate variance based on confidence"""
        if self.confidence == 0:
            return 0.25  # Maximum uncertainty
        return (1.0 - self.confidence) * 0.25
    
    def get_count(self) -> float:
        """Convert confidence to count-like measure"""
        if self.confidence == 0:
            return 0.0
        return self.confidence * 100.0  # Arbitrary scaling
    
    def is_true(self, threshold: float = 0.5) -> bool:
        """Check if truth value indicates truth above threshold"""
        return self.strength >= threshold and self.confidence > 0.1
    
    def is_false(self, threshold: float = 0.5) -> bool:
        """Check if truth value indicates falsehood"""
        return self.strength < threshold and self.confidence > 0.1
    
    def is_unknown(self, confidence_threshold: float = 0.1) -> bool:
        """Check if truth value is too uncertain to determine"""
        return self.confidence <= confidence_threshold
    
    def __str__(self) -> str:
        return f"TV(s={self.strength:.3f}, c={self.confidence:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TruthValue):
            return False
        return (abs(self.strength - other.strength) < 1e-6 and 
                abs(self.confidence - other.confidence) < 1e-6)
    
    def __hash__(self) -> int:
        return hash((round(self.strength, 6), round(self.confidence, 6)))


class TruthValueOperations:
    """Operations on truth values for reasoning"""
    
    @staticmethod
    def conjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """Logical AND of two truth values"""
        strength = tv1.strength * tv2.strength
        confidence = min(tv1.confidence, tv2.confidence)
        return TruthValue(strength, confidence)
    
    @staticmethod
    def disjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """Logical OR of two truth values"""
        strength = tv1.strength + tv2.strength - (tv1.strength * tv2.strength)
        confidence = min(tv1.confidence, tv2.confidence)
        return TruthValue(strength, confidence)
    
    @staticmethod
    def negation(tv: TruthValue) -> TruthValue:
        """Logical NOT of truth value"""
        return TruthValue(1.0 - tv.strength, tv.confidence)
    
    @staticmethod
    def implication(antecedent: TruthValue, consequent: TruthValue) -> TruthValue:
        """Logical implication"""
        # P → Q ≡ ¬P ∨ Q
        not_p = TruthValueOperations.negation(antecedent)
        return TruthValueOperations.disjunction(not_p, consequent)
    
    @staticmethod
    def revision(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """Revise truth value with new evidence"""
        if tv1.confidence == 0:
            return tv2
        if tv2.confidence == 0:
            return tv1
            
        # Weighted average based on confidence
        total_conf = tv1.confidence + tv2.confidence
        strength = (tv1.strength * tv1.confidence + tv2.strength * tv2.confidence) / total_conf
        confidence = min(1.0, total_conf / 2.0)  # Conservative confidence update
        
        return TruthValue(strength, confidence)
    
    @staticmethod
    def deduction(premise1: TruthValue, premise2: TruthValue) -> TruthValue:
        """Deductive inference: If A→B and A, then B"""
        strength = premise1.strength * premise2.strength
        confidence = min(premise1.confidence, premise2.confidence) * 0.9  # Slight confidence reduction
        return TruthValue(strength, confidence)


# Common truth value constants
TRUE = TruthValue(1.0, 1.0)
FALSE = TruthValue(0.0, 1.0)
UNKNOWN = TruthValue(0.5, 0.0)
LIKELY = TruthValue(0.8, 0.8)
UNLIKELY = TruthValue(0.2, 0.8)