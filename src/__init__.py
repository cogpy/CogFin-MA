"""
CogFin-MA: OpenCog-inspired Retrieval-Augmented Multi-Agent Framework 
for Fundamental Company Analysis and Financial Insight
"""

__version__ = "1.0.0"
__author__ = "CogFin-MA Development Team"

from .opencog_framework import AtomSpace, Atom, ConceptNode, PredicateNode, EvaluationLink
from .agents import BaseAgent, SentimentAgent, PriceCorrelationAgent, NewsAgent, FinancialHealthAgent
from .knowledge_base import KnowledgeBase

__all__ = [
    'AtomSpace',
    'Atom', 
    'ConceptNode',
    'PredicateNode',
    'EvaluationLink',
    'BaseAgent',
    'SentimentAgent',
    'PriceCorrelationAgent', 
    'NewsAgent',
    'FinancialHealthAgent',
    'KnowledgeBase'
]