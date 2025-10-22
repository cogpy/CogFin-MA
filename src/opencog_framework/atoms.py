"""
OpenCog-inspired atom types for knowledge representation
"""

from typing import List, Union, Optional, Any, Dict
from abc import ABC, abstractmethod
import uuid
from .truth_values import TruthValue


class Atom(ABC):
    """Base class for all atoms in the AtomSpace"""
    
    def __init__(self, name: str = None, truth_value: TruthValue = None):
        self.id = str(uuid.uuid4())
        self.name = name or f"atom_{self.id[:8]}"
        self.truth_value = truth_value or TruthValue(1.0, 1.0)
        self.incoming_set = set()  # Atoms that reference this atom
        self.metadata = {}
        
    @abstractmethod
    def get_type(self) -> str:
        """Return the type name of this atom"""
        pass
        
    def __str__(self) -> str:
        return f"{self.get_type()}({self.name})"
        
    def __repr__(self) -> str:
        return self.__str__()
        
    def __hash__(self) -> int:
        return hash(self.id)
        
    def __eq__(self, other) -> bool:
        return isinstance(other, Atom) and self.id == other.id


class Node(Atom):
    """Base class for node atoms"""
    
    def __init__(self, name: str, truth_value: TruthValue = None):
        super().__init__(name, truth_value)


class Link(Atom):
    """Base class for link atoms that connect other atoms"""
    
    def __init__(self, outgoing: List[Atom], truth_value: TruthValue = None, name: str = None):
        super().__init__(name, truth_value)
        self.outgoing = outgoing or []
        
        # Update incoming sets of connected atoms
        for atom in self.outgoing:
            atom.incoming_set.add(self)
    
    def get_outgoing(self) -> List[Atom]:
        """Get the list of atoms this link connects to"""
        return self.outgoing
        
    def get_arity(self) -> int:
        """Get the number of atoms this link connects"""
        return len(self.outgoing)


class ConceptNode(Node):
    """Represents a concept in the knowledge base"""
    
    def get_type(self) -> str:
        return "ConceptNode"


class PredicateNode(Node):
    """Represents a predicate/relationship"""
    
    def get_type(self) -> str:
        return "PredicateNode"


class NumberNode(Node):
    """Represents a numerical value"""
    
    def __init__(self, value: Union[int, float], name: str = None, truth_value: TruthValue = None):
        self.value = value
        name = name or str(value)
        super().__init__(name, truth_value)
        
    def get_type(self) -> str:
        return "NumberNode"
        
    def get_value(self) -> Union[int, float]:
        return self.value


class ListLink(Link):
    """Represents an ordered list of atoms"""
    
    def get_type(self) -> str:
        return "ListLink"


class EvaluationLink(Link):
    """Represents an evaluation of a predicate with arguments"""
    
    def __init__(self, predicate: PredicateNode, arguments: Union[ListLink, List[Atom]], 
                 truth_value: TruthValue = None, name: str = None):
        if isinstance(arguments, list):
            arguments = ListLink(arguments)
        
        outgoing = [predicate, arguments]
        super().__init__(outgoing, truth_value, name)
        
    def get_predicate(self) -> PredicateNode:
        """Get the predicate being evaluated"""
        return self.outgoing[0]
        
    def get_arguments(self) -> ListLink:
        """Get the arguments to the predicate"""
        return self.outgoing[1]
        
    def get_type(self) -> str:
        return "EvaluationLink"


class InheritanceLink(Link):
    """Represents inheritance relationship between concepts"""
    
    def __init__(self, child: ConceptNode, parent: ConceptNode, 
                 truth_value: TruthValue = None, name: str = None):
        super().__init__([child, parent], truth_value, name)
        
    def get_child(self) -> ConceptNode:
        return self.outgoing[0]
        
    def get_parent(self) -> ConceptNode:
        return self.outgoing[1]
        
    def get_type(self) -> str:
        return "InheritanceLink"


class SimilarityLink(Link):
    """Represents similarity relationship between concepts"""
    
    def __init__(self, concept1: ConceptNode, concept2: ConceptNode,
                 truth_value: TruthValue = None, name: str = None):
        super().__init__([concept1, concept2], truth_value, name)
        
    def get_type(self) -> str:
        return "SimilarityLink"


class ImplicationLink(Link):
    """Represents logical implication"""
    
    def __init__(self, antecedent: Atom, consequent: Atom,
                 truth_value: TruthValue = None, name: str = None):
        super().__init__([antecedent, consequent], truth_value, name)
        
    def get_antecedent(self) -> Atom:
        return self.outgoing[0]
        
    def get_consequent(self) -> Atom:
        return self.outgoing[1]
        
    def get_type(self) -> str:
        return "ImplicationLink"