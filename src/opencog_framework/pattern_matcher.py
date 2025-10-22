"""
Pattern matching for querying the AtomSpace
"""

from typing import Dict, List, Set, Optional, Any, Union, Callable
from .atoms import Atom, Node, Link, ConceptNode, PredicateNode, EvaluationLink
from .atomspace import AtomSpace
from .truth_values import TruthValue


class Variable:
    """Represents a variable in pattern matching"""
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return f"${self.name}"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name


class BindingSet:
    """Represents variable bindings in pattern matching"""
    
    def __init__(self, bindings: Dict[Variable, Atom] = None):
        self.bindings = bindings or {}
    
    def bind(self, variable: Variable, atom: Atom):
        """Bind a variable to an atom"""
        self.bindings[variable] = atom
    
    def get_binding(self, variable: Variable) -> Optional[Atom]:
        """Get the atom bound to a variable"""
        return self.bindings.get(variable)
    
    def is_bound(self, variable: Variable) -> bool:
        """Check if a variable is bound"""
        return variable in self.bindings
    
    def get_variables(self) -> Set[Variable]:
        """Get all bound variables"""
        return set(self.bindings.keys())
    
    def copy(self) -> 'BindingSet':
        """Create a copy of this binding set"""
        return BindingSet(self.bindings.copy())
    
    def __str__(self):
        bindings_str = ", ".join(f"{var}: {atom}" for var, atom in self.bindings.items())
        return f"BindingSet({bindings_str})"


class Pattern:
    """Represents a pattern for matching against the AtomSpace"""
    
    def __init__(self, template: Union[Atom, Variable], variables: Set[Variable] = None):
        self.template = template
        self.variables = variables or set()
        
        # Extract variables from template if not provided
        if not self.variables:
            self.variables = self._extract_variables(template)
    
    def _extract_variables(self, element: Any) -> Set[Variable]:
        """Extract all variables from a template recursively"""
        variables = set()
        
        if isinstance(element, Variable):
            variables.add(element)
        elif isinstance(element, Link):
            for outgoing in element.get_outgoing():
                variables.update(self._extract_variables(outgoing))
        elif isinstance(element, list):
            for item in element:
                variables.update(self._extract_variables(item))
        
        return variables
    
    def get_variables(self) -> Set[Variable]:
        """Get all variables in this pattern"""
        return self.variables


class PatternMatcher:
    """Pattern matcher for querying the AtomSpace"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
    
    def match(self, pattern: Pattern) -> List[BindingSet]:
        """Find all matches for a pattern in the AtomSpace"""
        results = []
        
        # Start with empty binding set
        initial_bindings = BindingSet()
        
        # Try to match the pattern template
        self._match_recursive(pattern.template, initial_bindings, results)
        
        return results
    
    def _match_recursive(self, template: Union[Atom, Variable], 
                        bindings: BindingSet, results: List[BindingSet]) -> bool:
        """Recursively match template against atoms"""
        
        if isinstance(template, Variable):
            # Variable matches any atom
            if bindings.is_bound(template):
                # Variable already bound - check consistency
                return True
            else:
                # Try binding to all compatible atoms
                for atom in self.atomspace.atoms.values():
                    new_bindings = bindings.copy()
                    new_bindings.bind(template, atom)
                    results.append(new_bindings)
                return True
        
        elif isinstance(template, ConceptNode):
            # Match concept nodes by name
            matches = self.atomspace.get_atoms_by_name(template.name)
            for match in matches:
                if isinstance(match, ConceptNode):
                    results.append(bindings.copy())
            return len(matches) > 0
        
        elif isinstance(template, PredicateNode):
            # Match predicate nodes by name
            matches = self.atomspace.get_atoms_by_name(template.name)
            for match in matches:
                if isinstance(match, PredicateNode):
                    results.append(bindings.copy())
            return len(matches) > 0
        
        elif isinstance(template, EvaluationLink):
            # Match evaluation links
            return self._match_evaluation_link(template, bindings, results)
        
        elif isinstance(template, Link):
            # Match other types of links
            return self._match_link(template, bindings, results)
        
        return False
    
    def _match_evaluation_link(self, template: EvaluationLink, 
                              bindings: BindingSet, results: List[BindingSet]) -> bool:
        """Match evaluation links specifically"""
        found_matches = False
        
        # Get all evaluation links
        eval_links = self.atomspace.get_atoms_by_type(EvaluationLink)
        
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                # Try to match predicate and arguments
                template_pred = template.get_predicate()
                template_args = template.get_arguments()
                
                link_pred = link.get_predicate()
                link_args = link.get_arguments()
                
                # Check if predicates match
                pred_match = False
                if isinstance(template_pred, Variable):
                    # Variable predicate matches any predicate
                    pred_match = True
                elif template_pred.name == link_pred.name:
                    pred_match = True
                
                if pred_match:
                    # Check argument matching
                    new_bindings = bindings.copy()
                    if self._match_arguments(template_args.get_outgoing(), 
                                           link_args.get_outgoing(), new_bindings):
                        results.append(new_bindings)
                        found_matches = True
        
        return found_matches
    
    def _match_link(self, template: Link, bindings: BindingSet, 
                   results: List[BindingSet]) -> bool:
        """Match general links"""
        found_matches = False
        
        # Get all links of the same type
        links = self.atomspace.get_atoms_by_type(template.get_type())
        
        for link in links:
            if isinstance(link, Link) and link.get_arity() == template.get_arity():
                new_bindings = bindings.copy()
                if self._match_arguments(template.get_outgoing(), 
                                       link.get_outgoing(), new_bindings):
                    results.append(new_bindings)
                    found_matches = True
        
        return found_matches
    
    def _match_arguments(self, template_args: List[Union[Atom, Variable]], 
                        actual_args: List[Atom], bindings: BindingSet) -> bool:
        """Match lists of arguments"""
        if len(template_args) != len(actual_args):
            return False
        
        for template_arg, actual_arg in zip(template_args, actual_args):
            if isinstance(template_arg, Variable):
                if bindings.is_bound(template_arg):
                    # Check consistency
                    if bindings.get_binding(template_arg) != actual_arg:
                        return False
                else:
                    # Bind variable
                    bindings.bind(template_arg, actual_arg)
            elif isinstance(template_arg, Atom):
                # Direct comparison - simplified
                if template_arg.name != actual_arg.name or template_arg.get_type() != actual_arg.get_type():
                    return False
        
        return True
    
    def query_by_predicate(self, predicate_name: str) -> List[EvaluationLink]:
        """Find all evaluations of a specific predicate"""
        results = []
        
        eval_links = self.atomspace.get_atoms_by_type(EvaluationLink)
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                pred = link.get_predicate()
                if pred.name == predicate_name:
                    results.append(link)
        
        return results
    
    def query_by_concept(self, concept_name: str) -> List[Atom]:
        """Find all atoms related to a concept"""
        results = []
        
        # Find the concept
        concepts = self.atomspace.get_atoms_by_name(concept_name)
        for concept in concepts:
            if isinstance(concept, ConceptNode):
                # Find incoming links
                incoming = self.atomspace.get_incoming_set(concept)
                results.extend(incoming)
                
                # Add the concept itself
                results.append(concept)
        
        return results
    
    def find_related_concepts(self, concept_name: str, relation_type: str = None) -> List[ConceptNode]:
        """Find concepts related to a given concept"""
        results = []
        
        concepts = self.atomspace.get_atoms_by_name(concept_name)
        for concept in concepts:
            if isinstance(concept, ConceptNode):
                incoming = self.atomspace.get_incoming_set(concept)
                
                for link in incoming:
                    if relation_type is None or link.get_type() == relation_type:
                        # Find other concepts in the same link
                        for outgoing in link.get_outgoing():
                            if (isinstance(outgoing, ConceptNode) and 
                                outgoing != concept and outgoing not in results):
                                results.append(outgoing)
        
        return results