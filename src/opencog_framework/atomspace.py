"""
AtomSpace: Central knowledge repository for OpenCog-inspired framework
"""

from typing import Dict, List, Set, Optional, Type, Union, Callable
import threading
from collections import defaultdict
import json
import pickle
from .atoms import Atom, Node, Link, ConceptNode, PredicateNode, EvaluationLink
from .truth_values import TruthValue, TruthValueOperations


class AtomSpace:
    """Central repository for atoms and knowledge"""
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}  # atom_id -> atom
        self.atoms_by_type: Dict[str, Set[str]] = defaultdict(set)  # type -> set of atom_ids
        self.atoms_by_name: Dict[str, Set[str]] = defaultdict(set)  # name -> set of atom_ids
        self.incoming_sets: Dict[str, Set[str]] = defaultdict(set)  # atom_id -> incoming link ids
        self.outgoing_sets: Dict[str, Set[str]] = defaultdict(set)  # atom_id -> outgoing atom ids
        self._lock = threading.RLock()
        
    def add(self, atom: Atom) -> Atom:
        """Add an atom to the AtomSpace"""
        with self._lock:
            if atom.id in self.atoms:
                # Atom already exists, merge truth values if different
                existing = self.atoms[atom.id]
                if existing.truth_value != atom.truth_value:
                    existing.truth_value = TruthValueOperations.revision(
                        existing.truth_value, atom.truth_value
                    )
                return existing
            
            # Add new atom
            self.atoms[atom.id] = atom
            self.atoms_by_type[atom.get_type()].add(atom.id)
            if atom.name:
                self.atoms_by_name[atom.name].add(atom.id)
            
            # Handle links - update incoming/outgoing sets
            if isinstance(atom, Link):
                for target_atom in atom.get_outgoing():
                    self.incoming_sets[target_atom.id].add(atom.id)
                    self.outgoing_sets[atom.id].add(target_atom.id)
                    # Ensure target atoms are also in the atomspace
                    if target_atom.id not in self.atoms:
                        self.add(target_atom)
            
            return atom
    
    def remove(self, atom: Atom) -> bool:
        """Remove an atom from the AtomSpace"""
        with self._lock:
            if atom.id not in self.atoms:
                return False
            
            # Remove from indices
            del self.atoms[atom.id]
            self.atoms_by_type[atom.get_type()].discard(atom.id)
            if atom.name:
                self.atoms_by_name[atom.name].discard(atom.id)
            
            # Clean up incoming/outgoing references
            if isinstance(atom, Link):
                for target_atom in atom.get_outgoing():
                    self.incoming_sets[target_atom.id].discard(atom.id)
                    self.outgoing_sets[atom.id].discard(target_atom.id)
            
            # Remove from other atoms' outgoing sets
            for link_id in list(self.incoming_sets.get(atom.id, [])):
                if link_id in self.atoms:
                    link = self.atoms[link_id]
                    if isinstance(link, Link):
                        self.outgoing_sets[link_id].discard(atom.id)
            
            # Clean up empty sets
            if atom.id in self.incoming_sets:
                del self.incoming_sets[atom.id]
            if atom.id in self.outgoing_sets:
                del self.outgoing_sets[atom.id]
                
            return True
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get atom by ID"""
        return self.atoms.get(atom_id)
    
    def get_atoms_by_type(self, atom_type: Union[str, Type[Atom]]) -> List[Atom]:
        """Get all atoms of a specific type"""
        if isinstance(atom_type, type):
            type_name = atom_type.__name__
        else:
            type_name = atom_type
            
        atom_ids = self.atoms_by_type.get(type_name, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """Get all atoms with a specific name"""
        atom_ids = self.atoms_by_name.get(name, set())
        return [self.atoms[aid] for aid in atom_ids if aid in self.atoms]
    
    def get_incoming_set(self, atom: Atom) -> List[Link]:
        """Get all links that point to this atom"""
        link_ids = self.incoming_sets.get(atom.id, set())
        return [self.atoms[lid] for lid in link_ids 
                if lid in self.atoms and isinstance(self.atoms[lid], Link)]
    
    def get_outgoing_set(self, atom: Atom) -> List[Atom]:
        """Get all atoms that this atom points to (if it's a link)"""
        if not isinstance(atom, Link):
            return []
        return atom.get_outgoing()
    
    def find_atom(self, atom_type: Union[str, Type[Atom]], name: str = None, 
                  truth_value: TruthValue = None) -> Optional[Atom]:
        """Find an atom by type, name, and optionally truth value"""
        candidates = self.get_atoms_by_type(atom_type)
        
        if name is not None:
            candidates = [a for a in candidates if a.name == name]
        
        if truth_value is not None:
            candidates = [a for a in candidates if a.truth_value == truth_value]
        
        return candidates[0] if candidates else None
    
    def create_concept(self, name: str, truth_value: TruthValue = None) -> ConceptNode:
        """Create or retrieve a concept node"""
        existing = self.find_atom(ConceptNode, name)
        if existing:
            return existing
        
        concept = ConceptNode(name, truth_value)
        return self.add(concept)
    
    def create_predicate(self, name: str, truth_value: TruthValue = None) -> PredicateNode:
        """Create or retrieve a predicate node"""
        existing = self.find_atom(PredicateNode, name)
        if existing:
            return existing
        
        predicate = PredicateNode(name, truth_value)
        return self.add(predicate)
    
    def create_evaluation(self, predicate: Union[str, PredicateNode], 
                         arguments: List[Atom], truth_value: TruthValue = None) -> EvaluationLink:
        """Create an evaluation link"""
        if isinstance(predicate, str):
            predicate = self.create_predicate(predicate)
        
        eval_link = EvaluationLink(predicate, arguments, truth_value)
        return self.add(eval_link)
    
    def query(self, pattern: Atom) -> List[Dict[str, Atom]]:
        """Simple pattern matching query"""
        # This is a simplified version - real OpenCog has sophisticated pattern matching
        results = []
        
        if isinstance(pattern, ConceptNode):
            # Find concepts with same name or similar
            matches = self.get_atoms_by_name(pattern.name)
            for match in matches:
                if isinstance(match, ConceptNode):
                    results.append({pattern.name: match})
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the AtomSpace"""
        return {
            'total_atoms': len(self.atoms),
            'atoms_by_type': {t: len(ids) for t, ids in self.atoms_by_type.items()},
            'unique_names': len(self.atoms_by_name),
            'total_links': sum(1 for a in self.atoms.values() if isinstance(a, Link)),
            'total_nodes': sum(1 for a in self.atoms.values() if isinstance(a, Node))
        }
    
    def clear(self):
        """Clear all atoms from the AtomSpace"""
        with self._lock:
            self.atoms.clear()
            self.atoms_by_type.clear()
            self.atoms_by_name.clear()
            self.incoming_sets.clear()
            self.outgoing_sets.clear()
    
    def size(self) -> int:
        """Get the number of atoms in the AtomSpace"""
        return len(self.atoms)
    
    def export_to_dict(self) -> Dict:
        """Export AtomSpace to dictionary format"""
        return {
            'atoms': {
                atom_id: {
                    'type': atom.get_type(),
                    'name': atom.name,
                    'truth_value': {
                        'strength': atom.truth_value.strength,
                        'confidence': atom.truth_value.confidence
                    },
                    'outgoing': [a.id for a in atom.outgoing] if isinstance(atom, Link) else [],
                    'metadata': atom.metadata
                }
                for atom_id, atom in self.atoms.items()
            }
        }
    
    def save_to_file(self, filepath: str, format: str = 'json'):
        """Save AtomSpace to file"""
        data = self.export_to_dict()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_from_file(self, filepath: str, format: str = 'json'):
        """Load AtomSpace from file"""
        if format.lower() == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif format.lower() == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Reconstruct atoms from saved data
        self.clear()
        # Implementation would reconstruct the atom hierarchy
        # This is simplified for the basic framework
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"AtomSpace(atoms={stats['total_atoms']}, types={len(stats['atoms_by_type'])})"
    
    def __repr__(self) -> str:
        return self.__str__()