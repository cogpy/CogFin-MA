"""
Base agent class for the multi-agent financial analysis system
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Set
import asyncio
import logging
import json
import threading
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..opencog_framework.atomspace import AtomSpace
from ..opencog_framework.atoms import Atom, ConceptNode, PredicateNode, EvaluationLink
from ..opencog_framework.truth_values import TruthValue


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    COMMUNICATING = "communicating"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class Message:
    """Inter-agent communication message"""
    sender: str
    receiver: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'message_type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority
        }


class SharedMemory:
    """Shared memory system for agent communication and knowledge sharing"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.agent_data: Dict[str, Dict[str, Any]] = {}
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.company_data: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def store_agent_result(self, agent_id: str, analysis_type: str, results: Dict[str, Any]):
        """Store analysis results from an agent"""
        with self._lock:
            if agent_id not in self.agent_data:
                self.agent_data[agent_id] = {}
            self.agent_data[agent_id][analysis_type] = {
                'results': results,
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id
            }
            
            # Also store in analysis results by type
            if analysis_type not in self.analysis_results:
                self.analysis_results[analysis_type] = {}
            self.analysis_results[analysis_type][agent_id] = results
    
    def get_agent_results(self, agent_id: str, analysis_type: str = None) -> Optional[Dict[str, Any]]:
        """Get results from a specific agent"""
        with self._lock:
            if agent_id not in self.agent_data:
                return None
            
            if analysis_type:
                return self.agent_data[agent_id].get(analysis_type)
            return self.agent_data[agent_id]
    
    def get_analysis_results(self, analysis_type: str) -> Dict[str, Any]:
        """Get all results for a specific analysis type"""
        with self._lock:
            return self.analysis_results.get(analysis_type, {})
    
    def store_company_data(self, company: str, data_type: str, data: Dict[str, Any]):
        """Store company-specific data"""
        with self._lock:
            if company not in self.company_data:
                self.company_data[company] = {}
            self.company_data[company][data_type] = data
    
    def get_company_data(self, company: str, data_type: str = None) -> Optional[Dict[str, Any]]:
        """Get company data"""
        with self._lock:
            if company not in self.company_data:
                return None
            
            if data_type:
                return self.company_data[company].get(data_type)
            return self.company_data[company]
    
    def add_knowledge(self, subject: str, predicate: str, object_: str, 
                     truth_value: TruthValue = None, metadata: Dict[str, Any] = None):
        """Add knowledge to the AtomSpace"""
        subject_node = self.atomspace.create_concept(subject)
        object_node = self.atomspace.create_concept(object_)
        predicate_node = self.atomspace.create_predicate(predicate)
        
        eval_link = self.atomspace.create_evaluation(
            predicate_node, [subject_node, object_node], truth_value
        )
        
        if metadata:
            eval_link.metadata.update(metadata)
        
        return eval_link


class AgentCommunication:
    """Message passing system for agent communication"""
    
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, Set[str]] = {}  # topic -> set of agent_ids
        self.message_history: List[Message] = []
        self._lock = threading.RLock()
    
    def register_agent(self, agent_id: str):
        """Register an agent for communication"""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = asyncio.Queue()
    
    async def send_message(self, message: Message):
        """Send a message to an agent"""
        if message.receiver in self.message_queues:
            await self.message_queues[message.receiver].put(message)
            with self._lock:
                self.message_history.append(message)
    
    async def receive_message(self, agent_id: str, timeout: float = None) -> Optional[Message]:
        """Receive a message for an agent"""
        if agent_id not in self.message_queues:
            return None
        
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.message_queues[agent_id].get(), timeout=timeout
                )
            else:
                return await self.message_queues[agent_id].get()
        except asyncio.TimeoutError:
            return None
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = set()
        self.subscribers[topic].add(agent_id)
    
    async def publish(self, topic: str, message: Message):
        """Publish a message to all subscribers of a topic"""
        if topic in self.subscribers:
            for agent_id in self.subscribers[topic]:
                message.receiver = agent_id
                await self.send_message(message)


class BaseAgent(ABC):
    """Base class for all financial analysis agents"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.shared_memory = shared_memory
        self.communication = communication
        self.config = config or {}
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"Agent.{agent_id}")
        self.analysis_results: Dict[str, Any] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Register with communication system
        self.communication.register_agent(self.agent_id)
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup agent-specific logging"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'%(asctime)s - Agent.{self.agent_id} - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform agent-specific analysis"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return list of required data types for analysis"""
        pass
    
    async def run_analysis(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete analysis workflow"""
        self.start_time = datetime.now()
        self.state = AgentState.ANALYZING
        
        try:
            self.logger.info(f"Starting analysis for {company}")
            
            # Validate required data
            required_data = self.get_required_data()
            missing_data = [req for req in required_data if req not in data]
            if missing_data:
                self.logger.warning(f"Missing required data: {missing_data}")
            
            # Perform analysis
            results = await self.analyze(company, data)
            
            # Store results in shared memory
            self.analysis_results = results
            self.shared_memory.store_agent_result(
                self.agent_id, self.get_analysis_type(), results
            )
            
            # Add knowledge to AtomSpace
            await self._add_knowledge_to_atomspace(company, results)
            
            self.state = AgentState.COMPLETED
            self.end_time = datetime.now()
            
            self.logger.info(f"Analysis completed for {company}")
            return results
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Analysis failed for {company}: {str(e)}")
            raise
    
    async def _add_knowledge_to_atomspace(self, company: str, results: Dict[str, Any]):
        """Add analysis results to the AtomSpace as structured knowledge"""
        analysis_type = self.get_analysis_type()
        
        # Add basic facts
        for key, value in results.items():
            if isinstance(value, (str, int, float)):
                truth_value = TruthValue(0.8, 0.9)  # High confidence in own analysis
                self.shared_memory.add_knowledge(
                    company, f"{analysis_type}_{key}", str(value), 
                    truth_value, {"agent": self.agent_id, "timestamp": datetime.now().isoformat()}
                )
    
    async def communicate_with_agent(self, target_agent: str, message_type: str, 
                                   content: Dict[str, Any], priority: int = 1):
        """Send a message to another agent"""
        self.state = AgentState.COMMUNICATING
        
        message = Message(
            sender=self.agent_id,
            receiver=target_agent,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority
        )
        
        await self.communication.send_message(message)
        self.logger.info(f"Sent message to {target_agent}: {message_type}")
    
    async def listen_for_messages(self, timeout: float = 1.0) -> Optional[Message]:
        """Listen for incoming messages"""
        return await self.communication.receive_message(self.agent_id, timeout)
    
    def get_analysis_type(self) -> str:
        """Return the type of analysis this agent performs"""
        return self.__class__.__name__.replace('Agent', '').lower()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        metrics = {
            'agent_id': self.agent_id,
            'state': self.state.value,
            'analysis_type': self.get_analysis_type()
        }
        
        if self.start_time:
            metrics['start_time'] = self.start_time.isoformat()
        
        if self.end_time:
            metrics['end_time'] = self.end_time.isoformat()
            metrics['duration'] = (self.end_time - self.start_time).total_seconds()
        
        return metrics
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, state={self.state.value})"