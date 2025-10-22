"""
Main CogFin-MA System: OpenCog-inspired Retrieval-Augmented Multi-Agent Framework
for Fundamental Company Analysis and Financial Insight
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

from .opencog_framework.atomspace import AtomSpace
from .agents.base_agent import SharedMemory, AgentCommunication
from .agents.sentiment_agent import SentimentAgent
from .agents.price_correlation_agent import PriceCorrelationAgent
from .agents.news_agent import NewsAgent
from .agents.financial_health_agent import FinancialHealthAgent
from .agents.report_synthesis_agent import ReportSynthesisAgent
from .knowledge_base.knowledge_base import KnowledgeBase


class CogFinMA:
    """
    Main CogFin-MA Framework class
    
    Orchestrates the multi-agent system for comprehensive financial analysis
    using OpenCog-inspired symbolic reasoning and retrieval-augmented generation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the CogFin-MA framework
        
        Args:
            config: Configuration dictionary for the framework
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize core components
        self.atomspace = AtomSpace()
        self.knowledge_base = KnowledgeBase(config.get('knowledge_base', {}))
        self.shared_memory = SharedMemory(self.atomspace)
        self.communication = AgentCommunication()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Framework state
        self.is_initialized = True
        self.analysis_history = []
        
        self.logger.info("CogFin-MA Framework initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the framework"""
        logger = logging.getLogger("CogFinMA")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - CogFinMA - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agent instances"""
        agents = {}
        
        # Agent configurations
        agent_configs = self.config.get('agents', {})
        
        try:
            # Sentiment Analysis Agent
            agents['sentiment'] = SentimentAgent(
                agent_id='sentiment_agent',
                shared_memory=self.shared_memory,
                communication=self.communication,
                config=agent_configs.get('sentiment', {})
            )
            
            # Price Correlation Agent
            agents['price_correlation'] = PriceCorrelationAgent(
                agent_id='price_correlation_agent',
                shared_memory=self.shared_memory,
                communication=self.communication,
                config=agent_configs.get('price_correlation', {})
            )
            
            # News Analysis Agent
            agents['news'] = NewsAgent(
                agent_id='news_agent',
                shared_memory=self.shared_memory,
                communication=self.communication,
                config=agent_configs.get('news', {})
            )
            
            # Financial Health Agent
            agents['financial_health'] = FinancialHealthAgent(
                agent_id='financial_health_agent',
                shared_memory=self.shared_memory,
                communication=self.communication,
                config=agent_configs.get('financial_health', {})
            )
            
            # Report Synthesis Agent
            agents['synthesis'] = ReportSynthesisAgent(
                agent_id='synthesis_agent',
                shared_memory=self.shared_memory,
                communication=self.communication,
                config=agent_configs.get('synthesis', {})
            )
            
            self.logger.info(f"Initialized {len(agents)} agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
        
        return agents
    
    async def analyze_company(self, company: str, data: Dict[str, Any], 
                            agents_to_use: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive company analysis using multi-agent system
        
        Args:
            company: Company name or ticker symbol
            data: Input data for analysis (documents, financial data, etc.)
            agents_to_use: Optional list of specific agents to use
            
        Returns:
            Comprehensive analysis results
        """
        analysis_id = f"analysis_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Starting analysis {analysis_id} for {company}")
        
        analysis_results = {
            'analysis_id': analysis_id,
            'company': company,
            'start_time': datetime.now().isoformat(),
            'agent_results': {},
            'synthesis_results': {},
            'knowledge_base_updates': {},
            'performance_metrics': {},
            'status': 'running'
        }
        
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Run individual agent analyses
            agent_results = await self._run_agent_analyses(
                company, data, agents_to_use
            )
            analysis_results['agent_results'] = agent_results
            
            # Update knowledge base with results
            kb_updates = await self._update_knowledge_base(company, agent_results)
            analysis_results['knowledge_base_updates'] = kb_updates
            
            # Run synthesis agent to coordinate and synthesize results
            synthesis_results = await self._run_synthesis(company, data, agent_results)
            analysis_results['synthesis_results'] = synthesis_results
            
            # Generate performance metrics
            performance_metrics = self._calculate_performance_metrics(agent_results)
            analysis_results['performance_metrics'] = performance_metrics
            
            # Update analysis status
            analysis_results['status'] = 'completed'
            analysis_results['end_time'] = datetime.now().isoformat()
            
            # Store in analysis history
            self.analysis_history.append(analysis_results)
            
            self.logger.info(f"Analysis {analysis_id} completed successfully")
            
        except Exception as e:
            analysis_results['status'] = 'failed'
            analysis_results['error'] = str(e)
            analysis_results['end_time'] = datetime.now().isoformat()
            self.logger.error(f"Analysis {analysis_id} failed: {e}")
            raise
        
        return analysis_results
    
    def _validate_input_data(self, data: Dict[str, Any]):
        """Validate input data format and completeness"""
        required_fields = ['company_name']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Check data types
        document_fields = ['annual_reports', 'quarterly_reports', 'news_articles', 'earnings_transcripts']
        for field in document_fields:
            if field in data and not isinstance(data[field], list):
                raise ValueError(f"Field {field} must be a list")
    
    async def _run_agent_analyses(self, company: str, data: Dict[str, Any], 
                                agents_to_use: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run analyses across specified agents"""
        if agents_to_use is None:
            agents_to_use = ['sentiment', 'price_correlation', 'news', 'financial_health']
        
        agent_results = {}
        analysis_tasks = []
        
        # Create async tasks for each agent
        for agent_name in agents_to_use:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                task = asyncio.create_task(
                    agent.run_analysis(company, data),
                    name=f"{agent_name}_analysis"
                )
                analysis_tasks.append((agent_name, task))
                self.logger.info(f"Started {agent_name} analysis")
            else:
                self.logger.warning(f"Agent {agent_name} not found, skipping")
        
        # Wait for all agent analyses to complete
        for agent_name, task in analysis_tasks:
            try:
                result = await task
                agent_results[agent_name] = result
                self.logger.info(f"Completed {agent_name} analysis")
            except Exception as e:
                self.logger.error(f"Agent {agent_name} analysis failed: {e}")
                agent_results[agent_name] = {'error': str(e), 'status': 'failed'}
        
        return agent_results
    
    async def _update_knowledge_base(self, company: str, 
                                   agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge base with analysis results"""
        kb_updates = {
            'concepts_added': 0,
            'evaluations_added': 0,
            'updates_by_agent': {}
        }
        
        for agent_name, results in agent_results.items():
            if 'error' in results:
                continue
                
            agent_updates = {'concepts': 0, 'evaluations': 0}
            
            try:
                if agent_name == 'sentiment':
                    self._add_sentiment_to_kb(company, results)
                    agent_updates['evaluations'] += 1
                
                elif agent_name == 'price_correlation':
                    self._add_correlations_to_kb(company, results)
                    agent_updates['evaluations'] += len(results.get('correlations', {}))
                
                elif agent_name == 'news':
                    self._add_news_to_kb(company, results)
                    agent_updates['evaluations'] += len(results.get('key_events', []))
                
                elif agent_name == 'financial_health':
                    self._add_financial_data_to_kb(company, results)
                    agent_updates['evaluations'] += 1
                
                kb_updates['updates_by_agent'][agent_name] = agent_updates
                kb_updates['concepts_added'] += agent_updates['concepts']
                kb_updates['evaluations_added'] += agent_updates['evaluations']
                
            except Exception as e:
                self.logger.error(f"Failed to update KB with {agent_name} results: {e}")
        
        return kb_updates
    
    def _add_sentiment_to_kb(self, company: str, results: Dict[str, Any]):
        """Add sentiment analysis results to knowledge base"""
        overall_sentiment = results.get('overall_sentiment', {})
        if isinstance(overall_sentiment, dict):
            sentiment = overall_sentiment.get('sentiment', 'neutral')
            score = overall_sentiment.get('score', 0.5)
            confidence = overall_sentiment.get('confidence', 0.5)
            
            self.knowledge_base.add_sentiment_data(
                company, sentiment, score, 'sentiment_agent', confidence
            )
    
    def _add_correlations_to_kb(self, company: str, results: Dict[str, Any]):
        """Add price correlation results to knowledge base"""
        correlations = results.get('correlations', {})
        for event_type, correlation_data in correlations.items():
            if isinstance(correlation_data, dict):
                correlation_coef = correlation_data.get('correlation_coefficient', 0)
                confidence = 1.0 if correlation_data.get('is_significant', False) else 0.5
                
                self.knowledge_base.add_price_correlation(
                    company, event_type, correlation_coef, confidence
                )
    
    def _add_news_to_kb(self, company: str, results: Dict[str, Any]):
        """Add news analysis results to knowledge base"""
        key_events = results.get('key_events', [])
        for event in key_events:
            if isinstance(event, dict):
                impact_score = event.get('impact_score', 0.5)
                date = event.get('date', datetime.now().isoformat())
                news_category = event.get('category', 'general')
                
                self.knowledge_base.add_news_impact(
                    company, news_category, impact_score, date
                )
    
    def _add_financial_data_to_kb(self, company: str, results: Dict[str, Any]):
        """Add financial health results to knowledge base"""
        financial_metrics = results.get('financial_metrics', {})
        health_score = results.get('financial_health_score', 50)
        
        # Add overall health score
        self.knowledge_base.add_financial_data(
            company, 'FinancialHealthScore', health_score, 
            datetime.now().strftime('%Y'), confidence=0.8
        )
        
        # Add specific metrics
        for category, metrics in financial_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        self.knowledge_base.add_financial_data(
                            company, metric_name, metric_value,
                            datetime.now().strftime('%Y'), confidence=0.7
                        )
    
    async def _run_synthesis(self, company: str, data: Dict[str, Any], 
                           agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run synthesis agent to coordinate and synthesize results"""
        try:
            synthesis_agent = self.agents['synthesis']
            
            # Prepare data for synthesis
            synthesis_data = {
                **data,
                'agent_results': agent_results,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            synthesis_results = await synthesis_agent.run_analysis(company, synthesis_data)
            self.logger.info("Synthesis analysis completed")
            
            return synthesis_results
            
        except Exception as e:
            self.logger.error(f"Synthesis analysis failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _calculate_performance_metrics(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis"""
        metrics = {
            'total_agents': len(self.agents),
            'successful_agents': 0,
            'failed_agents': 0,
            'average_confidence': 0.0,
            'agent_performance': {}
        }
        
        confidence_scores = []
        
        for agent_name, results in agent_results.items():
            if 'error' in results:
                metrics['failed_agents'] += 1
                metrics['agent_performance'][agent_name] = 'failed'
            else:
                metrics['successful_agents'] += 1
                metrics['agent_performance'][agent_name] = 'success'
                
                # Extract confidence if available
                confidence_data = results.get('confidence_scores', {})
                if isinstance(confidence_data, dict):
                    overall_confidence = confidence_data.get('overall_confidence', 0.5)
                    confidence_scores.append(overall_confidence)
        
        if confidence_scores:
            metrics['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        metrics['success_rate'] = metrics['successful_agents'] / max(len(agent_results), 1)
        
        return metrics
    
    def get_company_insights(self, company: str) -> Dict[str, Any]:
        """Get comprehensive insights for a company from knowledge base"""
        try:
            return self.knowledge_base.get_investment_insights(company)
        except Exception as e:
            self.logger.error(f"Failed to get insights for {company}: {e}")
            return {'error': str(e)}
    
    def find_similar_companies(self, company: str, threshold: float = 0.7) -> List[str]:
        """Find companies similar to the given company"""
        try:
            return self.knowledge_base.find_similar_companies(company, threshold)
        except Exception as e:
            self.logger.error(f"Failed to find similar companies for {company}: {e}")
            return []
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        return self.analysis_history[-limit:] if limit > 0 else self.analysis_history
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'framework_stats': {
                'total_analyses': len(self.analysis_history),
                'agents_initialized': len(self.agents),
                'is_initialized': self.is_initialized
            },
            'atomspace_stats': self.atomspace.get_statistics(),
            'knowledge_base_stats': self.knowledge_base.get_statistics(),
            'memory_stats': {
                'total_agent_results': len(self.shared_memory.agent_data),
                'total_company_data': len(self.shared_memory.company_data)
            }
        }
    
    def export_knowledge(self, filepath: str):
        """Export knowledge base to file"""
        try:
            self.knowledge_base.export_knowledge(filepath)
            self.logger.info(f"Knowledge base exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export knowledge base: {e}")
            raise
    
    def import_knowledge(self, filepath: str):
        """Import knowledge base from file"""
        try:
            self.knowledge_base.import_knowledge(filepath)
            self.logger.info(f"Knowledge base imported from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to import knowledge base: {e}")
            raise
    
    async def batch_analyze(self, companies: List[str], 
                          data_provider: callable, 
                          max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Analyze multiple companies in batch with concurrency control
        
        Args:
            companies: List of company names/tickers
            data_provider: Function that takes company name and returns data dict
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            Batch analysis results
        """
        batch_results = {
            'total_companies': len(companies),
            'completed': 0,
            'failed': 0,
            'results': {},
            'start_time': datetime.now().isoformat()
        }
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single_company(company: str):
            async with semaphore:
                try:
                    data = data_provider(company)
                    result = await self.analyze_company(company, data)
                    batch_results['results'][company] = result
                    batch_results['completed'] += 1
                    self.logger.info(f"Batch analysis completed for {company}")
                except Exception as e:
                    batch_results['results'][company] = {'error': str(e)}
                    batch_results['failed'] += 1
                    self.logger.error(f"Batch analysis failed for {company}: {e}")
        
        # Run all analyses concurrently with semaphore control
        tasks = [analyze_single_company(company) for company in companies]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['success_rate'] = batch_results['completed'] / len(companies)
        
        self.logger.info(f"Batch analysis completed: {batch_results['completed']}/{len(companies)} successful")
        
        return batch_results
    
    def __str__(self) -> str:
        return f"CogFinMA(agents={len(self.agents)}, analyses={len(self.analysis_history)})"
    
    def __repr__(self) -> str:
        return self.__str__()