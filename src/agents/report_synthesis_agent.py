"""
Report Synthesis Agent for coordinating and synthesizing multi-agent analysis results
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

from .base_agent import BaseAgent, SharedMemory, AgentCommunication, Message
from ..opencog_framework.truth_values import TruthValue, TruthValueOperations


class ReportSynthesisAgent(BaseAgent):
    """Meta-agent that coordinates other agents and synthesizes final reports"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        super().__init__(agent_id, shared_memory, communication, config)
        
        self.agent_weights = config.get('agent_weights', {
            'sentiment': 0.2,
            'price_correlation': 0.25,
            'news': 0.2,
            'financial_health': 0.35
        })
        
        self.synthesis_templates = self._initialize_synthesis_templates()
        self.coordination_timeout = config.get('coordination_timeout', 300)  # 5 minutes
        
    def _initialize_synthesis_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for different types of synthesis reports"""
        return {
            'executive_summary': {
                'sections': ['overview', 'key_findings', 'investment_thesis', 'risks', 'recommendation'],
                'max_length': 1000
            },
            'detailed_analysis': {
                'sections': ['financial_health', 'market_sentiment', 'news_impact', 'price_dynamics', 'swot'],
                'max_length': 5000
            },
            'risk_assessment': {
                'sections': ['financial_risks', 'market_risks', 'operational_risks', 'regulatory_risks'],
                'max_length': 2000
            },
            'investment_recommendation': {
                'sections': ['recommendation', 'target_price', 'time_horizon', 'key_catalysts', 'risks'],
                'max_length': 1500
            }
        }
    
    def get_required_data(self) -> List[str]:
        """Return required data types for report synthesis"""
        return ['company_name', 'stock_ticker', 'analysis_date']
    
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent analysis and synthesize final report"""
        self.logger.info(f"Starting report synthesis coordination for {company}")
        
        # Coordinate with other agents
        agent_results = await self._coordinate_agent_analysis(company, data)
        
        # Synthesize results
        synthesis_results = await self._synthesize_results(company, agent_results, data)
        
        # Generate final reports
        final_reports = await self._generate_final_reports(company, synthesis_results, agent_results)
        
        # Perform quality assurance
        qa_results = self._perform_quality_assurance(final_reports, agent_results)
        
        results = {
            'coordination_summary': agent_results,
            'synthesis': synthesis_results,
            'reports': final_reports,
            'quality_assurance': qa_results,
            'overall_confidence': self._calculate_overall_confidence(agent_results),
            'completion_time': datetime.now().isoformat()
        }
        
        self.logger.info(f"Report synthesis completed for {company}")
        return results
    
    async def _coordinate_agent_analysis(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate analysis across multiple agents"""
        coordination_results = {
            'agents_contacted': [],
            'agents_completed': [],
            'agents_failed': [],
            'results': {},
            'coordination_messages': []
        }
        
        # Define target agents
        target_agents = ['sentiment_agent', 'price_correlation_agent', 'news_agent', 'financial_health_agent']
        
        # Request analysis from each agent
        for agent_id in target_agents:
            try:
                # Send coordination message
                message_content = {
                    'request_type': 'analysis_request',
                    'company': company,
                    'data': data,
                    'coordination_id': self.agent_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                await self.communicate_with_agent(agent_id, 'analysis_request', message_content, priority=2)
                coordination_results['agents_contacted'].append(agent_id)
                
                # Log coordination message
                coordination_results['coordination_messages'].append({
                    'agent': agent_id,
                    'action': 'analysis_requested',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Failed to contact agent {agent_id}: {e}")
                coordination_results['agents_failed'].append(agent_id)
        
        # Wait for responses or timeout
        await self._wait_for_agent_responses(coordination_results, target_agents)
        
        # Collect results from shared memory
        for agent_id in coordination_results['agents_completed']:
            agent_result = self.shared_memory.get_agent_results(agent_id)
            if agent_result:
                coordination_results['results'][agent_id] = agent_result
        
        return coordination_results
    
    async def _wait_for_agent_responses(self, coordination_results: Dict[str, Any], 
                                      target_agents: List[str]):
        """Wait for agent responses with timeout"""
        start_time = datetime.now()
        timeout_seconds = self.coordination_timeout
        
        while len(coordination_results['agents_completed']) < len(target_agents):
            # Check for timeout
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > timeout_seconds:
                remaining_agents = set(target_agents) - set(coordination_results['agents_completed'])
                coordination_results['agents_failed'].extend(list(remaining_agents))
                self.logger.warning(f"Timeout waiting for agents: {remaining_agents}")
                break
            
            # Listen for messages
            message = await self.listen_for_messages(timeout=1.0)
            if message and message.message_type == 'analysis_complete':
                sender = message.sender
                if sender in target_agents and sender not in coordination_results['agents_completed']:
                    coordination_results['agents_completed'].append(sender)
                    coordination_results['coordination_messages'].append({
                        'agent': sender,
                        'action': 'analysis_completed',
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.info(f"Received completion notification from {sender}")
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.1)
    
    async def _synthesize_results(self, company: str, agent_results: Dict[str, Any], 
                                data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents using OpenCog-style reasoning"""
        synthesis = {
            'overall_sentiment': 'neutral',
            'financial_health_score': 0.0,
            'market_impact_assessment': 'neutral',
            'investment_thesis': {},
            'risk_profile': {},
            'opportunity_profile': {},
            'consensus_analysis': {},
            'conflict_resolution': {},
            'confidence_weighted_scores': {}
        }
        
        results = agent_results.get('results', {})
        
        # Synthesize sentiment analysis
        synthesis['overall_sentiment'] = await self._synthesize_sentiment(results)
        
        # Synthesize financial health
        synthesis['financial_health_score'] = self._synthesize_financial_health(results)
        
        # Synthesize market impact
        synthesis['market_impact_assessment'] = self._synthesize_market_impact(results)
        
        # Generate investment thesis
        synthesis['investment_thesis'] = await self._generate_investment_thesis(results, company)
        
        # Synthesize risk profile
        synthesis['risk_profile'] = self._synthesize_risk_profile(results)
        
        # Synthesize opportunity profile
        synthesis['opportunity_profile'] = self._synthesize_opportunity_profile(results)
        
        # Perform consensus analysis
        synthesis['consensus_analysis'] = self._perform_consensus_analysis(results)
        
        # Resolve conflicts between agents
        synthesis['conflict_resolution'] = self._resolve_agent_conflicts(results)
        
        # Calculate confidence-weighted scores
        synthesis['confidence_weighted_scores'] = self._calculate_weighted_scores(results)
        
        return synthesis
    
    async def _synthesize_sentiment(self, results: Dict[str, Any]) -> str:
        """Synthesize sentiment across different agents"""
        sentiment_inputs = []
        
        # Get sentiment from sentiment agent
        if 'sentiment_agent' in results:
            sentiment_result = results['sentiment_agent']
            if isinstance(sentiment_result, dict):
                # Look for sentiment in various possible structures
                sentiment_data = None
                for key in ['overall_sentiment', 'sentiment_analysis', 'results']:
                    if key in sentiment_result:
                        sentiment_data = sentiment_result[key]
                        break
                
                if sentiment_data and isinstance(sentiment_data, dict):
                    agent_sentiment = sentiment_data.get('sentiment', 'neutral')
                    confidence = sentiment_data.get('confidence', 0.5)
                    sentiment_inputs.append({
                        'sentiment': agent_sentiment,
                        'confidence': confidence,
                        'source': 'sentiment_agent'
                    })
        
        # Get sentiment from news agent
        if 'news_agent' in results:
            news_result = results['news_agent']
            if isinstance(news_result, dict):
                news_data = news_result.get('results', news_result)
                if 'impact_assessment' in news_data:
                    impact = news_data['impact_assessment'].get('overall_impact', 'neutral')
                    # Convert impact to sentiment
                    sentiment_map = {'high': 'positive', 'medium': 'neutral', 'low': 'neutral'}
                    sentiment_inputs.append({
                        'sentiment': sentiment_map.get(impact, 'neutral'),
                        'confidence': 0.7,
                        'source': 'news_agent'
                    })
        
        # Synthesize using weighted voting
        if not sentiment_inputs:
            return 'neutral'
        
        sentiment_scores = {'positive': 0, 'neutral': 0, 'negative': 0}
        total_weight = 0
        
        for input_data in sentiment_inputs:
            sentiment = input_data['sentiment']
            confidence = input_data['confidence']
            
            sentiment_scores[sentiment] += confidence
            total_weight += confidence
        
        # Normalize and find consensus
        if total_weight > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_weight
        
        return max(sentiment_scores.items(), key=lambda x: x[1])[0]
    
    def _synthesize_financial_health(self, results: Dict[str, Any]) -> float:
        """Synthesize financial health score"""
        if 'financial_health_agent' in results:
            fh_result = results['financial_health_agent']
            if isinstance(fh_result, dict):
                fh_data = fh_result.get('results', fh_result)
                return fh_data.get('financial_health_score', 50.0)
        
        # Fallback: estimate from other agents
        estimated_score = 50.0  # Neutral baseline
        
        # Adjust based on sentiment
        sentiment = self._get_sentiment_from_results(results)
        if sentiment == 'positive':
            estimated_score += 10
        elif sentiment == 'negative':
            estimated_score -= 10
        
        return max(0.0, min(100.0, estimated_score))
    
    def _synthesize_market_impact(self, results: Dict[str, Any]) -> str:
        """Synthesize market impact assessment"""
        impact_indicators = []
        
        # From price correlation agent
        if 'price_correlation_agent' in results:
            pc_result = results['price_correlation_agent']
            if isinstance(pc_result, dict):
                pc_data = pc_result.get('results', pc_result)
                correlations = pc_data.get('correlations', {})
                
                # Check for significant correlations
                significant_count = 0
                for correlation_data in correlations.values():
                    if isinstance(correlation_data, dict) and correlation_data.get('is_significant'):
                        significant_count += 1
                
                if significant_count > 2:
                    impact_indicators.append('high')
                elif significant_count > 0:
                    impact_indicators.append('medium')
                else:
                    impact_indicators.append('low')
        
        # From news agent
        if 'news_agent' in results:
            news_result = results['news_agent']
            if isinstance(news_result, dict):
                news_data = news_result.get('results', news_result)
                impact_assessment = news_data.get('impact_assessment', {})
                overall_impact = impact_assessment.get('overall_impact', 'medium')
                impact_indicators.append(overall_impact)
        
        # Synthesize impact levels
        if not impact_indicators:
            return 'medium'
        
        impact_counts = {'high': 0, 'medium': 0, 'low': 0}
        for impact in impact_indicators:
            impact_counts[impact] += 1
        
        return max(impact_counts.items(), key=lambda x: x[1])[0]
    
    async def _generate_investment_thesis(self, results: Dict[str, Any], company: str) -> Dict[str, Any]:
        """Generate comprehensive investment thesis"""
        thesis = {
            'recommendation': 'hold',
            'conviction_level': 'medium',
            'key_strengths': [],
            'key_concerns': [],
            'catalysts': [],
            'target_price_factors': {},
            'time_horizon': '12 months'
        }
        
        # Gather key strengths from all agents
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                
                # Extract strengths
                strengths = self._extract_strengths_from_agent(agent_data, agent_id)
                thesis['key_strengths'].extend(strengths)
                
                # Extract concerns
                concerns = self._extract_concerns_from_agent(agent_data, agent_id)
                thesis['key_concerns'].extend(concerns)
        
        # Determine recommendation
        thesis['recommendation'] = self._determine_investment_recommendation(results)
        
        # Set conviction level
        thesis['conviction_level'] = self._determine_conviction_level(results)
        
        # Identify catalysts
        thesis['catalysts'] = self._identify_investment_catalysts(results)
        
        return thesis
    
    def _extract_strengths_from_agent(self, agent_data: Dict[str, Any], agent_id: str) -> List[str]:
        """Extract strengths from individual agent results"""
        strengths = []
        
        if agent_id == 'financial_health_agent':
            fh_strengths = agent_data.get('strengths', [])
            for strength in fh_strengths:
                if isinstance(strength, dict):
                    strengths.append(strength.get('strength', ''))
                else:
                    strengths.append(str(strength))
        
        elif agent_id == 'sentiment_agent':
            sentiment_data = agent_data.get('overall_sentiment', {})
            if isinstance(sentiment_data, dict) and sentiment_data.get('sentiment') == 'positive':
                strengths.append('Positive market sentiment and communication tone')
        
        elif agent_id == 'news_agent':
            opportunities = agent_data.get('opportunity_indicators', {})
            if isinstance(opportunities, dict):
                for opp_type, opp_list in opportunities.items():
                    if isinstance(opp_list, list) and len(opp_list) > 0:
                        strengths.append(f'Multiple {opp_type.replace("_", " ")} identified in news')
        
        return strengths[:3]  # Limit to top 3 per agent
    
    def _extract_concerns_from_agent(self, agent_data: Dict[str, Any], agent_id: str) -> List[str]:
        """Extract concerns from individual agent results"""
        concerns = []
        
        if agent_id == 'financial_health_agent':
            red_flags = agent_data.get('red_flags', [])
            for flag in red_flags:
                if isinstance(flag, dict):
                    concerns.append(flag.get('flag', ''))
                else:
                    concerns.append(str(flag))
        
        elif agent_id == 'news_agent':
            risks = agent_data.get('risk_indicators', {})
            if isinstance(risks, dict):
                for risk_type, risk_list in risks.items():
                    if isinstance(risk_list, list) and len(risk_list) > 2:
                        concerns.append(f'Multiple {risk_type.replace("_", " ")} identified')
        
        return concerns[:3]  # Limit to top 3 per agent
    
    def _determine_investment_recommendation(self, results: Dict[str, Any]) -> str:
        """Determine overall investment recommendation"""
        positive_signals = 0
        negative_signals = 0
        total_weight = 0
        
        for agent_id, agent_result in results.items():
            if not isinstance(agent_result, dict):
                continue
                
            agent_data = agent_result.get('results', agent_result)
            weight = self.agent_weights.get(agent_id.replace('_agent', ''), 0.25)
            
            # Evaluate agent signal
            signal = self._evaluate_agent_signal(agent_data, agent_id)
            
            if signal > 0:
                positive_signals += signal * weight
            elif signal < 0:
                negative_signals += abs(signal) * weight
            
            total_weight += weight
        
        if total_weight == 0:
            return 'hold'
        
        net_signal = (positive_signals - negative_signals) / total_weight
        
        if net_signal > 0.3:
            return 'buy'
        elif net_signal < -0.3:
            return 'sell'
        else:
            return 'hold'
    
    def _evaluate_agent_signal(self, agent_data: Dict[str, Any], agent_id: str) -> float:
        """Evaluate signal from individual agent (-1 to 1)"""
        if agent_id == 'financial_health_agent':
            health_score = agent_data.get('financial_health_score', 50)
            return (health_score - 50) / 50  # Normalize to -1 to 1
        
        elif agent_id == 'sentiment_agent':
            sentiment_data = agent_data.get('overall_sentiment', {})
            if isinstance(sentiment_data, dict):
                sentiment = sentiment_data.get('sentiment', 'neutral')
                if sentiment == 'positive':
                    return 0.5
                elif sentiment == 'negative':
                    return -0.5
        
        elif agent_id == 'news_agent':
            impact_assessment = agent_data.get('impact_assessment', {})
            if isinstance(impact_assessment, dict):
                impact_score = impact_assessment.get('impact_score', 0.5)
                return (impact_score - 0.5) * 2  # Convert to -1 to 1
        
        return 0.0  # Neutral
    
    def _determine_conviction_level(self, results: Dict[str, Any]) -> str:
        """Determine conviction level based on consensus and confidence"""
        confidence_scores = []
        
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                confidence_data = agent_data.get('confidence_scores', {})
                
                if isinstance(confidence_data, dict):
                    overall_confidence = confidence_data.get('overall_confidence', 0.5)
                    confidence_scores.append(overall_confidence)
        
        if not confidence_scores:
            return 'medium'
        
        avg_confidence = np.mean(confidence_scores)
        
        if avg_confidence > 0.8:
            return 'high'
        elif avg_confidence > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _identify_investment_catalysts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential investment catalysts"""
        catalysts = []
        
        # From news analysis
        if 'news_agent' in results:
            news_result = results['news_agent']
            if isinstance(news_result, dict):
                news_data = news_result.get('results', news_result)
                key_events = news_data.get('key_events', [])
                
                for event in key_events[:3]:  # Top 3 events
                    if isinstance(event, dict):
                        catalysts.append({
                            'type': 'news_event',
                            'description': event.get('title', ''),
                            'impact': event.get('impact_score', 0),
                            'timeframe': 'short_term'
                        })
        
        # From financial health analysis
        if 'financial_health_agent' in results:
            fh_result = results['financial_health_agent']
            if isinstance(fh_result, dict):
                fh_data = fh_result.get('results', fh_result)
                future_outlook = fh_data.get('future_outlook', {})
                
                if isinstance(future_outlook, dict) and future_outlook.get('growth_prospects') == 'strong':
                    catalysts.append({
                        'type': 'fundamental_improvement',
                        'description': 'Strong growth prospects indicated by financial metrics',
                        'impact': 0.7,
                        'timeframe': 'medium_term'
                    })
        
        return catalysts
    
    def _synthesize_risk_profile(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize comprehensive risk profile"""
        risk_profile = {
            'overall_risk': 'medium',
            'financial_risk': 'medium',
            'market_risk': 'medium',
            'operational_risk': 'medium',
            'regulatory_risk': 'medium',
            'key_risk_factors': [],
            'risk_mitigation_factors': []
        }
        
        # Aggregate risk indicators from all agents
        risk_indicators = defaultdict(list)
        
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                agent_risks = self._extract_risks_from_agent(agent_data, agent_id)
                
                for risk_type, risk_level in agent_risks.items():
                    risk_indicators[risk_type].append(risk_level)
        
        # Synthesize risk levels
        for risk_type, risk_levels in risk_indicators.items():
            if risk_levels:
                # Use most conservative (highest) risk assessment
                risk_mapping = {'low': 1, 'medium': 2, 'high': 3}
                max_risk_level = max(risk_levels, key=lambda x: risk_mapping.get(x, 2))
                risk_profile[f'{risk_type}_risk'] = max_risk_level
        
        # Calculate overall risk
        risk_levels = [
            risk_profile.get('financial_risk', 'medium'),
            risk_profile.get('market_risk', 'medium'),
            risk_profile.get('operational_risk', 'medium'),
            risk_profile.get('regulatory_risk', 'medium')
        ]
        
        high_risk_count = risk_levels.count('high')
        if high_risk_count >= 2:
            risk_profile['overall_risk'] = 'high'
        elif high_risk_count == 0 and risk_levels.count('low') >= 2:
            risk_profile['overall_risk'] = 'low'
        
        return risk_profile
    
    def _extract_risks_from_agent(self, agent_data: Dict[str, Any], agent_id: str) -> Dict[str, str]:
        """Extract risk assessments from individual agents"""
        risks = {}
        
        if agent_id == 'financial_health_agent':
            risk_assessment = agent_data.get('risk_assessment', {})
            if isinstance(risk_assessment, dict):
                risks['financial'] = risk_assessment.get('overall_risk', 'medium')
        
        elif agent_id == 'news_agent':
            risk_indicators = agent_data.get('risk_indicators', {})
            if isinstance(risk_indicators, dict):
                overall_risk = risk_indicators.get('overall_risk_level', 'medium')
                risks['regulatory'] = overall_risk
                risks['operational'] = overall_risk
        
        return risks
    
    def _synthesize_opportunity_profile(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize opportunity profile"""
        opportunity_profile = {
            'growth_opportunities': [],
            'market_opportunities': [], 
            'strategic_opportunities': [],
            'overall_opportunity_level': 'medium'
        }
        
        # Collect opportunities from all agents
        all_opportunities = []
        
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                opportunities = self._extract_opportunities_from_agent(agent_data, agent_id)
                all_opportunities.extend(opportunities)
        
        # Categorize opportunities
        for opportunity in all_opportunities:
            opp_type = opportunity.get('type', 'strategic')
            opp_list_key = f'{opp_type}_opportunities'
            
            if opp_list_key in opportunity_profile:
                opportunity_profile[opp_list_key].append(opportunity)
            else:
                opportunity_profile['strategic_opportunities'].append(opportunity)
        
        # Assess overall opportunity level
        total_opportunities = sum(len(opps) for opps in [
            opportunity_profile['growth_opportunities'],
            opportunity_profile['market_opportunities'],
            opportunity_profile['strategic_opportunities']
        ])
        
        if total_opportunities > 10:
            opportunity_profile['overall_opportunity_level'] = 'high'
        elif total_opportunities < 3:
            opportunity_profile['overall_opportunity_level'] = 'low'
        
        return opportunity_profile
    
    def _extract_opportunities_from_agent(self, agent_data: Dict[str, Any], agent_id: str) -> List[Dict[str, Any]]:
        """Extract opportunities from individual agents"""
        opportunities = []
        
        if agent_id == 'news_agent':
            opp_indicators = agent_data.get('opportunity_indicators', {})
            if isinstance(opp_indicators, dict):
                for opp_type, opp_list in opp_indicators.items():
                    if isinstance(opp_list, list):
                        for opp in opp_list[:2]:  # Limit per category
                            if isinstance(opp, dict):
                                opportunities.append({
                                    'type': opp_type.replace('_opportunities', ''),
                                    'description': opp.get('title', ''),
                                    'source': agent_id
                                })
        
        return opportunities
    
    def _perform_consensus_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consensus and disagreements between agents"""
        consensus = {
            'agreement_level': 'medium',
            'consensus_signals': [],
            'disagreement_areas': [],
            'reliability_assessment': {}
        }
        
        # Compare sentiment assessments
        sentiment_assessments = []
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                sentiment = self._get_sentiment_indicator(agent_data, agent_id)
                if sentiment:
                    sentiment_assessments.append((agent_id, sentiment))
        
        # Calculate sentiment consensus
        if len(sentiment_assessments) > 1:
            sentiment_values = [s[1] for s in sentiment_assessments]
            unique_sentiments = len(set(sentiment_values))
            
            if unique_sentiments == 1:
                consensus['consensus_signals'].append('All agents agree on sentiment direction')
            elif unique_sentiments == len(sentiment_assessments):
                consensus['disagreement_areas'].append('No consensus on sentiment direction')
        
        # Assess overall agreement
        disagreement_count = len(consensus['disagreement_areas'])
        if disagreement_count == 0:
            consensus['agreement_level'] = 'high'
        elif disagreement_count > 2:
            consensus['agreement_level'] = 'low'
        
        return consensus
    
    def _get_sentiment_indicator(self, agent_data: Dict[str, Any], agent_id: str) -> Optional[str]:
        """Get sentiment indicator from agent data"""
        if agent_id == 'sentiment_agent':
            sentiment_data = agent_data.get('overall_sentiment', {})
            if isinstance(sentiment_data, dict):
                return sentiment_data.get('sentiment')
        
        elif agent_id == 'news_agent':
            impact_assessment = agent_data.get('impact_assessment', {})
            if isinstance(impact_assessment, dict):
                impact = impact_assessment.get('overall_impact', 'medium')
                # Convert impact to sentiment
                impact_map = {'high': 'positive', 'medium': 'neutral', 'low': 'negative'}
                return impact_map.get(impact)
        
        return None
    
    def _resolve_agent_conflicts(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts between agent analyses"""
        conflicts = {
            'identified_conflicts': [],
            'resolution_strategy': {},
            'final_reconciliation': {}
        }
        
        # Implementation would identify and resolve specific conflicts
        # This is a simplified version
        
        return conflicts
    
    def _calculate_weighted_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence-weighted scores across agents"""
        weighted_scores = {}
        
        for metric in ['overall_assessment', 'risk_level', 'opportunity_level']:
            scores = []
            weights = []
            
            for agent_id, agent_result in results.items():
                if isinstance(agent_result, dict):
                    agent_data = agent_result.get('results', agent_result)
                    
                    # Get agent's confidence
                    confidence_data = agent_data.get('confidence_scores', {})
                    confidence = 0.5
                    if isinstance(confidence_data, dict):
                        confidence = confidence_data.get('overall_confidence', 0.5)
                    
                    # Get agent's assessment (simplified scoring)
                    agent_score = self._get_agent_metric_score(agent_data, agent_id, metric)
                    if agent_score is not None:
                        scores.append(agent_score)
                        weights.append(confidence * self.agent_weights.get(agent_id.replace('_agent', ''), 0.25))
            
            if scores and weights:
                weighted_avg = np.average(scores, weights=weights)
                weighted_scores[metric] = float(weighted_avg)
        
        return weighted_scores
    
    def _get_agent_metric_score(self, agent_data: Dict[str, Any], agent_id: str, metric: str) -> Optional[float]:
        """Get numerical score for a metric from agent data"""
        if metric == 'overall_assessment':
            if agent_id == 'financial_health_agent':
                return agent_data.get('financial_health_score', 50.0) / 100.0
            elif agent_id == 'sentiment_agent':
                sentiment_data = agent_data.get('overall_sentiment', {})
                if isinstance(sentiment_data, dict):
                    sentiment = sentiment_data.get('sentiment', 'neutral')
                    sentiment_map = {'positive': 0.8, 'neutral': 0.5, 'negative': 0.2}
                    return sentiment_map.get(sentiment, 0.5)
        
        return None
    
    async def _generate_final_reports(self, company: str, synthesis: Dict[str, Any], 
                                    agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive reports"""
        reports = {}
        
        for report_type, template in self.synthesis_templates.items():
            report_content = await self._generate_report_section(
                report_type, template, company, synthesis, agent_results
            )
            reports[report_type] = report_content
        
        return reports
    
    async def _generate_report_section(self, report_type: str, template: Dict[str, Any], 
                                     company: str, synthesis: Dict[str, Any], 
                                     agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a specific report section"""
        sections = template.get('sections', [])
        max_length = template.get('max_length', 2000)
        
        report_content = {
            'company': company,
            'report_type': report_type,
            'generated_at': datetime.now().isoformat(),
            'sections': {}
        }
        
        for section in sections:
            section_content = self._generate_section_content(
                section, company, synthesis, agent_results
            )
            report_content['sections'][section] = section_content
        
        # Ensure report doesn't exceed max length
        total_length = sum(len(str(content)) for content in report_content['sections'].values())
        if total_length > max_length:
            report_content['truncated'] = True
        
        return report_content
    
    def _generate_section_content(self, section: str, company: str, 
                                synthesis: Dict[str, Any], agent_results: Dict[str, Any]) -> str:
        """Generate content for a specific report section"""
        if section == 'overview':
            return f"Analysis of {company} based on multi-agent financial intelligence framework."
        
        elif section == 'key_findings':
            findings = []
            findings.append(f"Overall sentiment: {synthesis.get('overall_sentiment', 'neutral')}")
            findings.append(f"Financial health score: {synthesis.get('financial_health_score', 0):.1f}/100")
            findings.append(f"Market impact: {synthesis.get('market_impact_assessment', 'medium')}")
            return " | ".join(findings)
        
        elif section == 'recommendation':
            investment_thesis = synthesis.get('investment_thesis', {})
            recommendation = investment_thesis.get('recommendation', 'hold')
            conviction = investment_thesis.get('conviction_level', 'medium')
            return f"Recommendation: {recommendation.upper()} with {conviction} conviction"
        
        elif section == 'risks':
            risk_profile = synthesis.get('risk_profile', {})
            overall_risk = risk_profile.get('overall_risk', 'medium')
            return f"Overall risk level: {overall_risk}"
        
        elif section == 'financial_health':
            health_score = synthesis.get('financial_health_score', 0)
            return f"Financial health assessment: {health_score:.1f}/100"
        
        else:
            return f"Analysis for {section} section"
    
    def _perform_quality_assurance(self, reports: Dict[str, Any], 
                                 agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality assurance on generated reports"""
        qa_results = {
            'completeness_check': {},
            'consistency_check': {},
            'confidence_assessment': {},
            'overall_quality_score': 0.0
        }
        
        # Check report completeness
        for report_type, report in reports.items():
            template = self.synthesis_templates.get(report_type, {})
            expected_sections = set(template.get('sections', []))
            actual_sections = set(report.get('sections', {}).keys())
            
            completeness = len(actual_sections & expected_sections) / len(expected_sections) if expected_sections else 0
            qa_results['completeness_check'][report_type] = completeness
        
        # Check consistency across reports
        consistency_score = self._check_report_consistency(reports)
        qa_results['consistency_check']['overall_consistency'] = consistency_score
        
        # Assess confidence based on agent results
        confidence_scores = []
        for agent_id, agent_result in agent_results.get('results', {}).items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                confidence_data = agent_data.get('confidence_scores', {})
                if isinstance(confidence_data, dict):
                    overall_confidence = confidence_data.get('overall_confidence', 0.5)
                    confidence_scores.append(overall_confidence)
        
        if confidence_scores:
            qa_results['confidence_assessment']['average_confidence'] = np.mean(confidence_scores)
        else:
            qa_results['confidence_assessment']['average_confidence'] = 0.5
        
        # Calculate overall quality score
        quality_components = [
            np.mean(list(qa_results['completeness_check'].values())),
            consistency_score,
            qa_results['confidence_assessment']['average_confidence']
        ]
        
        qa_results['overall_quality_score'] = np.mean(quality_components)
        
        return qa_results
    
    def _check_report_consistency(self, reports: Dict[str, Any]) -> float:
        """Check consistency across different reports"""
        # Simplified consistency check
        # In practice, this would check for contradictions between reports
        return 0.8  # Placeholder score
    
    def _calculate_overall_confidence(self, agent_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score across all agents"""
        confidence_scores = []
        
        results = agent_results.get('results', {})
        for agent_id, agent_result in results.items():
            if isinstance(agent_result, dict):
                agent_data = agent_result.get('results', agent_result)
                confidence_data = agent_data.get('confidence_scores', {})
                
                if isinstance(confidence_data, dict):
                    overall_confidence = confidence_data.get('overall_confidence', 0.5)
                    # Weight by agent importance
                    weight = self.agent_weights.get(agent_id.replace('_agent', ''), 0.25)
                    confidence_scores.append(overall_confidence * weight)
        
        return sum(confidence_scores) if confidence_scores else 0.5
    
    def _get_sentiment_from_results(self, results: Dict[str, Any]) -> str:
        """Helper to extract overall sentiment from results"""
        if 'sentiment_agent' in results:
            sentiment_result = results['sentiment_agent']
            if isinstance(sentiment_result, dict):
                sentiment_data = sentiment_result.get('results', sentiment_result)
                overall_sentiment = sentiment_data.get('overall_sentiment', {})
                if isinstance(overall_sentiment, dict):
                    return overall_sentiment.get('sentiment', 'neutral')
        
        return 'neutral'