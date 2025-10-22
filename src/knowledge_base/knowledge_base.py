"""
Central Knowledge Base for the CogFin-MA system
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import json
import logging
from datetime import datetime

from ..opencog_framework.atomspace import AtomSpace
from ..opencog_framework.atoms import ConceptNode, PredicateNode, EvaluationLink, InheritanceLink
from ..opencog_framework.truth_values import TruthValue
from ..opencog_framework.pattern_matcher import PatternMatcher, Variable, Pattern


class KnowledgeBase:
    """Central knowledge repository integrating OpenCog AtomSpace with financial domain knowledge"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.atomspace = AtomSpace()
        self.pattern_matcher = PatternMatcher(self.atomspace)
        self.logger = logging.getLogger(__name__)
        
        # Initialize domain knowledge
        self._initialize_financial_ontology()
        self._initialize_reasoning_rules()
        
    def _initialize_financial_ontology(self):
        """Initialize core financial concepts and relationships"""
        # Core financial concepts
        financial_concepts = [
            'Company', 'Stock', 'Revenue', 'Profit', 'Assets', 'Liabilities', 'Equity',
            'CashFlow', 'Earnings', 'Market', 'Industry', 'Sector', 'Investment',
            'Risk', 'Return', 'Volatility', 'Growth', 'Valuation', 'Ratio'
        ]
        
        for concept_name in financial_concepts:
            concept = self.atomspace.create_concept(concept_name)
            
        # Financial metrics hierarchy
        self._create_metric_hierarchy()
        
        # Risk categories
        self._create_risk_taxonomy()
        
        # Market relationships
        self._create_market_relationships()
        
        self.logger.info("Financial ontology initialized")
    
    def _create_metric_hierarchy(self):
        """Create hierarchy of financial metrics"""
        # Profitability metrics
        profitability = self.atomspace.create_concept("ProfitabilityMetric")
        
        profit_metrics = ["ROE", "ROA", "ROIC", "GrossMargin", "OperatingMargin", "NetMargin"]
        for metric in profit_metrics:
            metric_node = self.atomspace.create_concept(metric)
            inheritance = InheritanceLink(metric_node, profitability)
            self.atomspace.add(inheritance)
        
        # Liquidity metrics
        liquidity = self.atomspace.create_concept("LiquidityMetric")
        
        liquidity_metrics = ["CurrentRatio", "QuickRatio", "CashRatio"]
        for metric in liquidity_metrics:
            metric_node = self.atomspace.create_concept(metric)
            inheritance = InheritanceLink(metric_node, liquidity)
            self.atomspace.add(inheritance)
        
        # Leverage metrics
        leverage = self.atomspace.create_concept("LeverageMetric")
        
        leverage_metrics = ["DebtToEquity", "DebtToAssets", "InterestCoverage"]
        for metric in leverage_metrics:
            metric_node = self.atomspace.create_concept(metric)
            inheritance = InheritanceLink(metric_node, leverage)
            self.atomspace.add(inheritance)
    
    def _create_risk_taxonomy(self):
        """Create risk classification taxonomy"""
        risk = self.atomspace.create_concept("Risk")
        
        risk_types = {
            "FinancialRisk": ["CreditRisk", "LiquidityRisk", "MarketRisk"],
            "OperationalRisk": ["BusinessRisk", "CompetitiveRisk", "TechnologicalRisk"],
            "RegulatoryRisk": ["ComplianceRisk", "PolicyRisk", "LegalRisk"]
        }
        
        for main_type, sub_types in risk_types.items():
            main_risk = self.atomspace.create_concept(main_type)
            main_inheritance = InheritanceLink(main_risk, risk)
            self.atomspace.add(main_inheritance)
            
            for sub_type in sub_types:
                sub_risk = self.atomspace.create_concept(sub_type)
                sub_inheritance = InheritanceLink(sub_risk, main_risk)
                self.atomspace.add(sub_inheritance)
    
    def _create_market_relationships(self):
        """Create market and industry relationships"""
        # Market sectors
        market = self.atomspace.create_concept("Market")
        
        sectors = ["Technology", "Healthcare", "Finance", "Energy", "ConsumerGoods", "Telecommunications"]
        for sector_name in sectors:
            sector = self.atomspace.create_concept(sector_name)
            belongs_to = self.atomspace.create_predicate("BelongsTo")
            
            evaluation = EvaluationLink(belongs_to, [sector, market])
            self.atomspace.add(evaluation)
    
    def _initialize_reasoning_rules(self):
        """Initialize reasoning rules for financial analysis"""
        # This would contain OpenCog-style reasoning rules
        # Simplified for this implementation
        pass
    
    def add_company_knowledge(self, company: str, knowledge_data: Dict[str, Any]):
        """Add company-specific knowledge to the knowledge base"""
        company_node = self.atomspace.create_concept(company)
        
        # Add company attributes
        for attribute, value in knowledge_data.items():
            if isinstance(value, (str, int, float)):
                predicate = self.atomspace.create_predicate(f"Has{attribute.title()}")
                value_node = self.atomspace.create_concept(str(value))
                
                evaluation = EvaluationLink(predicate, [company_node, value_node])
                self.atomspace.add(evaluation)
    
    def add_financial_data(self, company: str, metric: str, value: float, 
                          period: str, confidence: float = 0.9):
        """Add financial metric data with truth values"""
        company_node = self.atomspace.create_concept(company)
        metric_node = self.atomspace.create_concept(metric)
        value_node = self.atomspace.create_concept(str(value))
        period_node = self.atomspace.create_concept(period)
        
        # Create evaluation with truth value
        has_metric = self.atomspace.create_predicate("HasMetric")
        truth_value = TruthValue(min(value / 100.0, 1.0), confidence)
        
        evaluation = EvaluationLink(
            has_metric, 
            [company_node, metric_node, value_node, period_node],
            truth_value
        )
        self.atomspace.add(evaluation)
    
    def add_sentiment_data(self, company: str, sentiment: str, score: float, 
                          source: str, confidence: float = 0.8):
        """Add sentiment analysis results"""
        company_node = self.atomspace.create_concept(company)
        sentiment_node = self.atomspace.create_concept(sentiment.title() + "Sentiment")
        source_node = self.atomspace.create_concept(source)
        
        has_sentiment = self.atomspace.create_predicate("HasSentiment")
        truth_value = TruthValue(abs(score), confidence)
        
        evaluation = EvaluationLink(
            has_sentiment,
            [company_node, sentiment_node, source_node],
            truth_value
        )
        self.atomspace.add(evaluation)
    
    def add_news_impact(self, company: str, news_type: str, impact_score: float,
                       date: str, confidence: float = 0.7):
        """Add news impact analysis"""
        company_node = self.atomspace.create_concept(company)
        news_type_node = self.atomspace.create_concept(news_type)
        date_node = self.atomspace.create_concept(date)
        
        has_news_impact = self.atomspace.create_predicate("HasNewsImpact")
        truth_value = TruthValue(impact_score, confidence)
        
        evaluation = EvaluationLink(
            has_news_impact,
            [company_node, news_type_node, date_node],
            truth_value
        )
        self.atomspace.add(evaluation)
    
    def add_price_correlation(self, company: str, event_type: str, 
                            correlation_strength: float, confidence: float = 0.6):
        """Add price correlation data"""
        company_node = self.atomspace.create_concept(company)
        event_type_node = self.atomspace.create_concept(event_type)
        
        has_correlation = self.atomspace.create_predicate("HasPriceCorrelation")
        truth_value = TruthValue(abs(correlation_strength), confidence)
        
        evaluation = EvaluationLink(
            has_correlation,
            [company_node, event_type_node],
            truth_value
        )
        self.atomspace.add(evaluation)
    
    def query_company_metrics(self, company: str, metric_type: str = None) -> List[Dict[str, Any]]:
        """Query financial metrics for a company"""
        results = []
        
        # Find all metric evaluations for the company
        has_metric = self.atomspace.create_predicate("HasMetric")
        company_node = self.atomspace.create_concept(company)
        
        # Get all evaluation links with this predicate
        eval_links = self.pattern_matcher.query_by_predicate("HasMetric")
        
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                args = link.get_arguments().get_outgoing()
                if len(args) >= 4 and args[0].name == company:
                    metric_name = args[1].name
                    metric_value = args[2].name
                    period = args[3].name
                    
                    if not metric_type or metric_type.lower() in metric_name.lower():
                        results.append({
                            'company': company,
                            'metric': metric_name,
                            'value': metric_value,
                            'period': period,
                            'truth_value': {
                                'strength': link.truth_value.strength,
                                'confidence': link.truth_value.confidence
                            }
                        })
        
        return results
    
    def query_sentiment_analysis(self, company: str) -> List[Dict[str, Any]]:
        """Query sentiment analysis results for a company"""
        results = []
        
        eval_links = self.pattern_matcher.query_by_predicate("HasSentiment")
        
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                args = link.get_arguments().get_outgoing()
                if len(args) >= 3 and args[0].name == company:
                    sentiment_type = args[1].name
                    source = args[2].name
                    
                    results.append({
                        'company': company,
                        'sentiment': sentiment_type,
                        'source': source,
                        'truth_value': {
                            'strength': link.truth_value.strength,
                            'confidence': link.truth_value.confidence
                        }
                    })
        
        return results
    
    def query_news_impact(self, company: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Query recent news impact for a company"""
        results = []
        
        eval_links = self.pattern_matcher.query_by_predicate("HasNewsImpact")
        
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                args = link.get_arguments().get_outgoing()
                if len(args) >= 3 and args[0].name == company:
                    news_type = args[1].name
                    date = args[2].name
                    
                    results.append({
                        'company': company,
                        'news_type': news_type,
                        'date': date,
                        'impact_score': link.truth_value.strength,
                        'confidence': link.truth_value.confidence
                    })
        
        return results
    
    def query_correlations(self, company: str) -> List[Dict[str, Any]]:
        """Query price correlations for a company"""
        results = []
        
        eval_links = self.pattern_matcher.query_by_predicate("HasPriceCorrelation")
        
        for link in eval_links:
            if isinstance(link, EvaluationLink):
                args = link.get_arguments().get_outgoing()
                if len(args) >= 2 and args[0].name == company:
                    event_type = args[1].name
                    
                    results.append({
                        'company': company,
                        'event_type': event_type,
                        'correlation_strength': link.truth_value.strength,
                        'confidence': link.truth_value.confidence
                    })
        
        return results
    
    def find_similar_companies(self, company: str, similarity_threshold: float = 0.7) -> List[str]:
        """Find companies with similar financial profiles"""
        # This would implement similarity matching based on financial metrics
        # Simplified implementation
        similar_companies = []
        
        company_metrics = self.query_company_metrics(company)
        if not company_metrics:
            return similar_companies
        
        # Get all companies in the knowledge base
        all_companies = set()
        for atom in self.atomspace.get_atoms_by_type(ConceptNode):
            # Check if this might be a company (heuristic)
            if len(atom.name) > 2 and atom.name[0].isupper():
                all_companies.add(atom.name)
        
        # Compare metrics (simplified)
        for other_company in all_companies:
            if other_company != company:
                other_metrics = self.query_company_metrics(other_company)
                similarity = self._calculate_company_similarity(company_metrics, other_metrics)
                
                if similarity >= similarity_threshold:
                    similar_companies.append(other_company)
        
        return similar_companies
    
    def _calculate_company_similarity(self, metrics1: List[Dict[str, Any]], 
                                    metrics2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two companies based on metrics"""
        if not metrics1 or not metrics2:
            return 0.0
        
        # Create metric dictionaries
        dict1 = {m['metric']: float(m['value']) for m in metrics1}
        dict2 = {m['metric']: float(m['value']) for m in metrics2}
        
        # Find common metrics
        common_metrics = set(dict1.keys()) & set(dict2.keys())
        if not common_metrics:
            return 0.0
        
        # Calculate similarity for common metrics
        similarities = []
        for metric in common_metrics:
            val1, val2 = dict1[metric], dict2[metric]
            if val1 != 0 and val2 != 0:
                # Use relative difference
                similarity = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                similarities.append(max(0.0, similarity))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def get_investment_insights(self, company: str) -> Dict[str, Any]:
        """Generate investment insights using knowledge base reasoning"""
        insights = {
            'financial_strength': 'neutral',
            'sentiment_trend': 'neutral',
            'news_impact': 'neutral',
            'price_momentum': 'neutral',
            'overall_assessment': 'neutral',
            'confidence': 0.5,
            'reasoning': []
        }
        
        # Analyze financial metrics
        financial_metrics = self.query_company_metrics(company)
        if financial_metrics:
            insights['financial_strength'] = self._assess_financial_strength(financial_metrics)
            insights['reasoning'].append(f"Financial assessment based on {len(financial_metrics)} metrics")
        
        # Analyze sentiment
        sentiment_data = self.query_sentiment_analysis(company)
        if sentiment_data:
            insights['sentiment_trend'] = self._assess_sentiment_trend(sentiment_data)
            insights['reasoning'].append(f"Sentiment analysis from {len(sentiment_data)} sources")
        
        # Analyze news impact
        news_data = self.query_news_impact(company)
        if news_data:
            insights['news_impact'] = self._assess_news_impact(news_data)
            insights['reasoning'].append(f"News impact from {len(news_data)} events")
        
        # Calculate overall assessment
        assessments = [
            insights['financial_strength'],
            insights['sentiment_trend'], 
            insights['news_impact']
        ]
        
        positive_count = assessments.count('positive')
        negative_count = assessments.count('negative')
        
        if positive_count > negative_count:
            insights['overall_assessment'] = 'positive'
        elif negative_count > positive_count:
            insights['overall_assessment'] = 'negative'
        
        # Calculate confidence based on data availability
        data_points = len(financial_metrics) + len(sentiment_data) + len(news_data)
        insights['confidence'] = min(data_points / 20.0, 1.0)  # Max confidence at 20 data points
        
        return insights
    
    def _assess_financial_strength(self, metrics: List[Dict[str, Any]]) -> str:
        """Assess financial strength from metrics"""
        if not metrics:
            return 'neutral'
        
        # Count positive vs negative indicators (simplified)
        positive_indicators = 0
        negative_indicators = 0
        
        for metric in metrics:
            tv = metric.get('truth_value', {})
            strength = tv.get('strength', 0.5)
            
            if strength > 0.6:
                positive_indicators += 1
            elif strength < 0.4:
                negative_indicators += 1
        
        if positive_indicators > negative_indicators:
            return 'positive'
        elif negative_indicators > positive_indicators:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_sentiment_trend(self, sentiment_data: List[Dict[str, Any]]) -> str:
        """Assess sentiment trend"""
        if not sentiment_data:
            return 'neutral'
        
        positive_sentiment = 0
        negative_sentiment = 0
        
        for sentiment in sentiment_data:
            if 'positive' in sentiment['sentiment'].lower():
                positive_sentiment += sentiment['truth_value']['strength']
            elif 'negative' in sentiment['sentiment'].lower():
                negative_sentiment += sentiment['truth_value']['strength']
        
        if positive_sentiment > negative_sentiment:
            return 'positive'
        elif negative_sentiment > positive_sentiment:
            return 'negative'
        else:
            return 'neutral'
    
    def _assess_news_impact(self, news_data: List[Dict[str, Any]]) -> str:
        """Assess overall news impact"""
        if not news_data:
            return 'neutral'
        
        avg_impact = sum(item['impact_score'] for item in news_data) / len(news_data)
        
        if avg_impact > 0.6:
            return 'positive'
        elif avg_impact < 0.4:
            return 'negative'
        else:
            return 'neutral'
    
    def export_knowledge(self, filepath: str):
        """Export knowledge base to file"""
        try:
            self.atomspace.save_to_file(filepath, format='json')
            self.logger.info(f"Knowledge base exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export knowledge base: {e}")
    
    def import_knowledge(self, filepath: str):
        """Import knowledge base from file"""
        try:
            self.atomspace.load_from_file(filepath, format='json')
            self.logger.info(f"Knowledge base imported from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to import knowledge base: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            'atomspace_stats': self.atomspace.get_statistics(),
            'total_evaluations': len(self.atomspace.get_atoms_by_type(EvaluationLink)),
            'total_concepts': len(self.atomspace.get_atoms_by_type(ConceptNode)),
            'total_predicates': len(self.atomspace.get_atoms_by_type(PredicateNode))
        }