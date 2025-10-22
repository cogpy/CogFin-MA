"""
News Analysis Agent for processing and analyzing financial news
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import requests
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import numpy as np
from urllib.parse import urlparse

from .base_agent import BaseAgent, SharedMemory, AgentCommunication
from ..opencog_framework.truth_values import TruthValue


class NewsAgent(BaseAgent):
    """Agent specialized in analyzing financial news and market events"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        super().__init__(agent_id, shared_memory, communication, config)
        
        self.relevance_threshold = config.get('relevance_threshold', 0.6)
        self.impact_weights = config.get('impact_weights', {
            'earnings': 0.8,
            'merger': 0.9,
            'regulatory': 0.7,
            'product': 0.6,
            'management': 0.5,
            'partnership': 0.6,
            'lawsuit': 0.7,
            'analyst': 0.4
        })
        
        self.credibility_sources = self._initialize_source_credibility()
        self.news_categories = self._initialize_news_categories()
    
    def _initialize_source_credibility(self) -> Dict[str, float]:
        """Initialize credibility scores for different news sources"""
        return {
            'reuters.com': 0.95,
            'bloomberg.com': 0.95,
            'wsj.com': 0.9,
            'ft.com': 0.9,
            'cnbc.com': 0.8,
            'marketwatch.com': 0.8,
            'yahoo.com': 0.7,
            'fool.com': 0.6,
            'seekingalpha.com': 0.7,
            'benzinga.com': 0.6,
            'default': 0.5
        }
    
    def _initialize_news_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize news categorization patterns"""
        return {
            'earnings': {
                'keywords': ['earnings', 'quarterly results', 'revenue', 'eps', 'guidance', 'forecast'],
                'impact_level': 'high',
                'typical_duration': 3  # days
            },
            'merger_acquisition': {
                'keywords': ['merger', 'acquisition', 'buyout', 'takeover', 'acquire', 'merge'],
                'impact_level': 'very_high',
                'typical_duration': 30
            },
            'regulatory': {
                'keywords': ['fda', 'sec', 'regulation', 'approval', 'compliance', 'investigation'],
                'impact_level': 'high',
                'typical_duration': 7
            },
            'product_launch': {
                'keywords': ['launch', 'product', 'release', 'unveil', 'announce', 'introduce'],
                'impact_level': 'medium',
                'typical_duration': 5
            },
            'management_change': {
                'keywords': ['ceo', 'cfo', 'resign', 'appoint', 'hire', 'executive', 'leadership'],
                'impact_level': 'medium',
                'typical_duration': 7
            },
            'partnership': {
                'keywords': ['partnership', 'collaboration', 'joint venture', 'alliance', 'agreement'],
                'impact_level': 'medium',
                'typical_duration': 5
            },
            'legal': {
                'keywords': ['lawsuit', 'legal', 'court', 'settlement', 'litigation', 'patent'],
                'impact_level': 'medium',
                'typical_duration': 10
            },
            'analyst': {
                'keywords': ['analyst', 'upgrade', 'downgrade', 'rating', 'target price', 'recommendation'],
                'impact_level': 'low',
                'typical_duration': 2
            }
        }
    
    def get_required_data(self) -> List[str]:
        """Return required data types for news analysis"""
        return ['news_articles', 'company_name', 'stock_ticker', 'industry']
    
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive news analysis"""
        self.logger.info(f"Starting news analysis for {company}")
        
        results = {
            'news_summary': {},
            'impact_assessment': {},
            'credibility_analysis': {},
            'temporal_analysis': {},
            'sentiment_trends': {},
            'key_events': [],
            'risk_indicators': {},
            'opportunity_indicators': {},
            'confidence_scores': {}
        }
        
        news_articles = data.get('news_articles', [])
        if not news_articles:
            self.logger.warning(f"No news articles provided for {company}")
            return results
        
        # Process and categorize news
        processed_news = await self._process_news_articles(news_articles, company)
        results['news_summary'] = self._create_news_summary(processed_news)
        
        # Assess impact of different news types
        results['impact_assessment'] = self._assess_news_impact(processed_news, data)
        
        # Analyze source credibility
        results['credibility_analysis'] = self._analyze_source_credibility(processed_news)
        
        # Temporal analysis
        results['temporal_analysis'] = self._analyze_temporal_patterns(processed_news)
        
        # Identify key events
        results['key_events'] = self._identify_key_events(processed_news)
        
        # Risk and opportunity indicators
        results['risk_indicators'] = self._identify_risk_indicators(processed_news)
        results['opportunity_indicators'] = self._identify_opportunity_indicators(processed_news)
        
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_confidence_scores(processed_news)
        
        self.logger.info(f"News analysis completed for {company}")
        return results
    
    async def _process_news_articles(self, articles: List[Dict[str, Any]], 
                                   company: str) -> List[Dict[str, Any]]:
        """Process and enrich news articles with analysis"""
        processed_articles = []
        
        for article in articles:
            try:
                processed = await self._process_single_article(article, company)
                if processed:
                    processed_articles.append(processed)
            except Exception as e:
                self.logger.error(f"Error processing article: {e}")
                continue
        
        self.logger.info(f"Processed {len(processed_articles)} articles")
        return processed_articles
    
    async def _process_single_article(self, article: Dict[str, Any], 
                                    company: str) -> Optional[Dict[str, Any]]:
        """Process a single news article"""
        title = article.get('title', '')
        content = article.get('content', '') or article.get('text', '')
        url = article.get('url', '')
        published_date = article.get('published_date') or article.get('date')
        source = article.get('source', '')
        
        if not title and not content:
            return None
        
        # Extract source domain from URL if not provided
        if not source and url:
            try:
                source = urlparse(url).netloc.lower()
            except Exception:
                source = 'unknown'
        
        # Parse published date
        parsed_date = self._parse_date(published_date)
        
        # Categorize the news
        category = self._categorize_news(title + ' ' + content)
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(title + ' ' + content, company)
        
        # Assess sentiment
        sentiment_analysis = self._assess_news_sentiment(title + ' ' + content)
        
        # Get source credibility
        credibility = self._get_source_credibility(source)
        
        # Calculate impact score
        impact_score = self._calculate_impact_score(category, sentiment_analysis, credibility)
        
        return {
            'title': title,
            'content': content[:1000],  # Truncate for storage
            'url': url,
            'source': source,
            'published_date': parsed_date.isoformat() if parsed_date else None,
            'category': category,
            'relevance_score': relevance_score,
            'sentiment': sentiment_analysis,
            'credibility': credibility,
            'impact_score': impact_score,
            'key_phrases': self._extract_key_phrases(title + ' ' + content)
        }
    
    def _parse_date(self, date_str: Any) -> Optional[datetime]:
        """Parse various date formats"""
        if not date_str:
            return None
        
        if isinstance(date_str, datetime):
            return date_str
        
        if isinstance(date_str, str):
            try:
                # Try different formats
                formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S'
                ]
                
                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                
                # Try pandas parsing as fallback
                import pandas as pd
                return pd.to_datetime(date_str)
            except Exception:
                pass
        
        return None
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news article based on content"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, config in self.news_categories.items():
            score = 0
            for keyword in config['keywords']:
                score += text_lower.count(keyword.lower())
            category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) == 0:
            return 'general'
        
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_relevance_score(self, text: str, company: str) -> float:
        """Calculate how relevant the news is to the company"""
        text_lower = text.lower()
        company_lower = company.lower()
        
        relevance_score = 0.0
        
        # Direct company name mentions
        company_mentions = text_lower.count(company_lower)
        relevance_score += min(company_mentions * 0.3, 1.0)
        
        # Industry-specific keywords (simplified)
        industry_keywords = [
            'technology', 'pharmaceutical', 'automotive', 'finance', 'healthcare',
            'energy', 'retail', 'telecommunications', 'aerospace', 'biotech'
        ]
        
        for keyword in industry_keywords:
            if keyword in text_lower:
                relevance_score += 0.1
        
        # Financial relevance keywords
        financial_keywords = [
            'revenue', 'profit', 'earnings', 'stock', 'market', 'shares',
            'investor', 'financial', 'quarterly', 'annual'
        ]
        
        for keyword in financial_keywords:
            if keyword in text_lower:
                relevance_score += 0.05
        
        return min(relevance_score, 1.0)
    
    def _assess_news_sentiment(self, text: str) -> Dict[str, Any]:
        """Assess sentiment of news article using keyword analysis"""
        text_lower = text.lower()
        
        positive_keywords = [
            'profit', 'growth', 'increase', 'beat', 'exceed', 'strong', 'positive',
            'success', 'gain', 'rise', 'up', 'breakthrough', 'innovation', 'award',
            'partnership', 'expansion', 'approval', 'launch', 'win'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'decrease', 'miss', 'weak', 'negative', 'fall',
            'drop', 'concern', 'risk', 'challenge', 'problem', 'investigation',
            'lawsuit', 'delay', 'cancel', 'layoff', 'bankruptcy'
        ]
        
        positive_score = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_score = sum(1 for kw in negative_keywords if kw in text_lower)
        
        total_score = positive_score + negative_score
        
        if total_score == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        sentiment_score = (positive_score - negative_score) / total_score
        
        if sentiment_score > 0.2:
            sentiment = 'positive'
        elif sentiment_score < -0.2:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence = min(total_score / 10.0, 1.0)
        
        return {
            'sentiment': sentiment,
            'score': float(sentiment_score),
            'confidence': float(confidence),
            'positive_signals': positive_score,
            'negative_signals': negative_score
        }
    
    def _get_source_credibility(self, source: str) -> float:
        """Get credibility score for news source"""
        source_lower = source.lower()
        
        for known_source, credibility in self.credibility_sources.items():
            if known_source in source_lower:
                return credibility
        
        return self.credibility_sources['default']
    
    def _calculate_impact_score(self, category: str, sentiment: Dict[str, Any], 
                               credibility: float) -> float:
        """Calculate overall impact score for news article"""
        base_impact = self.impact_weights.get(category, 0.5)
        sentiment_multiplier = 1.0 + abs(sentiment.get('score', 0)) * 0.5
        credibility_multiplier = credibility
        confidence_multiplier = sentiment.get('confidence', 0.5)
        
        impact_score = (base_impact * sentiment_multiplier * 
                       credibility_multiplier * confidence_multiplier)
        
        return min(impact_score, 1.0)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from news text"""
        # Simplified key phrase extraction
        text_lower = text.lower()
        
        financial_phrases = [
            'quarterly earnings', 'revenue growth', 'market share', 'profit margin',
            'cash flow', 'debt ratio', 'stock price', 'market cap', 'ipo',
            'merger and acquisition', 'product launch', 'regulatory approval',
            'clinical trial', 'patent filing', 'partnership agreement'
        ]
        
        found_phrases = []
        for phrase in financial_phrases:
            if phrase in text_lower:
                found_phrases.append(phrase)
        
        return found_phrases[:10]  # Limit to top 10
    
    def _create_news_summary(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of processed news"""
        if not processed_news:
            return {}
        
        summary = {
            'total_articles': len(processed_news),
            'by_category': defaultdict(int),
            'by_sentiment': defaultdict(int),
            'average_relevance': 0.0,
            'average_credibility': 0.0,
            'date_range': {},
            'top_sources': []
        }
        
        relevance_scores = []
        credibility_scores = []
        sources = []
        dates = []
        
        for article in processed_news:
            summary['by_category'][article['category']] += 1
            summary['by_sentiment'][article['sentiment']['sentiment']] += 1
            
            relevance_scores.append(article['relevance_score'])
            credibility_scores.append(article['credibility'])
            sources.append(article['source'])
            
            if article['published_date']:
                dates.append(article['published_date'])
        
        if relevance_scores:
            summary['average_relevance'] = np.mean(relevance_scores)
        
        if credibility_scores:
            summary['average_credibility'] = np.mean(credibility_scores)
        
        if dates:
            summary['date_range'] = {
                'earliest': min(dates),
                'latest': max(dates)
            }
        
        # Top sources by article count
        source_counts = Counter(sources)
        summary['top_sources'] = [
            {'source': source, 'count': count}
            for source, count in source_counts.most_common(5)
        ]
        
        return dict(summary)
    
    def _assess_news_impact(self, processed_news: List[Dict[str, Any]], 
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the potential impact of news on the company"""
        impact_assessment = {
            'overall_impact': 'neutral',
            'impact_score': 0.0,
            'high_impact_news': [],
            'category_impacts': {},
            'temporal_impact': {},
            'risk_level': 'medium'
        }
        
        if not processed_news:
            return impact_assessment
        
        high_impact_articles = []
        category_impacts = defaultdict(list)
        
        for article in processed_news:
            impact_score = article['impact_score']
            category = article['category']
            
            category_impacts[category].append(impact_score)
            
            if impact_score > 0.7:  # High impact threshold
                high_impact_articles.append({
                    'title': article['title'],
                    'impact_score': impact_score,
                    'category': category,
                    'sentiment': article['sentiment']['sentiment'],
                    'date': article['published_date']
                })
        
        # Calculate overall impact
        all_impacts = [article['impact_score'] for article in processed_news]
        impact_assessment['impact_score'] = np.mean(all_impacts) if all_impacts else 0.0
        
        # Determine overall impact level
        if impact_assessment['impact_score'] > 0.7:
            impact_assessment['overall_impact'] = 'high'
        elif impact_assessment['impact_score'] > 0.4:
            impact_assessment['overall_impact'] = 'medium'
        else:
            impact_assessment['overall_impact'] = 'low'
        
        # Category-wise impacts
        for category, scores in category_impacts.items():
            impact_assessment['category_impacts'][category] = {
                'average_impact': np.mean(scores),
                'max_impact': max(scores),
                'article_count': len(scores)
            }
        
        impact_assessment['high_impact_news'] = sorted(
            high_impact_articles, key=lambda x: x['impact_score'], reverse=True
        )[:10]
        
        return impact_assessment
    
    def _analyze_source_credibility(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze credibility of news sources"""
        credibility_analysis = {
            'average_credibility': 0.0,
            'high_credibility_ratio': 0.0,
            'source_breakdown': {},
            'reliability_score': 0.0
        }
        
        if not processed_news:
            return credibility_analysis
        
        credibility_scores = [article['credibility'] for article in processed_news]
        credibility_analysis['average_credibility'] = np.mean(credibility_scores)
        
        high_credibility_count = sum(1 for score in credibility_scores if score > 0.8)
        credibility_analysis['high_credibility_ratio'] = high_credibility_count / len(credibility_scores)
        
        # Source breakdown
        source_credibility = defaultdict(list)
        for article in processed_news:
            source_credibility[article['source']].append(article['credibility'])
        
        for source, scores in source_credibility.items():
            credibility_analysis['source_breakdown'][source] = {
                'average_credibility': np.mean(scores),
                'article_count': len(scores)
            }
        
        # Overall reliability score
        credibility_analysis['reliability_score'] = (
            credibility_analysis['average_credibility'] * 0.7 +
            credibility_analysis['high_credibility_ratio'] * 0.3
        )
        
        return credibility_analysis
    
    def _analyze_temporal_patterns(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in news coverage"""
        temporal_analysis = {
            'coverage_trend': 'stable',
            'peak_periods': [],
            'sentiment_evolution': {},
            'recency_bias': 0.0
        }
        
        articles_with_dates = [
            article for article in processed_news 
            if article['published_date']
        ]
        
        if len(articles_with_dates) < 5:
            return temporal_analysis
        
        # Sort by date
        articles_with_dates.sort(key=lambda x: x['published_date'])
        
        # Analyze coverage trend (simplified)
        recent_articles = [
            article for article in articles_with_dates
            if (datetime.now() - datetime.fromisoformat(article['published_date'])).days <= 30
        ]
        
        older_articles = [
            article for article in articles_with_dates
            if (datetime.now() - datetime.fromisoformat(article['published_date'])).days > 30
        ]
        
        if older_articles:
            recent_count = len(recent_articles)
            older_count = len(older_articles)
            
            if recent_count > older_count * 1.5:
                temporal_analysis['coverage_trend'] = 'increasing'
            elif recent_count < older_count * 0.5:
                temporal_analysis['coverage_trend'] = 'decreasing'
        
        # Recency bias (more weight to recent news)
        total_weight = 0
        weighted_impact = 0
        
        for article in articles_with_dates:
            days_ago = (datetime.now() - datetime.fromisoformat(article['published_date'])).days
            weight = 1 / (1 + days_ago * 0.1)  # Exponential decay
            
            total_weight += weight
            weighted_impact += article['impact_score'] * weight
        
        if total_weight > 0:
            temporal_analysis['recency_bias'] = weighted_impact / total_weight
        
        return temporal_analysis
    
    def _identify_key_events(self, processed_news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key events from news articles"""
        key_events = []
        
        # Sort by impact score
        high_impact_news = sorted(
            processed_news, 
            key=lambda x: x['impact_score'], 
            reverse=True
        )[:20]  # Top 20 high-impact articles
        
        for article in high_impact_news:
            if article['impact_score'] > 0.6:  # Key event threshold
                key_events.append({
                    'title': article['title'],
                    'category': article['category'],
                    'impact_score': article['impact_score'],
                    'sentiment': article['sentiment']['sentiment'],
                    'date': article['published_date'],
                    'source': article['source'],
                    'credibility': article['credibility'],
                    'key_phrases': article['key_phrases']
                })
        
        return key_events
    
    def _identify_risk_indicators(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify potential risk indicators from news"""
        risk_indicators = {
            'regulatory_risks': [],
            'competitive_risks': [],
            'operational_risks': [],
            'financial_risks': [],
            'overall_risk_level': 'low'
        }
        
        risk_categories = {
            'regulatory': ['investigation', 'lawsuit', 'violation', 'fine', 'penalty'],
            'competitive': ['competitor', 'market share', 'pricing pressure'],
            'operational': ['recall', 'manufacturing', 'supply chain', 'disruption'],
            'financial': ['debt', 'liquidity', 'cash flow', 'bankruptcy', 'loss']
        }
        
        for article in processed_news:
            if article['sentiment']['sentiment'] == 'negative':
                content = (article['title'] + ' ' + article['content']).lower()
                
                for risk_type, keywords in risk_categories.items():
                    for keyword in keywords:
                        if keyword in content:
                            risk_info = {
                                'title': article['title'],
                                'impact_score': article['impact_score'],
                                'date': article['published_date'],
                                'keyword_matched': keyword
                            }
                            
                            if risk_type == 'regulatory':
                                risk_indicators['regulatory_risks'].append(risk_info)
                            elif risk_type == 'competitive':
                                risk_indicators['competitive_risks'].append(risk_info)
                            elif risk_type == 'operational':
                                risk_indicators['operational_risks'].append(risk_info)
                            elif risk_type == 'financial':
                                risk_indicators['financial_risks'].append(risk_info)
        
        # Calculate overall risk level
        total_risks = sum(len(risks) for risks in [
            risk_indicators['regulatory_risks'],
            risk_indicators['competitive_risks'],
            risk_indicators['operational_risks'],
            risk_indicators['financial_risks']
        ])
        
        if total_risks > 10:
            risk_indicators['overall_risk_level'] = 'high'
        elif total_risks > 5:
            risk_indicators['overall_risk_level'] = 'medium'
        
        return risk_indicators
    
    def _identify_opportunity_indicators(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify potential opportunities from news"""
        opportunity_indicators = {
            'growth_opportunities': [],
            'partnership_opportunities': [],
            'market_opportunities': [],
            'innovation_opportunities': [],
            'overall_opportunity_level': 'medium'
        }
        
        opportunity_categories = {
            'growth': ['expansion', 'growth', 'new market', 'international'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'alliance'],
            'market': ['market opportunity', 'demand', 'adoption', 'penetration'],
            'innovation': ['innovation', 'patent', 'technology', 'breakthrough']
        }
        
        for article in processed_news:
            if article['sentiment']['sentiment'] == 'positive':
                content = (article['title'] + ' ' + article['content']).lower()
                
                for opp_type, keywords in opportunity_categories.items():
                    for keyword in keywords:
                        if keyword in content:
                            opp_info = {
                                'title': article['title'],
                                'impact_score': article['impact_score'],
                                'date': article['published_date'],
                                'keyword_matched': keyword
                            }
                            
                            if opp_type == 'growth':
                                opportunity_indicators['growth_opportunities'].append(opp_info)
                            elif opp_type == 'partnership':
                                opportunity_indicators['partnership_opportunities'].append(opp_info)
                            elif opp_type == 'market':
                                opportunity_indicators['market_opportunities'].append(opp_info)
                            elif opp_type == 'innovation':
                                opportunity_indicators['innovation_opportunities'].append(opp_info)
        
        # Calculate overall opportunity level
        total_opportunities = sum(len(opps) for opps in [
            opportunity_indicators['growth_opportunities'],
            opportunity_indicators['partnership_opportunities'],
            opportunity_indicators['market_opportunities'],
            opportunity_indicators['innovation_opportunities']
        ])
        
        if total_opportunities > 15:
            opportunity_indicators['overall_opportunity_level'] = 'high'
        elif total_opportunities < 5:
            opportunity_indicators['overall_opportunity_level'] = 'low'
        
        return opportunity_indicators
    
    def _calculate_confidence_scores(self, processed_news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence scores for the analysis"""
        confidence_scores = {
            'data_quality': 0.0,
            'source_reliability': 0.0,
            'sample_size': 0.0,
            'temporal_coverage': 0.0,
            'overall_confidence': 0.0
        }
        
        if not processed_news:
            return confidence_scores
        
        # Data quality score
        articles_with_content = sum(1 for article in processed_news if len(article['content']) > 100)
        confidence_scores['data_quality'] = articles_with_content / len(processed_news)
        
        # Source reliability score
        avg_credibility = np.mean([article['credibility'] for article in processed_news])
        confidence_scores['source_reliability'] = avg_credibility
        
        # Sample size score
        confidence_scores['sample_size'] = min(len(processed_news) / 50.0, 1.0)  # Max confidence at 50 articles
        
        # Temporal coverage score
        articles_with_dates = [article for article in processed_news if article['published_date']]
        if articles_with_dates and len(articles_with_dates) > 1:
            dates = [datetime.fromisoformat(article['published_date']) for article in articles_with_dates]
            date_range = (max(dates) - min(dates)).days
            confidence_scores['temporal_coverage'] = min(date_range / 365.0, 1.0)  # Max at 1 year coverage
        
        # Overall confidence
        confidence_scores['overall_confidence'] = np.mean([
            confidence_scores['data_quality'],
            confidence_scores['source_reliability'],
            confidence_scores['sample_size'],
            confidence_scores['temporal_coverage']
        ])
        
        return confidence_scores