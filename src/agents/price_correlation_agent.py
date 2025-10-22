"""
Stock Price Correlation Agent for linking textual events to market movements
"""

from typing import Dict, List, Any, Optional, Tuple
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from scipy import stats
import yfinance as yf

from .base_agent import BaseAgent, SharedMemory, AgentCommunication
from ..opencog_framework.truth_values import TruthValue


class PriceCorrelationAgent(BaseAgent):
    """Agent specialized in correlating textual events with stock price movements"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        super().__init__(agent_id, shared_memory, communication, config)
        
        self.correlation_window = config.get('correlation_window_days', 5)
        self.min_correlation_threshold = config.get('min_correlation_threshold', 0.3)
        self.price_data_cache = {}
        self.event_impact_models = self._initialize_impact_models()
    
    def _initialize_impact_models(self) -> Dict[str, Dict[str, float]]:
        """Initialize models for different event types and their typical impacts"""
        return {
            'earnings_announcement': {
                'positive_keywords': ['beat', 'exceed', 'outperform', 'growth', 'strong'],
                'negative_keywords': ['miss', 'decline', 'weak', 'challenge', 'loss'],
                'typical_impact': 0.05  # 5% typical price movement
            },
            'product_launch': {
                'positive_keywords': ['launch', 'introduce', 'innovation', 'breakthrough'],
                'negative_keywords': ['delay', 'cancel', 'problem', 'issue'],
                'typical_impact': 0.03
            },
            'regulatory_news': {
                'positive_keywords': ['approval', 'cleared', 'authorized', 'agreement'],
                'negative_keywords': ['reject', 'deny', 'investigation', 'violation'],
                'typical_impact': 0.04
            },
            'management_change': {
                'positive_keywords': ['appoint', 'hire', 'promote', 'join'],
                'negative_keywords': ['resign', 'departure', 'fired', 'leave'],
                'typical_impact': 0.02
            },
            'financial_guidance': {
                'positive_keywords': ['raise', 'increase', 'upgrade', 'optimistic'],
                'negative_keywords': ['lower', 'reduce', 'downgrade', 'cautious'],
                'typical_impact': 0.06
            }
        }
    
    def get_required_data(self) -> List[str]:
        """Return required data types for price correlation analysis"""
        return ['stock_ticker', 'earnings_transcripts', 'press_releases', 'news_articles', 'stock_prices']
    
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform price correlation analysis"""
        self.logger.info(f"Starting price correlation analysis for {company}")
        
        results = {
            'correlations': {},
            'event_impacts': {},
            'causal_hypotheses': [],
            'anomaly_detection': {},
            'predictive_signals': {},
            'confidence_scores': {}
        }
        
        ticker = data.get('stock_ticker')
        if not ticker:
            self.logger.warning(f"No stock ticker provided for {company}")
            return results
        
        # Get or fetch stock price data
        price_data = await self._get_stock_price_data(ticker)
        if price_data is None or price_data.empty:
            self.logger.warning(f"No stock price data available for {ticker}")
            return results
        
        # Analyze different event types
        event_types = ['earnings_transcripts', 'press_releases', 'news_articles']
        
        for event_type in event_types:
            if event_type in data and data[event_type]:
                self.logger.info(f"Analyzing {event_type} correlations")
                
                correlations = await self._analyze_event_price_correlations(
                    data[event_type], price_data, event_type, company
                )
                results['correlations'][event_type] = correlations
        
        # Generate causal hypotheses
        results['causal_hypotheses'] = self._generate_causal_hypotheses(
            results['correlations'], data
        )
        
        # Detect price anomalies
        results['anomaly_detection'] = self._detect_price_anomalies(
            price_data, data
        )
        
        # Extract predictive signals
        results['predictive_signals'] = self._extract_predictive_signals(
            results['correlations'], price_data
        )
        
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_correlation_confidence(
            results['correlations']
        )
        
        self.logger.info(f"Price correlation analysis completed for {company}")
        return results
    
    async def _get_stock_price_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get stock price data for the ticker"""
        if ticker in self.price_data_cache:
            return self.price_data_cache[ticker]
        
        try:
            # Fetch 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            stock = yf.Ticker(ticker)
            price_data = stock.history(start=start_date, end=end_date)
            
            if not price_data.empty:
                # Calculate additional metrics
                price_data['Returns'] = price_data['Close'].pct_change()
                price_data['Volatility'] = price_data['Returns'].rolling(window=20).std()
                price_data['Price_Change'] = price_data['Close'].diff()
                price_data['Price_Change_Pct'] = price_data['Close'].pct_change() * 100
                
                self.price_data_cache[ticker] = price_data
                self.logger.info(f"Fetched price data for {ticker}: {len(price_data)} records")
                return price_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch price data for {ticker}: {e}")
        
        return None
    
    async def _analyze_event_price_correlations(self, events: List[Dict[str, Any]], 
                                              price_data: pd.DataFrame, 
                                              event_type: str, company: str) -> Dict[str, Any]:
        """Analyze correlations between events and price movements"""
        correlations = {
            'event_impacts': [],
            'average_impact': 0.0,
            'correlation_coefficient': 0.0,
            'significant_events': [],
            'temporal_patterns': {}
        }
        
        event_impacts = []
        
        for event in events:
            impact_analysis = await self._analyze_single_event_impact(
                event, price_data, event_type
            )
            if impact_analysis:
                event_impacts.append(impact_analysis)
                correlations['event_impacts'].append(impact_analysis)
        
        if event_impacts:
            # Calculate average impact
            impacts = [e['price_impact'] for e in event_impacts if 'price_impact' in e]
            if impacts:
                correlations['average_impact'] = np.mean(impacts)
                correlations['impact_std'] = np.std(impacts)
                
                # Calculate correlation coefficient
                sentiment_scores = [e.get('sentiment_score', 0) for e in event_impacts]
                if len(impacts) == len(sentiment_scores) and len(impacts) > 2:
                    corr_coef, p_value = stats.pearsonr(sentiment_scores, impacts)
                    correlations['correlation_coefficient'] = float(corr_coef)
                    correlations['p_value'] = float(p_value)
                    correlations['is_significant'] = p_value < 0.05
            
            # Identify significant events
            correlations['significant_events'] = [
                e for e in event_impacts 
                if abs(e.get('price_impact', 0)) > self.min_correlation_threshold
            ]
            
            # Analyze temporal patterns
            correlations['temporal_patterns'] = self._analyze_temporal_impact_patterns(
                event_impacts
            )
        
        return correlations
    
    async def _analyze_single_event_impact(self, event: Dict[str, Any], 
                                         price_data: pd.DataFrame, 
                                         event_type: str) -> Optional[Dict[str, Any]]:
        """Analyze the impact of a single event on stock price"""
        event_date = self._extract_event_date(event)
        if not event_date:
            return None
        
        # Find price data around the event date
        event_window = self._get_price_window(price_data, event_date, self.correlation_window)
        if event_window.empty:
            return None
        
        # Calculate price impact
        pre_event_price = self._get_pre_event_price(event_window, event_date)
        post_event_price = self._get_post_event_price(event_window, event_date)
        
        if pre_event_price is None or post_event_price is None:
            return None
        
        price_impact = (post_event_price - pre_event_price) / pre_event_price
        
        # Analyze event content for sentiment
        event_text = event.get('text', '') or event.get('content', '')
        sentiment_analysis = self._analyze_event_sentiment(event_text, event_type)
        
        # Calculate volume impact if available
        volume_impact = self._calculate_volume_impact(event_window, event_date)
        
        return {
            'event_date': event_date.isoformat(),
            'event_type': event_type,
            'price_impact': float(price_impact),
            'volume_impact': volume_impact,
            'pre_event_price': float(pre_event_price),
            'post_event_price': float(post_event_price),
            'sentiment_score': sentiment_analysis['score'],
            'event_classification': sentiment_analysis['classification'],
            'confidence': sentiment_analysis['confidence'],
            'event_summary': event_text[:200] if event_text else ''
        }
    
    def _extract_event_date(self, event: Dict[str, Any]) -> Optional[datetime]:
        """Extract date from event data"""
        date_fields = ['date', 'timestamp', 'published_date', 'release_date']
        
        for field in date_fields:
            if field in event and event[field]:
                try:
                    if isinstance(event[field], str):
                        # Try different date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']:
                            try:
                                return datetime.strptime(event[field], fmt)
                            except ValueError:
                                continue
                        
                        # Try pandas date parser as fallback
                        return pd.to_datetime(event[field])
                    elif isinstance(event[field], datetime):
                        return event[field]
                except Exception:
                    continue
        
        return None
    
    def _get_price_window(self, price_data: pd.DataFrame, event_date: datetime, 
                         window_days: int) -> pd.DataFrame:
        """Get price data window around event date"""
        start_date = event_date - timedelta(days=window_days)
        end_date = event_date + timedelta(days=window_days)
        
        mask = (price_data.index >= start_date) & (price_data.index <= end_date)
        return price_data[mask]
    
    def _get_pre_event_price(self, window: pd.DataFrame, event_date: datetime) -> Optional[float]:
        """Get price before event (closest trading day)"""
        pre_event = window[window.index < event_date]
        if not pre_event.empty:
            return float(pre_event['Close'].iloc[-1])
        return None
    
    def _get_post_event_price(self, window: pd.DataFrame, event_date: datetime) -> Optional[float]:
        """Get price after event (closest trading day)"""
        post_event = window[window.index >= event_date]
        if not post_event.empty:
            return float(post_event['Close'].iloc[0])
        return None
    
    def _analyze_event_sentiment(self, text: str, event_type: str) -> Dict[str, Any]:
        """Analyze sentiment of event text using keyword matching"""
        if not text:
            return {'score': 0.0, 'classification': 'neutral', 'confidence': 0.0}
        
        text_lower = text.lower()
        model = self.event_impact_models.get(event_type, {})
        
        positive_keywords = model.get('positive_keywords', [])
        negative_keywords = model.get('negative_keywords', [])
        
        positive_score = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_score = sum(1 for kw in negative_keywords if kw in text_lower)
        
        total_keywords = positive_score + negative_score
        
        if total_keywords == 0:
            return {'score': 0.0, 'classification': 'neutral', 'confidence': 0.0}
        
        sentiment_score = (positive_score - negative_score) / total_keywords
        
        if sentiment_score > 0.2:
            classification = 'positive'
        elif sentiment_score < -0.2:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        confidence = min(total_keywords / 10.0, 1.0)  # Max confidence with 10+ keywords
        
        return {
            'score': float(sentiment_score),
            'classification': classification,
            'confidence': float(confidence)
        }
    
    def _calculate_volume_impact(self, window: pd.DataFrame, event_date: datetime) -> float:
        """Calculate volume impact around event"""
        try:
            pre_volume = window[window.index < event_date]['Volume'].mean()
            post_volume = window[window.index >= event_date]['Volume'].mean()
            
            if pre_volume > 0:
                return float((post_volume - pre_volume) / pre_volume)
        except Exception:
            pass
        
        return 0.0
    
    def _generate_causal_hypotheses(self, correlations: Dict[str, Dict[str, Any]], 
                                  data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate causal hypotheses based on correlations"""
        hypotheses = []
        
        for event_type, correlation_data in correlations.items():
            if correlation_data.get('is_significant', False):
                corr_coef = correlation_data.get('correlation_coefficient', 0)
                avg_impact = correlation_data.get('average_impact', 0)
                
                hypothesis = {
                    'event_type': event_type,
                    'hypothesis': f"{event_type} shows significant correlation with stock price",
                    'correlation_strength': abs(corr_coef),
                    'average_impact': avg_impact,
                    'direction': 'positive' if corr_coef > 0 else 'negative',
                    'confidence': min(abs(corr_coef) * 2, 1.0),
                    'supporting_events': len(correlation_data.get('significant_events', []))
                }
                
                hypotheses.append(hypothesis)
        
        # Sort by confidence
        hypotheses.sort(key=lambda x: x['confidence'], reverse=True)
        return hypotheses
    
    def _detect_price_anomalies(self, price_data: pd.DataFrame, 
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unusual price movements that might correlate with events"""
        anomalies = {
            'significant_moves': [],
            'volatility_spikes': [],
            'volume_anomalies': []
        }
        
        if price_data.empty:
            return anomalies
        
        # Detect significant price movements (> 2 standard deviations)
        returns = price_data['Returns'].dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        threshold = 2 * std_return
        significant_moves = price_data[abs(price_data['Returns'] - mean_return) > threshold]
        
        for date, row in significant_moves.iterrows():
            anomalies['significant_moves'].append({
                'date': date.isoformat(),
                'price_change_pct': float(row['Returns'] * 100),
                'magnitude': float(abs(row['Returns'] - mean_return) / std_return)
            })
        
        # Detect volatility spikes
        volatility = price_data['Volatility'].dropna()
        vol_threshold = volatility.quantile(0.95)
        vol_spikes = price_data[price_data['Volatility'] > vol_threshold]
        
        for date, row in vol_spikes.iterrows():
            anomalies['volatility_spikes'].append({
                'date': date.isoformat(),
                'volatility': float(row['Volatility']),
                'percentile': float(volatility[volatility <= row['Volatility']].count() / len(volatility))
            })
        
        return anomalies
    
    def _extract_predictive_signals(self, correlations: Dict[str, Dict[str, Any]], 
                                   price_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract signals that might predict future price movements"""
        signals = {
            'leading_indicators': [],
            'pattern_recognition': {},
            'momentum_signals': {}
        }
        
        # Identify event types that consistently lead price movements
        for event_type, correlation_data in correlations.items():
            if correlation_data.get('correlation_coefficient', 0) > 0.5:
                signals['leading_indicators'].append({
                    'event_type': event_type,
                    'predictive_power': correlation_data.get('correlation_coefficient'),
                    'average_lead_time': self.correlation_window,  # Simplified
                    'reliability': correlation_data.get('is_significant', False)
                })
        
        return signals
    
    def _analyze_temporal_impact_patterns(self, event_impacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in event impacts"""
        patterns = {
            'time_to_impact': {},
            'decay_pattern': {},
            'seasonal_effects': {}
        }
        
        if len(event_impacts) < 5:
            return patterns
        
        # Analyze impact timing patterns
        impacts_by_hour = {}
        for impact in event_impacts:
            try:
                event_date = datetime.fromisoformat(impact['event_date'].replace('Z', '+00:00'))
                hour = event_date.hour
                if hour not in impacts_by_hour:
                    impacts_by_hour[hour] = []
                impacts_by_hour[hour].append(impact['price_impact'])
            except Exception:
                continue
        
        # Calculate average impact by hour
        for hour, impacts in impacts_by_hour.items():
            patterns['time_to_impact'][hour] = {
                'average_impact': float(np.mean(impacts)),
                'count': len(impacts)
            }
        
        return patterns
    
    def _calculate_correlation_confidence(self, correlations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence scores for correlations"""
        confidence_scores = {}
        
        for event_type, correlation_data in correlations.items():
            confidence = 0.0
            
            # Base confidence on correlation strength
            corr_coef = abs(correlation_data.get('correlation_coefficient', 0))
            confidence += corr_coef * 0.4
            
            # Add confidence based on number of events
            event_count = len(correlation_data.get('event_impacts', []))
            confidence += min(event_count / 20.0, 0.3)  # Max 0.3 for sample size
            
            # Add confidence based on statistical significance
            if correlation_data.get('is_significant', False):
                confidence += 0.3
            
            confidence_scores[event_type] = min(confidence, 1.0)
        
        # Overall confidence
        if confidence_scores:
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores