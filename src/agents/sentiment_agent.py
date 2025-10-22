"""
Sentiment Analysis Agent for financial text analysis
"""

from typing import Dict, List, Any, Optional
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
import numpy as np

from .base_agent import BaseAgent, SharedMemory, AgentCommunication
from ..opencog_framework.truth_values import TruthValue


class SentimentAgent(BaseAgent):
    """Agent specialized in financial sentiment analysis"""
    
    def __init__(self, agent_id: str, shared_memory: SharedMemory, 
                 communication: AgentCommunication, config: Dict[str, Any] = None):
        super().__init__(agent_id, shared_memory, communication, config)
        
        self.model_name = config.get('model_name', 'yiyanghkust/finbert-tone')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 8)
        
        # Initialize model
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model for sentiment analysis"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.logger.info(f"Loaded sentiment model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def get_required_data(self) -> List[str]:
        """Return required data types for sentiment analysis"""
        return ['earnings_transcripts', 'press_releases', 'annual_reports', 'news_articles']
    
    async def analyze(self, company: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sentiment analysis on financial texts"""
        self.logger.info(f"Starting sentiment analysis for {company}")
        
        results = {
            'overall_sentiment': None,
            'sentiment_scores': {},
            'temporal_analysis': {},
            'key_themes': [],
            'confidence_scores': {},
            'detailed_analysis': {}
        }
        
        # Process different types of documents
        document_types = ['earnings_transcripts', 'press_releases', 'annual_reports', 'news_articles']
        
        for doc_type in document_types:
            if doc_type in data and data[doc_type]:
                self.logger.info(f"Analyzing {doc_type} for sentiment")
                
                doc_sentiment = await self._analyze_document_type(
                    data[doc_type], doc_type, company
                )
                results['sentiment_scores'][doc_type] = doc_sentiment
        
        # Calculate overall sentiment
        results['overall_sentiment'] = self._calculate_overall_sentiment(
            results['sentiment_scores']
        )
        
        # Extract key themes
        results['key_themes'] = await self._extract_key_themes(data, company)
        
        # Temporal analysis
        results['temporal_analysis'] = self._analyze_temporal_patterns(
            results['sentiment_scores']
        )
        
        # Confidence scoring
        results['confidence_scores'] = self._calculate_confidence_scores(
            results['sentiment_scores']
        )
        
        self.logger.info(f"Sentiment analysis completed for {company}")
        return results
    
    async def _analyze_document_type(self, documents: List[Dict[str, Any]], 
                                   doc_type: str, company: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific document type"""
        if not documents:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        all_sentiments = []
        all_scores = []
        temporal_data = []
        
        for doc in documents:
            text = doc.get('text', '') or doc.get('content', '')
            if not text:
                continue
            
            # Split into chunks for processing
            chunks = self._split_text_into_chunks(text)
            
            chunk_sentiments = []
            chunk_scores = []
            
            for chunk in chunks:
                sentiment_result = await self._analyze_text_sentiment(chunk)
                chunk_sentiments.append(sentiment_result['sentiment'])
                chunk_scores.append(sentiment_result['score'])
            
            if chunk_sentiments:
                # Aggregate chunk results
                doc_sentiment = self._aggregate_sentiment_scores(
                    chunk_sentiments, chunk_scores
                )
                all_sentiments.append(doc_sentiment['sentiment'])
                all_scores.append(doc_sentiment['score'])
                
                # Add temporal information if available
                if 'date' in doc or 'timestamp' in doc:
                    temporal_data.append({
                        'date': doc.get('date') or doc.get('timestamp'),
                        'sentiment': doc_sentiment['sentiment'],
                        'score': doc_sentiment['score']
                    })
        
        if not all_sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # Calculate overall sentiment for document type
        overall_sentiment = self._aggregate_sentiment_scores(all_sentiments, all_scores)
        overall_sentiment['temporal_data'] = temporal_data
        overall_sentiment['document_count'] = len(documents)
        
        return overall_sentiment
    
    async def _analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text chunk"""
        if not text.strip():
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=self.max_length
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = predictions.cpu().numpy()[0]
            
            # Map to sentiment labels (FinBERT typically has negative, neutral, positive)
            sentiment_labels = ['negative', 'neutral', 'positive']
            max_idx = np.argmax(scores)
            
            return {
                'sentiment': sentiment_labels[max_idx],
                'score': float(scores[max_idx]),
                'confidence': float(np.max(scores)),
                'detailed_scores': {
                    'negative': float(scores[0]),
                    'neutral': float(scores[1]),
                    'positive': float(scores[2])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 400) -> List[str]:
        """Split text into manageable chunks for analysis"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _aggregate_sentiment_scores(self, sentiments: List[str], 
                                  scores: List[float]) -> Dict[str, Any]:
        """Aggregate multiple sentiment scores"""
        if not sentiments or not scores:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # Count sentiment occurrences
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for sentiment in sentiments:
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        # Calculate weighted average score
        avg_score = np.mean(scores)
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])
        overall_sentiment = max_sentiment[0]
        
        # Calculate confidence based on consistency
        confidence = max_sentiment[1] / len(sentiments) if sentiments else 0.0
        
        return {
            'sentiment': overall_sentiment,
            'score': float(avg_score),
            'confidence': float(confidence),
            'sentiment_distribution': sentiment_counts,
            'total_analyzed': len(sentiments)
        }
    
    def _calculate_overall_sentiment(self, sentiment_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall sentiment across all document types"""
        if not sentiment_scores:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        all_sentiments = []
        all_scores = []
        all_confidences = []
        
        for doc_type, scores in sentiment_scores.items():
            if scores and 'sentiment' in scores:
                all_sentiments.append(scores['sentiment'])
                all_scores.append(scores.get('score', 0.0))
                all_confidences.append(scores.get('confidence', 0.0))
        
        if not all_sentiments:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # Aggregate across document types
        overall = self._aggregate_sentiment_scores(all_sentiments, all_scores)
        overall['average_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        
        return overall
    
    async def _extract_key_themes(self, data: Dict[str, Any], company: str) -> List[str]:
        """Extract key themes from the analyzed texts"""
        # Simplified theme extraction based on common financial terms
        financial_themes = {
            'growth': ['growth', 'expand', 'increase', 'rise', 'improve'],
            'risk': ['risk', 'concern', 'challenge', 'decline', 'decrease'],
            'innovation': ['innovation', 'technology', 'development', 'new', 'launch'],
            'competition': ['competition', 'market share', 'competitor', 'rival'],
            'regulation': ['regulation', 'compliance', 'regulatory', 'policy'],
            'profitability': ['profit', 'margin', 'earnings', 'revenue', 'income']
        }
        
        theme_scores = {theme: 0 for theme in financial_themes}
        
        # Count theme occurrences across all documents
        for doc_type, documents in data.items():
            if isinstance(documents, list):
                for doc in documents:
                    text = (doc.get('text', '') or doc.get('content', '')).lower()
                    for theme, keywords in financial_themes.items():
                        for keyword in keywords:
                            theme_scores[theme] += text.count(keyword)
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:5] if score > 0]
    
    def _analyze_temporal_patterns(self, sentiment_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment patterns over time"""
        temporal_analysis = {
            'trend': 'stable',
            'volatility': 0.0,
            'recent_sentiment': 'neutral'
        }
        
        # Extract temporal data from sentiment scores
        all_temporal_data = []
        for doc_type, scores in sentiment_scores.items():
            if 'temporal_data' in scores:
                all_temporal_data.extend(scores['temporal_data'])
        
        if len(all_temporal_data) > 1:
            # Sort by date
            all_temporal_data.sort(key=lambda x: x.get('date', ''))
            
            # Calculate trend
            recent_scores = [item['score'] for item in all_temporal_data[-3:]]
            earlier_scores = [item['score'] for item in all_temporal_data[:-3]]
            
            if recent_scores and earlier_scores:
                recent_avg = np.mean(recent_scores)
                earlier_avg = np.mean(earlier_scores)
                
                if recent_avg > earlier_avg + 0.1:
                    temporal_analysis['trend'] = 'improving'
                elif recent_avg < earlier_avg - 0.1:
                    temporal_analysis['trend'] = 'declining'
                
            # Calculate volatility
            all_scores = [item['score'] for item in all_temporal_data]
            temporal_analysis['volatility'] = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
            
            # Recent sentiment
            if all_temporal_data:
                temporal_analysis['recent_sentiment'] = all_temporal_data[-1]['sentiment']
        
        return temporal_analysis
    
    def _calculate_confidence_scores(self, sentiment_scores: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence scores for the analysis"""
        confidence_scores = {}
        
        for doc_type, scores in sentiment_scores.items():
            if scores:
                confidence_scores[doc_type] = scores.get('confidence', 0.0)
        
        # Overall confidence
        if confidence_scores:
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        else:
            confidence_scores['overall'] = 0.0
        
        return confidence_scores