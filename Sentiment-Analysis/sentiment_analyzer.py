import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Optional
import textwrap
import re
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    """An agent that performs sentiment analysis on financial texts using FinBERT."""
    
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone", max_length: int = 512):
        """Initialize the agent with a model and tokenizer."""
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer, self.model = self._load_finbert()
        logger.info(f"Sentiment Analyzer Agent initialized with model: {model_name}")

    def _load_finbert(self) -> tuple:
        """Load the FinBERT model and tokenizer."""
        try:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertForSequenceClassification.from_pretrained(self.model_name)
            return tokenizer, model
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            raise

    def _extract_text_from_pdf_like_content(self, pdf_text: str) -> str:
        """Extract and clean text from PDF-like content."""
        content_blocks = re.findall(r'<CONTENT_FROM_OCR>(.*?)</CONTENT_FROM_OCR>', pdf_text, re.DOTALL)
        full_text = ""
        for block in content_blocks:
            if any(keyword in block.lower() for keyword in ["safe harbor", "e-voting", "instructions", "notice of the"]):
                continue
            cleaned_block = re.sub(r'(the red box\s+)+', ' ', block.strip())
            full_text += cleaned_block + "\n\n"
        return full_text.strip()

    def _preprocess_text(self, text: str) -> List[str]:
        """Split text into chunks for FinBERT processing."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.max_length:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _analyze_sentiment(self, chunks: List[str]) -> List[Dict]:
        """Run sentiment analysis on text chunks."""
        self.model.eval()
        sentiment_results = []
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        with torch.no_grad():
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).tolist()[0]
                
                pred_label_idx = torch.argmax(logits, dim=1).item()
                pred_label = label_map[pred_label_idx]
                confidence = probs[pred_label_idx]
                
                sentiment_results.append({
                    "text": chunk,
                    "sentiment": pred_label,
                    "confidence": confidence,
                    "probabilities": dict(zip(["negative", "neutral", "positive"], probs))
                })
        
        return sentiment_results

    def _aggregate_sentiment(self, sentiment_results: List[Dict]) -> Dict:
        """Aggregate sentiment across chunks."""
        avg_probs = {"negative": 0, "neutral": 0, "positive": 0}
        n_chunks = len(sentiment_results)
        
        for result in sentiment_results:
            for label, prob in result["probabilities"].items():
                avg_probs[label] += prob / n_chunks
        
        overall_sentiment = max(avg_probs, key=avg_probs.get)
        overall_confidence = avg_probs[overall_sentiment]
        
        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "average_probabilities": avg_probs,
            "detailed_results": sentiment_results
        }

    def run(self, input_data: str, source_type: str = "pdf") -> Dict:
        """
        Process input text and return sentiment analysis results.
        Args:
            input_data: Raw text (e.g., PDF content, earnings call, X posts).
            source_type: Type of input ('pdf', 'text', etc.) for preprocessing.
        Returns:
            Dict with sentiment analysis results.
        """
        logger.info(f"Processing input data (source: {source_type})...")
        
        # Preprocess based on source type
        if source_type == "pdf":
            processed_text = self._extract_text_from_pdf_like_content(input_data)
        else:
            processed_text = input_data  # For plain text inputs (e.g., X posts)
        
        logger.info(f"Extracted text length: {len(processed_text)} characters")
        
        # Split into chunks
        chunks = self._preprocess_text(processed_text)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Analyze sentiment
        sentiment_results = self._analyze_sentiment(chunks)
        logger.info("Sentiment analysis completed")
        
        # Aggregate results
        aggregated_result = self._aggregate_sentiment(sentiment_results)
        logger.info(f"Overall sentiment: {aggregated_result['overall_sentiment']} "
                   f"(Confidence: {aggregated_result['overall_confidence']:.2f})")
        
        # Add metadata for Master AI
        output = {
            "agent": "SentimentAnalyzer",
            "source_type": source_type,
            "result": aggregated_result
        }
        
        return output

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = SentimentAnalyzerAgent()
    
    # Load your PDF-like content (replace with your file path or full text)
    with open("annual-report-2024.pdf", "r", encoding="utf-8") as f:
        pdf_content = f.read()
    
    # Run the agent
    result = agent.run(pdf_content, source_type="pdf")
    
    # Print results
    print("\n=== Sentiment Analyzer Agent Output ===")
    print(json.dumps(result, indent=2, default=str))  # Pretty-print JSON
    
    # Summary for human readability
    print(f"\nSummary:")
    print(f"Overall Sentiment: {result['result']['overall_sentiment'].capitalize()} "
          f"(Confidence: {result['result']['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in result['result']['average_probabilities'].items():
        print(f"  {label.capitalize()}: {prob:.2f}")
    
    print("\nDetailed Results (First 3 Chunks):")
    for i, detail in enumerate(result['result']['detailed_results'][:3], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(detail['text'], width=50, placeholder='...')}")
        print(f"  Sentiment: {detail['sentiment'].capitalize()} "
              f"(Confidence: {detail['confidence']:.2f})")