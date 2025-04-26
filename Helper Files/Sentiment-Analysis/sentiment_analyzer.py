import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Optional
import textwrap
import re
import logging
import json
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import os
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

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

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract and return text from a real PDF file."""
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n\n"
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise

    def _extract_text_from_pdf_like_content(self, pdf_text: str) -> str:
        """Extract and clean text from PDF-like OCR-tagged content."""
        content_blocks = re.findall(r'<CONTENT_FROM_OCR>(.*?)</CONTENT_FROM_OCR>', pdf_text, re.DOTALL)
        full_text = ""
        for block in content_blocks:
            if any(keyword in block.lower() for keyword in ["safe harbor", "e-voting", "instructions", "notice of the"]):
                continue
            cleaned_block = re.sub(r'(the red box\s+)+', ' ', block.strip())
            full_text += cleaned_block + "\n\n"
        return full_text.strip()

    def _preprocess_text(self, text: str) -> List[str]:
        """Split text into chunks using sentence-aware token-based logic."""
        sentences = sent_tokenize(text)
        chunks, current_chunk = [], []

        for sentence in sentences:
            tokens = self.tokenizer.encode(' '.join(current_chunk + [sentence]), add_special_tokens=False)
            if len(tokens) <= self.max_length:
                current_chunk.append(sentence)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

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

    def export_to_csv(self, results: List[Dict], filename: str = "sentiment_output.csv"):
        """Export chunk-level sentiment results to a CSV file."""
        rows = []
        for i, r in enumerate(results, 1):
            rows.append({
                "Chunk No.": i,
                "Sentiment": r["sentiment"],
                "Confidence": r["confidence"],
                "Negative": r["probabilities"]["negative"],
                "Neutral": r["probabilities"]["neutral"],
                "Positive": r["probabilities"]["positive"],
                "Text": r["text"]
            })
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        logger.info(f"Results exported to {os.path.abspath(filename)}")

    def visualize_sentiments(self, results: List[Dict]):
        """Generate and display sentiment distribution pie and bar charts."""
        counts = {"positive": 0, "neutral": 0, "negative": 0}
        for r in results:
            counts[r["sentiment"]] += 1

        labels = list(counts.keys())
        values = list(counts.values())

        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("Sentiment Distribution (Pie Chart)")
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.title("Sentiment Count per Label (Bar Chart)")
        plt.ylabel("Count")
        plt.xlabel("Sentiment")
        plt.show()

    def show_top_negative_chunks(self, results: List[Dict], top_n: int = 5):
        """Print top N chunks with highest negative sentiment."""
        sorted_res = sorted(results, key=lambda x: x["probabilities"]["negative"], reverse=True)[:top_n]
        print("\n=== Top Negative Chunks ===")
        for i, res in enumerate(sorted_res, 1):
            print(f"Chunk {i} - Neg Score: {res['probabilities']['negative']:.2f}")
            print(textwrap.shorten(res["text"], width=120, placeholder="..."))
            print()

    def run(self, input_data: str, source_type: str = "pdf") -> Dict:
        """
        Process input text and return sentiment analysis results.
        Args:
            input_data: Raw text (e.g., PDF content or OCR tagged).
            source_type: Type of input ('pdf', 'text', etc.).
        Returns:
            Dict with sentiment analysis results.
        """
        logger.info(f"Processing input data (source: {source_type})...")

        if source_type == "pdf":
            processed_text = self._extract_text_from_pdf_like_content(input_data)
        else:
            processed_text = input_data

        logger.info(f"Extracted text length: {len(processed_text)} characters")

        chunks = self._preprocess_text(processed_text)
        logger.info(f"Split into {len(chunks)} chunks")

        sentiment_results = self._analyze_sentiment(chunks)
        logger.info("Sentiment analysis completed")

        aggregated_result = self._aggregate_sentiment(sentiment_results)
        logger.info(f"Overall sentiment: {aggregated_result['overall_sentiment']} (Confidence: {aggregated_result['overall_confidence']:.2f})")

        return {
            "agent": "SentimentAnalyzer",
            "source_type": source_type,
            "result": aggregated_result
        }

if __name__ == "__main__":
    agent = SentimentAnalyzerAgent()

    # Use real PDF path here
    pdf_path = "annual-report-2024.pdf"
    pdf_text = agent._extract_text_from_pdf(pdf_path)

    result = agent.run(pdf_text, source_type="text")

    print("\n=== Sentiment Analyzer Agent Output ===")
    print(json.dumps(result, indent=2, default=str))

    print(f"\nSummary:")
    print(f"Overall Sentiment: {result['result']['overall_sentiment'].capitalize()} (Confidence: {result['result']['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in result['result']['average_probabilities'].items():
        print(f"  {label.capitalize()}: {prob:.2f}")

    print("\nDetailed Results (First 3 Chunks):")
    for i, detail in enumerate(result['result']['detailed_results'][:3], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(detail['text'], width=50, placeholder='...')}")
        print(f"  Sentiment: {detail['sentiment'].capitalize()} (Confidence: {detail['confidence']:.2f})")

    agent.export_to_csv(result['result']['detailed_results'])
    agent.visualize_sentiments(result['result']['detailed_results'])
    agent.show_top_negative_chunks(result['result']['detailed_results'])