import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from typing import List, Dict
import textwrap
import re
import pdfplumber

# Load FinBERT model and tokenizer
def load_finbert():
    """Loads the FinBERT model and tokenizer from Hugging Face."""
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Extract text from PDF-like content
def extract_text_from_pdf_like_content(pdf_text: str) -> str:
    """Extracts content from <CONTENT_FROM_OCR> tags, cleaning up boilerplate."""
    # Use regex to extract content between <CONTENT_FROM_OCR> tags
    content_blocks = re.findall(r'<CONTENT_FROM_OCR>(.*?)</CONTENT_FROM_OCR>', pdf_text, re.DOTALL)
    
    # Combine and clean the extracted text
    full_text = ""
    for block in content_blocks:
        # Skip boilerplate sections (e.g., AGM instructions, Safe Harbor)
        if any(keyword in block.lower() for keyword in ["safe harbor", "e-voting", "instructions", "notice of the"]):
            continue
        # Remove redundant repetitions (e.g., "the red box" spam)
        cleaned_block = re.sub(r'(the red box\s+)+', ' ', block.strip())
        full_text += cleaned_block + "\n\n"
    
    return full_text.strip()

# Preprocess text into manageable chunks
def preprocess_text(text: str, max_length: int = 512) -> List[str]:
    """Splits text into chunks that fit FinBERT's 512-token limit."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Perform sentiment analysis on text chunks
def analyze_sentiment(chunks: List[str], tokenizer, model) -> List[Dict]:
    """Runs sentiment analysis on each chunk using FinBERT."""
    model.eval()
    sentiment_results = []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
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

# Aggregate sentiment results
def aggregate_sentiment(sentiment_results: List[Dict]) -> Dict:
    """Aggregates sentiment across chunks into an overall score."""
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

# Main function to run the Sentiment Analyzer on PDF-like content
def sentiment_analyzer_pdf(pdf_content: str) -> Dict:
    """Analyzes sentiment of a PDF-like financial report."""
    print("Loading FinBERT model...")
    tokenizer, model = load_finbert()
    
    print("Using extracted PDF text...")
    financial_report = pdf_content
    print(f"Extracted text length: {len(financial_report)} characters")
    
    print("Preprocessing text...")
    chunks = preprocess_text(financial_report)
    print(f"Split into {len(chunks)} chunks.")
    
    print("Analyzing sentiment...")
    sentiment_results = analyze_sentiment(chunks, tokenizer, model)
    
    print("Aggregating results...")
    aggregated_result = aggregate_sentiment(sentiment_results)
    
    return aggregated_result

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from an actual PDF file."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Skip boilerplate pages (e.g., AGM notices)
                if any(keyword in text.lower() for keyword in ["safe harbor", "e-voting", "notice of the"]):
                    continue
                full_text += text + "\n\n"
    return full_text.strip()

# Example usage with your provided document
if __name__ == "__main__":
    # Your provided document (truncated for brevity here)
    # with open("infosys_report.txt", "r", encoding="utf-8") as f:
    #     pdf_content = f.read()
    
    pdf_path = "annual-report-2024.pdf"  # Replace with your PDF path
    pdf_content = extract_text_from_pdf(pdf_path)
    
    # Run the Sentiment Analyzer
    result = sentiment_analyzer_pdf(pdf_content)
    
    # Print results
    print("\n=== Sentiment Analyzer Output ===")
    print(f"Overall Sentiment: {result['overall_sentiment'].capitalize()} "
          f"(Confidence: {result['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in result['average_probabilities'].items():
        print(f"  {label.capitalize()}: {prob:.2f}")
    
    print("\nDetailed Results (First 3 Chunks):")
    for i, detail in enumerate(result['detailed_results'][:3], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(detail['text'], width=50, placeholder='...')}")
        print(f"  Sentiment: {detail['sentiment'].capitalize()} "
              f"(Confidence: {detail['confidence']:.2f})")