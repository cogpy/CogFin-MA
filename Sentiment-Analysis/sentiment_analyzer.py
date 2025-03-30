import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from typing import List, Dict
import textwrap

# Load FinBERT model and tokenizer
def load_finbert():
    """
    Loads the FinBERT model and tokenizer from Hugging Face.
    Using 'yiyanghkust/finbert-tone' for nuanced tone analysis (positive, neutral, negative).
    """
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Preprocess text into manageable chunks
def preprocess_text(text: str, max_length: int = 512) -> List[str]:
    """
    Splits a long financial report into chunks that fit FinBERT's 512-token limit.
    """
    # Split text into sentences or smaller chunks
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
    """
    Runs sentiment analysis on each chunk using FinBERT.
    Returns a list of dictionaries with sentiment scores and labels.
    """
    model.eval()  # Set to evaluation mode
    sentiment_results = []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}  # FinBERT-tone labels
    
    with torch.no_grad():  # Disable gradient computation for inference
        for chunk in chunks:
            # Tokenize and encode the text
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Run model inference
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).tolist()[0]  # Convert to probabilities
            
            # Get predicted label and confidence
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
    """
    Aggregates sentiment across chunks into an overall score for the report.
    """
    avg_probs = {"negative": 0, "neutral": 0, "positive": 0}
    n_chunks = len(sentiment_results)
    
    for result in sentiment_results:
        for label, prob in result["probabilities"].items():
            avg_probs[label] += prob / n_chunks
    
    # Determine overall sentiment
    overall_sentiment = max(avg_probs, key=avg_probs.get)
    overall_confidence = avg_probs[overall_sentiment]
    
    return {
        "overall_sentiment": overall_sentiment,
        "overall_confidence": overall_confidence,
        "average_probabilities": avg_probs,
        "detailed_results": sentiment_results
    }

# Main function to run the Sentiment Analyzer
def sentiment_analyzer(financial_report: str) -> Dict:
    """
    Main function to analyze sentiment of a financial report.
    Returns results formatted for the Master AI Agent.
    """
    # Load model and tokenizer
    print("Loading FinBERT model...")
    tokenizer, model = load_finbert()
    
    # Preprocess the report
    print("Preprocessing text...")
    chunks = preprocess_text(financial_report)
    print(f"Split into {len(chunks)} chunks.")
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    sentiment_results = analyze_sentiment(chunks, tokenizer, model)
    
    # Aggregate results
    print("Aggregating results...")
    aggregated_result = aggregate_sentiment(sentiment_results)
    
    return aggregated_result

# Example usage
if __name__ == "__main__":
    # Sample financial report (e.g., earnings call excerpt)
    sample_report = """
    Good afternoon, everyone. We’re pleased to report a strong Q1 2025 with revenue up 10% to $25 billion, 
    driven by robust demand for our new electric vehicle lineup. However, supply chain constraints have 
    squeezed our margins, with costs rising 15% year-over-year. Management remains optimistic about 
    resolving these issues by Q3, and we’re confident in our growth trajectory despite macroeconomic 
    headwinds like rising interest rates. Analysts have mixed views, with some citing production delays 
    as a concern, but we believe our innovation pipeline will keep us ahead of competitors.
    """
    
    # Run the Sentiment Analyzer
    result = sentiment_analyzer(sample_report)
    
    # Print results in a readable format
    print("\n=== Sentiment Analyzer Output ===")
    print(f"Overall Sentiment: {result['overall_sentiment'].capitalize()} "
          f"(Confidence: {result['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in result['average_probabilities'].items():
        print(f"  {label.capitalize()}: {prob:.2f}")
    
    print("\nDetailed Results:")
    for i, detail in enumerate(result['detailed_results'], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(detail['text'], width=50, placeholder='...')}")
        print(f"  Sentiment: {detail['sentiment'].capitalize()} "
              f"(Confidence: {detail['confidence']:.2f})")