import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from typing import List, Dict
import textwrap
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def load_finbert():
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                if any(keyword in text.lower() for keyword in ["safe harbor", "e-voting", "notice of the"]):
                    continue
                full_text += text + "\n\n"
    return full_text.strip()

def better_preprocess(text: str, max_tokens: int = 512, tokenizer=None) -> List[str]:
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        tokens = tokenizer.encode(' '.join(current_chunk + [sentence]), add_special_tokens=False)
        if len(tokens) <= max_tokens:
            current_chunk.append(sentence)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def analyze_sentiment(chunks: List[str], tokenizer, model) -> List[Dict]:
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

def aggregate_sentiment(sentiment_results: List[Dict]) -> Dict:
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

def sentiment_analyzer_pdf(pdf_path: str) -> Dict:
    print("Loading FinBERT model...")
    tokenizer, model = load_finbert()

    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text length: {len(full_text)} characters")

    print("Preprocessing text with sentence-aware chunking...")
    chunks = better_preprocess(full_text, tokenizer=tokenizer)
    print(f"Split into {len(chunks)} chunks.")

    print("Analyzing sentiment...")
    sentiment_results = analyze_sentiment(chunks, tokenizer, model)

    print("Aggregating results...")
    aggregated_result = aggregate_sentiment(sentiment_results)

    return aggregated_result

if __name__ == "__main__":
    pdf_path = "annual-report-2024.pdf"  # Replace with your path
    result = sentiment_analyzer_pdf(pdf_path)

    print("\n=== Sentiment Analyzer Output ===")
    print(f"Overall Sentiment: {result['overall_sentiment'].capitalize()} (Confidence: {result['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in result['average_probabilities'].items():
        print(f"  {label.capitalize()}: {prob:.2f}")

    print("\nDetailed Results (First 3 Chunks):")
    for i, detail in enumerate(result['detailed_results'][:3], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(detail['text'], width=50, placeholder='...')}")
        print(f"  Sentiment: {detail['sentiment'].capitalize()} (Confidence: {detail['confidence']:.2f})")