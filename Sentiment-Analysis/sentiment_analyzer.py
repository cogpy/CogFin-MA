import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import re
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
import os

nltk.download("punkt")

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

def preprocess_text_sent_aware(text: str, max_tokens=512, tokenizer=None):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        tokens = tokenizer.encode(" ".join(current_chunk + [sentence]), add_special_tokens=False)
        if len(tokens) <= max_tokens:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def analyze_sentiment(chunks, tokenizer, model):
    model.eval()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    results = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).tolist()[0]

            pred_label_idx = torch.argmax(logits, dim=1).item()
            pred_label = label_map[pred_label_idx]
            confidence = probs[pred_label_idx]

            results.append({
                "text": chunk,
                "sentiment": pred_label,
                "confidence": confidence,
                "probabilities": {
                    "negative": probs[0],
                    "neutral": probs[1],
                    "positive": probs[2]
                }
            })

    return results

def aggregate_sentiment(results):
    avg_probs = {"negative": 0, "neutral": 0, "positive": 0}
    for res in results:
        for label in avg_probs:
            avg_probs[label] += res["probabilities"][label] / len(results)

    overall = max(avg_probs, key=avg_probs.get)
    return {
        "overall_sentiment": overall,
        "overall_confidence": avg_probs[overall],
        "average_probabilities": avg_probs,
        "detailed_results": results
    }

def visualize_sentiments(results):
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        counts[r["sentiment"]] += 1

    labels = list(counts.keys())
    values = list(counts.values())

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Sentiment Distribution")
    plt.show()

    # Bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Sentiment Count per Label")
    plt.ylabel("Count")
    plt.xlabel("Sentiment")
    plt.show()

def export_to_csv(results, filename="sentiment_output.csv"):
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
    print(f"\nCSV exported to {os.path.abspath(filename)}")

def show_top_negative_chunks(results, top_n=5):
    sorted_res = sorted(results, key=lambda x: x["probabilities"]["negative"], reverse=True)[:top_n]
    print("\n=== Top Negative Chunks ===")
    for i, res in enumerate(sorted_res, 1):
        print(f"Chunk {i} - Neg Score: {res['probabilities']['negative']:.2f}")
        print(textwrap.shorten(res["text"], width=120, placeholder="..."))
        print()

# === MAIN PIPELINE ===
if __name__ == "__main__":
    pdf_path = "annual-report-2024.pdf"  # Change if needed
    print("Loading FinBERT model...")
    tokenizer, model = load_finbert()

    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted text length: {len(pdf_text)} characters")

    print("Preprocessing text with sentence-aware chunking...")
    chunks = preprocess_text_sent_aware(pdf_text, tokenizer=tokenizer)
    print(f"Split into {len(chunks)} chunks.")

    print("Analyzing sentiment...")
    results = analyze_sentiment(chunks, tokenizer, model)

    print("Aggregating results...")
    aggregated = aggregate_sentiment(results)

    # === OUTPUT ===
    print("\n=== Sentiment Analyzer Output ===")
    print(f"Overall Sentiment: {aggregated['overall_sentiment'].capitalize()} (Confidence: {aggregated['overall_confidence']:.2f})")
    print("Average Probabilities:")
    for label, prob in aggregated["average_probabilities"].items():
        print(f"  {label.capitalize()}: {prob:.2f}")

    print("\nDetailed Results (First 3 Chunks):")
    for i, res in enumerate(aggregated["detailed_results"][:3], 1):
        print(f"Chunk {i}:")
        print(f"  Text: {textwrap.shorten(res['text'], width=60)}")
        print(f"  Sentiment: {res['sentiment'].capitalize()} (Confidence: {res['confidence']:.2f})")

    # Export CSV
    export_to_csv(aggregated["detailed_results"])

    # Visualize sentiment distribution
    visualize_sentiments(aggregated["detailed_results"])

    # Show top 5 negative chunks
    show_top_negative_chunks(aggregated["detailed_results"])