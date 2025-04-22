# Requires: pip install PyPDF2

import PyPDF2
import subprocess
import json
import os
import tempfile

def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def call_deepseek_model(prompt: str) -> str:
    cmd = ["ollama", "run", "deepseek-r1"]
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"Model inference failed: {result.stderr}")
    return result.stdout.strip()

def analyze_annual_report(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")
    report_text = extract_text_from_pdf(pdf_path)
    prompt = (
        "You are a financial analyst. "
        "Carefully read the following annual report and extract the most important sections and details "
        "that are essential for conducting a fundamental analysis of the company. "
        "These may include insights from the income statement, balance sheet, cash flow statement, management commentary, "
        "risk factors, business model, revenue drivers, and forward-looking statements. "
        "Present the output as a structured summary highlighting each relevant section and its significance.\n\n"
        f"{report_text}"
    )
    return call_deepseek_model(prompt)

if __name__ == "__main__": 
    pdf_file = os.path.join(os.path.dirname(__file__), "annual-report-2024.pdf")
    analysis = analyze_annual_report(pdf_file)
    print(analysis)