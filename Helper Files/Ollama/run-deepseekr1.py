# Requires: pip install PyPDF2

import PyPDF2
import subprocess
import json
import os
import tempfile
import datetime

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
        encoding="utf-8"    # <-- ensure UTF-8 encoding on Windows
    )
    if result.returncode != 0:
        raise RuntimeError(f"Model inference failed: {result.stderr}")
    return result.stdout.strip()

def analyze_annual_report(pdf_path: str) -> dict:
    """Analyze annual report without hardcoded sections"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} not found")
    
    report_text = extract_text_from_pdf(pdf_path)
    
    # First, ask the model to identify the key sections in the document
    section_prompt = (
        "You are a financial analyst reviewing an annual report. First, identify 5-7 key sections or themes "
        "that appear in this document. Output ONLY a JSON array of section names. For example: "
        "[\"Financial Highlights\", \"Business Overview\", \"Risk Factors\"].\n\n"
        f"Here is the annual report:\n{report_text[:10000]}..."  # Send a truncated version to identify sections
    )
    
    section_response = call_deepseek_model(section_prompt)
    
    # Try to parse the response as JSON
    try:
        # Look for JSON array in the response
        import re
        array_match = re.search(r'(\[.*\])', section_response, re.DOTALL)
        if array_match:
            section_response = array_match.group(1)
        
        sections = json.loads(section_response)
        if not isinstance(sections, list):
            raise ValueError("Expected a list of sections")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❗ Failed to parse sections: {e}")
        print(f"Raw response: {repr(section_response)}")
        # Fallback to simple sections
        sections = ["Overview", "Financial Performance", "Strategy", "Risks", "Outlook"]
    
    # Now analyze each section
    results = {}
    
    # Create a prompt for detailed analysis of all sections
    analysis_prompt = (
        "You are a financial analyst. Analyze this annual report and extract key points for each section.\n\n"
        "IMPORTANT: Your output must be ONLY a valid JSON object where each key is one of these sections "
        f"identified in the document: {', '.join(sections)}.\n\n"
        "For each section key, include an array of bullet points as strings. Each bullet should be a "
        "significant insight or data point from that section.\n\n"
        "Example format:\n{\n"
        f"  \"{sections[0]}\": [\"key point 1\", \"key point 2\"],\n"
        f"  \"{sections[1]}\": [\"key point 1\", \"key point 2\"],\n"
        "  ...\n}\n\n"
        f"Here is the annual report to analyze:\n{report_text}"
    )
    
    raw = call_deepseek_model(analysis_prompt)
    
    # Handle the response
    if not raw or not raw.strip():
        raise RuntimeError("❗ Model returned empty response.")
    
    # Try to extract JSON
    import re
    json_match = re.search(r'(\{[\s\S]*\})', raw)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("❗ Failed to parse JSON. Raw model output (repr):")
        print(repr(raw))
        with open("last_model_output.txt", "w", encoding="utf-8") as f:
            f.write(raw)
        
        # Create fallback with the identified sections
        print("\n\n🔄 Creating fallback response with identified sections...")
        fallback = {section: ["Could not parse model output"] for section in sections}
        return fallback

def save_as_markdown(summary: dict, pdf_name: str) -> str:
    """Save the analysis results to a nicely formatted Markdown file"""
    
    # Extract the PDF filename without extension
    pdf_basename = os.path.splitext(os.path.basename(pdf_name))[0]
    output_file = f"{pdf_basename}_summary.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write a header
        f.write(f"# Analysis of {pdf_basename}\n\n")
        f.write(f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d')}*\n\n")
        
        # Write each section
        for section, points in summary.items():
            f.write(f"## {section}\n\n")
            for point in points:
                f.write(f"- {point}\n")
            f.write("\n")
    
    return output_file

def interactive_summary(summary: dict, pdf_path: str):
    # Save to Markdown first
    md_file = save_as_markdown(summary, pdf_path)
    print(f"✅ Analysis saved to {md_file}")
    
    # Continue with interactive view
    sections = list(summary.keys())
    while True:
        print("\nSelect a section to view (or 0 to exit):")
        for i, sec in enumerate(sections, 1):
            print(f"  {i}. {sec}")
        choice = input("Enter choice: ").strip()
        if choice == "0":
            break
        if not choice.isdigit() or not (1 <= int(choice) <= len(sections)):
            print("Invalid choice, try again.")
            continue
        sec = sections[int(choice) - 1]
        print(f"\n--- {sec} ---")
        for line in summary[sec]:
            print(" •", line)
    print("Goodbye!")

if __name__ == "__main__":
    pdf_file = os.path.join(os.path.dirname(__file__), "MIDAS Technology Sector Infosys.pdf")
    summary = analyze_annual_report(pdf_file)
    interactive_summary(summary, pdf_file)