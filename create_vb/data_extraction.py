import os
import json
from tqdm import tqdm
import pdfplumber
from PyPDF2 import PdfReader
import camelot

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_DIR = "/home2/sathya.marisetti/lma_major/finance"       # 🔁 Folder containing PDFs
OUTPUT_FILE = "/home2/sathya.marisetti/lma_major/knowledge_base.jsonl"   # 🔁 Output JSONL file
MAX_PAGES = None  # Limit for testing, set to None for all pages

# ----------------------------
# HELPER FUNCTION
# ----------------------------
def extract_from_pdf(file_path):
    """
    Extracts text, tables, and metadata from a single PDF file.
    Returns a list of structured JSON objects (one per page).
    """
    file_name = os.path.basename(file_path)
    results = []

    try:
        pdf_reader = PdfReader(file_path)
        metadata = pdf_reader.metadata or {}
    except Exception as e:
        print(f"[ERROR] Cannot read metadata for {file_name}: {e}")
        metadata = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            if MAX_PAGES:
                total_pages = min(total_pages, MAX_PAGES)

            # Inner tqdm: per-page progress
            for page_num in tqdm(range(total_pages), desc=f"Processing {file_name}", leave=False):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""

                # Extract tables using Camelot
                tables = []
                try:
                    tables_data = camelot.read_pdf(file_path, pages=str(page_num + 1))
                    for i, t in enumerate(tables_data):
                        tables.append({
                            "table_id": f"{file_name}_p{page_num+1}_t{i+1}",
                            "content": t.df.values.tolist()
                        })
                except Exception:
                    pass  # Skip table errors

                entry = {
                    "doc_id": f"{file_name}_p{page_num+1}",
                    "source_file": file_name,
                    "page_number": page_num + 1,
                    "text": text.strip(),
                    "tables": tables,
                    "metadata": {
                        "title": metadata.get('/Title') or metadata.get('title'),
                        "author": metadata.get('/Author') or metadata.get('author'),
                        "subject": metadata.get('/Subject') or metadata.get('subject'),
                        "keywords": metadata.get('/Keywords') or metadata.get('keywords'),
                    }
                }
                results.append(entry)

    except Exception as e:
        print(f"[ERROR] Cannot open PDF {file_name}: {e}")

    return results


# ----------------------------
# MAIN EXECUTION
# ----------------------------
def main():
    pdf_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found in folder!")
        return

    print(f"🔍 Found {len(pdf_files)} PDF files in {INPUT_DIR}")

    # Remove old output if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"🧹 Old output file '{OUTPUT_FILE}' deleted.")

    # Outer tqdm: per-PDF progress
    for pdf_path in tqdm(pdf_files, desc="Extracting all PDFs", unit="file"):
        try:
            extracted_data = extract_from_pdf(pdf_path)
            
            # Write immediately after each PDF
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for record in extracted_data:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"💾 Saved {len(extracted_data)} entries from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to extract from {pdf_path}: {e}")

    print(f"\n✅ All PDFs processed! Results continuously saved in '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
