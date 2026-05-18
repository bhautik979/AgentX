import os
import json
import re
from tqdm import tqdm
import nltk
import tiktoken

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

# ----------------------------
# CONFIGURATION
# ----------------------------
INPUT_FILE = "/home2/sathya.marisetti/lma_major/knowledge_base.jsonl"
OUTPUT_FILE = "/home2/sathya.marisetti/lma_major/knowledge_base_cleaned_segmented.jsonl"
EMBED_MODEL = "text-embedding-3-small"  # model tokenizer reference
MAX_TOKENS_PER_SEGMENT = 500  # optional: split long paragraphs

# Initialize tokenizer
enc = tiktoken.encoding_for_model(EMBED_MODEL)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def clean_and_deduplicate(text):
    """Remove duplicates, unwanted artifacts, and noisy formatting."""
    if not text:
        return ""

    # Remove non-breaking spaces
    text = text.replace("\xa0", " ")

    # Deduplicate repeated lines (common in OCR or header/footer repetition)
    lines = text.splitlines()
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            unique_lines.append(line)
    text = "\n".join(unique_lines)  # keep line breaks for paragraph segmentation

    # Clean stray symbols and encoding noise
    text = text.encode("utf-8", "ignore").decode()
    return text.strip()


def normalize_text(text):
    """Lowercase and normalize punctuation."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace within lines
    text = re.sub(r'\s([?.!,;:])', r'\1', text)  # fix space before punctuation
    return text.strip()


def tokenize_count(text):
    """Return token count using the chosen tokenizer."""
    return len(enc.encode(text))


def segment_text(text, max_tokens=MAX_TOKENS_PER_SEGMENT):
    """Segment text into paragraphs, optionally split long paragraphs by sentences."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    segments = []

    for para in paragraphs:
        sentences = sent_tokenize(para)
        current_segment = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = tokenize_count(sent)
            if current_tokens + sent_tokens > max_tokens and current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_tokens = 0
            current_segment.append(sent)
            current_tokens += sent_tokens

        if current_segment:
            segments.append(" ".join(current_segment))

    return segments


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    processed_records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for file_idx, line in enumerate(tqdm(infile, desc="Cleaning & Segmenting"), 1):
            try:
                record = json.loads(line)
                raw_text = record.get("text", "")

                # Step 1: Clean + deduplicate
                clean_text = clean_and_deduplicate(raw_text)

                # Step 2: Normalize
                norm_text = normalize_text(clean_text)

                # Step 3: Segment
                segments = segment_text(norm_text)

                for i, seg in enumerate(segments, 1):
                    processed_records.append({
                        "doc_id": f"{record.get('doc_id', 'doc')}_{file_idx}_seg{i}",
                        "source_file": record.get("source_file"),
                        "page_number": record.get("page_number"),
                        "text": seg,
                        "token_count": tokenize_count(seg),
                        "metadata": record.get("metadata", {})
                    })

            except Exception as e:
                print(f"[ERROR] Failed to process record: {e}")

    # Save cleaned + segmented text
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in processed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Done! {len(processed_records)} cleaned & segmented entries saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
