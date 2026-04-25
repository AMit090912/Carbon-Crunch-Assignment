"""
Receipt OCR Pipeline
Carbon Crunch Shortlisting Assignment
"""

import os
import json
import re
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import easyocr
import pytesseract
from PIL import Image

# 1. IMAGE PREPROCESSING

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Prepares a receipt image for OCR:
    - Converts to grayscale
    - Denoises
    - Deskews
    - Improves contrast (CLAHE)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 11
    )

    # Deskew
    deskewed = deskew(binary)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(deskewed)

    return enhanced


def deskew(image: np.ndarray) -> np.ndarray:
    """Detects and corrects skew angle using Hough lines."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return image  # no meaningful skew
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# 2. TEXT DETECTION & RECOGNITION

# Initialize EasyOCR once (expensive)
_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def extract_text_easyocr(processed_img: np.ndarray) -> list[dict]:
    """
    Runs EasyOCR and returns list of:
    { text, bbox, confidence }
    """
    reader = get_reader()
    results = reader.readtext(processed_img)
    extracted = []
    for (bbox, text, conf) in results:
        extracted.append({
            "text": text.strip(),
            "bbox": bbox,
            "confidence": round(float(conf), 4)
        })
    return extracted


def extract_text_tesseract(processed_img: np.ndarray) -> list[dict]:
    """
    Runs Tesseract with confidence scores as fallback.
    """
    pil_img = Image.fromarray(processed_img)
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    extracted = []
    for i, text in enumerate(data['text']):
        text = text.strip()
        conf = data['conf'][i]
        if text and conf > 0:
            extracted.append({
                "text": text,
                "confidence": round(conf / 100.0, 4),
                "bbox": None
            })
    return extracted


def extract_raw_text(image_path: str) -> tuple[list[dict], str]:
    """
    Tries EasyOCR first, falls back to Tesseract.
    Returns (word_blocks, full_text_string)
    """
    processed = preprocess_image(image_path)
    try:
        blocks = extract_text_easyocr(processed)
        engine = "easyocr"
    except Exception:
        blocks = extract_text_tesseract(processed)
        engine = "tesseract"

    full_text = " ".join(b["text"] for b in blocks if b["text"])
    return blocks, full_text, engine


# 3. KEY INFORMATION EXTRACTION WITH CONFIDENCE

DATE_PATTERNS = [
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4})\b",
]

TOTAL_PATTERNS = [
    r"(?:total|grand\s*total|amount\s*due|total\s*amount|rounded?\s*total)[^\d]*([0-9]+[.,][0-9]{2})",
    r"(?:total)[^\d]*([0-9]+\.[0-9]{2})",
    r"\btotal\b.*?(\d+\.\d{2})",
]

CURRENCY_PATTERNS = [
    (r"\bRM\b", "MYR"),
    (r"\bMYR\b", "MYR"),
    (r"\$", "USD"),
    (r"\b(?:USD)\b", "USD"),
    (r"£", "GBP"),
    (r"€", "EUR"),
    (r"\bINR\b|\bRs\.?\b", "INR"),
]

STORE_SKIP = {"receipt", "cash", "bill", "invoice", "tax", "gst", "thank", "you"}


def field_confidence(ocr_conf: float, pattern_matched: bool, heuristic_score: float) -> float:
    """
    Combines OCR confidence, pattern match, and heuristic into a single score.
    """
    base = ocr_conf if ocr_conf > 0 else 0.5
    pattern_bonus = 0.15 if pattern_matched else 0.0
    score = (base * 0.6) + (heuristic_score * 0.25) + pattern_bonus
    return round(min(score, 1.0), 3)


def extract_store_name(blocks: list[dict], full_text: str) -> dict:
    """Heuristic: store name is usually in early, large-font blocks."""
    candidates = []
    for i, b in enumerate(blocks[:8]):
        t = b["text"].strip()
        if len(t) > 3 and not any(skip in t.lower() for skip in STORE_SKIP):
            # Prefer longer, earlier, high-confidence text
            score = b["confidence"] * (1.0 - i * 0.08)
            candidates.append((t, score, b["confidence"]))

    if not candidates:
        return {"value": "Unknown", "confidence": 0.2}

    best = max(candidates, key=lambda x: x[1])
    conf = field_confidence(best[2], False, 0.75)
    return {"value": best[0], "confidence": conf}


def extract_date(blocks: list[dict], full_text: str) -> dict:
    """Regex-based date extraction with pattern validation."""
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            raw_date = match.group(1)
            # Find OCR confidence for the token containing this date
            ocr_conf = next(
                (b["confidence"] for b in blocks if raw_date[:4] in b["text"]),
                0.75
            )
            conf = field_confidence(ocr_conf, True, 0.9)
            return {"value": raw_date, "confidence": conf}

    return {"value": None, "confidence": 0.1}


def extract_total(blocks: list[dict], full_text: str) -> dict:
    """Looks for total keywords + amount patterns."""
    text_lower = full_text.lower()
    for pattern in TOTAL_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            amount = match.group(1).replace(",", ".")
            ocr_conf = next(
                (b["confidence"] for b in blocks if amount[:3] in b["text"]),
                0.75
            )
            conf = field_confidence(ocr_conf, True, 0.95)
            return {"value": amount, "confidence": conf}

    # Fallback: find largest number in text
    amounts = re.findall(r"\b\d+\.\d{2}\b", full_text)
    if amounts:
        biggest = max(amounts, key=lambda x: float(x))
        return {"value": biggest, "confidence": 0.45}

    return {"value": None, "confidence": 0.1}


def extract_currency(full_text: str) -> dict:
    for pattern, currency in CURRENCY_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            return {"value": currency, "confidence": 0.92}
    return {"value": "Unknown", "confidence": 0.3}


def extract_items(blocks: list[dict], full_text: str) -> list[dict]:
    """
    Extracts line items: text + price pairs.
    Looks for lines that have a word + amount pattern.
    """
    items = []
    # Split full text into lines by joining nearby bboxes
    lines = group_into_lines(blocks)

    item_pattern = re.compile(
        r"^(.+?)\s+(\d+[.,]\d{2})\s*$"
    )
    skip_keywords = {"total", "subtotal", "tax", "gst", "change", "cash", "discount",
                     "rounding", "amount", "price", "qty", "item", "desc"}

    for line in lines:
        line = line.strip()
        match = item_pattern.match(line)
        if match:
            name = match.group(1).strip()
            price = match.group(2).replace(",", ".")
            if any(kw in name.lower() for kw in skip_keywords):
                continue
            if len(name) < 2:
                continue
            # OCR confidence for this line
            ocr_conf = next(
                (b["confidence"] for b in blocks if name[:5].lower() in b["text"].lower()),
                0.7
            )
            conf = field_confidence(ocr_conf, True, 0.8)
            items.append({"name": name, "price": price, "confidence": round(conf, 3)})

    return items


def group_into_lines(blocks: list[dict]) -> list[str]:
    """Groups word blocks into lines using Y-coordinate proximity."""
    if not blocks or blocks[0].get("bbox") is None:
        # No bbox info — just return text split by spaces as lines
        full = " ".join(b["text"] for b in blocks)
        return full.split("  ")  # double-space as line separator

    # Sort by top-Y of bbox
    def top_y(b):
        bbox = b["bbox"]
        return min(p[1] for p in bbox) if bbox else 0

    sorted_blocks = sorted(blocks, key=top_y)
    lines = []
    current_line = []
    prev_y = None
    threshold = 15  # pixels

    for b in sorted_blocks:
        y = top_y(b)
        if prev_y is None or abs(y - prev_y) < threshold:
            current_line.append(b["text"])
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [b["text"]]
        prev_y = y

    if current_line:
        lines.append(" ".join(current_line))

    return lines


# 4. CONFIDENCE SCORING & FLAGS

LOW_CONFIDENCE_THRESHOLD = 0.7

def flag_low_confidence(result: dict) -> list[str]:
    """Flags any field with confidence below threshold."""
    flags = []
    scalar_fields = ["store_name", "date", "total_amount", "currency"]
    for field in scalar_fields:
        if field in result and result[field].get("confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD:
            flags.append(f"low_confidence_{field}")

    for i, item in enumerate(result.get("items", [])):
        if item.get("confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD:
            flags.append(f"low_confidence_item_{i}")

    return flags


def overall_receipt_confidence(result: dict) -> float:
    """Average confidence across all fields."""
    scores = []
    for field in ["store_name", "date", "total_amount", "currency"]:
        if field in result:
            scores.append(result[field].get("confidence", 0))
    for item in result.get("items", []):
        scores.append(item.get("confidence", 0))
    return round(sum(scores) / len(scores), 3) if scores else 0.0

# 5. MAIN PIPELINE

def process_receipt(image_path: str) -> dict:
    """
    Full pipeline for a single receipt image.
    Returns structured JSON with confidence scores.
    """
    path = Path(image_path)
    result = {
        "filename": path.name,
        "processed_at": datetime.now().isoformat(),
    }

    # Edge case: file doesn't exist
    if not path.exists():
        return {**result, "error": "File not found", "overall_confidence": 0.0}

    try:
        # Step 1: Extract text
        blocks, full_text, engine = extract_raw_text(image_path)
        result["ocr_engine"] = engine
        result["raw_text"] = full_text

        # Step 2: Extract fields
        result["store_name"] = extract_store_name(blocks, full_text)
        result["date"] = extract_date(blocks, full_text)
        result["items"] = extract_items(blocks, full_text)
        result["total_amount"] = extract_total(blocks, full_text)
        result["currency"] = extract_currency(full_text)

        # Step 3: Flags
        result["flags"] = flag_low_confidence(result)

        # Step 4: Overall confidence
        result["overall_confidence"] = overall_receipt_confidence(result)

    except Exception as e:
        result["error"] = str(e)
        result["overall_confidence"] = 0.0

    return result


def process_all_receipts(input_dir: str, output_dir: str) -> list[dict]:
    """
    Processes all images in input_dir, saves individual JSONs,
    and returns list of all results.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    all_results = []

    for img_file in sorted(image_files):
        print(f"Processing: {img_file.name}")
        result = process_receipt(str(img_file))
        all_results.append(result)

        # Save individual JSON
        out_file = output_path / f"{img_file.stem}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → Saved: {out_file.name} | confidence: {result.get('overall_confidence', 0):.2f}")

    return all_results



# 6. FINANCIAL SUMMARY


def generate_financial_summary(results: list[dict]) -> dict:
    """
    Aggregates extracted data into an expense summary.
    """
    valid = [r for r in results if "error" not in r and r.get("total_amount", {}).get("value")]

    total_spend = 0.0
    store_spend = {}

    for r in valid:
        try:
            amount = float(r["total_amount"]["value"].replace(",", "."))
        except (ValueError, AttributeError):
            continue

        total_spend += amount
        store = r.get("store_name", {}).get("value", "Unknown")
        if store not in store_spend:
            store_spend[store] = {"transactions": 0, "total": 0.0}
        store_spend[store]["transactions"] += 1
        store_spend[store]["total"] = round(store_spend[store]["total"] + amount, 2)

    return {
        "generated_at": datetime.now().isoformat(),
        "total_receipts_processed": len(results),
        "successful_extractions": len(valid),
        "failed_extractions": len(results) - len(valid),
        "total_spend": round(total_spend, 2),
        "number_of_transactions": len(valid),
        "average_spend_per_receipt": round(total_spend / len(valid), 2) if valid else 0.0,
        "spend_per_store": {
            store: {
                "transactions": data["transactions"],
                "total_spend": data["total"]
            }
            for store, data in sorted(store_spend.items(), key=lambda x: -x[1]["total"])
        }
    }

# ENTRY POINT

if __name__ == "__main__":
    INPUT_DIR = "receipts"
    OUTPUT_DIR = "outputs"

    print("=" * 50)
    print("Receipt OCR Pipeline — Carbon Crunch Assignment")
    print("=" * 50)

    results = process_all_receipts(INPUT_DIR, OUTPUT_DIR)

    if results:
        summary = generate_financial_summary(results)

        # Save summary
        summary_path = Path(OUTPUT_DIR) / "financial_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save combined output
        combined_path = Path(OUTPUT_DIR) / "all_receipts.json"
        with open(combined_path, "w") as f:
            json.dump({"receipts": results, "summary": summary}, f, indent=2)

        print("\n" + "=" * 50)
        print("FINANCIAL SUMMARY")
        print("=" * 50)
        print(f"Total receipts:     {summary['total_receipts_processed']}")
        print(f"Successful:         {summary['successful_extractions']}")
        print(f"Total spend:        {summary['total_spend']}")
        print(f"Transactions:       {summary['number_of_transactions']}")
        print(f"Average per bill:   {summary['average_spend_per_receipt']}")
        print(f"\nOutputs saved to:   {OUTPUT_DIR}/")
    else:
        print("No receipts processed. Add images to the 'receipts/' folder.")
