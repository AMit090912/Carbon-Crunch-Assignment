# Receipt OCR Extraction System
Carbon Crunch Shortlisting Assignment

## What it does
- Extracts store name, date, items, total, currency from receipt images
- Confidence score (0–1) for every field
- Flags low confidence fields automatically
- Saves one JSON per receipt + financial summary

## Setup
```bash
pip install -r requirements.txt
sudo apt install tesseract-ocr  # Ubuntu
# brew install tesseract        # macOS
```

## Usage
Add images to `receipts/` folder then:
```bash
python ocr_pipeline.py
```
Results saved to `outputs/`

## Output format
```json
{
  "store_name":   { "value": "BOOK TAK SDN BHD", "confidence": 0.91 },
  "date":         { "value": "25/12/2018",        "confidence": 0.95 },
  "items":        [{ "name": "KF MODELLING CLAY", "price": "9.00", "confidence": 0.88 }],
  "total_amount": { "value": "9.00", "confidence": 0.97 },
  "currency":     { "value": "MYR",  "confidence": 0.92 },
  "flags": [],
  "overall_confidence": 0.926
}
```

## Stack
EasyOCR · Tesseract · OpenCV · Pillow
