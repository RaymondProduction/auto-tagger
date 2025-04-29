
# WD14/WD-VIT Tagger - Local Batch Image Tagging

This script automatically tags a batch of images using a local ONNX model (`wd14-vit` or `wd-v1-4-moat`) with HuggingFace-style tag lists.

It loads a `.onnx` model and `.csv` tag list, processes all images in a folder, and generates `.txt` files containing the filtered tags for each image.

---

## 📦 Requirements

Install dependencies:

```bash
pip install numpy onnxruntime-gpu Pillow
```

*(If you do not have a GPU, install `onnxruntime` instead of `onnxruntime-gpu`.)*

---

## 📂 Directory Structure

```plaintext
./models/
    model.onnx
    selected_tags.csv
./images/
    image1.jpg
    image2.png
...
```

---

## ⚙️ Configuration

Edit the following variables at the top of the script:

| Variable | Purpose | Example |
|:--|:--|:--|
| `MODEL_NAME` | Model name without extension | `"wd-vit-tagger-v3"` |
| `MODEL_DIR` | Path to directory containing `.onnx` and `.csv` files | `"./models"` |
| `IMAGES_DIR` | Path to directory containing images to tag | `"./images"` |
| `TAGS_DIR` | Path to output `.txt` files | `"./images"` |
| `THRESHOLD` | Minimum confidence for general tags | `0.35` |
| `CHARACTER_THRESHOLD` | Minimum confidence for character tags | `0.85` |
| `REPLACE_UNDERSCORE` | Replace underscores with spaces | `True` |
| `TRAILING_COMMA` | Add trailing commas after tags | `False` |
| `ORT_PROVIDERS` | ONNX Runtime providers | `["CUDAExecutionProvider", "CPUExecutionProvider"]` |

---

## 🚀 How It Works

- Loads the ONNX model and selected_tags.csv.
- Processes each image:
  - Resizes to a square canvas (white background) maintaining aspect ratio.
  - Converts to BGR and runs inference through ONNX model.
  - Applies `sigmoid` activation on model outputs.
- Filters:
  - General tags by `THRESHOLD`
  - Character tags by `CHARACTER_THRESHOLD`
- Outputs:
  - `.txt` file containing selected tags for each image.

---

## 📋 Example output (`image1.txt`)

```plaintext
1girl, long_hair, blue_eyes, smile, looking_at_viewer
```

---

## ❗ Notes

- Model expects images resized to a square with center-padding (NOT simple stretch).
- Tags are sorted by type: **character** tags first (priority), then **general**.
- You can tweak thresholds to control strictness of tagging.

---
