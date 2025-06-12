
# WD14/WD-VIT Tagger - Local Batch Image Tagging with Metadata Merge 🏷️

This script automatically tags a batch of images using a local ONNX model (`wd14-vit` or `wd-v1-4-moat`) and merges them with any existing metadata (like prompts stored by Stable Diffusion tools such as AUTOMATIC1111).  

🧼 It also cleans prompts by:
- Removing `<lora:...>` tags
- Stripping brackets (`()`, `{}`, `[]`)
- Keeping only the prompt part (excluding "Negative prompt" and below)
- Deduplicating tags

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Note**:  
- Use `onnxruntime-gpu` if you have a GPU  
- Or replace it with `onnxruntime` if you're on CPU only

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

Edit these variables in `auto-tagger.py`:

| Variable | Purpose | Example |
|:--|:--|:--|
| `MODEL_NAME` | Model file name without extension | `"wd-vit-tagger-v3"` |
| `MODEL_DIR` | Directory containing the `.onnx` and `.csv` | `"./models"` |
| `IMAGES_DIR` | Input image folder | `"./images"` |
| `TAGS_DIR` | Where `.txt` files with tags will be saved | `"./images"` |
| `THRESHOLD` | Confidence threshold for general tags | `0.35` |
| `CHARACTER_THRESHOLD` | Threshold for character tags | `0.85` |
| `REPLACE_UNDERSCORE` | Replace `_` with space | `True` |
| `TRAILING_COMMA` | Add comma at end of tag strings | `False` |
| `ORT_PROVIDERS` | ONNX Runtime providers | `["CUDAExecutionProvider", "CPUExecutionProvider"]` |

---

## 🚀 How It Works

1. Downloads the model and tag CSV if missing
2. Processes all images in the `IMAGES_DIR`:
   - Loads each image and any prompt metadata (if embedded)
   - Cleans metadata prompt:
     - Removes `<lora:...>` tags
     - Keeps only portion before "Negative prompt:"
     - Removes `()`, `[]`, `{}` brackets
   - Runs ONNX model and filters tags by threshold
   - Merges both sets of tags (metadata + model) and removes duplicates
   - Writes final tag list to a `.txt` file

---

## 📋 Example Output (`image1.txt`)

```plaintext
bookcase, cat, cozy room, sunny light, tabby fur
```

This output could be a cleaned combination of:
- Prompt from embedded metadata
- Tags predicted by the ONNX model

---

## 📝 Example Console Output

```plaintext
=== image1.png ===
📦 From Metadata:
cat, sunny light, bookshelf, tabby fur, cozy room
🤖 From Model:
cat, bookshelf, cozy room, tabby fur
📦 🔃 🤖 Combined tags:
bookcase, cat, cozy room, sunny light, tabby fur
```

---

## ❗ Notes

- If no metadata is found, only model-generated tags are used
- All duplicates are removed (case-insensitive match)
- Brackets like `(a)`, `{b}` are stripped to improve consistency
- Tags are sorted alphabetically

---

## 🛠️ Build as Executable with PyInstaller

To create a standalone executable from this script using **PyInstaller**, follow these steps:

### 1. Install PyInstaller

```bash
pip install pyinstaller
```

### 2. Create the Executable (Linux example)

```bash
pyinstaller --onefile \
  --add-data="venv/lib/python*/site-packages/onnxruntime:onnxruntime" \
  auto-tagger.py
```

Or use the helper script:

```bash
bash compile.sh
```

> Adjust the path to `onnxruntime` depending on your Python version and OS.

---

### 3. Troubleshooting

If you see an error like `ModuleNotFoundError: No module named 'onnxruntime'`, it's likely due to missing native libraries or incorrect inclusion.

You may also need to add:

```bash
--hidden-import=onnxruntime
```

### 4. CUDA/TensorRT Warnings

If you see warnings about missing libraries like `libcublas.so.12` or `libcudnn.so`, and you're not using GPU acceleration, **you can ignore these** as long as `CPUExecutionProvider` is in your `ORT_PROVIDERS`.

---

## ✅ Final Note

Your executable will be located in:

```plaintext
./dist/auto-tagger
```

You can run it from the terminal:

```bash
./dist/auto-tagger
```

Or double-click it (on Windows) if you use `--noconsole`.

---

Your final `.txt` tag files will be saved next to the images in the same folder (`TAGS_DIR`).  
Output is easy to batch-edit, analyze, or reuse in training datasets.