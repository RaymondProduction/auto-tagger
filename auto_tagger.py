import os
import numpy as np
import csv
import onnxruntime as ort
import urllib.request
from PIL import Image
import re
import argparse
from Levenshtein import distance

from tag_categories import CATEGORY_KEYWORDS  # Define categories and keywords for filtering

# Configuration
#MODEL_NAME = "wd-v1-4-moat-tagger-v2"  # Without ".onnx"
#MODEL_NAME = "wd-vit-tagger-v3"
# Argument parsing
parser = argparse.ArgumentParser(description="Batch tag images with optional metadata prompt merging.")


# Defaults
parser.set_defaults(prompt_from_meta=True, analys_by_ai=True)

# Include flags to disable features
parser.add_argument("-np", "--no-prompt-from-meta", dest="prompt_from_meta", action="store_false",
                    help="❌ Disable tags from embedded prompt metadata (default: ON)")
parser.add_argument("-nai", "--no-analys-by-ai", dest="analys_by_ai", action="store_false",
                    help="❌ Disable AI-based image analysis (default: ON)")
parser.add_argument("-d", "--diff-mode", action="store_true",
                    help="🧩 Enable diff mode: generate 'different.txt' and 'merged_clean_prompt.txt' from tag files")
parser.add_argument("-i", "--input", type=str, default="./images", help="Input folder with images")
parser.add_argument("-o", "--output", type=str, default="./images", help="Output folder for tag text files")
parser.add_argument("-m", "--model-dir", type=str, default="./models", help="Directory containing ONNX model and tags CSV")
parser.add_argument("-nc", "--no-clean", action="store_true", help="Do not clean prompt metadata (keep brackets, <lora>, etc.)")
parser.add_argument("-fc", "--filter-category", type=str,
                    help="Comma-separated categories to keep: bg,pose,nsfw,shape,place,accessories,emotion,composition,quality,clothing,objects,character")

args = parser.parse_args()

# Diff Mode Logic (Early Exit)
if args.diff_mode:
    TAGS_DIR = args.output
    DIFF_OUTPUT_PATH = os.path.join(TAGS_DIR, "different.txt")
    STRIPPED_PROMPT_OUTPUT_PATH = os.path.join(TAGS_DIR, "merged_clean_prompt.txt")

    print("🧩 Running diff mode...")

    tag_files = [
        f for f in os.listdir(TAGS_DIR)
        if f.endswith(".txt") and not f.startswith(("different", "merged", "unknown"))
    ]

    tags_per_file = {}
    for filename in tag_files:
        path = os.path.join(TAGS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                tags = {t.strip() for t in f.read().split(",") if t.strip()}
                tags_per_file[filename] = tags
        except Exception as e:
            print(f"⚠️ Failed to read {filename}: {e}")

    if not tags_per_file:
        print("❌ No tag files found in output directory.")
    else:
        all_tags = set.union(*tags_per_file.values())
        common_tags = set.intersection(*tags_per_file.values())

        with open(DIFF_OUTPUT_PATH, "w", encoding="utf-8") as diff_file:
            for filename, tags in tags_per_file.items():
                diff = sorted(tags - common_tags)
                diff_file.write(f"=== {filename} ===\n")
                diff_file.write("\n".join(diff))
                diff_file.write("\n\n")

        with open(STRIPPED_PROMPT_OUTPUT_PATH, "w", encoding="utf-8") as stripped_file:
            for filename, tags in tags_per_file.items():
                kept_tags = sorted(tags & common_tags)
                prompt = ", ".join(kept_tags)
                stripped_file.write(f"=== {filename.replace('.txt', '.png')} ===\n")
                stripped_file.write(f"{prompt} [X] Negative prompt: ...\n\n")

        print(f"\n✅ Diff mode complete.")
        print(f"  → {DIFF_OUTPUT_PATH}")
        print(f"  → {STRIPPED_PROMPT_OUTPUT_PATH}")

    exit(0)  # Skip main logic if diff mode is active

# Configuration from arguments
MAX_TAG_EDIT_DISTANCE = 1  # Allow one typo (for example, "sceneri" => "scenery")
MODEL_NAME = "model"
MODEL_URL = "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/model.onnx"
MODEL_DIR = args.model_dir
IMAGES_DIR = args.input
TAGS_DIR = args.output
TAGS_CSV = "selected_tags.csv"
TAGS_CSV_URL = "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/main/selected_tags.csv"

THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
REPLACE_UNDERSCORE = True
TRAILING_COMMA = False
ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Prepare folders
for path in [MODEL_DIR, IMAGES_DIR, TAGS_DIR]:
    os.makedirs(path, exist_ok=True)

# Paths
model_path = os.path.join(MODEL_DIR, MODEL_NAME + ".onnx")
tags_path = os.path.join(MODEL_DIR, TAGS_CSV)

# Download model if needed
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Downloading from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model downloaded.")

# Download tags if needed
if not os.path.exists(tags_path):
    print(f"Tags not found at {tags_path}. Downloading from {TAGS_CSV_URL}...")
    urllib.request.urlretrieve(TAGS_CSV_URL, tags_path)
    print("Tags downloaded.")

# Load model
print(f"Loading model from {model_path}...")
model = ort.InferenceSession(model_path, providers=ORT_PROVIDERS)

# Load tags
tags = []
general_index = None
character_index = None
with open(tags_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if general_index is None and row[2] == "0":
            general_index = reader.line_num - 2
        elif character_index is None and row[2] == "4":
            character_index = reader.line_num - 2
        tag = row[1].replace("_", " ") if REPLACE_UNDERSCORE else row[1]
        tags.append(tag)

print("Model and tags loaded.")

def filter_tags_by_category(tag_list):
    if not args.filter_category:
        return tag_list, set()

    categories = {c.strip() for c in args.filter_category.split(",")}
    allowed_keywords = set()
    all_keywords = set()  # Gathering all keywords from all categories
    for cat in CATEGORY_KEYWORDS:
        all_keywords.update(CATEGORY_KEYWORDS[cat])
        if cat in categories:
            allowed_keywords.update(CATEGORY_KEYWORDS[cat])

    filtered = []
    unknown = set()

    for tag in tag_list:
        normalized_tag = tag.lower().replace(" ", "")
        matched_any = False

        # Check if tag matches any keyword in any category
        for keyword in all_keywords:
            normalized_keyword = keyword.replace(" ", "").lower()
            if normalized_keyword in normalized_tag or distance(normalized_tag, normalized_keyword) <= MAX_TAG_EDIT_DISTANCE:
                matched_any = True
                break

        # If it matches any allowed keyword, add to filtered
        if matched_any:
            for keyword in allowed_keywords:
                normalized_keyword = keyword.replace(" ", "").lower()
                if normalized_keyword in normalized_tag or distance(normalized_tag, normalized_keyword) <= MAX_TAG_EDIT_DISTANCE:
                    filtered.append(tag)
                    break

        # If no match found in any category, add to unknown
        if not matched_any:
            unknown.add(tag)
    return filtered, unknown

# Function to extract prompt metadata and clean Lora tags
def extract_clean_prompt(img):
    try:
        raw = img.info.get("parameters")
        if not raw:
            return None

        # Retrive base  prompt before "Negative prompt:"
        raw_main_prompt = raw.split("Negative prompt:")[0]

        if args.no_clean:
            # Don't clean the prompt if --no-clean is specified
            return raw_main_prompt.strip()

        # Remove <lora:...>
        cleaned = re.sub(r"<lora:[^>]+?>", "", raw_main_prompt)

        # Remove all types of brackets
        cleaned = re.sub(r"[\[\]\(\)\{\}]", "", cleaned)

        return cleaned.strip()
    except Exception as e:
        return None

# Function to tag a single image
def tag_image(image):
    input = model.get_inputs()[0]
    height = input.shape[1]

    # Prepare the image
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image_resized = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image_resized, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
    image_np = np.array(square).astype(np.float32)
    image_np = image_np[:, :, ::-1]  # RGB -> BGR
    image_np = np.expand_dims(image_np, 0)

    # Inference
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image_np})[0]

    result = list(zip(tags, probs[0]))

    # Filter tags
    general = [item for item in result[general_index:character_index] if item[1] > THRESHOLD]
    character = [item for item in result[character_index:] if item[1] > CHARACTER_THRESHOLD]
    all_tags = character + general

    # Create a string of tags
    if TRAILING_COMMA:
        tags_string = ", ".join(tag[0] + "," for tag in all_tags)
    else:
        tags_string = ", ".join(tag[0] for tag in all_tags)

    return tags_string.strip()

# Process all images
image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

print(f"Tagging images in {IMAGES_DIR}...")

all_unknown_tags = set()
for filename in os.listdir(IMAGES_DIR):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(IMAGES_DIR, filename)

        # Open image once
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
            continue

        # Read and clean metadata prompt
        print(f"\n=== {filename} ===")
        cleaned_prompt = None 
        if args.prompt_from_meta:
            cleaned_prompt = extract_clean_prompt(img)
            if cleaned_prompt:
                print("📦 From Metadata:")
                print(cleaned_prompt)
            else:
                print("📦 No metadata found.")

        auto_tags = None
        if args.analys_by_ai:
            auto_tags = tag_image(img)
            print("🤖 From Model:")
            print(auto_tags)

        if cleaned_prompt and auto_tags and auto_tags.strip().lower() == cleaned_prompt.lower():
            print("⚠️ Duplicate tags (prompt matches model tags)")

        # Save tags to file
        tags_filename = os.path.splitext(filename)[0] + ".txt"
        tags_filepath = os.path.join(TAGS_DIR, tags_filename)

        combined_tags_set = set()

        # Add only if -pm is specified
        if args.prompt_from_meta and cleaned_prompt:
            meta_tags_raw = [tag.strip() for tag in cleaned_prompt.split(",") if tag.strip()]
            meta_tags_filtered, meta_unknown = filter_tags_by_category(meta_tags_raw)
            combined_tags_set.update(meta_tags_filtered)
            all_unknown_tags.update(meta_unknown)

        # Add only if -ai is specified
        if args.analys_by_ai and auto_tags:
            auto_tags_raw = [tag.strip() for tag in auto_tags.split(",") if tag.strip()]
            auto_tags_filtered, auto_unknown = filter_tags_by_category(auto_tags_raw)
            combined_tags_set.update(auto_tags_filtered)
            all_unknown_tags.update(auto_unknown)

        # Convert to string
        combined_tags_string = ", ".join(sorted(combined_tags_set))

        print("📦 🔃 🤖 Combined tags:")
        print(combined_tags_string)

        # Save to file
        with open(tags_filepath, "w", encoding="utf-8") as f:
            f.write(combined_tags_string)

print("\nAll done!")

# Save unknown tags
if all_unknown_tags:
    unknown_tags_path = os.path.join(TAGS_DIR, "unknown-tags.txt")
    with open(unknown_tags_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_unknown_tags)))
    print(f"\n⚠️ Unknown tags saved to {unknown_tags_path}")