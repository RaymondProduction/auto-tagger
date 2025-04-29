import os
import numpy as np
import csv
import sys
import onnxruntime as ort
from PIL import Image

# Configuration
#MODEL_NAME = "wd-v1-4-moat-tagger-v2"  # Without ".onnx"
MODEL_NAME = "wd-vit-tagger-v3" 
MODEL_DIR = "./models"                 # Where model.onnx and selected_tags.csv are located
IMAGES_DIR = "./images"                # Folder with images
TAGS_DIR = IMAGES_DIR                  # Where to write text files with tags
THRESHOLD = 0.35
CHARACTER_THRESHOLD = 0.85
REPLACE_UNDERSCORE = True
TRAILING_COMMA = False
ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# Prepare folders
os.makedirs(TAGS_DIR, exist_ok=True)

# Load the model
print(f"Loading model {MODEL_NAME}...")
model_path = os.path.join(MODEL_DIR, MODEL_NAME + ".onnx")
tags_path = os.path.join(MODEL_DIR, MODEL_NAME + ".csv")
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

# Function to tag a single image
def tag_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input = model.get_inputs()[0]
    height = input.shape[1]

    # Prepare the image
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))
    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    # Inference
    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

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

    return tags_string

# Process all images
image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]

print(f"Tagging images in {IMAGES_DIR}...")

for filename in os.listdir(IMAGES_DIR):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(IMAGES_DIR, filename)
        tags_string = tag_image(image_path)

        tags_filename = os.path.splitext(filename)[0] + ".txt"
        tags_filepath = os.path.join(TAGS_DIR, tags_filename)

        with open(tags_filepath, "w", encoding="utf-8") as f:
            f.write(tags_string.strip())

        print(f"Tagged {filename}: {tags_string}")

print("All done!")