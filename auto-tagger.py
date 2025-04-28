import os
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import open_clip


# python -m site
# python3 -m venv myenv
# source myenv/bin/activate   # (Linux, Mac)
# myenv\Scripts\activate.bat  # (Windows)


# Parameters
input_folder = "./path"  # Folder with images
threshold = 0.2  # Threshold for tags

# Loading the model and tags
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# Download wd14 tags
tag_list_path = hf_hub_download(repo_id="SmilingWolf/wd-v1-4-convnext-tagger", filename="selected_tags.csv")

with open(tag_list_path, 'r', encoding='utf-8') as f:
    tag_list = [line.strip().split(',')[0] for line in f.readlines()]

# Prepare text format
def predict_tags(img):
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        img_features = model.encode_image(img)

    img_features /= img_features.norm(dim=-1, keepdim=True)

    text = tokenizer(tag_list)
    text_features = model.encode_text(text)

    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (img_features @ text_features.T).squeeze(0)
    scores = similarity.softmax(dim=0)

    tags_scores = list(zip(tag_list, scores.tolist()))
    return tags_scores

# Process all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(input_folder, filename)
        print(f"Processing: {filename}")

        img = Image.open(image_path).convert('RGB')
        tags_scores = predict_tags(img)

        selected_tags = [tag for tag, score in tags_scores if score > threshold]
        output_path = os.path.join(input_folder, os.path.splitext(filename)[0] + ".txt")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(", ".join(selected_tags))

print("Done!")