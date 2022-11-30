import torch
from PIL import Image
import open_clip
import os

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')


image = '000000419052.jpg'
prompts = ['The pizza is not above the couch.','The pizza is above the couch.']
url = os.path.join('data', 'trainval2017', image)
image = preprocess(Image.open(url)).unsqueeze(0)
text = tokenizer(prompts)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    text_probs = (100.0 * image_features @ text_features.T)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]