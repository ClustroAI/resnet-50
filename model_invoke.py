import json
import requests
from PIL import Image

import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

def invoke(input_text):
    # Parsing input
    input_json = json.loads(input_text)
    image_url = input_json['image_url']
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = logits.argmax(-1).item()

    # Return the result
    return model.config.id2label[predicted_label]
