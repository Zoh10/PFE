import ultralytics
import paddleocr  # Assuming paddleocr is used for OCR
from PIL import Image
import numpy as np
import json
import flair.data
import flair.models
import re

# Define classes for potential invoice elements
CLASSES = ['details', 'logo', 'receiver', 'sender', 'table', 'total']

# Function to visualize and extract specific image region
def visualize(x, y, h, w, image):
    return image.crop((x, y, x + w, y + h))

# Function to remove duplicate bounding boxes
def cleaner(table):
    seen = set()
    cleaned_table = []
    for box in table:
        if tuple(box) not in seen:
            seen.add(tuple(box))
            cleaned_table.append(box)
    return cleaned_table

# Function to format extracted text
def format_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.replace('@', '_')  # Replace '@' with '_'
    text = re.sub(r"(?P<=\w)([A-Z])", r" \g", text).strip()  # Separate concatenated words
    return text

# Load pre-trained models (outside loops for one-time execution)
model = ultralytics.Detector()  # Assuming ultralytics for object detection
reader = paddleocr.PaddleOCR(lang='eng')  # Assuming English language
NER_text = flair.models.TextClassifier.load('your_ner_model_for_text')  # Replace with your NER model
NER_tab = flair.models.TextClassifier.load('your_ner_model_for_table')  # Replace with your NER model

def parse_invoice(image_path):
    start = time.time()  # Start timer (optional)

    # Load image
    image = Image.open(image_path)
    image = np.array(image)

    # Predict bounding boxes
    results = model(image)

    # Extract bounding boxes for relevant elements
    table = []
    for element in results.pandas().xyxy[0]:
        if element['name'] in CLASSES:
            table.append([int(x) for x in element[['xmin', 'ymin', 'xmax', 'ymax']]])

    # Remove duplicate bounding boxes
    table = cleaner(table)

    # Data structures to store extracted information
    final_table = [['Description', 'QTE', 'PU', 'PT']]  # Assuming table headers
    final_total = []
    toSend = {}

    # Process each element
    for box in table:
        image_patch = visualize(box[0], box[1], box[3] - box[1], box[2] - box[0], image)

        # Extract text using OCR
        text = reader(image_patch)
        text = ''.join([out['words'] for out in text])

        if box[0] in [b[0] for b in results.pandas().xyxy[0] if b['name'] == 'table']:
            # Identify table headers and data columns using pre-trained NER model
            flair_data = flair.data.Sentence(text)
            NER_tab.predict(flair_data)
            headers = []
            for entity in flair_data.entities:
                if entity.label_ == 'HEADER':
                    headers.append(entity.text)
            # Update header indices based on identified headers
            for i, header in enumerate(headers):
                if header in final_table[0]:
                    final_table[0][i] = header
            # Extract table data based on headers
            for line in text.splitlines():
                data = [format_text(w) for w in line.split()]
                if len(data) == len(final_table[0]):
                    final_table.append(data)

        elif box[0] in [b[0] for b in results.pandas().xyxy[0] if b['name'] == 'total']:
            # Extract total values using OCR and potentially NER
            final_total.append(format_text(text))

        elif box[0] in [b[0] for b in results.pandas().xyxy[0] if b['name'] == 'logo']:
            # Store logo bounding box coordinates
                        toSend['logo'] = box

        else:
            # Extract text and perform NER for sender, receiver, etc.
            flair_data = flair.data.Sentence(text)
            NER_text.predict(flair_data)
            for entity in flair_data.entities:
                if entity.label_ == 'SENDER' or entity.label_ == 'RECEIVER' or entity.label_ == 'DETAILS':  # Adjust labels based on your NER model
                    toSend[entity.label_.lower()] = format_text(entity.text)

    # Save extracted information (replace with your saving logic)
    # ...

    # Stop timer and print processing time (optional)
    # print(f"Processing time: {time.time() - start}")

    return toSend

# Example usage
image_path = "path/to/your/invoice.jpg"
extracted_data = parse_invoice(image_path)
print(extracted_data)

