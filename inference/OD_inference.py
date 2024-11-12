from PIL import Image
import  torch
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bbox(image, data):
   # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()

checkpoint="../model/lora_model"
EXAMPLE_IMAGE_PATH="../test_image/5 (35).jpg"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

print("Applying the OD")
image = Image.open(EXAMPLE_IMAGE_PATH)
task = "<OD>"
text = "<OD>"

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt"
).to(device)
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    num_beams=3)
generated_text = processor.batch_decode(
    generated_ids, skip_special_tokens=False)[0]
response = processor.post_process_generation(
    generated_text,
    task=task,
    image_size=image.size)
"""
detections = sv.Detections.from_lmm(
    sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

bounding_box_annotator = sv.BoundingBoxAnnotator(
    color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX)

image = bounding_box_annotator.annotate(image, detections)
image = label_annotator.annotate(image, detections)
image.thumbnail((600, 600))

cv.imshow("image", image)
"""

print(response)
plot_bbox(image, response['<OD>'])