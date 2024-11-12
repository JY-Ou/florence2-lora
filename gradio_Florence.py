import gradio as gr
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import torch
import  argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

class Florence2:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    def __call__(self, task_prompt, image, text_input=None):
        """
        Calling the Microsoft Florence2 model
        """
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        decoded_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(decoded_output)

        generated_text = self.processor.batch_decode(generated_ids,
                                                skip_special_tokens=False)[0]
        print(generated_text)
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height))

        return parsed_answer

parser = argparse.ArgumentParser("Florence_inference", add_help=True)
# path
parser.add_argument("--checkpoint", type=str, default="../model/lora_all1_o_1", help="path to model")
args = parser.parse_args()
checkpoint = args.checkpoint
print("load Florence2")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
print("load Florence2 finish")
florence2= Florence2(model, processor, device)


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.
    Parameters:
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.
    Returns:
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {'bboxes': bboxes, 'labels': labels}

    return od_results

def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    draw = ImageDraw.Draw(image)
    scale = 1

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = "lime"
        fill_color = "lime" if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def draw_ocr_bboxes(image, prediction):
    """
    Draw OCR BBox
    """
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']

    for box, label in zip(bboxes, labels):
        color = 'lime'
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=4, outline=color)
        draw.text((new_box[0] + 8, new_box[1] + 2),
                  "{}".format(label),
                  align="right",
                  fill=color)

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def process_data(input_img,data):
    # Process data
    # Create a figure with the correct aspect ratio
    fig, ax = plt.subplots(figsize=(input_img.width / 100, input_img.height / 100))  # Adjust the divisor to match your desired dpi

    width, height = input_img.size
    defect=[]
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        if(label=='broken-weft'):
            # x1:465 ,y2:990
            if(x1 < round((465/ 1000)* width) and y2 < round((990/ 1000)* height)):
                defect.append('broken-weft')
                # Draw bounding box
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')

                ax.add_patch(rect)

                # Add label text
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        if (label == "drop"):
            #:x1<790
            if (x1 < round((790 / 1000) * width)):
                defect.append("drop")
                # Draw bounding box
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')

                ax.add_patch(rect)

                # Add label text
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        else:
            defect.append(label)
            # Draw bounding box
            rect = patches.Rectangle((x1, y1),
                                     x2 - x1,
                                     y2 - y1,
                                     linewidth=2,
                                     edgecolor='lime',
                                     facecolor='none')

            ax.add_patch(rect)

            # Add label text
            plt.text(x1,
                     y1,
                     label,
                     color='black',
                     fontsize=8,
                     bbox=dict(facecolor='lime', alpha=1))
    # Turn off axis
    ax.axis('off')

    # Get the figure canvas
    fig.canvas.draw()

    # Convert the figure to a numpy array
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return defect, Image.fromarray(img_data)




def task_selector(task, input_img, input_text=None):
    if task == "Object Detection":
        task_prompt='<OD>'
        result = florence2(task_prompt, input_img, input_text)
        output_img, output_text=process_data(input_img,result[task_prompt])
    elif task == "Caption":
        task_prompt = '<CAPTION>'
        output_img = draw_boxes(input_img)
        output_text = ""
    elif task == "Detailed Caption":
        task_prompt = '<DETAILED_CAPTION>'
        output_img = draw_boxes(input_img)
        output_text = ""
    elif task == "More Detailed Caption":
        task_prompt = '<MORE_DETAILED_CAPTION>'
        output_img = draw_boxes(input_img)
        output_text = ""
    elif task == "Referrinig Expression Segmentation":
        task_prompt = '<REFERRING_EXPRESSION_SEGMENTATION>'
        output_img = apply_segmentation(input_img)
        output_text = ""
    elif task == "Region To Segmentation":
        task_prompt = '<REGION_TO_SEGMENTATION>'
        output_img = apply_segmentation(input_img)
        output_text = ""
    elif task == "OCR":
        task_prompt = '<OCR>'
        output_img = None
        output_text = process_text(input_text)
    elif task == "OCR With Region":
        task_prompt = '<OCR_WITH_REGION>'
        output_img = apply_segmentation(input_img)
        output_text = ""
    return output_img, output_text

interface = gr.Interface(
    fn=task_selector,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Dropdown(["Object Detection", "Referrinig Expression Segmentation", "Region To Segmentation", "Captioning", "Detailed Caption", "More Detailed Caption", "OCR", "OCR With Region"], label="Select Task"),
        gr.Textbox(label="Input Text")
    ],
    outputs=[
        gr.Image(label="Output Image"),
        gr.Textbox(label="Output Text"),
    ],
    title="Image and Text Processing",
    description="Select a task and upload an image or input text"
)

interface.launch()

"""
def process_data(input_img,data):
    # Process data
    # Create a figure with the correct aspect ratio
    dpi = 30
    width, height = input_img.size
    fig, ax = plt.subplots(figsize=(width/10 , height/10 ))  # Adjust the divisor to match your desired dpi

    # Display the image
    ax.imshow(input_img)
    defect=[]
    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        if(label=='broken-weft'):
            # x1:465 ,y2:990
            if(x1 < round((465/ 1000)* width) and y2 < round((990/ 1000)* height)):
                defect.append('broken-weft')
                # Draw bounding box
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')

                ax.add_patch(rect)

                # Add label text
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        if (label == "drop"):
            #:x1<790
            if (x1 < round((790 / 1000) * width)):
                defect.append("drop")
                # Draw bounding box
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')

                ax.add_patch(rect)

                # Add label text
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        else:
            defect.append(label)
            # Draw bounding box
            rect = patches.Rectangle((x1, y1),
                                     x2 - x1,
                                     y2 - y1,
                                     linewidth=2,
                                     edgecolor='lime',
                                     facecolor='none')

            ax.add_patch(rect)

            # Add label text
            plt.text(x1,
                     y1,
                     label,
                     color='black',
                     fontsize=8,
                     bbox=dict(facecolor='lime', alpha=1))
    # Turn off axis
    ax.axis('off')

    # Get the figure canvas
    fig.canvas.draw()
    plt.show()
    # Convert the figure to a numpy array
    img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Close the figure
    plt.close(fig)

    return Image.fromarray(img_data), defect
"""