import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import torch
import  argparse
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
from IPython.display import display


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_bbox(image, data):
    """
    Plot BBox
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    width, height = image.size
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        if(label=='broken-weft'):
            # x1:465 ,y2:990
            if(x1 < round((465/ 1000)* width) and y2 < round((990/ 1000)* height)):
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')
                ax.add_patch(rect)
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        if(label=="drop"):
            #:x1<790
            if(x1 < round((790/ 1000)* width)):
                rect = patches.Rectangle((x1, y1),
                                         x2 - x1,
                                         y2 - y1,
                                         linewidth=2,
                                         edgecolor='lime',
                                         facecolor='none')
                ax.add_patch(rect)
                plt.text(x1,
                         y1,
                         label,
                         color='black',
                         fontsize=8,
                         bbox=dict(facecolor='lime', alpha=1))
        else:
            rect = patches.Rectangle((x1, y1),
                                     x2 - x1,
                                     y2 - y1,
                                     linewidth=2,
                                     edgecolor='lime',
                                     facecolor='none')
            ax.add_patch(rect)
            plt.text(x1,
                     y1,
                     label,
                     color='black',
                     fontsize=8,
                     bbox=dict(facecolor='lime', alpha=1))

    ax.axis('off')
    plt.show()

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


def main(args):
    checkpoint = args.checkpoint
    image_path = args.image_path
    task_prompt = args.task_prompt
    text_input = args.text_input
    image = Image.open(image_path)

    print(f'task:{task_prompt}')
    print("load Florence2")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    print("load Florence2 finish")

    florence2= Florence2(model, processor, device)

    result = florence2(task_prompt, image, text_input)

    match task_prompt:
        case '<CAPTION>' :
            print(f"result: {result}")
        case '<DETAILED_CAPTION>':
            print(result)
        case '<MORE_DETAILED_CAPTION>' :
            print(result[task_prompt])
        case '<OD>':
            print(f"result: {result}")
            plot_bbox(image, result[task_prompt])
        case '<DENSE_REGION_CAPTION>':
            print(result)
            plot_bbox(image, result[task_prompt])
        case '<REGION_PROPOSAL>':
            print(result)
            plot_bbox(image, result[task_prompt])
        case '<CAPTION_TO_PHRASE_GROUNDING>':
            print(result)
            plot_bbox(image, result[task_prompt])
        case '<REFERRING_EXPRESSION_SEGMENTATION>':
            print(result)
            output_image = copy.deepcopy(image)
            draw_polygons(output_image,
                          result[task_prompt],
                          fill_mask=True)
        case '<REGION_TO_SEGMENTATION>':
            print(result)
            output_image = copy.deepcopy(image)
            draw_polygons(output_image,
                          result[task_prompt],
                          fill_mask=True)
        case '<OPEN_VOCABULARY_DETECTION>':
            print(result)
            bbox_result = convert_to_od_format(result[task_prompt])
            print(bbox_result)
            plot_bbox(image, bbox_result)
        case '<REGION_TO_CATEGORY>':
            print(result)
        case '<REGION_TO_DESCRIPTION>':
            print(result)
        case '<OCR>':
            print(result)
        case '<OCR_WITH_REGION>':
            print(result)
            print(result[task_prompt])
            output_image = copy.deepcopy(image)
            draw_ocr_bboxes(output_image, result[task_prompt])
"""
        case '<VISUAL_GROUNDING>':
            print(result)
            bbox_result = convert_to_od_format(result[task_prompt])
            print(bbox_result)
            plot_bbox(image, bbox_result)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Florence_inference", add_help=True)
    # path
    parser.add_argument("--checkpoint", type=str, default="../model/lora_all1_o_1", help="path to model")
    parser.add_argument("--image_path", type=str, default="../test_image/frame19974.jpg", help="path to image")
    # inference wrinkle<loc_452><loc_708><loc_522><loc_735>wrinkle<loc_554><loc_651><loc_723><loc_715>
    parser.add_argument("--task_prompt", type=str, default='<MORE_DETAILED_CAPTION>',
                        help="choose task"
                             " '<CAPTION>' '<DETAILED_CAPTION>' '<MORE_DETAILED_CAPTION>' '<OD>' '<DENSE_REGION_CAPTION>' <REGION_PROPOSAL>'"
                             " '<CAPTION_TO_PHRASE_GROUNDING>' '<REFERRING_EXPRESSION_SEGMENTATION>' '<REGION_TO_SEGMENTATION>'  "
                             " '<OPEN_VOCABULARY_DETECTION>' '<REGION_TO_DESCRIPTION>' "
                             " '<REGION_TO_CATEGORY>' '<OCR>' '<OCR_WITH_REGION>' ")

    parser.add_argument("--text_input", type=str, default=None , help="")
    """ 
    ################** need text input **################
    CAPTION_TO_PHRASE_GROUNDING
    REFERRING_EXPRESSION_SEGMENTATION:in put describe object
    REGION_TO_SEGMENTATION:in put bbox
    OPEN_VOCABULARY_DETECTION
    REGION_TO_DESCRIPTION
    REGION_TO_CATEGORY
    ################** need text input **################
    """
    args = parser.parse_args()
    setup_seed(111)
    main(args)
