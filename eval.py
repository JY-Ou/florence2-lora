# @title Collect predictions
import torch
import numpy as np
import random
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
import supervision as sv
import re
from dataset import DetectionDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PATTERN = r'(wrinkle|drop|broken-weft)<loc_\d+>'

def extract_classes(dataset: DetectionDataset):
    class_set = set()
    for i in range(len(dataset.dataset)):
        image, data = dataset.dataset[i]
        suffix = data["suffix"]
        classes = re.findall(PATTERN, suffix)
        class_set.update(classes)
    return sorted(class_set)

def main(args):
    # config
    checkpoint = args.checkpoint
    revision = 'refs/pr/6'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset_path
    dataset_location = args.dataset_location
    #model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, revision=revision).to(device)
    # AutoProcessor crop_image_size=768
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, revision=revision)
    print("finish load model")
    # load dataset
    train_dataset = DetectionDataset(
        jsonl_file_path=f"{dataset_location}/train_od1s.jsonl",
        image_directory_path=f"{dataset_location}/train/"
    )
    val_dataset = DetectionDataset(
        jsonl_file_path=f"{dataset_location}/valid_od1s.jsonl",
        image_directory_path=f"{dataset_location}/valid/"
    )

    CLASSES = extract_classes(train_dataset)

    targets = []
    predictions = []

    for i in tqdm(range(len(val_dataset.dataset)), desc="Processing dataset"):
        image, data = val_dataset.dataset[i]
        prefix = data['prefix']
        suffix = data['suffix']

        inputs = processor(text=prefix, images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        prediction = processor.post_process_generation(generated_text, task='<OD>', image_size=image.size)
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, prediction, resolution_wh=image.size)
        prediction = prediction[np.isin(prediction['class_name'], CLASSES)]
        prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
        prediction.confidence = np.ones(len(prediction))

        target = processor.post_process_generation(suffix, task='<OD>', image_size=image.size)
        target = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, target, resolution_wh=image.size)
        target.class_id = np.array([CLASSES.index(class_name) for class_name in target['class_name']])

        targets.append(target)
        predictions.append(prediction)

    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    print(f"map50_95: {mean_average_precision.map50_95:.2f}")
    print(f"map50: {mean_average_precision.map50:.2f}")
    print(f"map75: {mean_average_precision.map75:.2f}")

    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=CLASSES
    )

    confusion_matrix.plot()
    plt.gcf().savefig(f'{checkpoint}/confusion_matrix.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Florence_lora_train_OD", add_help=False)
    # path
    parser.add_argument("--dataset_location", type=str, default="./dataset/fabric_B", help="path to dataset")
    parser.add_argument("--checkpoint", type=str, default='./model/lora_od1_6', help="path to model")

    args = parser.parse_args()

    setup_seed(111)
    main(args)