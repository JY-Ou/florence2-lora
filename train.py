import os
from functools import partial
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from dataset import DetectionDataset
import logging
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_loss_from_file(file_path, save_path, plt_config):
    epochs, train_losses, val_losses = [], [], []
    with open(file_path, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            epoch, train_loss, val_loss = line.strip().split(',')
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='train_loss')
    plt.plot(epochs, val_losses, 'r-', label='val_loss')
    plt.title('train and val curve')
    plt.xlabel(f"Epoch  r:{plt_config['lora_rank']} alpha:{plt_config['lora_alpha']} "
               f"lora_dropout:{plt_config['lora_dropout']}, lr:{plt_config['learning_rate']}")
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{save_path}/loss_curve_from_file.png')

def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

def train_model(train_loader, val_loader, model, processor, device, logger, save_path,epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # render_inference_results(peft_model, val_loader.dataset, 6)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch + 1} training loss: {avg_train_loss}")
        print(f"Average Training Loss: {avg_train_loss}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(device)

                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            logger.info(f"Epoch {epoch + 1} validation loss: {avg_val_loss}")
            print(f"Average Validation Loss: {avg_val_loss}")

           # render_inference_results(peft_model, val_loader.dataset, 6)

        # 保存每个epoch的模型
        output_dir = f"{save_path}/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir,use_safetensors=False)

    return train_losses, val_losses



def main(args):
    # config
    batch_size = args.batch_size  #原 6
    num_workers = args.num_workers
    epochs = args.epochs # 10
    lr = args.learning_rate
    # plt
    plt_config = {
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'learning_rate': lr
    }

    checkpoint = args.checkpoint
    revision = 'refs/pr/6'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset_path
    dataset_location = args.dataset_location
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')

    #lora_config
    config = LoraConfig(
        use_mora=True,
        mora_type=6,
        r=args.lora_rank,
        # lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "out_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2", "fc1"],
        task_type="CAUSAL_LM",
        lora_dropout=args.lora_dropout,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision=revision
    )

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f'Script description: {parser.description}')
    logger.info("Model lora configuration: {}".format(config))
    logger.info("Training parameters: batch size = {}, epochs = {}, learning rate = {}".format(batch_size, epochs, lr))

    #model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, revision=revision).to(device)
    # AutoProcessor crop_image_size=768
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=True, revision=revision)

    peft_model = get_peft_model(model, config).to(device)
    trainable_parameters  = peft_model.print_trainable_parameters()
    # logger.info("Trainable parameters: {}".format(trainable_parameters))
    print(trainable_parameters)

    # load dataset
    train_dataset = DetectionDataset(
        jsonl_file_path=f"{dataset_location}/train_od.jsonl",
        image_directory_path=f"{dataset_location}/train/"
    )
    val_dataset = DetectionDataset(
        jsonl_file_path=f"{dataset_location}/valid_od.jsonl",
        image_directory_path=f"{dataset_location}/valid/"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              collate_fn=partial(collate_fn, processor=processor, device=device),
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            collate_fn=partial(collate_fn, processor=processor, device=device),
                            num_workers=num_workers)

    train_losses, val_losses = train_model(train_loader, val_loader, peft_model, processor, device, logger, save_path, epochs, lr)

        # 保存损失值到txt文件
    with open(f'{save_path}/loss_history.txt', 'w') as f:
        f.write("Epoch,Train Loss,Validation Loss\n")
        for i in range(epochs):
            f.write(f"{i + 1},{train_losses[i]},{val_losses[i]}\n")
    plot_loss_from_file(f'{save_path}/loss_history.txt', save_path, plt_config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Florence_lora_train", add_help=False)
    # path
    parser.add_argument("--dataset_location", type=str, default="./dataset/fabric", help="path to dataset")
    parser.add_argument("--save_path", type=str, default='./output/model_checkpoints', help='path to save log')
    parser.add_argument("--checkpoint", type=str, default='./model/Florence', help="path to model")
    # hyper-parameter
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
    # lora
    parser.add_argument("--lora_rank", type=int, default=8, help="lora_rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="learning lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="lora_dropout")
    args = parser.parse_args()

    setup_seed(111)
    main(args)


