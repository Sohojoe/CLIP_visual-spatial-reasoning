#  from https://github.com/openai/CLIP/issues/83
# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from torch import nn
import torch.optim as optim
import os
import json
import wandb
import numpy as np

# BATCH_SIZE must larger than 1

EPOCH = 1024*3
BATCH_SIZE = 384
BASE_MODEL = "ViT-B/32"
# LEARNING_RATE = 5e-5
LEARNING_RATE = 2e-5
EVAL_STEP = 10

# If using GPU then use mixed precision training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu" # for debugging
# Must set jit=False for training
model, preprocess = clip.load(BASE_MODEL, device=device, jit=False)

class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):

        self.image_path = list_image_path
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Image from PIL module
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title


def create_training_dataset(json_path, img_path):
    data_json = []
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            j_line = json.loads(line)
            data_json.append(j_line)
    list_image_path = []
    list_txt = []
    for data in data_json:
        list_image_path.append(os.path.join(img_path, data['image']))
        caption = data['caption']
        if data['label'] == 1:
            caption = caption + ' (True)'
        else:
            caption = caption + ' (False)'
        list_txt.append(caption)
    dataset = image_title_dataset(list_image_path, list_txt)
    return dataset

def create_validation_dataset(json_path, img_path):
    data_json = []
    with open(json_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            j_line = json.loads(line)
            data_json.append(j_line)
    list_image_path = []
    list_txt = []
    for data in data_json:
        list_image_path.append(os.path.join(img_path, data['image']))
        list_image_path.append(os.path.join(img_path, data['image']))
        caption = data['caption']
        if data['label'] == 1:
            caption = caption + ' (True)'
            list_txt.append(caption)
            caption = caption + ' (False)'
            list_txt.append(caption)
        else:
            caption = caption + ' (False)'
            list_txt.append(caption)
            caption = caption + ' (True)'
            list_txt.append(caption)
    dataset = image_title_dataset(list_image_path, list_txt)
    return dataset

# use your own data
# json_path = os.path.join('data', 'splits', 'random', 'dev.jsonl')
json_path = os.path.join('data', 'splits', 'random', 'train.jsonl')
img_path = os.path.join('data', 'trainval2017')
dataset = create_training_dataset(json_path, img_path)
validation_json_path = os.path.join('data', 'splits', 'random', 'test.jsonl')
validation_img_path = os.path.join('data', 'trainval2017')
# validation_dataset = create_validation_dataset(validation_json_path, validation_img_path)
validation_dataset = create_training_dataset(validation_json_path, validation_img_path)

# Define your own dataloader
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_dataset, batch_size=32)

# https://github.com/openai/CLIP/issues/57


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


if device == "cpu":
    model.float()
else:
    # Actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                       betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
config = {
    "learning_rate": optimizer.param_groups[0]['lr'],
    "betas": optimizer.param_groups[0]['betas'],
    "eps": optimizer.param_groups[0]['eps'],
    "weight_decay": optimizer.param_groups[0]['weight_decay'],
    # "learning_rate": 5e-5,
    "epochs": EPOCH,
    "batch_size": BATCH_SIZE,
    "device": device,
    "base_model": BASE_MODEL,
}
# wandb.init(project="clip-visual-spatial-reasoning", entity="sohojoe")
wandb.init(project="clip-visual-spatial-reasoning", config=config)

total_loss = 0
best_loss = np.Inf
best_iter = 0
checkpoint_path = None
step_global = 0
# add your own code to track the training progress.
for epoch in range(EPOCH):
    for batch in train_dataloader:
        # break
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(
            len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth))/2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    # send multiple parms to wandb
    lr = optimizer.param_groups[0]['lr']
    wandb_log = {"loss": total_loss, "lr": lr}

    print("Epoch: {:04d}, Loss: {}".format(epoch, total_loss))

    
    step_global += 1

    # save model

    # validate model
    if step_global % EVAL_STEP == 0:
        # model_path = os.path.join('model_checkpoint', 'model_1850.pt')
        # model_path = os.path.join('model_checkpoint', 'model_0961.pt')
        # checkpoint = torch.load(model_path)
        model.eval()

        # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = model.visual.input_resolution #default is 224
        # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
        # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

        # model.load_state_dict(checkpoint['model_state_dict'])
        num_correct = 0
        num_errors = 0
        validation_losses = []
        for batch in validation_dataloader:
            # optimizer.zero_grad()

            images, texts = batch

            images = images.to(device)
            texts = texts.to(device)

            with torch.no_grad():
                logits_per_image, logits_per_text = model(images, texts)

                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device)

                total_loss = (loss_img(logits_per_image, ground_truth) +
                    loss_txt(logits_per_text, ground_truth))/2
                validation_losses.append(total_loss.item())


            # i = 0
            # while i < len(logits_per_image):
            #     if torch.argmax(logits_per_image[i]) == torch.argmax(logits_per_text[i]):
            #         num_correct += 1
            #     else:
            #         num_errors += 1
            #     i += 2
        model.train()
        # acc = float(num_correct) / float(num_correct + num_errors)
        acc = np.mean(validation_losses)
        print (f"====== evaliuate ======")
        print (f"epoch: {epoch}, global step: {step_global}, val performance: {acc}")
        print (f"=======================")
        wandb_log["eval_acc"] = acc
        if acc < best_loss:
            best_iter = epoch+1
            best_loss = acc

            checkpoint_dir = os.path.join(f"model_checkpoint")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            old_checkpoint_path = checkpoint_path
            checkpoint_path = os.path.join(checkpoint_dir, 'model_{:04d}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                }, checkpoint_path) #just change to your preferred folder/filename
            if old_checkpoint_path is not None:
                os.remove(old_checkpoint_path)
            print (f"===== best model saved! =======")
    wandb.log(wandb_log)




# # old
# from eval000 import evaluate
# from data000 import ImageTextClassificationDataset
# import torch
# import clip
# import os

# from transformers import AutoProcessor, AutoModel


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu" # force CPU
# val_json_path = os.path.join('data', 'splits', 'random', 'test.jsonl')
# img_feature_path = os.path.join('data', 'trainval2017')
# model_path = os.path.join('model_checkpoint', 'model_1850.pt')

# # processor = AutoProcessor.from_pretrained(model_url)
# # model = AutoModel.from_pretrained(model_url)
# model, processor = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
# checkpoint = torch.load(model_path)

# # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# # checkpoint['model_state_dict']["input_resolution"] = model.visual.input_resolution #default is 224
# # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

# model.load_state_dict(checkpoint['model_state_dict'])



# def collate_fn_batch_clip(batch):
#     imgs, captions, labels, filenames = zip(*batch)
#     # inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
#     # captions = [captions[0]+' (False)', captions[0]+' (True)']
#     # inputs = processor(list(captions[0]), images=list(imgs), return_tensors="pt", padding=True, truncation=True)
#     # inputs = processor(list(captions[0]), images=list(imgs), return_tensors="pt", padding=True)
#     inputs = processor(captions[0], images=imgs, return_tensors="pt", padding=True)
#     labels = torch.tensor(labels)
#     # images = torch.tensor(imgs)
#     # return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels
#     return inputs.input_ids, inputs.pixel_values, labels, list(captions[0]), filenames




# dataset_val = ImageTextClassificationDataset(img_feature_path, val_json_path)



# val_loader = torch.utils.data.DataLoader(
#     dataset_val,
#     collate_fn = collate_fn_batch_clip,
#     batch_size=64,
#     shuffle=False,
#     num_workers=1,)

# if __name__ == "__main__":

#     # evaluate and save
#     # if step_global % args.eval_step == 0:
#     # evaluate
#     acc, _, _, _ = evaluate(val_loader, model, device)
#     step_global = 1850
#     epoch = 1850
#     print (f"====== evaliuate ======")
#     print (f"epoch: {epoch}, global step: {step_global}, val performance: {acc}")
#     print (f"=======================")
#     # wandb.log({"eval_acc": acc})
#     # if val_best_score < acc:
#     #     val_best_score = acc
#     # else:
#     #     continue
#     # checkpoint_dir = os.path.join(args.output_dir, f"best_checkpoint")
#     # if not os.path.exists(checkpoint_dir):
#     #     os.makedirs(checkpoint_dir)
#     # if model_type == "visualbert":
#     #     model.save_pretrained(checkpoint_dir)
#     # elif model_type == "lxmert":
#     #     model.lxmert.save_pretrained(checkpoint_dir)
#     # elif model_type == "vilt":
#     #     processor.save_pretrained(checkpoint_dir)
#     #     model.save_pretrained(checkpoint_dir)
#     # print (f"===== best model saved! =======")