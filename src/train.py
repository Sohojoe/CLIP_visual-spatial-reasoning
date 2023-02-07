#  from https://github.com/openai/CLIP/issues/83

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
import argparse
import sys

class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):

        self.image_path = list_image_path
        self.caption_tokens = clip.tokenize(list_txt)

        self.negative_caption_tokens = {}

        for i in range (len(list_txt)):
            caption = list_txt[i]
            if caption.endswith('(True)'):
                caption = caption.replace('(True)', '(False)')
            else:
                caption = caption.replace('(False)', '(True)')
            self.negative_caption_tokens[str(self.caption_tokens[i].numpy())] = clip.tokenize(caption)


    def __len__(self):
        return len(self.caption_tokens)

    def __getitem__(self, idx):
        # Image from PIL module
        image = preprocess(Image.open(self.image_path[idx]))
        caption_token = self.caption_tokens[idx]
        return image, caption_token

    def get_inverse_prompt_token(self, caption_token):
        return self.negative_caption_tokens[str(caption_token.numpy())]


def create_dataset(json_path, img_path):
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

# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def train(args, device, model, train_dataloader, test_dataloader, test_dataset):

    if device == "cpu":
        model.float()
    else:
        # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                        betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    
    wandb_name = "clip-visual-spatial-reasoning"
    if args.debugging:
        wandb_name = "debugging-" + wandb_name
    wandb.init(project=wandb_name, config=args)

    total_loss = 0
    best_accuracy = -np.Inf
    best_iter = 0
    checkpoint_path = None
    step_global = 0
    total_epoch = 1 if args.only_evaluate else args.epoch
    for epoch in range(total_epoch):
        # send multiple parms to wandb
        wandb_log = {}

        # validate epoch
        validate = (args.only_evaluate or step_global % args.eval_step == 0) and not args.skip_evaluate
        if validate:
            model.eval()
            num_correct = 0
            num_errors = 0
            test_losses = []
            for batch in test_dataloader:
                # optimizer.zero_grad()

                images, texts = batch

                for i in range(len(images)):
                    caption_token = texts[i]
                    negative_caption_token = test_dataset.get_inverse_prompt_token(caption_token)
                    # texts.append(negative_caption_token)
                    # append negative_caption_token (shape is [1,77]) to texts (shape is [32,77])
                    texts = torch.cat((texts, negative_caption_token), 0)


                images = images.to(device)
                texts = texts.to(device)

                with torch.no_grad():
                    logits_per_image, logits_per_text = model(images, texts)

                # calculate loss
                len_for_loss = int(logits_per_image.shape[1]/2)
                logits_per_image_for_loss = logits_per_image[:,:len_for_loss]
                logits_per_text_for_loss = logits_per_text[:len_for_loss,:]
                ground_truth = torch.arange(
                    len(images), dtype=torch.long, device=device)
                total_loss = (loss_img(logits_per_image_for_loss, ground_truth) +
                    loss_txt(logits_per_text_for_loss, ground_truth))/2
                test_losses.append(total_loss.item())

                # calculate accuracy
                for i in range(len(logits_per_image)):
                    possitive = logits_per_text[i,i]
                    negative = logits_per_text[i+len(logits_per_image),i]
                    if possitive > negative:
                        num_correct += 1
                    else:
                        num_errors += 1
            model.train()
            test_accuracy = float(num_correct) / float(num_correct + num_errors)
            test_loss = np.mean(test_losses)
            print (f"====== evaliuate ======")
            print (f"epoch: {epoch}, global step: {step_global}, test_loss: {test_loss}, test_accuracy: {test_accuracy}")
            print (f"=======================")
            wandb_log["test_loss"] = test_loss
            wandb_log["test_accuracy"] = test_accuracy
            if test_accuracy > best_accuracy:
                best_iter = epoch+1
                best_accuracy = test_accuracy

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
        
        # training epoch
        if not args.only_evaluate:
            for batch in train_dataloader:
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

        wandb_log["loss"] = total_loss
        wandb_log["lr"] = optimizer.param_groups[0]['lr']
        print("Epoch: {:04d}, Loss: {}".format(epoch, total_loss))

        # save model
        # TODO save model every n epoch

        step_global += 1
        wandb.log(wandb_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--base_model', type=str, default="ViT-B/32")
    parser.add_argument('--betas', type=float, default=(0.9, 0.98))
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--only_evaluate', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, required=False)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--freeze_visual', type=bool, default=True)
    parser.add_argument('--skip_evaluate', type=bool, default=False)
    parser.add_argument('--debugging', type=bool, default=False)

    args = parser.parse_args()
    

    # debug settings
    # args.learning_rate = 4e-6
    # args.eval_step = 1
    args.base_model = "ViT-L/14@336px"
    args.batch_size = 27
    # args.skip_evaluate = True
    # args.only_evaluate = True
    # args.checkpoint_path = os.path.join('model_checkpoint', 'model_0005 - lr1e-4.pt')
    # args.checkpoint_path = os.path.join('model_checkpoint', 'model_1850.pt')
    # args.checkpoint_path = os.path.join('model_checkpoint', 'model_0961.pt')
    # args.device = "cpu"

    if sys._getframe().f_back:
        args.debugging = True

    torch.manual_seed(args.random_seed)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Must set jit=False for training
    model, preprocess = clip.load(args.base_model, device=device, jit=False)

    # handle loading model from checkpoint
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path)
        # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, 
        # # if you set context_length to 100 since your string is very long during training, 
        # # then assign 100 to checkpoint['model_state_dict']["context_length"] 
        # checkpoint['model_state_dict']["input_resolution"] = model.visual.input_resolution #default is 224
        # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
        # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 
        model.load_state_dict(checkpoint['model_state_dict'])

    # freeze visual model
    if args.freeze_visual:
        for k in model.visual.transformer.parameters():  
            k.requires_grad=False

    # use your own data
    # json_path = os.path.join('data', 'splits', 'random', 'dev.jsonl')
    json_path = os.path.join('data', 'splits', 'random', 'train.jsonl')
    img_path = os.path.join('data', 'trainval2017')
    dataset = create_dataset(json_path, img_path)
    test_json_path = os.path.join('data', 'splits', 'random', 'test.jsonl')
    test_img_path = os.path.join('data', 'trainval2017')
    test_dataset = create_dataset(test_json_path, test_img_path)

    # Define your own dataloader
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    train(args, device, model, train_dataloader, test_dataloader, test_dataset)
