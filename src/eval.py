
import os
import argparse
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from lxmert_for_classification import LxmertForBinaryClassification
from data import ImageTextClassificationDataset
# import clip
from transformers import CLIPProcessor, CLIPModel

def evaluate(data_loader, model, model_type="clip"):
    model.cuda()
    model.eval()

    correct, total, all_true = 0, 0, 0
    preds = []
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
    # if True:
    #     i=0
    #     data = data_loader[0]
        if model_type == "visualbert":
            batch_cap, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            img_attention_mask = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            img_token_type_ids = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            batch_inputs.update({
                "visual_embeds": batch_img.cuda(),
                "visual_token_type_ids": img_token_type_ids.cuda(),
                "visual_attention_mask": img_attention_mask.cuda(),
                })

        elif  model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
                })
        elif model_type == "clip":
            input_ids, pixel_values, y = data
        y = y.cuda()
        with torch.no_grad():
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif model_type == "clip":
                batch_cap = input_ids.cuda()
                batch_img = pixel_values.cuda()
                outputs = model(input_ids=batch_cap, 
                        pixel_values=batch_img)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        scores = outputs.logits_per_image
        
        preds_current = torch.argmax(scores, dim=1)
        correct_this_batch = int(sum(y == preds_current))
        correct += correct_this_batch
        preds += preds_current.cpu().numpy().tolist()
        total+=batch_img.shape[0]
        all_true += sum(y)
        ave_score = correct / float(total)
        debug = 'sd'

        # print errors
        #print (y != torch.argmax(scores, dim=1))

    # TODO: save also predictions
    return ave_score, total, all_true, preds
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--model_type', type=str, default='clip')
    # parser.add_argument('--checkpoint_path', type=str, required=True)
    # parser.add_argument('--img_feature_path', type=str, required=True)
    # parser.add_argument('--test_json_path', type=str, required=True)
    parser.add_argument('--output_preds', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = args.model_type
    # load model
    if model_type == "clip":
        print("Loading CLIP model...")
        # clip_model_name = 'ViT-L/14' #@param ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64'] {type:'string'}
        # model, clip_preprocess = clip.load(clip_model_name, device=device)
        # model.to(device).eval()
        # tokenizer = clip.tokenize
        # clip_str = "openai/clip-vit-base-patch32"
        # clip_str = "openai/clip-vit-large-patch14"
        clip_str = "openai/clip-vit-large-patch14-336"
        model = CLIPModel.from_pretrained(clip_str)
        processor = CLIPProcessor.from_pretrained(clip_str)
    
    json_path=os.path.join('data', 'splits', 'random', 'test.jsonl')
    # json_path=os.path.join('data', 'splits', 'zeroshot', 'test.jsonl')
    img_path=os.path.join('data', 'images')
    # img_path=os.path.join('data', 'trainval2017')
    dataset = ImageTextClassificationDataset(img_path, json_path, model_type=model_type)
    # for image, caption, label in dataset:
    #     print(caption, 'True' if label else 'False') 
    # load data
    def collate_fn_batch_clip(batch):
        imgs, captions, labels = zip(*batch)
        # inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        # captions = [captions[0]+' (False)', captions[0]+' (True)']
        inputs = processor(list(captions[0]), images=list(imgs), return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        # images = torch.tensor(imgs)
        # return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels
        return inputs.input_ids, inputs.pixel_values, labels
        

    # img_feature_path = args.img_feature_path
    # json_path = args.test_json_path
    if model_type == "clip":
        collate_fn_batch = collate_fn_batch_clip

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=1,
        shuffle=False,
        # persistent_workers=True,
        num_workers=0,)
        # num_workers=16,)
    acc, total, all_true, preds = evaluate(test_loader, model, model_type=model_type)
    print (f"total example: {total}, # true example: {all_true}, acccuracy: {acc}")

    # save preds
    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")
        


