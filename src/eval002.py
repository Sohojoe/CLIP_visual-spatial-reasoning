
import os
import argparse
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from lxmert_for_classification import LxmertForBinaryClassification
from data002 import ImageTextClassificationDataset
# import clip
from transformers import CLIPProcessor, CLIPModel

def evaluate(data_loader, model):
    model.cuda()
    model.eval()

    correct, total, all_true = 0, 0, 0
    preds = []
    errors = list()
    inital_vs_final_disagreements = 0
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
    # if True:
    #     i=0
    #     data = data_loader[0]
        input_ids, pixel_values, y, captions, filenames = data
        y = y.cuda()
        # if sum(y) != 0:
        #     continue
        with torch.no_grad():
            batch_cap = input_ids.cuda()
            batch_img = pixel_values.cuda()
            outputs = model(input_ids=batch_cap, 
                pixel_values=batch_img)

        # reproduce huggingface webapp
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # raw_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        raw_scores = (100.0 * image_features @ text_features.T)
        # scores = outputs.logits_per_image

        scores = torch.as_tensor([[raw_scores[0][0], raw_scores[0][1]]]).cuda()
        initial_preds_current = torch.argmax(scores, dim=1)
        initial_correct_this_batch = int(sum(y == initial_preds_current))

        # scores[0][0] -= raw_scores[0][2] # subtract bias for the relation
        # scores[0][1] -= raw_scores[0][3] # subtract bias for the relation
        scores = raw_scores
        scores = scores.softmax(dim=-1)
        
        preds_current = torch.argmax(scores, dim=1)
        correct_this_batch = int(sum(y == preds_current))

        if initial_correct_this_batch != correct_this_batch:
            inital_vs_final_disagreements += 1
        correct += correct_this_batch
        preds += preds_current.cpu().numpy().tolist()
        total+=batch_img.shape[0]
        all_true += sum(y)
        ave_score = correct / float(total)
        if correct_this_batch != batch_img.shape[0]:
            errors.append(filenames[0]+' '+str(int(y[0]))+' '+captions[0]+', '+captions[1]+', '+str(float(scores[0][0]))+', '+str(float(scores[0][1])))

        # print errors
        #print (y != torch.argmax(scores, dim=1))

    # TODO: save also predictions
    return ave_score, total, all_true, preds, errors
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--model_url', type=str, default='laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    # parser.add_argument('--checkpoint_path', type=str, required=True)
    # parser.add_argument('--img_feature_path', type=str, required=True)
    # parser.add_argument('--test_json_path', type=str, required=True)
    parser.add_argument('--output_preds', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_url = args.model_url
    # load model
    print("Loading CLIP model...")
    # clip_model_name = 'ViT-L/14' #@param ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px', 'RN101', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64'] {type:'string'}
    # model, clip_preprocess = clip.load(clip_model_name, device=device)
    # model.to(device).eval()
    # tokenizer = clip.tokenize
    # model_url = "openai/clip-vit-base-patch32"
    # model_url = "openai/clip-vit-large-patch14"
    # model_url = "openai/clip-vit-large-patch14-336"
    # model_url = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # model = CLIPModel.from_pretrained(model_url)
    # processor = CLIPProcessor.from_pretrained(model_url)
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(model_url)
    model = AutoModel.from_pretrained(model_url)
    
    # json_path=os.path.join('data', 'splits', 'random', 'test.jsonl')
    # json_path=os.path.join('data', 'splits', 'zeroshot', 'test.jsonl')
    # json_path=os.path.join('data', 'data_files', 'all_vsr_validated_data.jsonl') # 10119 image text pairs
    # json_path=os.path.join('data', 'data_files', 'all_vsr_raw_data.jsonl') # 10119 image text pairs
    json_path=os.path.join('data', 'data_files', 'first_10_percent_vsr_validated_data.jsonl') # 1012 image text pairs
    # json_path=os.path.join('data', 'data_files', 'debug.jsonl') # 10 image text pairs (for debugging)
    # img_path=os.path.join('data', 'images')
    img_path=os.path.join('data', 'trainval2017')
    dataset = ImageTextClassificationDataset(img_path, json_path)
    # for image, caption, label in dataset:
    #     print(caption, 'True' if label else 'False') 
    # load data
    def collate_fn_batch_clip(batch):
        imgs, captions, labels, filenames = zip(*batch)
        # inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        # captions = [captions[0]+' (False)', captions[0]+' (True)']
        # inputs = processor(list(captions[0]), images=list(imgs), return_tensors="pt", padding=True, truncation=True)
        # inputs = processor(list(captions[0]), images=list(imgs), return_tensors="pt", padding=True)
        inputs = processor(captions[0], images=imgs, return_tensors="pt", padding=True)
        labels = torch.tensor(labels)
        # images = torch.tensor(imgs)
        # return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels
        return inputs.input_ids, inputs.pixel_values, labels, list(captions[0]), filenames
        

    # img_feature_path = args.img_feature_path
    # json_path = args.test_json_path
    collate_fn_batch = collate_fn_batch_clip

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=1,
        shuffle=False,
        # persistent_workers=True,
        num_workers=0,)
        # num_workers=16,)
    acc, total, all_true, preds, errors = evaluate(test_loader, model)
    print (f"total example: {total}, # true example: {all_true}, acccuracy: {acc}")

    # save preds
    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")
        
    # for e in errors:
    #     print (e)

