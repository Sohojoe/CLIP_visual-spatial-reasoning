
import os
import cv2
import glob
import json
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, model_type="clip", vilt_processor=None): 
        
        
        self.model_type = model_type
        self.img_path = img_path

        self.imgs = {}
        img_paths = glob.glob(img_path+"/*.jpg")
        print (f"load images...")
        for img_path in tqdm(img_paths):
            img_name = img_path.split(os.sep)[-1]
            #tmp = Image.open(img_path)
            #keep = tmp.copy()
            #tmp.close()
            self.imgs[img_name] = cv2.imread(img_path)
       
        self.data_json = []
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data_json.append(j_line)
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        stopwords = ['the', 'is']
        relation = data_point["relation"]
        subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
        subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
        subject_a  = [word for word in subject_a.split() if word.lower() not in stopwords][0].strip(".")
        subject_b  = [word for word in subject_b.split() if word.lower() not in stopwords][0].strip(".")
        captions = [
            subject_b + ' ' + relation + ' ' + subject_a, # false case first
            subject_a + ' ' + relation + ' ' + subject_b, # true case second
        ]
        return self.imgs[data_point["image"]], captions, data_point["label"]
        # return self.imgs[data_point["image"]], data_point["caption"], data_point["label"]

    def __len__(self):
        return len(self.data_json)


