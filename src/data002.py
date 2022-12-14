
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


# Adjacency
# adjacent to, alongside, at the side of, at the right side of, at the left side of,
# attached to, at the back of, ahead of, against, at the edge of
# Directional
# off, past, toward, down, away from, along, around, from∗, into, across,
# across from, through∗, down from
# Orientation
# facing, facing away from, parallel to, perpendicular to
# Proximity
# by, close to, near, far from, far away from
# Topological
# connected to, detached from, has as a part, part of, contains, within, at, on, in,
# with, surrounding, among, consists of, out of, between, inside, outside, touching
# Unallocated
# beyond,next to, opposite to, among, enclosed by

negate = {
    # Adjacency
    "adjacent to": "nonadjacent to", 
    "alongside": "away from", 
    "at the side of": "away from", 
    "at the right side of": "at the left side of", 
    "at the left side of": "at the right side of",
    "attached to": "disconnect from", 
    "at the back of": "at the front of", 
    "ahead of": "not ahead of", 
    "against": "away from", 
    "at the edge of": "far from the edge of", 
    # Directional
    "off": "on", 
    "past": "before", 
    "toward": "away from", 
    "down": "up", 
    "away from": "not away from", 
    "along": "not along", 
    "around": "not around", 
    "into": "not into", 
    "across": "not accross",
    "across from": "not across from", 
    "down from": "up from", 
    # Orientation
    "facing": "facing away from", 
    "facing away from": "facing", 
    "parallel to": "perpendicular to", 
    "perpendicular to": "parallel to", 
    # Proximity
    "by": "far away from", 
    "close to": "far from", 
    "near": "far from", 
    "far from": "close to", 
    "far away from": "by", 
    # Topological
    "connected to": "detached from", 
    "detached from": "connected to", 
    "has as a part": "does not have a part", 
    "part of": "not part of", 
    "contains": "does not contain", 
    "within": "outside of", 
    "at": "not at", 
    "on": "not on", 
    "in": "not in",
    "with": "not with", 
    "surrounding": "not surrounding", 
    "among": "not among", 
    "consists of": "does not consists of", 
    "out of": "not out of", 
    "between": "not between", 
    "inside": "outside", 
    "outside": "inside", 
    "touching": "not touching",
    # Unallocated
    "beyond": "inside",
    "next to": "far from", 
    "opposite to": "not opposite to", 
    "enclosed by": "not enclosed by", 
    # missing
    "above": "below",
    "below": "above",
    "behind": "in front of",
    "on top of": "not on top of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
}

# load image in real time version
class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_path, json_path, vilt_processor=None): 
        self.img_path = img_path

        self.imgs = {}
        # img_paths = glob.glob(img_path+"/*.jpg")
        # print (f"load images...")
        # for img_path in tqdm(img_paths):
        #     img_name = img_path.split(os.sep)[-1]
        #     #tmp = Image.open(img_path)
        #     #keep = tmp.copy()
        #     #tmp.close()
        #     self.imgs[img_name] = cv2.imread(img_path)
       
        self.data_json = []
        with open(json_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                j_line = json.loads(line)
                self.data_json.append(j_line)
            
    def __getitem__(self, idx):
        data_point = self.data_json[idx]
        relation = data_point["relation"]
        negative_relation = negate[relation]
        full_subject_a = data_point["caption"].split(' ' + relation + ' ')[0]
        full_subject_b = data_point["caption"].split(' ' + relation + ' ')[1]
        full_subject_a = full_subject_a.strip(".").lower()
        full_subject_b = full_subject_b.strip(".").lower()
        # stopwords = ['the']
        stopwords = []
        # remove stop words
        full_subject_a  = [word for word in full_subject_a.split() if word.lower() not in stopwords]
        full_subject_b  = [word for word in full_subject_b.split() if word.lower() not in stopwords]
        
        # wrap relation in quotes
        # relation, negative_relation = '"'+relation+'"', '"'+negative_relation+'"'
        
        # move modifier to the predicate
        modifier = [word for word in full_subject_a if word in ['is']]
        if len(modifier) > 0:
            full_subject_a.remove(modifier[0])
            relation = modifier[0] + " " + relation
            negative_relation = modifier[0] + " " + negative_relation
        else:
            # print ('relation:', relation, ':', data_point["caption"])
            pass
        # convert word array to string
        full_subject_a = " ".join(full_subject_a)
        full_subject_b = " ".join(full_subject_b)

        # contains
        stopwords = ['the', 'is']
        subject_a  = [word for word in full_subject_a.split() if word.lower() not in stopwords][0].strip(".")
        subject_b  = [word for word in full_subject_b.split() if word.lower() not in stopwords][0].strip(".")

        # # # use negative dict
        # captions = [
        #     # false case first
        #     data_point["caption"].split(' ' + relation + ' ')[0] \
        #         + ' ' + negate[relation] + ' ' + \
        #         data_point["caption"].split(' ' + relation + ' ')[1], 
        #     # true case second
        #     data_point["caption"].split(' ' + relation + ' ')[0] \
        #         + ' ' + relation + ' ' + \
        #         data_point["caption"].split(' ' + relation + ' ')[1], 
        # ]

        # # add captions that we want remove in vector space
        # captions.append(data_point["caption"].split(' ' + relation + ' ')[0] \
        #         + ' ' + negate[relation])               # remove bias for the negative relation
        # captions.append(data_point["caption"].split(' ' + relation + ' ')[0] \
        #         + ' ' + relation)                       # remove bias for the positive relation
        # # captions.append(negate[relation])               # remove bias for the negative relation
        # # captions.append(relation)                       # remove bias for the positive relation

        captions = [
            # false case first
            # full_subject_a + ' ' + negative_relation + ' ' + full_subject_b,
            # full_subject_a + ' ' + relation, 
            # full_subject_a + ' "' + negative_relation + '" ' + full_subject_b,
            full_subject_a + ' "' + relation + '"',
            # data_point["caption"].split(' ' + relation + ' ')[0] \
            #     + ' ' + relation,
            # "<" + subject_a + '> <' + negate[relation] + '> <' + subject_b + '>',
            # "subject: " + full_subject_a + '. ' + "predicate: " + negate[relation] + '. ' + "object: " + full_subject_b + ".",
            # subject_a + ', ' + relation,
            # "<" + subject_a + '> <' + relation + '>',
            # "<" + full_subject_a + '> <' + relation + '>',
            # "subject: " + full_subject_a + '. ' + "predicate: " + relation + '. ',

            # true case second
            # data_point["caption"], 
            # full_subject_a + ' ' + relation + ' ' + full_subject_b,
            full_subject_a + ' "' + relation + '" ' + full_subject_b,
            # subject_a + ', ' + relation + ', ' + subject_b,
            # "<" + subject_a + '> <' + relation + '> <' + subject_b + '>',
            # "<" + full_subject_a + '> <' + relation + '> <' + full_subject_b + '>',
            # "subject: " + full_subject_a + '. ' + "predicate: " + relation + '. ' + "object: " + full_subject_b + ".",
            
            # other false cases
            # data_point["caption"].split(' ' + relation + ' ')[0],
            # relation + ' ' + data_point["caption"].split(' ' + relation + ' ')[1],
            # relation,
        ]


        # load Image
        img_path = os.path.join(self.img_path, data_point["image"])
        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        return image, captions, data_point["label"], data_point["image"]
        # return self.imgs[data_point["image"]], captions, data_point["label"]
        # return self.imgs[data_point["image"]], data_point["caption"], data_point["label"]

    def __len__(self):
        return len(self.data_json)
