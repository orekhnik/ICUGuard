#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import tqdm
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.applications import ResNet50, Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[22]:


# root_dir = '/home/nicko/Downloads/kevin_annotations/30_sec_KOAK_BOX_5_4person/'
# root_dir = '/home/nicko/Downloads/kevin_annotations/cekarna'
# root_dir = '/home/nicko/Downloads/kevin_annotations/10_sec_KOAK_10_3pers'
# root_dir = '/home/nicko/Downloads/kevin_annotations/10_sec_KOAK_BOX_5_4person'
root_dir = '/home/nicko/Downloads/kevin_annotations/20_sec_KTJI_SHORT'
# root_dir = '/home/nicko/Downloads/kevin_annotations/1min_version_koak_box_5'
                                    

def read_json(path):
    with open(path) as f:
        return json.loads(f.read())
    
def select_part(urls, part):
    for u in urls:
        if part in u:
            return u
    return None

files = os.listdir(root_dir)
annotation_json_path = select_part(files, 'json')
assert annotation_json_path is not None
annotation_json_path = os.path.join(root_dir, annotation_json_path)

    
annotation_json = read_json(annotation_json_path)

print(len(annotation_json), 'detections')
# annotation_json[0]


# In[23]:


images_orig_dir = os.path.join(root_dir, 'images_orig')
shutil.rmtree(images_orig_dir, ignore_errors=True)

images_annot_dir = os.path.join(root_dir, 'images_annot')
shutil.rmtree(images_annot_dir, ignore_errors=True)

orig_video_path = select_part(files, 'original')
annot_video_path = select_part(files, 'annotated')
assert orig_video_path is not None and annot_video_path is not None
orig_video_path = os.path.join(root_dir, orig_video_path)
annot_video_path = os.path.join(root_dir, annot_video_path)

os.makedirs(images_orig_dir)
os.makedirs(images_annot_dir)

print(f'Input video: {orig_video_path}')
print(f'Annotated video: {annot_video_path}')
print()

get_ipython().system('cd $images_orig_dir && ffmpeg -i $orig_video_path %04d.jpg -hide_banner')
get_ipython().system('cd $images_annot_dir && ffmpeg -i $annot_video_path %04d.jpg -hide_banner')

orig_frames = len(os.listdir(images_orig_dir))
annot_frames = len(os.listdir(images_annot_dir))
frame_ratio = orig_frames / annot_frames

orig_frames, annot_frames, frame_ratio


# In[24]:


saved_model_personnel = '/data/datasets/hackathon/is_personnel/models/resnet50-categorical_crossentropy-avg_pool-weighted_False-rot_0-hflip_False-vflip_False-brirange_None.ckpt.h5'
personnel_model = load_model(saved_model_personnel)

saved_model_actions = '/data/datasets/hackathon/actions/models/xception-categorical_crossentropy-global_average_pooling2d_1-weighted_False-rot_0-hflip_False-vflip_False-brirange_None.ckpt.h5'
actions_model = load_model(saved_model_actions)


# In[25]:


prep_fun_personnel = preprocess_input_resnet
# prep_fun_personnel = preprocess_input_xception

prep_fun_actions = preprocess_input_xception


personnel_prediction_key, actions_prediction_key = 'is_personnel', 'action'

output_annotations_path = os.path.join(root_dir, 'annotations_processed.json')

personnel_idx_to_label, actions_idx_to_label = ['pacient', 'personal'], ["lezici", "sedici", "stojici"]



print(f'Saving new annotations to {output_annotations_path}')

def preproc(image, prep_fun, resize_to=224, resample=Image.NEAREST):
    image = Image.fromarray(image)
    image = image.resize((resize_to, resize_to), resample=resample)
    image = np.expand_dims(image, axis=0)
    return prep_fun(image)

def predict_label(model, image, prep_fun, idx_to_label):
    image = preproc(image, prep_fun)
    predictions = model.predict(image)[0]
    idx = np.argmax(predictions)
    return idx_to_label[idx], idx

objects = {}
for ann in tqdm.tqdm(annotation_json):
    image_num = int(ann['image_id'].split('.')[0])
    image_name = f'{(int(image_num*frame_ratio) + 1):>04}.jpg'
    image_path = os.path.join(images_orig_dir, image_name)

    img = plt.imread(image_path)
    
    # clip to zero because there're negative centroids
    bbox = np.clip(ann['box'], 0, 1e20).astype(int)

    cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    label_personnel, idx_personnel = predict_label(personnel_model, cut, prep_fun_personnel, personnel_idx_to_label)
    label_actions, idx_actions = predict_label(actions_model, cut, prep_fun_actions, actions_idx_to_label)
    
    # voting
    voting = objects.get(ann['idx'])
    if voting is None:
        voting = {personnel_prediction_key: [0, 0],
                  actions_prediction_key: [0, 0, 0]}
        objects[ann['idx']] = voting
    voting[personnel_prediction_key][idx_personnel] += 1
    voting[actions_prediction_key][idx_actions] += 1
    
print(f'Converting labels')
for ann in annotation_json:
    ann[personnel_prediction_key] = personnel_idx_to_label[np.argmax(objects[ann['idx']][personnel_prediction_key])]
    ann[actions_prediction_key] = actions_idx_to_label[np.argmax(objects[ann['idx']][actions_prediction_key])]

print(f'Dumping labels')
with open(output_annotations_path, 'w') as f:
    f.write(json.dumps(annotation_json))


# In[26]:


import cv2

prep_fun_personnel = preprocess_input_resnet
# prep_fun_personnel = preprocess_input_xception

prep_fun_actions = preprocess_input_xception

personnel_prediction_key, actions_prediction_key = 'is_personnel', 'action'
personnel_idx_to_label, actions_idx_to_label = ['pacient', 'personal'], ["lezici", "sedici", "stojici"]

n = 10
seed = 40



tmp = annotation_json.copy()
random.seed(seed)
random.shuffle(tmp)


output_images_dir = os.path.join(root_dir, 'images_final_predictions')
print(f'Saving images to {output_images_dir}')

output_video_path = os.path.join(root_dir, 'processed_video.mp4')
print(f'Saving video to {output_video_path}')



shutil.rmtree(output_images_dir, ignore_errors=True)
images_annot_dir = os.path.join(root_dir, 'images_annot')
shutil.copytree(images_annot_dir, output_images_dir)

def preproc(image, prep_fun, resize_to=224, resample=Image.NEAREST):
    image = Image.fromarray(image)
    image = image.resize((resize_to, resize_to), resample=resample)
    image = np.expand_dims(image, axis=0)
    return prep_fun(image)

def predict_label(model, image, prep_fun, idx_to_label):
    image = preproc(image, prep_fun)
    predictions = model.predict(image)[0]
    idx = np.argmax(predictions)
    return idx_to_label[idx], idx
# predictions[idx]

images = {}
objects = {}
# for ann in tqdm.tqdm(tmp[:n]):
for ann in tqdm.tqdm(annotation_json):
    image_num = int(ann['image_id'].split('.')[0])
    image_name = f'{(int(image_num*frame_ratio) + 1):>04}.jpg'
    image_path = os.path.join(images_orig_dir, image_name)

    img = plt.imread(image_path)
    
    # clip to zero because there're negative centroids
    bbox = np.clip(ann['box'], 0, 1e20).astype(int)

    cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    label_personnel, idx_personnel = predict_label(personnel_model, cut, prep_fun_personnel, personnel_idx_to_label)
    label_actions, idx_actions = predict_label(actions_model, cut, prep_fun_actions, actions_idx_to_label)
    
    # voting
    voting = objects.get(ann['idx'])
    if voting is None:
        voting = {personnel_prediction_key: [0, 0],
                  actions_prediction_key: [0, 0, 0]}
        objects[ann['idx']] = voting
    voting[personnel_prediction_key][idx_personnel] += 1
    voting[actions_prediction_key][idx_actions] += 1
    
#     print(f'{label_personnel} {score_personnel*100:.1f}%')
#     print(f'{label_actions} {score_actions*100:.1f}%')
#     plt.imshow(cut)
#     plt.show()

    vis_image_name = f'{(image_num + 1):>04}.jpg'
    vis_obj = images.get(vis_image_name)
    if vis_obj is None:
        vis_image_path = os.path.join(images_annot_dir, vis_image_name)
        vis_image_part = plt.imread(vis_image_path)
        vis_obj = {
            'image': vis_image_part,
            'idx': [],
            'box': []}
        images[vis_image_name] = vis_obj

    vis_obj['idx'].append(ann['idx'])
    vis_obj['box'].append((bbox[0], bbox[1]))
#     cv2.putText(vis_obj, f'{label_personnel} {score_personnel*100:.1f}%', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX,  1.8, (255,0,0), 10)
#     cv2.putText(vis_obj, f'{label_actions} {score_actions*100:.1f}%', (bbox[0], bbox[1]+65), cv2.FONT_HERSHEY_SIMPLEX,  1.8, (255,0,0), 10)


print(f'Collecting images')
for imname, vis_obj in images.items():
    img = vis_obj['image']
    for idx, box in zip(vis_obj['idx'], vis_obj['box']):
        cv2.putText(img, personnel_idx_to_label[np.argmax(objects[idx][personnel_prediction_key])], box, cv2.FONT_HERSHEY_SIMPLEX,  1.2, (255,0,0), 4)
        cv2.putText(img, actions_idx_to_label[np.argmax(objects[idx][actions_prediction_key])], (box[0], box[1]+65), cv2.FONT_HERSHEY_SIMPLEX,  1.2, (255,0,0), 4)
    plt.imsave(os.path.join(output_images_dir, imname), img)
    
print('Merging video')
get_ipython().system('cd $output_images_dir && ffmpeg -y -framerate 24 -i %04d.jpg $output_video_path')


# In[ ]:




