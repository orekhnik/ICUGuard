#!/usr/bin/env python
# coding: utf-8

# In[62]:


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


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[124]:


# root_dir = '/home/nicko/Downloads/kevin_annotations/30_sec_KOAK_BOX_5_4person/'
# root_dir = '/home/nicko/Downloads/kevin_annotations/cekarna'
# root_dir = '/home/nicko/Downloads/kevin_annotations/10_sec_KOAK_10_3pers'
# root_dir = '/home/nicko/Downloads/kevin_annotations/10_sec_KOAK_BOX_5_4person'
root_dir = '/home/nicko/Downloads/kevin_annotations/20_sec_KTJI_SHORT'
                                    

    

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


# # Split video into frames

# In[125]:


# from google.colab import auth
# auth.authenticate_user()
# # !gsutil -m cp -r gs://cee_hacks_2020/{folder_name}/ folder_whatever/


# In[126]:


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


# # Visualize bboxes

# In[127]:


seed = 43
n = 1


tmp = annotation_json.copy()
random.seed(seed)
random.shuffle(tmp)

for ann in tmp[:n]:
# for ann in tqdm.tqdm(annotation_json[-10:]):
    image_num = int(ann['image_id'].split('.')[0])
    image_name = f'{(int(image_num*frame_ratio) + 1):>04}.jpg'
    image_path = os.path.join(images_orig_dir, image_name)
#     image_path = '/tmp/1.JPG'
    print(image_path)

    img = plt.imread(image_path)
    bbox = np.clip(ann['box'], 0, 1e20).astype(int)
    
    print('image', img.shape[:2])
    print('bbox', bbox)
    plt.imshow(img)
    plt.show()
    
    cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
    plt.imshow(cut)
    plt.show()


# # Generate images

# In[128]:


output_dir = os.path.join(root_dir, 'images_processed')

shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

print(f'Saving images to {output_dir}')

for ann in tqdm.tqdm(annotation_json):
    image_num = int(ann['image_id'].split('.')[0])
    image_name = f'{(int(image_num*frame_ratio) + 1):>04}.jpg'
    image_path = os.path.join(images_orig_dir, image_name)
#     print(image_path)

    img = plt.imread(image_path)
#     bbox = np.array(ann['box'], int)
    bbox = np.clip(ann['box'], 0, 1e20).astype(int)
    
#     print('image', img.shape[:2])
#     print('bbox', bbox)
#     plt.imshow(img)
#     plt.show()
    
    cut = img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    
#     plt.imshow(cut)
#     plt.show()

    object_dir = os.path.join(output_dir, f'{ann["idx"]}')
    os.makedirs(object_dir, exist_ok=True)
    
    image_output_path = os.path.join(object_dir, image_name)
    plt.imsave(image_output_path, cut)


models_dir = os.path.join(root_dir, 'images_models')
os.makedirs(models_dir)

print('Copying images to is_personnel dir')
is_personnel_dir = os.path.join(models_dir, 'is_personnel')
shutil.copytree(output_dir, is_personnel_dir)
is_personnel_classes = ['personnel', 'patient']
for c in is_personnel_classes:
    os.makedirs(os.path.join(is_personnel_dir, c))
    
print('Copying images to actions dir')
actions_dir = os.path.join(models_dir, 'actions')
shutil.copytree(output_dir, actions_dir)
actions_classes = ['sitting', 'staying', 'laying down']
for c in actions_classes:
    os.makedirs(os.path.join(actions_dir, c))


# In[ ]:




