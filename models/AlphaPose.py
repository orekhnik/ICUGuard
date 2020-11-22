#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install -U torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html
get_ipython().system(' pip3 install torch==1.1.0 torchvision==0.3.0')
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/MVIG-SJTU/AlphaPose.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # clone and install dependencies
  get_ipython().system('git clone -q {git_repo_url}')
  get_ipython().system('pip install -q youtube-dl cython gdown')
  get_ipython().system('pip install -q -U PyYAML')
  get_ipython().system('apt-get install -y -q libyaml-dev')
  get_ipython().system('cd {project_name} && python setup.py build develop --user')
  
import sys
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False

from IPython.display import YouTubeVideo


# In[ ]:


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))


# In[ ]:


from google.cloud import storage
tracker = bucket.blob('ikem/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth')
get_ipython().system('mkdir -p {project_name}/trackers/weights/')
tracker.download_to_filename(join(project_name, 'trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth'))


# In[ ]:


get_ipython().system('pip install pillow==7.2.0.0')


# In[ ]:


yolo_pretrained_model_path = join(project_name,'detector/yolo/data/yolov3-spp.weights')
if not exists(yolo_pretrained_model_path):
  # download the YOLO weights
  get_ipython().system('mkdir -p {project_name}/detector/yolo/data')
  get_ipython().system('gdown -O {yolo_pretrained_model_path} https://drive.google.com/uc?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC')

tracker_pretrained_model_path = join(project_name,'detector/tracker/data/jde.1088x608.uncertainty.pt')
if not exists(tracker_pretrained_model_path):
 # tracker weights
 get_ipython().system('mkdir -p {project_name}/detector/tracker/data')
 get_ipython().system('gdown -O {tracker_pretrained_model_path} https://drive.google.com/uc?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA')

# ResNet152 backbone 73.3 AP
pretrained_model_path = join(project_name,'pretrained_models/fast_421_res152_256x192.pth')
pretrained_model_config_path = join(project_name,'configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml')
if not exists(pretrained_model_path):
  # download the pretrained model
  get_ipython().system('gdown -O {pretrained_model_path} https://drive.google.com/uc?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9')

# Halpe 69.0 AP
pretrained_model_halpe_path = join(project_name,'pretrained_models/halpe26_fast_res50_256x192.pth')
pretrained_model_halpe_config_path = join(project_name,'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml')
if not exists(pretrained_model_halpe_path):
  # download the pretrained model
  get_ipython().system('gdown -O {pretrained_model_halpe_path} https://drive.google.com/uc?id=1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb')


# In[ ]:


import sys
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False

from IPython.display import YouTubeVideo
get_ipython().system('rm -rf youtube.mp4')
#bucket = client.get_bucket('cee_hacks_2020')
# video = bucket.blob('ikem/KTJI_short.avi')
# video = bucket.blob('ikem/short_videos/download')
video.download_to_filename('youtube.avi')
# cut the first 5 seconds
get_ipython().system('ffmpeg -y -loglevel info -i youtube.avi  -vcodec libx264 -t 120  -fflags +igndts video_resampled.avi')
# run AlpaPose on these 5 seconds video
get_ipython().system('rm -rf AlphaPose_video.avi')
get_ipython().system('rm -r /content/vis')


# In[ ]:


get_ipython().system('cd {project_name} && python3 scripts/demo_inference.py --sp --video ../video.avi --outdir ../ --save_video --checkpoint ../{pretrained_model_path} --cfg ../{pretrained_model_config_path} --posebatch 20 --qsize 150 --pose_track --showbox --save_img')


# In[ ]:


# convert the result into MP4
get_ipython().system('ffmpeg -y -loglevel info -i AlphaPose_video.avi -vcodec libx264 AlphaPose_video.mp4')


# In[ ]:


def show_local_mp4_video(file_name, width=640, height=480):
  import io
  import base64
  from IPython.display import HTML
  video_encoded = base64.b64encode(io.open(file_name, 'rb').read())
  return HTML(data='''<video width="{0}" height="{1}" alt="test" controls>
                        <source src="data:video/mp4;base64,{2}" type="video/mp4" />
                      </video>'''.format(width, height, video_encoded.decode('ascii')))

show_local_mp4_video('AlphaPose_video.mp4', width=960, height=720)


# In[ ]:


from google.cloud import storage
client = storage.Client.from_service_account_json('/content/bookrecomedation-bq.json')
bucket = client.get_bucket('cee_hacks_2020')
main_fold = 'ikem'
folder_name = join(main_fold, '1min_version_koak_box_5')
orig_video = bucket.blob(join(folder_name,'original_short_vid.mp4'))
orig_video.upload_from_filename('video.avi')
annotation = bucket.blob(join(folder_name,'alphapose-results.json'))
annotation.upload_from_filename('alphapose-results.json')
annotated_video = bucket.blob(join(folder_name,'annotated_video.mp4'))
annotated_video.upload_from_filename('AlphaPose_video.mp4')
orig_video = bucket.blob(join(folder_name,'original_short_vid.mp4'))
get_ipython().system('gsutil -m cp -r vis/ gs://cee_hacks_2020/{folder_name}/ ')


# In[ ]:


from google.colab import auth
auth.authenticate_user()


# In[ ]:


bucket = client.get_bucket('cee_hacks_2020')
main_fold = 'ikem'
folder_name = join(main_fold, '1min_version_koak_box_5')
orig_video = bucket.blob(join(folder_name,'original_short_vid_resampled.mp4'))
orig_video.upload_from_filename('video_resampled.avi')

