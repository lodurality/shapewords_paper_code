import torch
import os
import random
import safetensors.torch
import numpy as np
import pandas as pd
import logging
import os
from collections import defaultdict
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import PIL
import safetensors
import torch.nn as nn
import pandas as pd
import logging
import random
import math
import os
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import json


# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
# from torchvision.transforms import v2
from torchvision.transforms import transforms
from tqdm.auto import tqdm
import time

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        # why this
        "linear": PIL.Image.BILINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# -------------
# 
# 
# 
# 
categories = {
    "02992529": "phone",
    "03761084": "microwave",
    "03991062": "pot",
    "03046257": "clock",
    "03513137": "helmet",
    "04379243": "table",
    "03593526": "jar",
    "04225987": "skateboard",
    "02958343": "car",
    "02876657": "bottle",
    "04460130": "tower",
    "03001627": "chair",
    "02871439": "bookshelf",
    "02942699": "camera",
    "02691156": "airplane",
    "03642806": "laptop",
    "02801938": "basket",
    "04256520": "sofa",
    "03624134": "knife",
    "02946921": "can",
    "04090263": "rifle",
    "04468005": "train",
    "03938244": "pillow",
    "03636649": "lamp",
    "02747177": "trash bin",
    # "02747177": "dumpster",
    "03710193": "mailbox",
    "04530566": "watercraft",
    # "04530566": "vessel",
    "03790512": "motorbike",
    "03207941": "dishwasher",
    "02828884": "bench",
    "03948459": "pistol",
    "04099429": "rocket",
    "03691459": "loudspeaker",
    # "03691459": "speaker",
    "03337140": "file cabinet",
    # "03337140": "cabinet",
    "02773838": "bag",
    "02933112": "cabinet",
    "02818832": "bed",
    "02843684": "birdhouse",
    # "02843684": "nest",
    "03211117": "display",
    "03928116": "piano",
    "03261776": "earphone",
    # "03261776": "headset",
    "04401088": "telephone",
    "04330267": "stove",
    "03759954": "microphone",
    "02924116": "bus",
    "03797390": "mug",
    "04074963": "remote",
    "02808440": "bathtub",
    "02880940": "bowl",
    "03085013": "keyboard",
    "03467517": "guitar",
    "04554684": "washer",
    "02834778": "bicycle",
    "03325088": "faucet",
    "04004475": "printer",
    "02954340": "cap"
}




def get_image_paths(directory, extensions=('.png', '.jpg', '.jpeg',)):
    image_paths = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            image_paths.extend(get_image_paths(full_path, extensions))
        elif entry.lower().endswith(extensions):
            image_paths.append(full_path)
    return image_paths



class ProjectionDataset(Dataset):
    def __init__(
        self,
        depth_images_root,
        preprocess,
        textured_images_root=None,
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        max_images = 10e6,
        min_crop_scale = 0.8,
        max_crop_scale = 1,
        use_transform=True,
        num_depth_views=1,
        pb_emb_path=None,
        use_pb_embs=False,
        use_clip_residual=False,
        max_shapes = 10e6,
        start_view = 0,
        end_view = 360,
        step_view = 12,
        from_memory=True,
        pad_border= 120,
        
    ):
        assert pb_emb_path is not None, "pb_emb_path should not be None"

        self.textured_images_root = textured_images_root
        self.depth_images_root = depth_images_root
        self.size = size
        self.flip_p = flip_p
        self.min_crop_scale = min_crop_scale
        self.max_crop_scale = max_crop_scale
        self.max_images = max_images
        self.textured_image_path = []
        self.model_dict = defaultdict(dict)
        self.preprocess = preprocess
        self.use_pb_embs = use_pb_embs
        self.pb_emb_path = pb_emb_path
        self.use_clip_residual = use_clip_residual
        self.depth_images_paths = depth_images_root
        self.use_transform = use_transform
        self.num_depth_views = num_depth_views
        self.from_memory = from_memory
        self.pad_border = pad_border
       

        if use_clip_residual:
            assert textured_images_root!=None, "to use clip residuals provide the textured path"
            self.textured_image_paths = get_image_paths(directory=textured_images_root)[:int(max_images)]

        if use_clip_residual:
            self.num_shapes = None
            self.num_images = len(self.textured_image_paths)
            self._length = self.num_images
            self._length = self.num_images * repeats # make sure on only training it

        if self.use_transform:
            self.transforms = transforms.Compose([
                    transforms.Pad(self.pad_border, fill=(255,255,255), padding_mode='constant'),
                    transforms.RandomResizedCrop(size=(self.size, self.size),scale=(self.min_crop_scale,self.max_crop_scale), antialias=True),
                    transforms.RandomHorizontalFlip(p=self.flip_p),
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Resize(size=(self.size, self.size), antialias=True),])
            
       
        path_template = '{}/{}'
        pb_dict = {}
        all_emb_files = sorted(os.listdir(pb_emb_path))
        for cur_file in all_emb_files:
            cur_data = np.load(path_template.format(pb_emb_path, cur_file))
            cur_dict = dict(zip(cur_data['ids'], cur_data['embs']))
            pb_dict = {**pb_dict, **cur_dict}


        self.pb_data = pb_dict
        self.shapes = list(self.pb_data.keys())
        self.shapes = random.sample(self.shapes, k=min(int(max_shapes),len(self.shapes)))
        # self.shapes = os.listdir('/scratch/workspace/dmpetrov_umass_edu-ulip/geometry-editing-2d3d/debug_training_images')
        self.views = [str(i).zfill(3) for i in range(start_view, end_view, step_view)] # end view is not inclusive like the ulip dataset
        self.ulip_template = '{}_r_{}_depth0001.png'

        
        if not use_clip_residual:
            self.num_images = None
            self.num_shapes = len(self.shapes)
            self._length = len(self.shapes)
            self._length = self._length * repeats # make sure on only training it
        
        print(f"{self.shapes[:5]=}")
        print(f"{self.views=}")
        print(f"{self._length=}")




    def __len__(self):
        return self._length
        # 'fore_03001627_1bec15f362b641ca7350b1b2f753f3a2_foreground_prompt_1_background_prompt_91_seed_5816_view_7.jpg'

    def __getitem__(self, i):
        if self.use_clip_residual: 
            return self.load_textured_mem(i)
        else:
            if not self.from_memory:
                return self.load_disk(i)
            else:
                return self.load_memory(i)



    def load_memory(self, i):
        example = {}
        shape_id = self.shapes[i % self.num_shapes]
        view_ids =  random.sample(self.views, k=self.num_depth_views)
        if shape_id not in self.model_dict:
            self.model_dict[shape_id]={}
        depth_images_shape=[]
        for each_view_id in view_ids:
            if each_view_id not in self.model_dict[shape_id]:
                each_path = os.path.join(self.depth_images_paths,self.ulip_template.format(shape_id.replace("_","-"),str(each_view_id)))
                cur_image = np.array(Image.open(each_path).convert("RGB"))
                self.model_dict[shape_id][each_view_id] = cur_image
            else:
                cur_image = self.model_dict[shape_id][each_view_id]
            depth_images_shape.append(cur_image)

        for j,each_depth_image in enumerate(depth_images_shape): # cropping the images, if they are not square
            crop = min(each_depth_image.shape[0], each_depth_image.shape[1]) 
            h, w, = (
                each_depth_image.shape[0],
                each_depth_image.shape[1],
            )
            each_depth_image = each_depth_image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            depth_images_shape[j] = each_depth_image

            
        depth_images_shape = [Image.fromarray(each_depth_image)for each_depth_image in depth_images_shape] # coverting it into image and then transforming
        depth_images_shape = [self.transforms(each_depth_image) for each_depth_image in depth_images_shape]
            
        # some sanity check
        # saving_directory = f'./random/{i}'
        # os.makedirs(saving_directory,exist_ok=True)
        # for j, each_depth_image in enumerate(depth_images_shape):
        #     each_depth_image.save(os.path.join(saving_directory,f'_{j}.png'))

        # converting it into array, normalizing and then tensors
        depth_images_shape = [np.array(each_depth_image) for each_depth_image in depth_images_shape]
        depth_images_shape = np.array(depth_images_shape).astype(np.uint8)
        depth_images_shape = (depth_images_shape / 127.5 - 1.0).astype(np.float32)
        example["depth_images"] = torch.from_numpy(depth_images_shape).permute(0,3, 1, 2) # for depth images these should not  matter, only lets keep this away for now
        if self.use_pb_embs:
            pb_embs = torch.from_numpy(self.pb_data[shape_id])
            example["pb_embs"] = pb_embs
        return example


    def load_disk(self, i):
        example = {}
        shape_id = self.shapes[i % self.num_shapes]
        view_ids =  random.sample(self.views, k=self.num_depth_views)
        depth_images_shape=[]
        for each_view_id in view_ids:
            each_path = os.path.join(self.depth_images_paths,self.ulip_template.format(shape_id.replace("_","-"),str(each_view_id)))
            depth_images_shape.append(np.array(Image.open(each_path).convert("RGB")))


        for j,each_depth_image in enumerate(depth_images_shape): # cropping the images, if they are not square
            crop = min(each_depth_image.shape[0], each_depth_image.shape[1]) 
            h, w, = (
                each_depth_image.shape[0],
                each_depth_image.shape[1],
            )
            each_depth_image = each_depth_image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            depth_images_shape[j] = each_depth_image

            
        depth_images_shape = [Image.fromarray(each_depth_image)for each_depth_image in depth_images_shape] # coverting it into image and then transforming
        depth_images_shape = [self.transforms(each_depth_image) for each_depth_image in depth_images_shape]
            
        # # some sanity check
        # saving_directory = f'./random/{i}'
        # os.makedirs(saving_directory,exist_ok=True)
        # for j, each_depth_image in enumerate(depth_images_shape):
        #     each_depth_image.save(os.path.join(saving_directory,f'_{j}.png'))

        # converting it into array, normalizing and then tensors
        depth_images_shape = [np.array(each_depth_image) for each_depth_image in depth_images_shape]
        depth_images_shape = np.array(depth_images_shape).astype(np.uint8)
        depth_images_shape = (depth_images_shape / 127.5 - 1.0).astype(np.float32)
        example["depth_images"] = torch.from_numpy(depth_images_shape).permute(0,3, 1, 2) # for depth images these should not  matter, only lets keep this away for now
        if self.use_pb_embs:
            pb_embs = torch.from_numpy(self.pb_data[shape_id])
            example["pb_embs"] = pb_embs
        return example
    




        
    def load_textured_mem(self, i):
        example = {}
        textured_image_path = self.textured_image_paths[i % self.num_images]
        textured_image_id = os.path.basename(textured_image_path)
        textured_image_id = textured_image_id.split("_")
        shape_id = f"{textured_image_id[1]}_{textured_image_id[2]}"
        view_id = int(textured_image_id[-1].split(".")[0])
        if shape_id not in self.model_dict or textured_image_path not in self.model_dict[shape_id]: # checking if the shape and corresponding textured image is already cached
                
            textured_image = Image.open(textured_image_path).convert("RGB")
            self.model_dict[shape_id][textured_image_path] = textured_image
            if 'depth_images' not in self.model_dict[shape_id]:
                depth_image_path = os.path.join(self.depth_images_paths,f"{shape_id}_depth_views.npz")
                depth_images_all_shape = np.load(depth_image_path)['arr_0']
                processed_depth_images_all_shape=[]
                for j,image in enumerate(depth_images_all_shape): # coverting it into rgb and then np.int for cropping transforming etc
                    cur_depth_image = Image.fromarray(image).convert("RGB")
                    processed_depth_images_all_shape.append(np.array(cur_depth_image).astype(np.uint8))
                self.model_dict[shape_id]['depth_images'] = np.array(processed_depth_images_all_shape)
       
        textured_image = self.model_dict[shape_id][textured_image_path] # loading the textured and the corresponding depth_image
        depth_images_all_shape = self.model_dict[shape_id]['depth_images']

        sampled_indexes = [((view_id+5*v) % 20) for v in range(self.num_depth_views)] # sampling the depth images
        depth_images_shape = depth_images_all_shape[np.array(sampled_indexes)]

        for j,each_depth_image in enumerate(depth_images_shape): # cropping the images, if they are not square
            crop = min(each_depth_image.shape[0], each_depth_image.shape[1]) 
            h, w, = (
                each_depth_image.shape[0],
                each_depth_image.shape[1],
            )
            each_depth_image = each_depth_image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
            depth_images_shape[j] = each_depth_image

            
        depth_images_shape = [Image.fromarray(each_depth_image)for each_depth_image in depth_images_shape] # coverting it into image and then transforming
        depth_images_shape = [self.transforms(each_depth_image) for each_depth_image in depth_images_shape]
            
        # some sanity check
        # saving_directory = f'./random/{i}'
        # os.makedirs(saving_directory,exist_ok=True)
        # textured_image.save(os.path.join(saving_directory,f"textured_image_{i}.png"))
        # for j, each_depth_image in enumerate(depth_images_shape):
        #     each_depth_image.save(os.path.join(saving_directory,f'_{j}.png'))

        # converting it into array, normalizing and then tensors
        depth_images_shape = [np.array(each_depth_image) for each_depth_image in depth_images_shape]
        depth_images_shape = np.array(depth_images_shape).astype(np.uint8)
        depth_images_shape = (depth_images_shape / 127.5 - 1.0).astype(np.float32)
        example["depth_images"] = torch.from_numpy(depth_images_shape).permute(0,3, 1, 2) # for depth images these should not  matter, only lets keep this away for now
        example["textured_images"] = self.preprocess(textured_image)
        if self.use_pb_embs:
            pb_embs = torch.from_numpy(self.pb_data[shape_id])
            example["pb_embs"] = pb_embs
        return example


class ULIPControlNetDataset(Dataset):
    def __init__(
            self,
            preprocess,
            textured_images_root=None,
            prompts_path=None,
            size=512,
            repeats=1,
            interpolation="bicubic",
            flip_p=0.5,
            max_images=10e6,
            min_crop_scale=0.8,
            max_crop_scale=1,
            use_transform=True,
            num_depth_views=1,
            pb_emb_path=None,
            categories_path=None,
            use_pb_embs=True,
            use_clip_residual=False,
            max_shapes=10e6,
            start_view=0,
            end_view=360,
            step_view=12,
            from_memory=True,
            pad_border=120,
            cache_data=False,
            split_ids=None

    ):
        assert pb_emb_path is not None, "pb_emb_path should not be None"

        self.textured_images_root = textured_images_root
        self.size = size
        self.flip_p = flip_p
        self.min_crop_scale = min_crop_scale
        self.max_crop_scale = max_crop_scale
        self.max_images = max_images
        self.textured_image_path = []
        self.model_dict = defaultdict(dict)
        self.preprocess = preprocess
        self.use_pb_embs = use_pb_embs
        self.pb_emb_path = pb_emb_path
        self.use_clip_residual = use_clip_residual
        self.use_transform = use_transform
        self.num_depth_views = num_depth_views
        self.from_memory = from_memory
        self.pad_border = pad_border
        self.cache_data = cache_data
        self.split_ids = split_ids
        if prompts_path is not None:
            self.prompts = np.loadtxt(prompts_path, dtype=str, delimiter='\t')

        assert textured_images_root != None, "to use clip residuals provide the textured path"
        if split_ids is None:
            print('parsing paths')
            self.textured_image_paths = get_image_paths(directory=textured_images_root)  # [:int(max_images)]
            self.textured_image_paths = [item for item in self.textured_image_paths if 'combined' in item][:int(max_images)]
        else:
            print('using splits')
            data_path = textured_images_root

            #split_ids = train_val_arr[:100]

            all_paths = []
            for shape_id in split_ids:
                cat_id, model_id = shape_id.split('_')

                image_path = f'{data_path}/{cat_id}/{shape_id}/combined/'
                if os.path.exists(image_path):
                    cur_paths = [f'{image_path}/{item}' for item in sorted(os.listdir(image_path))]
                else:
                    print(f'Image path for shape {shape_id} does not exist.')
                    cur_paths = []
                all_paths += cur_paths
            self.textured_image_paths = all_paths
            # for item

        if self.use_transform:
            self.transforms = transforms.Compose([
                transforms.Pad(self.pad_border, fill=(255, 255, 255), padding_mode='constant'),
                transforms.RandomResizedCrop(size=(self.size, self.size),
                                             scale=(self.min_crop_scale, self.max_crop_scale), antialias=True),
                transforms.RandomHorizontalFlip(p=self.flip_p),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(size=(self.size, self.size), antialias=True), ])

        path_template = '{}/{}'
        pb_dict = {}
        all_emb_files = sorted(os.listdir(pb_emb_path))
        for cur_file in all_emb_files:
            cur_data = np.load(path_template.format(pb_emb_path, cur_file))
            cur_dict = dict(zip(cur_data['ids'], cur_data['embs']))
            pb_dict = {**pb_dict, **cur_dict}

        self.categories_path = categories_path
        if categories_path is not None:
            with open(categories_path, 'r') as f:
                self.categories = json.load(f)


        self.pb_data = pb_dict
        self.shapes = list(self.pb_data.keys())
        self.shapes = random.sample(self.shapes, k=min(int(max_shapes), len(self.shapes)))
        # self.shapes = os.listdir('/scratch/workspace/dmpetrov_umass_edu-ulip/geometry-editing-2d3d/debug_training_images')
        self.views = [str(i).zfill(3) for i in
                      range(start_view, end_view, step_view)]  # end view is not inclusive like the ulip dataset
        self.ulip_template = '{}_r_{}_depth0001.png'

        self.num_images = len(self.textured_image_paths)
        self.num_shapes = len(self.shapes)
        self._length = self.num_images
        self._length = self._length * repeats  # make sure on only training it

        print(f"{self.shapes[:5]=}")
        print(f"{self.views=}")
        print(f"{self._length=}")

    def __len__(self):
        return self._length
        # 'fore_03001627_1bec15f362b641ca7350b1b2f753f3a2_foreground_prompt_1_background_prompt_91_seed_5816_view_7.jpg'

    def __getitem__(self, i):
        return self.load_data(i)

    def load_data(self, i):
        example = {}
        textured_image_path = self.textured_image_paths[i % self.num_images]
        #print(textured_image_path)
        textured_image_id = os.path.basename(textured_image_path)
        textured_image_id = textured_image_id.split("_")
        shape_id = textured_image_path.split('/combined')[0].split('/')[-1]
        cat_id = shape_id.split('_')[0]
        view_id = textured_image_path.split('angle_')[-1].split('_')[0]
        prompt_id = int(textured_image_path.split('prompt_')[-1].split('_')[0])
        #print(shape_id, view_id, prompt_id)
        if shape_id not in self.model_dict or textured_image_path not in self.model_dict[
            shape_id]:  # checking if the shape and corresponding textured image is already cached

            textured_image = Image.open(textured_image_path).convert("RGB")
            if self.cache_data:
                self.model_dict[shape_id][textured_image_path] = textured_image

        #textured_image = self.model_dict[shape_id][
        #    textured_image_path]  # loading the textured and the corresponding depth_image

        if self.preprocess is not None:
            example["textured_images"] = self.preprocess(textured_image)
        else:
            example['textured_images'] = textured_image
        pb_embs = torch.from_numpy(self.pb_data[shape_id])
        example["pb_embs"] = pb_embs
        example['category'] = self.categories[cat_id]
        example["prompts"] = self.prompts[prompt_id].replace('object', self.categories[cat_id])
        example['angles'] = int(view_id)
        return example