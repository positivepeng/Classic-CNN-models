# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 21:10:23 2019

@author: hp
"""

import numpy as np
import os
import random
from tqdm import tqdm
from matplotlib.pylab import imread, imshow
from skimage import transform
import pickle

path = r"E:\recently\国创\备份\kaggle_cats_and_dogs\PetImages";
    

def load_data(root_path, pic_num = 0):
    # pic_num 指定每种类别的数量，为0则默认为全部
    images = []
    labels = []
    for category_id, category_name in enumerate(os.listdir(root_path)):
        category_path = root_path + os.sep + category_name    
        image_cnt = 0
        for image_name in os.listdir(category_path):
            try:
                image_path = category_path + os.sep + image_name
                images.append(transform.resize(imread(image_path), (300,300,3)))
                labels.append(category_id)
                image_cnt += 1
                print(image_cnt)
                if pic_num != 0 and image_cnt >= pic_num:
                    break
            except:
                print("error", image_path)
    return np.array(images), np.array(labels)

def show_one_image(img, category_id):
    print(category_id)
    imshow(img)


if __name__ == '__main__':
    category_num = 500
    images, labels = load_data(path, category_num)
    data = {
        "images" : images,
        "labels" : labels
    }
    with open("cat_and_dog_1000.pickle", "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("cat_and_dog_1000.pickle", "rb") as f:
        data_read_from_pickle = pickle.load(f)
    
    print(data_read_from_pickle["images"].shape)
    
    
    


