import os
import glob
import numpy as np 
import cv2
import sys

import ntpath

from keras.preprocessing.image import img_to_array


def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            x[i, j, labels[i][j]] = 1
    #x = x.reshape(dims[0] * dims[1], n_labels)
    return x


def data_gen_small(img_dir, mask_dir, batch_size, dims, n_labels):
    while True:
        #ix = np.random.choice(np.arange(len(lists)), batch_size)
        images =  glob.glob( os.path.join(img_dir,"*.png")  ) 
        ix = np.random.choice(images, batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            i_name = ntpath.basename(i)
            img_path = i
            original_img = cv2.imread(img_path)[:, :, ::-1]
            resized_img = cv2.resize(original_img, (dims[1],dims[0]))
            #resized_img = original_img
            #print("Resizedi: " + str(resized_img.shape))
            array_img = img_to_array(resized_img)/255
            imgs.append(array_img)
            # masks
            original_mask = cv2.imread(mask_dir + i_name,0)
            #print("Imread: " + str(original_mask.shape))
            resized_mask = cv2.resize(original_mask, (dims[1],dims[0]))
            #resized_mask = original_mask
            #print("Resized: "+str(resized_mask.shape))
            array_mask = category_label(resized_mask, dims, n_labels)
            labels.append(array_mask)
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels

def get_data_small(img_dir, mask_dir, batch_size, dims, n_labels):
    images =  glob.glob( os.path.join(img_dir,"*.png")  ) 
    ix = np.random.choice(images, batch_size)
    imgs = []
    labels = []
    for i in ix:
        # images
        i_name = ntpath.basename(i)
        print(i_name)
        sys.stdout.flush()
        img_path = i
        original_img = cv2.imread(img_path)[:, :, ::-1]
        resized_img = cv2.resize(original_img, (dims[1],dims[0]))
        #resized_img = original_img
        #print("Resizedi: " + str(resized_img.shape))
        array_img = img_to_array(resized_img)/255
        imgs.append(array_img)
        # masks
        original_mask = cv2.imread(mask_dir + i_name,0)
        #print("Imread: " + str(original_mask.shape))
        resized_mask = cv2.resize(original_mask, (dims[1],dims[0]))
        #resized_mask = original_mask
        #print("Resized: "+str(resized_mask.shape))
        array_mask = category_label(resized_mask, dims, n_labels)
        labels.append(array_mask)
    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels