import os
import glob
import cv2
import numpy as np 

#dataToSeg("./data_png/", "./data_png/", "val")
def dataToSeg(data_path, seg_path, prefix):
    buildings_prefix = os.path.join(data_path,"buildings",prefix)
    roads_prefix = os.path.join(data_path,"roads",prefix)

    imgs = "images/"
    lbls = "labels/"
    B=1
    R=2

    prefix_seg = os.path.join(seg_path,prefix)

    for img in os.listdir(os.path.join(buildings_prefix,imgs)):
        if img.endswith('.png'):
            #check for building image
            imm = cv2.imread(os.path.join(buildings_prefix,imgs,img))
            im = cv2.imread(os.path.join(buildings_prefix,lbls,img))
            im = np.where(im==255,B,im)

            im2 = cv2.imread(os.path.join(roads_prefix,lbls,img))
            if not (im2 is None):
                im2 = np.where(im2==255,R,im2)
                im = np.add(im,im2)
            
            cv2.imwrite(os.path.join(prefix_seg,imgs,img),imm)
            cv2.imwrite(os.path.join(prefix_seg,lbls,img),im)

    for img in os.listdir(os.path.join(roads_prefix,imgs)):
        if img.endswith('.png'):
            if not os.path.isfile(os.path.join(prefix_seg,imgs,img)):
                imm = cv2.imread(os.path.join(roads_prefix,imgs,img))
                im = cv2.imread(os.path.join(roads_prefix,lbls,img))
                if not (im is None):
                    im = np.where(im==255,R,im)
                    cv2.imwrite(os.path.join(prefix_seg,imgs,img),imm)
                    cv2.imwrite(os.path.join(prefix_seg,lbls,img),im)

def dataset_to_seg_detaset():

    B = 1
    R = 2

    building_train = "./data_png/buildings/train/"
    building_test = "./data_png/buildings/test/"
    road_train = "./data_png/roads/train/"
    road_test = "./data_png/roads/test/"

    imgs = "images/"
    lbls = "labels/"

    train_seg = "./data_png/train/"
    test_seg = "./data_png/test/"


    for img in os.listdir(building_test+imgs):
        if img.endswith('.png'):
            #check for building image
            imm = cv2.imread(building_test+imgs+img)
            im = cv2.imread(building_test+lbls+img)
            im = np.where(im==255,B,im)

            im2 = cv2.imread(road_test+lbls+img)
            if not (im2 is None):
                im2 = np.where(im2==255,R,im2)
                im = np.add(im,im2)
            
            cv2.imwrite(test_seg+imgs+img,imm)
            cv2.imwrite(test_seg+lbls+img,im)

    for img in os.listdir(road_train+imgs):
        if img.endswith('.png'):
            if not os.path.isfile(train_seg+imgs+img):
                imm = cv2.imread(road_train+imgs+img)
                im = cv2.imread(road_train+lbls+img)
                if not (im is None):
                    im = np.where(im==255,R,im)
                    cv2.imwrite(train_seg+imgs+img,imm)
                    cv2.imwrite(train_seg+lbls+img,im)
            
