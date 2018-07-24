import numpy as np
import cv2
import os

loc_src_1 = "/home/muhsin/code/master_thes/human_activity_sdha/data/img_preprocessed_set_1/"
loc_src_2 = "/home/muhsin/code/master_thes/human_activity_sdha/data/img_preprocessed_set_2/"

def getClassNum(video_file_name):
    """does some string operations to obtain class num"""
    file_parts = video_file_name.split('_')
    class_of_file = int(file_parts[0])
    return class_of_file

def readImg(file_name, loc_src):
    file_loc = loc_src + file_name
    img = cv2.imread(file_loc)
    img = cv2.resize(img, (32, 32))
    # img = cv2.resize(img, (28, 28))     ## for autoencoder
    return img

def load_data():
    X, y = [], []
    file_names = os.listdir(loc_src_1)
    for file in file_names:
        y.append(getClassNum(file))
        img = readImg(file, loc_src_1)
        X.append(img[:,:,0])

    file_names = os.listdir(loc_src_2)
    for file in file_names:
        y.append(getClassNum(file))
        img = readImg(file, loc_src_2)
        X.append(img[:,:,0])
    return X, y

def main():
    X, y = load_data()

if __name__ == '__main__':
    main()
