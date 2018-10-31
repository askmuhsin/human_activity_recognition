"""
## end to end inference
## input video file -- output prediction class
## objectives
## 1. video --> ffmpeg --> frames in temp dir
## 2. frames --> preprocess --> frames in temp dir
## 3. frames --> CNN encoder --> encoded sequence in temp dir
## 4. seq --> LSTM RNN --> output class"""

import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model
from nn_model import LeNetSequenceGen

raw_frame_dst = './data/inference/raw_frames/'
processed_frame_dst = './data/inference/processed_frames/'
cnn_tf_model_loc = './model/'
# lstm_model_loc = './rnn_model/model_consensus_net.h5'
lstm_model_loc = './rnn_model/model_fc2_main.h5'
feauture_length = 84
padding_length = 100
video_file_name = '../data/segmented_set2/41_17_2.avi'   ## test file
actual_class = 2    ## for the default video

if len(sys.argv)>1:
    video_file_name = sys.argv[1]
    try:
        actual_class = video_file_name.split('/')[-1].split('_')[-1].split('.')[0]
    except:
        print("parameter formatting not a match!")

def generateImages(video_file_name):
    """to run ffpmeg command and generate images"""
    class_of_vid = getClassNum(video_file_name)
    seq_num = 99    ## arbitary
    ffmpeg_cmd = wrapFfmpegCmd(video_file_name, class_of_vid, seq_num)
    os.system(ffmpeg_cmd)

def getClassNum(video_file_name):
    """does some string operations to obtain classname"""
    file_parts = video_file_name.split('_')
    class_of_file = int(file_parts[-1].split('.')[0])
    return class_of_file

def wrapFfmpegCmd(file_name, cls_num, seq_num, loc_dst=raw_frame_dst, frame_rate=15):
    """Generates ffmpeg command to conver video into images"""
    file_loc = file_name
    gen_file_name = f"{cls_num}_{seq_num}_%d.png"
    gen_file_loc = loc_dst + gen_file_name
    command = f"ffmpeg -i {file_loc} -r {frame_rate} {gen_file_loc}"
    return command

def imgPreprocessor(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Size of original images
    # max-min range --> (332-216, 612, 244)
    # avg range --> (260, 380)
    img_gray = cv2.resize(img_gray, (70, 95))
    img_gauss = cv2.GaussianBlur(img_gray, (5,5), 0)
    img_norm = np.empty_like((img_gauss))
    img_norm = cv2.normalize(img_gauss, img_norm, 0, 255, cv2.NORM_MINMAX)
    return img_norm

def readImg(file_name, loc_src=raw_frame_dst):
    file_loc = loc_src + file_name
    img = cv2.imread(file_loc)
    return img

def writeImg(img, file_name, loc_dst=processed_frame_dst):
    file_loc = loc_dst + file_name
    cv2.imwrite(file_loc, img)

def preprocessAndWrite(file_name):
    img = readImg(file_name)
    img_p = imgPreprocessor(img)
    writeImg(img_p, file_name)

def readImgResized(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (32, 32))
    return img

def get_images(processed_file_name):
    for img in processed_file_name:
        file_name = processed_frame_dst + img
        if os.path.isfile(file_name):
            img = readImgResized(file_name)
            img = img[:,:,0]
            img = ((img-255)/255)
            test_img = img
            test_img = test_img.reshape(1, 32, 32, 1)
            yield test_img

def tf_inference(processed_file_name):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    fc2 = LeNetSequenceGen(x)
    logits = fc2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        X = []
        for item in get_images(processed_file_name):
            fc2_seq = sess.run(logits, feed_dict={x: item})
            X.append(fc2_seq[0])

    X = np.asarray(X)
    return X

def stackAndPad(array, feature_length):
    max_length = feature_length*padding_length   ## # len of encoded data per seq * max frames per Sequence
    if len(array)>padding_length:
        array = array[:padding_length]
    temp = np.concatenate(array, axis=0)
    pad_array = np.zeros((max_length - temp.shape[0]))
    padded = np.hstack((pad_array, temp))
    assert padded.shape[0]==max_length, "Sequence overflow"
    return padded

def main():
    os.system("rm ./data/inference/*/*")
    ## Generate frames from video
    generateImages(video_file_name)
    # apply preprocessing to frames and save to disk
    raw_file_name = os.listdir(raw_frame_dst)
    print("***********************************************************")
    print(f"Video file read! and {len(raw_file_name)} frames generated.")
    for img_file in raw_file_name:
        preprocessAndWrite(img_file)
    ## Neural net model CNN-encoder
    processed_file_name = os.listdir(processed_frame_dst)
    print("Frames preprocessing complete.")
    print("***********************************************************")
    X = tf_inference(processed_file_name)
    ## LSTM implementation
    X = stackAndPad(X, feauture_length)
    X = np.reshape(X, (1, padding_length, feauture_length))
    print("***********************************************************")
    print("CNN encoding generated.")
    model = load_model(lstm_model_loc)
    # print(model.summary())
    print("***********************************************************")
    print("Softmax:")
    print(model.predict(X))
    pred = model.predict_classes(X)[0]
    print("\n***********************************************************")
    print(f"Actual Class of the video --> {actual_class}, Model prediction --> {pred}")
    print()
    if model.predict(X)[0][pred] < 1e-1:
        print("Model prediction is week/wrong")
    else:
        if actual_class == pred:
            print("Model prediction is strong")
        # else:
        #     print("Model prediction is wrong")
    os.system("rm ./data/inference/*/*")

if __name__ == '__main__':
    main()
