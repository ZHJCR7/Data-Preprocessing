"""
Extract face image from input video and save the frame of the video and txt file

Author: Jeffrey Chao
Data: 2021.05.20
"""
import os
from os.path import join
import cv2
import dlib
from PIL import Image as pil_image
from tqdm import tqdm
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

def get_boundingbox(face):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :return: x1, x2, y1, y2 coordinates
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    return x1, x2, y1, y2

def get_landmark(face, frame, predictor):
    """
    Get the 68 landmarks of the extracted face
    :param face: dlib face class
    :param image: frame image
    :param predictor: dlib shape_predictor class to locate important landmark
    :return: the 68 landmarks numpy array
    """
    landmark = []
    landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, face).parts()])
    for idx, point in enumerate(landmarks):
        pos = [point[0, 0], point[0, 1]]
        landmark.append(pos)
    landmark = np.array(landmark)
    return landmark

def txt_file_write(list_file_save_path, img_path, label, xmin, xmax, ymin, ymax, landmark, mask=None):
    try:
        line = img_path + ',' + str(mask) + ',' + str(label) + ',' + \
               str(ymin) + ',' + str(ymax) + ',' + str(xmin) + ',' + str(xmax)
        for i in range(len(landmark)):
            line = line + ',' + str(landmark[i][0]) + ',' + str(landmark[i][1])

        with open(list_file_save_path, 'a+') as f:
            f.writelines(line)
            f.write('\n')

    except:
        pass


def crop_face_area_from_video(video_path, output_path, dat_path, label, list_file_save_path, start_frame=0, end_frame=None):

    print('Starting: {}'.format(video_path))

    # Read and write
    reader = cv2.VideoCapture(video_path)

    os.makedirs(output_path, exist_ok=True)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Face detector
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(dat_path)

    # Frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        ymin, ymax, xmin, xmax = 0, 0, 0, 0
        landmark = []
        height, width = image.shape[:2]

        # 2. Detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)

        for rotation in range(5):
            if len(faces):
                # For now only take biggest face
                face = faces[0]

                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                xmin, xmax, ymin, ymax = get_boundingbox(face)
                landmark = get_landmark(face, image, face_predictor)
                print('Find the face!')
                break
            else:
                image = cv2.transpose(image)
                image = cv2.flip(image, 0)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_detector(gray, 1)
                print('Rotate the frame ' + str(rotation) + 'times!')

        if rotation == 4 and len(faces) == 0:
            print('No face find!')
            break

        #write the image path, label, boundarybox, mask=none and landmark
        #1. creat the name of the new image
        filepath_img = output_path + "/" + str(frame_num).zfill(4) + '.jpg'
        txt_file_write(list_file_save_path, filepath_img, label, xmin, xmax, ymin, ymax, landmark)


        cv2.imwrite(filepath_img, image, [int(cv2.IMWRITE_JPEG_QUALITY),95])

    pbar.close()
    print('Finished! Output saved under {}'.format(output_path))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--extract_face_path', type=str, default='./output', help='Path to save the output frame')
    parser.add_argument('--video_pathname_list', type=str, default='./data_list', help='The list file of video')
    parser.add_argument('--dat_path', type=str, default='./shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--list_file_save_path', type=str, default='./data_list/train1.txt')
    parser.add_argument('--json_path', type=str, default='./train_label.json')


    args = parser.parse_args()

    ## Parameters parsing
    extract_face_path = args.extract_face_path
    video_pathname_list = args.video_pathname_list
    dat_path = args.dat_path
    list_file_save_path = args.list_file_save_path
    json_path = args.json_path

    video_path_list_filename = video_pathname_list + "/video_path_list.txt"

    if not os.path.exists(extract_face_path):
        os.makedirs(extract_face_path)

    if not os.path.exists(video_pathname_list):
        os.makedirs(video_pathname_list)

    if os.path.exists(list_file_save_path):
        os.remove(list_file_save_path)

    # extract the face and save the image to the file and print the output path list
    video_path_list = []
    with open(video_path_list_filename) as read_file:
        for line in read_file:
            video_path_list.append(line.strip())

    #load json file and get the label of the video
    load_json = open(json_path, "r")
    json_dictionary = json.load(load_json)

    for i in range(len(video_path_list)):
        video_path = video_path_list[i].split(".mp4")[0].split("./")[1]
        face_path = extract_face_path + "/" + video_path
        video_name = video_path_list[i].split("/")[-1]
        label = json_dictionary[video_name]
        if not os.path.exists(face_path):
            os.makedirs(face_path)
        crop_face_area_from_video(video_path_list[i], face_path, dat_path, label, list_file_save_path)

if __name__ == '__main__':
    main()