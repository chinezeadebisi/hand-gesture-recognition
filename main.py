#!/usr/bin/env python
# -*- coding: utf-8 -*-

# To run: python main.py
# reference script : https://google.github.io/mediapipe/solutions/hands.html
# reference code - https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
# Youtube video - https://youtu.be/f7uBsb-0sGQ

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils

import csv
from model import Classifier
import copy
from utils import CvFpsCalc
import itertools

# main functions for all code starts from here
def main():
    # initilize parameters
    cap_width = 960
    cap_height = 540
    cap_device = 0 
    use_static_image_mode = 'store_true'
    min_tracking_confidence = 0.5
    min_detection_confidence = 0.5
    
    # initilize webcam
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # initilize hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # initlize gesture classifier
    classifier = Classifier()

    # read lables of gestures
    with open('model/classifier/labels.csv', encoding='utf-8-sig') as f:
        labels = csv.reader(f)
        labels = [
            row[0] for row in labels
        ]

    # initilize FPS calculator
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0
    
    while True:
        # find FPS
        fps = cvFpsCalc.get()

        # read image from web camera
        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)  # flip image left to right

        # copy image for processing
        debug_img = copy.deepcopy(image)

        # set proper color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # detect hand from image
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # check if hand is found in image or not
        if results.multi_hand_landmarks is not None:
            # process on each hand
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # find box which contain hand
                brect = find_bounding_rect(debug_img, landmarks)

                # find landmarks
                landmark_list = find_landmarks(debug_img, landmarks)

                # preprocess landmark
                pre_processed_list = process_landmark_for_recognition(landmark_list)

                # recognize hand gesture
                hand_sign_no = classifier(pre_processed_list)

                # draw landmarks on live video
                cv2.rectangle(debug_img, (brect[0], brect[1]), (brect[2], brect[3]),(255, 0, 0), 1)
                mp_drawing.draw_landmarks(debug_img, landmarks, mp_hands.HAND_CONNECTIONS)
                debug_img = draw_text_info(debug_img, brect, handedness, labels[hand_sign_no],)
        
        # write text on image
        cv2.putText(debug_img, "FPS: " + str(fps), (11, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture Recognition Tool', debug_img)

        # press ESC to stop camera
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# find bounding rectangle 
def find_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# The input function to find hand landmark
def find_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# transform to valid landmark for recogntion
def process_landmark_for_recognition(point_list):
    temp_list = copy.deepcopy(point_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_list[index][0] = temp_list[index][0] - base_x
        temp_list[index][1] = temp_list[index][1] - base_y

    temp_list = list(itertools.chain.from_iterable(temp_list))

    max_value = max(list(map(abs, temp_list)))

    def normalize_(n):
        return n / max_value

    temp_list = list(map(normalize_, temp_list))

    return temp_list

# write text on video
def draw_text_info(image, brect, handedness, sign_text):
    hand_text = handedness.classification[0].label[0:]
    if sign_text != "":
        hand_text = hand_text + ': ' + sign_text
    cv2.putText(image, hand_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    return image

if __name__ == '__main__':
    main()
