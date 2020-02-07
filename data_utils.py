import cv2
import numpy as np
import config as cf
import os
import argparse
import keras


def getTestData():
    print('Loading age images test')
    test = np.load(os.path.join(os.getcwd(), 'data/test.npy'), allow_pickle=True)

    test_data = []

    for i in range(test.shape[0]):
        test_data.append(test[i])

    print('Number of age test data:', str(len(test_data)))
    return test_data


def data_test():
    datasets_test = getTestData()
    images = []
    age_labels = []
    gender_labels = []

    for i in range(len(datasets_test)):
        image = datasets_test[i][0] / 255.0
        age = datasets_test[i][1]
        gender = datasets_test[i][2]
        images.append(image)
        age_labels.append(age)
        gender_labels.append(gender)

    age_labels = keras.utils.to_categorical(age_labels, cf.NUM_AGE_CLASSES)
    gender_labels = keras.utils.to_categorical(gender_labels, cf.NUM_GENDER_CLASSES)

    return images, age_labels, gender_labels


def get_arguments():
    # argparse.ArgumentDefaultsHelpFormatter will add information about the default value of each of the arguments
    parser = argparse.ArgumentParser(description="This code detect faces from images and estimate age and gender ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight_file', type=str, default=cf.MODEL_PATH,
                        help='pretrained weight')
    parser.add_argument('--image_path', type=str, default='./image/selenagomez.jpg', help='path to image')
    parser.add_argument('--margin', type=float, default=0.1, help='margin around detected face ')
    parser.add_argument('--threshold', type=float, default=0.7, help='threshold to apply to detect faces')
    arguments = parser.parse_args()
    return arguments


def draw_labels_and_boxes(img, boxes, result, margin):
    class_ids_age = np.argmax(result[0], axis=1)
    class_ids_gender = np.argmax(result[1], axis=1)

    if len(class_ids_age) <= 0:
        print('No age predicted')
        return None

    if len(class_ids_gender) <= 0:
        print('No gender predicted')
        return None

    # Initialize color to perform each labels uniquely
    # colors = np.random.randint(0, 255, size=(num_age_label, 3), dtype='uint8')

    for i in range(len(class_ids_age)):
        # get the bounding box coordinates
        left, top, right, bottom = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        width = right - left
        height = bottom - top
        img_h, img_w = img.shape[:2]

        x1 = max(int(left - margin * width), 0)
        y1 = max(int(top - margin * height), 0)
        x2 = min(int(right + margin * width), img_w - 1)
        y2 = min(int(bottom + margin * height), img_h - 1)
        # Get the unique color for this class
        # color = [int(c) for c in colors[class_ids_age[i]]]
        # Color red
        color = (0, 0, 255)

        # classify label according to result

        if class_ids_age[i] == 0:
            age = '1-13'
        elif class_ids_age[i] == 1:
            age = '15-23'
        elif class_ids_age[i] == 2:
            age = '24-39'
        elif class_ids_age[i] == 3:
            age = '40-55'
        elif class_ids_age[i] == 4:
            age = '56-80'

        if class_ids_gender[i] == 0:
            gender = 'Male'
        else:
            gender = 'Female'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = 'Gender: {}. Age: {}'.format(gender, age)
        cv2.putText(img, text, (x1 - 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img




