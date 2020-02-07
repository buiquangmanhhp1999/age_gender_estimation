import numpy as np
import cv2
import argparse
from tqdm import tqdm
from utils import get_meta
import config as cf
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=cf.IMAGE_SIZE,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def create_test_data():
    img_dir = Path('./test/')
    img_size = cf.IMAGE_SIZE
    num1_13 = 0
    num14_23 = 0
    num24_39 = 0
    num40_55 = 0
    num56_80 = 0
    male = 0
    female = 0
    data = []
    for img_path in img_dir.glob('*.jpg'):
        name = img_path.name    # [age]_[gender]_[race]_[date&time].jpg
        age, gender = name.split('_')[:2]

        img = cv2.imread(str(img_path))
        age = int(age)

        if 1 <= age <= 11:
            num1_13 += 1
            label_age = 0
        elif 12 <= age <= 23:
            num14_23 += 1
            label_age = 1
        elif 24 <= age <= 39:
            num24_39 += 1
            label_age = 2
        elif 40 <= age <= 55:
            num40_55 += 1
            label_age = 3
        else:
            num56_80 += 1
            label_age = 4

        label_gender = int(gender)
        if label_gender == 3:
            label_gender = 1
        if label_gender == 0:
            male += 1
        else:
            female += 1

        label_gender = int(gender)
        if label_gender == 3:
            label_gender = 1
        img = cv2.resize(img, (img_size, img_size), cv2.INTER_AREA)
        data.append((img, label_age, label_gender))

    print('Number of test data')
    print('1-13: ', num1_13)
    print('14-23: ', num14_23)
    print('24-39: ', num24_39)
    print('40-55: ', num40_55)
    print('56-80: ', num56_80)
    print('male: ', male)
    print('female: ', female)

    np.save('test.npy', data)


def create_train_data():
    img_dir = Path('./train/')
    img_size = cf.IMAGE_SIZE
    num1_13 = 0
    num14_23 = 0
    num24_39 = 0
    num40_55 = 0
    num56_80 = 0
    male = 0
    female = 0
    data = []
    for img_path in img_dir.glob('*.jpg'):
        name = img_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender = name.split('_')[:2]
        img = cv2.imread(str(img_path))

        age = int(age)
        if age >= 86:
            continue

        if 1 <= age <= 13:
            num1_13 += 1
            label_age = 0
        elif 14 <= age <= 23:
            num14_23 += 1
            label_age = 1
        elif 24 <= age <= 39:
            num24_39 += 1
            label_age = 2
        elif 40 <= age <= 55:
            num40_55 += 1
            label_age = 3
        else:
            num56_80 += 1
            label_age = 4

        label_gender = int(gender)
        if label_gender == 3:
            label_gender = 1

        if label_gender == 0:
            male += 1
        else:
            female += 1

        img = cv2.resize(img, (img_size, img_size), cv2.INTER_AREA)
        data.append((img, label_age, label_gender))

    print('Number of training data')
    print('1-13: ', num1_13)
    print('14-23: ', num14_23)
    print('24-39: ', num24_39)
    print('40-55: ', num40_55)
    print('56-80: ', num56_80)
    print('male: ', male)
    print('female: ', female)
    np.save('train.npy', data)


def read_data():
    args = get_args()
    db = args.db
    min_score = args.min_score

    root_path = "./{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age, face_location = get_meta(mat_path, db)

    sample_num = len(face_score)
    valid_sample_num = 0

    # create database from WIKI Dataset
    for i in tqdm(range(sample_num)):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        if age[i] >= 86:
            continue
        if age[i] <= 0:
            continue

        img = cv2.imread(root_path + str(full_path[i][0]))

        if int(gender[i]) == 0:
            label_gender = 1
        else:
            label_gender = 0
        cv2.imwrite('./crop/' + str(age[i]) + '_' + str(label_gender) + '_' + str(i) + '.jpg', img)

        valid_sample_num += 1
    print(valid_sample_num)


if __name__ == '__main__':
    # read_data()
    create_train_data()
    create_test_data()
