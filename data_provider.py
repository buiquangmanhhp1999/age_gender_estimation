import keras
import Project.age_gender.config as cf
import os
import numpy as np


class Datasets(object):
    def __init__(self):
        self.datasets_train = self.getTrainData()
        self.all_data = []
        self.convert_data_format()

    def gen(self):
        np.random.shuffle(self.all_data)

        images = []
        age_labels = []
        gender_labels = []

        for i in range(len(self.all_data)):
            image, age, gender = self.all_data[i]
            images.append(image)
            age_labels.append(age)
            gender_labels.append(gender)

        age_labels = keras.utils.to_categorical(age_labels, num_classes=cf.NUM_AGE_CLASSES)
        gender_labels = keras.utils.to_categorical(gender_labels, num_classes=cf.NUM_GENDER_CLASSES)
        return images, age_labels, gender_labels

    @staticmethod
    def getTrainData():
        print('Loading age image...')
        train = np.load(os.path.join(os.getcwd(), 'data/train.npy'), allow_pickle=True)

        train_data = []

        for i in range(train.shape[0]):
            train_data.append(train[i])

        print('Number of age train data:', str(len(train_data)))

        return train_data

    def convert_data_format(self):
        # Age datasets:
        for i in range(len(self.datasets_train)):
            image = self.datasets_train[i][0] / 255.0
            age_labels = self.datasets_train[i][1]
            gender_labels = self.datasets_train[i][2]
            self.all_data.append((image, age_labels, gender_labels))


