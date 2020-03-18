from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import optimizers
from keras.utils.generic_utils import get_custom_objects
from keras.layers import GlobalMaxPool2D, Dropout, Dense, Activation, BatchNormalization
from keras.models import Model
import numpy as np
import config as cf
from keras.backend import sigmoid
# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5,...
# Higher the number, the more complex the model is
# Now in this project, I use model EfficientNetB4
import data_provider as data_provider
from efficientnet.keras import EfficientNetB4 as Net


class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish_act(x, beta=1):
    return x * sigmoid(beta * x)


class Efficient_Net(object):
    def __init__(self, trainable=True):
        self.trainable = trainable
        if self.trainable:
            self.train_data = data_provider.Datasets()

        self.model = self.build_model()

        # Compile the model
        losses = {
            "age_output": "categorical_crossentropy",
            "gender_output": "categorical_crossentropy"
        }

        opt = optimizers.Adam(1e-3)
        self.model.compile(loss=losses, optimizer=opt, metrics=['acc'])

        # Train the part you added
        if self.trainable:
            self.model.summary()

    @staticmethod
    def build_age_branch(x):
        # Output age branch
        predictions_age = Dense(cf.NUM_AGE_CLASSES, activation="softmax", name='age_output')(x)

        return predictions_age

    @staticmethod
    def build_gender_branch(x):
        # Output gender branch
        predictions_gender = Dense(cf.NUM_GENDER_CLASSES, activation="softmax", name='gender_output')(x)

        return predictions_gender

    def build_model(self):
        get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

        # Model
        model = Net(weights='imagenet', include_top=False, input_shape=cf.input_shape)

        # Adding 2 fully-connected layers to B4.
        x = model.output

        x = BatchNormalization()(x)
        x = GlobalMaxPool2D(name='gap1')(x)
        x = Dropout(cf.DROPOUT_RATE, name='dropout1')(x)

        # Output layer
        predictions_age = self.build_age_branch(x)
        predictions_gender = self.build_gender_branch(x)
        model_final = Model(inputs=model.input, outputs=[predictions_age, predictions_gender])

        return model_final

    def train(self):
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_age_output_acc', factor=cf.DECAY_LR_RATE, patience=5, verbose=1, )
        # Model Checkpoint
        cpt_save = ModelCheckpoint('./weight.h5', save_best_only=True, monitor='val_age_output_acc', mode='max')

        print("Training......")

        trainX, trainAgeY, trainGenderY = self.train_data.gen()
        trainX = np.array(trainX)

        self.model.fit(trainX, {"age_output": trainAgeY, "gender_output": trainGenderY}, validation_split=0.15,
                       callbacks=[cpt_save, reduce_lr], verbose=1, epochs=cf.NUM_EPOCHS, shuffle=True,
                       batch_size=cf.BATCH_SIZE)
