from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D

# This model has been inspired by VGG Net:
# https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
# https://arxiv.org/pdf/1409.1556v6.pdf

class VGGNet:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return VGGNet.img_rows, VGGNet.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], VGGNet.img_rows, VGGNet.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], VGGNet.img_rows, VGGNet.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(28,28,1)))
        # model.add(Conv2D(filters=64, kernel_size=1, activation='relu'))
        # model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
        # model.add(Conv2D(filters=128, kernel_size=1, activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(filters=256, kernel_size=1, activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(filters=512, kernel_size=1, activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        # model.add(Conv2D(filters=512, kernel_size=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=1000, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
