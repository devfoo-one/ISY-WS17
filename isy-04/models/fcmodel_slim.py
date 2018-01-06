from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D

class FCModelSlim:

    @staticmethod
    def load_inputshape():
        return FCModelSlim.img_rows, FCModelSlim.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        model = Sequential()
        model.add(Dense(units=784, activation='relu', input_dim=784))
        model.add(Dense(units=784, activation='relu'))
        model.add(Dense(units=784, activation='relu'))
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
