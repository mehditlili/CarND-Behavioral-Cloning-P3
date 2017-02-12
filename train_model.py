import numpy as np
import cv2

### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import csv
import glob
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.activations import relu
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def draw(img, title='', color=cv2.COLOR_BGR2RGB):
    color_img = cv2.cvtColor(img, color)
    """ Draw a single image with a title """
    f = plt.figure()
    plt.title(title)
    plt.imshow(color_img, cmap='gray')
    plt.show()
    plt.close(f)


def crop_image(img, x, y, w, h):
    y2 = y + h
    x2 = x + w
    return img[y:y2, x:x2]


def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def read_resize(path):
    image = cv2.imread(path)
    return resize_image(crop_image(image, 0, 50, image.shape[1], 80), 200, 66)
    #return crop_image(image, 0, 50, image.shape[1], 80)


def build_training_data():
    images = []
    steering_angles = []
    base_path = "/home/mehdi/Desktop/"
    folders = ["one_lap/", "avoiding_edges/", "few_laps/"]
    for folder in folders:
        # Read in the driving log CSV
        path = base_path + folder
        image_list = glob.glob(path + "IMG/center*.jpg")
        test_image = read_resize(image_list[0])
        height, width, channels = test_image.shape
        print("Image dimensions:", width, height)
        with open(path + 'driving_log.csv') as csvfile:
            log_data = csv.reader(csvfile)
            row_count = sum(1 for row in log_data)
            print("In total %s images were acquired" % row_count)
            print("In total %s images were found" % len(image_list))
            assert (row_count == len(image_list))
        with open(path + 'driving_log.csv') as csvfile:
            log_data = csv.reader(csvfile)
            zero_images = []
            for row in log_data:
                if np.fabs(float(row[3])) > 0.00000001:
                    image_file = row[0].split(os.sep)[-1]
                    print(image_file)
                    images.append(read_resize(path+"IMG/"+image_file))
                    steering_angles.append(1.0*float(row[3]))
                    images.append(cv2.flip(images[-1], 1))
                    steering_angles.append(-1.0 * float(row[3]))
                else:
                    image_file = row[0].split(os.sep)[-1]
                    zero_images.append(read_resize(path+"IMG/"+image_file))
        print("number of no steering images: %s" % len(zero_images))
        print("number of steering images: %s" % len(images))
        images += zero_images
        steering_angles += [0.0]*len(zero_images)
    return np.array(images), np.array(steering_angles)

from keras.layers.pooling import MaxPooling2D

def create_model3(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(83))
    model.add(Dense(1))
    model.compile("adam", loss="mse")
    return model

def create_model(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(6, 5, 5, activation="relu"))
    model.add(MaxPooling2D())
    #model.add(Activation('relu'))
    #model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    # model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    #model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    #model.add(Dense(1024))
    # model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.compile('adam', loss='mse')
    return model


def create_model2(input_shape):
    model = Sequential()

    model.add(Convolution2D(8, 4, 4, border_mode='valid', input_shape=input_shape))
    model.add(Convolution2D(16, 4, 4, border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.8))

    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    return model

def create_model4(input_shape):
    drive_model = Sequential()

    drive_model.add(Convolution2D(24, 5, 5, input_shape=input_shape, activation='relu'))
    drive_model.add(MaxPooling2D((2, 2), border_mode='same'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Convolution2D(36, 5, 5, activation='relu'))
    drive_model.add(MaxPooling2D((2, 2), border_mode='same'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Convolution2D(48, 5, 5, activation='relu'))
    drive_model.add(MaxPooling2D((2, 2), border_mode='same'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Convolution2D(64, 3, 3, activation='relu'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Convolution2D(64, 3, 3, activation='relu'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Flatten())

    drive_model.add(Dense(100, activation='relu'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Dense(50, activation='relu'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Dense(10, activation='tanh'))
    drive_model.add(Dropout(0.5))

    drive_model.add(Dense(1))
    # Compile and reload the model
    drive_model.compile(optimizer='Adam', loss='mse', lr=0.00002)
    return drive_model


images, steering_angles = build_training_data()
X_train, X_test, y_train, y_test = train_test_split(images, steering_angles, test_size=0.2)

print("Training images shape:")
print(X_train.shape)
print("Training steering angles:")
print(y_train.shape)

print("Testing images shape:")
print(X_test.shape)
print("Testing steering angles:")
print(y_test.shape)

X_train, y_train = shuffle(X_train, y_train)

model = create_model((images.shape[1], images.shape[2], images.shape[3]))
#model = create_model2((images.shape[1], images.shape[2], images.shape[3]))


def generate_batch_from_data(data, size=32):
    images = np.zeros((size, data[0][0].shape))
    steering_angles = np.zeros(size)
    while 1:
        for i in range(size):
            #row = data.iloc[[np.random.randint(len(data))]].reset_index()
            #images[i] = preprocess(read_image(row['center'][0]))
            images[i] = data[0][i]
            #steering_angles[i] = row['steering'][0]
            steering_angles[i] = data[1][i]
        print(images.shape)
        print(steering_angles)
        yield images, steering_angles


# Train the model
# History is a record of training loss and metrics
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=5, validation_split=0.2, shuffle=True)
#history = model.fit_generator(generate_batch_from_data([X_train, y_train]), samples_per_epoch=1000, nb_epoch=10)

# Calculate test score
test_score = model.evaluate(X_test, y_test)

print(test_score)

# history = model.fit_generator(generate_batch_from_data(data_train),
#                               samples_per_epoch=10000, nb_epoch=5,
#                               validation_data = generate_batch_from_data(data_val),
#                               nb_val_samples = 2048,
#                               verbose = 2)

print("Writing model.h5")
model.save('model.h5')
