import numpy as np
import cv2
import csv
from keras.models import Sequential, load_model
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from itertools import compress
from copy import deepcopy


def resize_image(img, width, height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def read_csv(base_path, data_folders_list, zero_ratio=0.05, side_angle_offset=0.1):
    """
    A smart csv reading function. Prepares the data for the generator
    If left and right camera images are available, they will be placed in separate lines to avoid confusion
    when it comes to calculating the total number of training samples
    :param base_path: path of the folder where training data sets are stored
    :param data_folders_list: list of different training data sets
    :param zero_ratio: how many training data with steering angle = 0 to keep
    :param side_angle_offset: when using left and right camera, what angle offset shoulf be added/deduced
    :return:
    """
    samples = []
    zero_steering = []
    steering_angles = []
    for data_folder in data_folders_list:
        folder_path = data_folder[0]
        csv_file = base_path + folder_path + "driving_log.csv"
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # Avoid reading row from udacity data that contains the columns titles
                if line[0].split('/')[-1] == "center":
                    continue
                # Update the path of the image, the one in the csv might be wrong
                line[0] = base_path + folder_path + "IMG/" + line[0].split('/')[-1]
                samples.append(line)
                steering_angles.append(float(line[3]))
                # Check if steering is close to zero
                if np.fabs(float(line[3])) < 0.0001:
                    zero_steering.append(len(samples) - 1)
    # Reduce the number of frames with zero steering
    plt.subplot(311)
    plt.hist(np.array(steering_angles), 100, color='green', alpha=0.8)
    plt.title("Before zero reducing")
    regu_samples, regu_steering_angles = discard_zeros(samples, steering_angles, zero_steering, zero_ratio)
    plt.subplot(312)
    plt.hist(np.array(regu_steering_angles), 100, color='green', alpha=0.8)
    plt.title("After zero reducing")
    print("len samples ", len(samples))
    print("len regulated samples ", len(regu_samples))
    # If left and right are also to be considered
    aug_samples = []
    aug_steerings = []
    if not only_center:
        for sample in regu_samples:
            # If left image available, update its angle and put it as fake center
            if sample[1] != "":
                line_left = deepcopy(sample)
                line_left[0] = sample[1]
                line_left[1] = ""
                line_left[2] = ""
                angle = float(line_left[3]) + side_angle_offset
                line_left[3] = str(angle)
                aug_steerings.append(angle)
                sample[1] = ""
                line_left[0] = base_path + folder_path + "IMG/" + line_left[0].split('/')[-1]
                aug_samples.append(line_left)
            if sample[2] != "":
                line_right = deepcopy(sample)
                line_right[0] = sample[2]
                line_right[1] = ""
                line_right[2] = ""
                angle = float(line_right[3]) - side_angle_offset
                line_right[3] = str(angle)
                aug_steerings.append(angle)
                sample[2] = ""
                line_right[0] = base_path + folder_path + "IMG/" + line_right[0].split('/')[-1]
                aug_samples.append(line_right)

    plt.subplot(313)
    plt.hist(np.array(aug_steerings+regu_steering_angles), 100, color='green', alpha=0.8)
    plt.title("After zero reducing and data augmentation")
    plt.tight_layout()
    plt.show()
    return regu_samples + aug_samples


def discard_zeros(samples, steering_angles, zero_steering, zero_ratio):
    shuffle(zero_steering)
    zero_discarding = zero_steering[int(len(zero_steering) * zero_ratio):]
    ranges = range(len(samples))
    keeping = [x not in zero_discarding for x in ranges]
    steering_angles = list(compress(steering_angles, keeping))
    samples = list(compress(samples, keeping))
    return samples, steering_angles


def create_model(input_shape):
    cnn_model = Sequential()
    cnn_model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=input_shape))
    cnn_model.add(Lambda(lambda x: x / 255.0 - 0.5))
    cnn_model.add(Convolution2D(6, 5, 5, activation="relu"))
    cnn_model.add(MaxPooling2D())
    cnn_model.add(Activation('relu'))
    cnn_model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Convolution2D(64, 3, 3))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Convolution2D(64, 3, 3))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1024))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(128))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(64))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(16))
    cnn_model.add(Activation('relu'))
    cnn_model.add(Dense(1))
    cnn_model.compile('adam', loss='mse')
    #cnn_model = load_model("model_test.h5")
    return cnn_model


def data_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(1.0 * center_angle)
                images.append(cv2.flip(center_image, 1))
                angles.append(-1.0 * center_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


base_path = "/home/mehdi/Desktop/project3/"
folders = [["udacity/", 1]]
only_center = False
samples = read_csv(base_path, folders, side_angle_offset=0.1)


print("Total number of samples: %s" % len(samples))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = data_generator(train_samples, batch_size=32)
validation_generator = data_generator(validation_samples, batch_size=32)
test_image = cv2.imread(samples[0][0])

model = create_model(tuple(test_image.shape))

# Train the model
# History is a record of training loss and metrics
history = model.fit_generator(train_generator, samples_per_epoch=2*len(train_samples),
                              validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

print("Writing model.h5")
model.save('model_test2.h5')
