import os
import csv

folders = ['provided_data', 'generated_1', 'generated_2']

samples = []
for folder in folders:
    with open('../' + folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            line.append(folder)
            samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples:
                for i in range(3):
                    source_path = sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = '../' + sample[7] + '/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                measurement = float(sample[3])
                measurements.append(measurement)
                measurements.append(measurement + 0.2)
                measurements.append(measurement - 0.2)
            augmented_images, augmented_measurements = [], []
            for image,measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples)*6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*6,
                    nb_epoch=5)
model.save('model.h5')
exit()
