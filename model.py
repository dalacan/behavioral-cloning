# Load modules
import csv
import copy
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Define hyper parameters
batch_size = 64
correction = 0.2
epochs = 3
model_file_name = 'model.h5'

class Model:
    def __init__(self):
        self.model = Sequential()

    def load_driving_data(self, filepath='./testdrive1/', augment_flip_data=False, correction_value=0.2):
        '''
        Load driving data from specified file path
        :param filepath:
        :param augment_flip_data:
        :param correction:
        :return:
        '''
        driving_log = []
        with open(filepath + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                # Load center image path and steering angle
                driving_log.append([filepath + line[0], float(line[3]), 1])

                # Load left image path and steering angle
                driving_log.append([filepath + line[1], float(line[3]) + correction_value, 1])

                # Load right image path and steering angle
                driving_log.append([filepath + line[2], float(line[3]) - correction_value, 1])

        # Add a duplicate of the driving log data as a flipped image & steering
        if augment_flip_data == True:
            augment_driving_log = copy.deepcopy(driving_log)
            for log in augment_driving_log:
                log[2] = 0  # Flag record as 'flipped' which will be used later for image flipping
                log[1] = str(float(log[1]) * -1)  # Flip steering measurement

            driving_log = driving_log + augment_driving_log

        # Return the data
        return driving_log

    def split_driving_data(self, data, test_size=0.2):
        train_samples, validation_samples = train_test_split(data, test_size=test_size)
        return train_samples, validation_samples

    def random_brightness(self, image):
        '''
        Apply random brightness
        :param image:
        :return:
        '''
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 2] = image[:, :, 2] * (.5 + np.random.uniform())
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    def generator(self, samples, batch_size=32, augment_flip_data=False, augment_brightness=False):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    # name = './testdrive1/IMG/'+batch_sample[0].split('/')[-1]
                    name = batch_sample[0]

                    center_image = cv2.imread(name)
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                    if augment_flip_data == True and batch_sample[2] == 0:
                        # Flip image
                        center_image = cv2.flip(center_image, 1)

                    # Apply random brightness
                    if augment_brightness == True:
                        center_image = self.random_brightness(center_image)

                    center_angle = float(batch_sample[1])
                    images.append(center_image)
                    angles.append(center_angle)
                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def load_data(self, augment_flip_data=False):
        '''
        Loads and augment the driving data and returns the training and validation generator
        :param augment_flip_data:
        :return:
        '''
        # Load the image and steering data
        # Load Track 1 driving log
        # driving_log = load_driving_data(filepath='./drivelogs/track-1/', augment_flip_data=augment_flip_data)
        driving_log = self.load_driving_data(filepath='./drivelogs/track-1-2/', augment_flip_data=augment_flip_data, correction_value=correction)

        # Load track 1 in reverse driving log
        driving_log = driving_log + self.load_driving_data(filepath='./drivelogs/track-1-reverse/', augment_flip_data=augment_flip_data, correction_value=correction)

        # Load track 1 recovery driving log
        # driving_log = driving_log + load_driving_data(filepath='./drivelogs/track-1-recovery/', augment_flip_data=augment_flip_data)
        # driving_log = driving_log + load_driving_data(filepath='./drivelogs/track-1-recovery-2/', augment_flip_data=augment_flip_data)

        # Load track 2 driving log
        driving_log = driving_log + self.load_driving_data(filepath='./drivelogs/track-2-2/', augment_flip_data=False, correction_value=correction)
        # driving_log = driving_log + load_driving_data(filepath='./drivelogs/track-2/', augment_flip_data=augment_flip_data)
        # driving_log = driving_log + load_driving_data(filepath='./drivelogs/track-2-1/', augment_flip_data=augment_flip_data)

        # Load track 2 in reverse
        driving_log = driving_log + self.load_driving_data(filepath='./drivelogs/track-2-reverse/', augment_flip_data=augment_flip_data, correction_value=correction)

        # Load track 2 recovery
        driving_log = driving_log + self.load_driving_data(filepath='./drivelogs/track-2-recovery/', augment_flip_data=augment_flip_data, correction_value=correction)

        print('Driving Log Samples: {}'.format(len(driving_log)))

        train_samples, validation_samples = self.split_driving_data(driving_log, 0.2)

        validation_samples, test_samples = self.split_driving_data(validation_samples, 0.3)

        # compile and train the model using the generator function
        train_generator = self.generator(train_samples, batch_size=batch_size, augment_flip_data=augment_flip_data, augment_brightness=True)
        validation_generator = self.generator(validation_samples, batch_size=batch_size, augment_flip_data=augment_flip_data)
        test_generator = self.generator(test_samples, batch_size=batch_size, augment_flip_data=augment_flip_data)

        return train_samples, validation_samples, test_samples, train_generator, validation_generator, test_generator


    def visualize_hist(self, data):
        '''
        Visualize the distribution of the label values
        :param data:
        :return:
        '''

        fig, axes = plt.subplots()

        #     axes.hist(classes, bins=10, histtype='stepfilled', stacked=True, alpha=0.8, density=True)
        axes.hist(data,  histtype='bar')
        axes.set_title('Class distribution')
        axes.legend(prop={'size': 10})

        fig.tight_layout()
        plt.show()

    def visualize_normal_dist(self, data):
        '''
        Visualize the normal distribution of the labels
        :param data:
        :return:
        '''
        x = np.arange(np.min(data), np.max(data), 0.001)
        y_train_mean = np.mean(data)
        y_train_sd = np.std(data)
        print('mean:', y_train_mean)
        print('standard deviation', y_train_sd)
        y_norm = scipy.stats.norm.pdf(x, y_train_mean, y_train_sd)

        fig, ax = plt.subplots()
        ax.plot(x, y_norm, '--')
        plt.show()

    def lenet(self):
        # Layer 1
        self.model.add(Conv2D(6, (5, 5), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))

        # Layer 2
        self.model.add(Conv2D(16, (5, 5), strides=1, activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))

        # Layer 3
        self.model.add(Flatten())

        # Fully connected layer
        self.model.add(Dense(120))
        self.model.add(Dense(84))

    def nvidia(self):
        # Convolution layer 1
        self.model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))

        # Convolution layer 2
        self.model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))

        # Convolution layer 3
        self.model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))

        # Convolution layer 4
        self.model.add(Conv2D(64, (3, 3), activation='relu'))

        # Convolution layer 5
        self.model.add(Conv2D(64, (3, 3), activation='relu'))

        # Flatten layer
        self.model.add(Flatten())

        # Connected layers
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))

    def run_model(self, train_samples, validation_samples, train_generator, validation_generator):
        # Preprocess incoming data, centered around zero with small standard deviation
        self.model.add(Lambda(lambda x: x / 255.5 - 0.5, input_shape=(160, 320, 3)))

        # Crop image. Crop 50 pixels from the top, 20 from the bottom, 0 from left, 0 from right
        self.model.add(Cropping2D(cropping=((50, 20), (0, 0))))

        # Load model
        self.nvidia()

        # Output to only predict steering measurement
        self.model.add(Dense(1))

        self.model.compile(loss='mse', optimizer='adam')

        history_object = self.model.fit_generator(train_generator,
                                                  steps_per_epoch=len(train_samples)//batch_size,
                                                  validation_data=validation_generator,
                                                  validation_steps=len(validation_samples)//batch_size,
                                                  epochs=epochs,
                                                  )

        # Output model summary
        self.model.summary()

        ### print the keys contained in the history object
        print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    def save_model(self, model_name):
        # Save model file
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model = load_model(model_name)


    def training_pipeline(self):
        # Load data
        train_samples, validation_samples, test_samples, train_generator, validation_generator, test_generator = self.load_data(True)

        # Visualize data
        steering_data = np.array(train_samples)[:, 1]
        self.visualize_hist(np.round(steering_data.astype(np.float), 6))
        self.visualize_normal_dist(np.round(steering_data.astype(np.float), 3))

        # Run model
        self.run_model(train_samples, validation_samples, train_generator, validation_generator)

        # Save model
        self.save_model(model_file_name)

        self.evaluate(test_generator, test_samples)

    def evaluation_pipeline(self):
        self.load_model(model_file_name)
        train_samples, validation_samples, test_samples, train_generator, validation_generator, test_generator = self.load_data(True)
        self.evaluate(test_generator, test_samples)

    def evaluate(self, test_generator, test_samples):
        score = self.model.evaluate_generator(test_generator, steps=len(test_samples) // batch_size)
        print('Test loss:', score)


model = Model()
# Run training pipline
model.training_pipeline()

# Run evaluation pipeline
# model.evaluation_pipeline()

