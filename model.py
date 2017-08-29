import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

filepath="./models/t1/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

lines = []

def read_img_path(csv_path, data_path=''):
    lines = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            path = data_path
            line[0] = path + line[0].replace(" ", "")
            line[1] = path + line[1].replace(" ", "")
            line[2] = path + line[2].replace(" ", "")
            lines.append(line)
            #print(line)
        return lines

lines = read_img_path('./data/driving_log.csv', './data/')
new_lines = read_img_path('./new_data/driving_log.csv')


lines.extend(new_lines)
print(np.array(lines).shape)
i=0
for line in lines:
    if len(line[0]) < 15:
        del lines[i] # Delete index
    i += 1
print(np.array(lines).shape)
train_lines, valid_lines = train_test_split(lines, test_size=0.2)

def process_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return img

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples[1:])
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
                
                steering_center = float(row[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                img_center = process_img(cv2.imread(row[0]))
                img_center_fliped = np.fliplr(img_center)
                img_left = process_img(cv2.imread(row[1]))
                img_left_fliped = np.fliplr(img_left)
                img_right = process_img(cv2.imread(row[2]))
                img_right_fliped = np.fliplr(img_right)

                images.append(img_center)
                angles.append(steering_center)
                images.append(img_left)
                angles.append(steering_left)
                images.append(img_right)
                angles.append(steering_right)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
train_generator = generator(train_lines, batch_size=64)
validation_generator = generator(valid_lines, batch_size=64)

import keras
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.activations import *

image_shape = (160,320,3)#X_train[0].shape

model = Sequential()
model.add(Lambda(lambda x: x / 225.0 - 0.5, input_shape=image_shape))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24, (5,5), strides=(2,2), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(SpatialDropout2D(0.5))

model.add(Conv2D(36, (5,5), strides=(2,2), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(SpatialDropout2D(0.5))

model.add(Conv2D(48, (5,5), strides=(2,2), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(SpatialDropout2D(0.5))

model.add(Conv2D(64, (3,3), strides=(1,1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(SpatialDropout2D(0.5))

model.add(Conv2D(64, (3,3), strides=(1,1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(SpatialDropout2D(0.5))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Dense(50,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Dense(10,kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', lr=0.000001)


filepath="./models/track_1/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history_object = model.fit_generator(train_generator, 
                                     steps_per_epoch=len(train_lines), 
                                     validation_data=validation_generator,
                                     validation_steps=len(valid_lines),
                                     epochs=100,
                                     callbacks=callbacks_list)

print(history_object.history.keys())
plt.rcParams['figure.figsize'] = (16, 9)

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'][1:])
plt.plot(history_object.history['val_loss'][1:])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('./models/track_1/final_model.h5')



