import csv
from scipy import misc
import numpy as np
import keras

IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120



lines = []
with open("./data/driving_log.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        filename = line[0].split('/')
        try:
            old_filename = '/'.join(['data',filename[-2],filename[-1]])
            new_filename = '/'.join(['data_cleaned',filename[-2],filename[-1]])
            img = misc.imread(old_filename)
            img = img[55:-30, :] #crop
            img = misc.imresize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interp='bicubic') #resize
            misc.imsave(new_filename, img)
            
        except Exception as e:
            print(e)
            print(old_filename, "::", new_filename)
        lines.append([new_filename, line[-4]])


print(len(lines))
fw = open("./data_cleaned/driving_log.csv", "w")
for line in lines:
    fw.write("{},{}\n".format(*line))
    #if  abs(float(line[1])) > 0.15:
    new_filename = line[0][:-4]+"_flipped.jpg"
    img = misc.imread(line[0])
    img = np.fliplr(img)
    fw.write("{},{}\n".format(new_filename, -1.0 * float(line[1])))
    misc.imsave(new_filename, img)
    
fw.close()

import csv

TRAIN_SPLIT_RATIO = 0.6
TEST_SPLIT_RATIO = 0.2
VALIDATION_SPLIT_RATIO = 0.2

data = []

with open("./data_cleaned/driving_log.csv") as f:
    reader = csv.reader(f)
    for line in reader:
        data.append(line)
        
data = np.array(data)
np.random.shuffle(data)
TRAIN_END = int(len(data)*TRAIN_SPLIT_RATIO)
VAL_END = TRAIN_END + int(len(data)*VALIDATION_SPLIT_RATIO)

train_data = data[:TRAIN_END]
validation_data = data[TRAIN_END:VAL_END]
test_data = data[VAL_END:]

total = 1.0* len(train_data)+len(test_data)+len(validation_data)
print(len(train_data), len(train_data)/total )
print(len(test_data), len(test_data)/total)
print(len(validation_data), len(validation_data)/total)

import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    samples = np.array(samples)
    np.random.shuffle(samples)
    print(samples[0])
    offset = 0
    while 1: # Loop forever so the generator never terminates
                    
        if offset >= num_samples:
            offset = 0
        vals = np.zeros((batch_size, 1))
        imgs = np.zeros((batch_size, IMAGE_HEIGHT,IMAGE_WIDTH, 3))
        index = 0
        while index < batch_size:
            batch_sample = samples[offset]
            offset += 1
            if offset >= num_samples:
                offset = 0
            filename = batch_sample[0]
            center_image = misc.imread(filename)
            imgs[index] = center_image
            vals[index][0] = float(batch_sample[1]) #steering angle
            index += 1

        X_train = imgs
        y_train = vals
            
        yield X_train, y_train
        

epochs = 100
batch_size = 128

train_data_size = len(train_data)
test_data_size = len(test_data)
validation_data_size = len(validation_data)

train_generator = generator(train_data, batch_size=batch_size)
validation_generator = generator(validation_data, batch_size=batch_size)
test_generator = generator(test_data, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.advanced_activations import ELU
import keras.regularizers as regularizers



model = Sequential()
#model.add(Cropping2D(cropping=((10,2),(0,0)), input_shape=(32, 32, 3)))
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH, 3)))
model.add(Conv2D(64, (5,5)))
model.add(ELU())
model.add(Conv2D(64, (5,5)))
model.add(ELU())
model.add(Conv2D(32, (3,3), strides=(2, 2)))
model.add(ELU())
model.add(Conv2D(32, (3,3), strides=(2, 2)))
model.add(ELU())
model.add(Flatten())
model.add(Dense(1024))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(64))
model.add(ELU())
model.add(Dense(1))

from keras.optimizers import Adam

model.compile(optimizer=Adam(lr=.0001), loss='mse', metrics=[ 'mse'])
import h5py
from keras.callbacks import ModelCheckpoint, Callback
checkpoint_callback = ModelCheckpoint('ckpts/model{epoch:02d}.h5')
history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=train_data_size/batch_size, \
                   validation_data=validation_generator, validation_steps=validation_data_size/batch_size, \
                    max_q_size=2, callbacks=[checkpoint_callback], initial_epoch=0)

metrics = model.evaluate_generator(test_generator, val_samples=test_data_size)
for i in zip(model.metrics_names, metrics):
    print(i)

model.save("model_12_31_17.h5")

#from keras.models import load_model
#model = load_model("model22.h5")