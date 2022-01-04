import glob
import numpy as np
import random

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import copy

def shuffle(xd, yl):
    n = int(len(xd)/2)
    while n > 0:
        i, j = random.randint(0, len(xd)-1), random.randint(0, len(xd)-1)
        xd[i], xd[j] = xd[j], xd[i]
        yl[i], yl[j] = yl[j], yl[i]
        n -= 1
    return xd, yl

path = './npy'
pose_data = []
label = []
label_cat = {'fall':0, 'sit':1, 'stand':2, 'walk':3}

batch_size = 16
epochs = 50
num_classes = len(label_cat)

for np_name in glob.glob(path + '/*.npy'):
    data_read = np.load(np_name, allow_pickle=True)
    print("Converting", np_name[len(path)+1:], "shape", data_read.shape)

    # remove index (keypoint id)
    data = []
    for pose_coords in data_read:
        data_temp = []
        for coord in pose_coords:
            data_temp.append(coord[1:3])
        data.append(data_temp)

    # centroid calculation for keypoint
    for pose_coords in data:
        for i in range(1, len(pose_coords)):
            pose_coords[i][0] = pose_coords[i][0] - pose_coords[0][0]
            pose_coords[i][1] = pose_coords[i][1] - pose_coords[0][1]
        pose_coords[0][0], pose_coords[0][1] = 0, 0

    # collect and label dataset
    pose_data_temp = []
    for pose_coords in data:
        if len(pose_data_temp) == batch_size:
            # acceleration calculation for keypoint
            pose_data_temp_cpy1 = copy.deepcopy(pose_data_temp)
            pose_data_temp_cpy2 = copy.deepcopy(pose_data_temp)
            for i in range(len(pose_data_temp_cpy1)-1):
                for j in range(len(pose_data_temp_cpy1[i])):
                    pose_data_temp_cpy1[i+1][j][0] = pose_data_temp_cpy1[i+1][j][0] - pose_data_temp_cpy2[i][j][0]
                    pose_data_temp_cpy1[i+1][j][1] = pose_data_temp_cpy1[i+1][j][1] - pose_data_temp_cpy2[i][j][1]

            pose_data.append(pose_data_temp_cpy1)
            label.append(label_cat[np_name[len(path)+1:-11]])
            pose_data_temp.pop(0)
        pose_data_temp.append(pose_coords)

print('Successful converting data')

pose_data, label = shuffle(pose_data, label)
x_dataset = np.array(pose_data)
y_label = np.array(label)

print('Original data shape: ', x_dataset.shape, y_label.shape)

X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_label, test_size=0.2,random_state=109) # 70% training and 30% test
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
print('Training data shape: ', X_train.shape, y_train.shape)
print('Testing data shape: ', X_test.shape, y_test.shape)
print('Total number of outputs : ', len(label_cat))
print('Output classes : ', label_cat)

X_train = X_train.reshape(-1, batch_size, 12, 2)
X_test = X_test.reshape(-1, batch_size, 12, 2)
print(X_train.shape, X_test.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.

train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)

X_train,valid_X,train_label,valid_label = train_test_split(X_train, train_Y_one_hot, test_size=0.2, random_state=13)

fall_model = Sequential()
fall_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(batch_size, 12, 2),padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))
fall_model.add(MaxPooling2D((2, 2),padding='same'))
fall_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))
fall_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fall_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fall_model.add(LeakyReLU(alpha=0.1))                  
fall_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fall_model.add(Flatten())
fall_model.add(Dense(128, activation='linear'))
fall_model.add(LeakyReLU(alpha=0.1))                  
fall_model.add(Dense(num_classes, activation='softmax'))
fall_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])
fall_model.summary()
fall_train = fall_model.fit(X_train, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
fall_model.save("./model_keras/gait_keras.h5py")

test_eval = fall_model.evaluate(X_test, test_Y_one_hot, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
# target_names = ["Class {}".format(i) for i in range(num_classes)]
target_names = ["{}".format(str(i)) for i in label_cat]
predicted_classes = fall_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(classification_report(y_test, predicted_classes, target_names=target_names))