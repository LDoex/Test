# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 6 - CNN example

# to try tensorflow, un-comment following two lines
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Convolution1D, \
                         ZeroPadding2D, MaxPooling1D,Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

trainfile_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtrain.csv'
testfile_path = 'D:/Users/oyyk/PycharmProjects/F_G_P/data/RawData/Muticlass_0-1-3_fewtest.csv'
#home_data = pd.read_csv(iowa_file_path)

#print(home_data.columns)

X = pd.read_csv(trainfile_path)
X_ = pd.read_csv(testfile_path)
dic_x = X.sample(frac=1)
dic_x_ = X_.sample(frac=1)
y = dic_x['label']
y_ = dic_x_['label']
dic_x.drop(dic_x.columns[-1], axis=1, inplace=True)
dic_x_.drop(dic_x.columns[-1], axis=1, inplace=True)
y_train = np.array(y)
X_train = np.array(dic_x)
y_test = np.array(y_)
X_test = np.array(dic_x_)
# downl oad the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(X_test, y_test, random_state = 3)
# data pre-processing
X_train = X_train.reshape((X_train.shape[0], 11, 11, 1))
X_test = X_test.reshape((X_test.shape[0], 11, 11, 1))
#y_train = y_train.reshape((y_train.shape[0], 1))
#y_test = y_test.reshape((y_test.shape[0], 1))
y_train = np_utils.to_categorical(y_train, num_classes=3)
y_test = np_utils.to_categorical(y_test, num_classes=3)


def scheduler(epoch):
    # 每隔20个epoch，学习率减小为原来的1/5
    if epoch % 20 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)

def getPrecision(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    N = (-1)*K.sum(K.round(K.clip(y_true-K.ones_like(y_true), -1, 0)))#N
    TN=K.sum(K.round(K.clip((y_true-K.ones_like(y_true))*(y_pred-K.ones_like(y_pred)), 0, 1)))#TN
    FP=N-TN
    precision = TP / (TP + FP + K.epsilon())#TT/P
    return precision

def getRecall(y_true, y_pred):
    TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))#TP
    P=K.sum(K.round(K.clip(y_true, 0, 1)))
    FN = P-TP #FN=P-TP
    recall = TP / (TP + FN + K.epsilon())#TP/(TP+FN)
    return recall
# Another way to build your CNN
model = Sequential()
model.add(ZeroPadding2D((1, 1), batch_input_shape=(None, 11, 11, 1)))
# Conv layer 1 output shape (128, 15, 1)
model.add(Convolution2D(
    filters=64,
    kernel_size=3,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_last',
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (128, 13, 1)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_last',
))

model.add(ZeroPadding2D((1, 1)))
# Conv layer 2 output shape (128, 7, 1)
model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (13, 3, 1)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))

# Fully connected layer 1 input shape (20 * 2 * 1) = (40), output shape (40)
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.6))
model.add(Activation('relu', name='Dense_1'))

# Fully connected layer 2 to shape (2) for 2 classes
model.add(Dense(3))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=10, batch_size=3, validation_split=0.1, callbacks=[reduce_lr], shuffle=True)

# preY=model.predict(X_train)
# print(y_train.T)
# print(preY.T)


print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

dense1_layer_model = Model(inputs=model.input,
                           outputs=model.get_layer('Dense_1').output)
# 以这个model的预测值作为输出
dense1_output = dense1_layer_model.predict(X_test) #取得‘Dense_1’的特征向量值


print (dense1_output.shape)

print(model.summary())
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)