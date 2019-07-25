#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import pickle
import gzip
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
np.random.seed(seed = 7)

# 读入数据
f = gzip.open(r'mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='unicode-escape')
f.close()

# 检查数据规格
print("train size ", len(train_set[0])," type ",type(train_set[0][0]))
print("valid size ", len(valid_set[0]))
print("test size ", len(test_set[0]))

# 提取训练集，验证集，测试集，并将label转换为独热码
train_x = train_set[0].reshape((-1,28,28,1))
train_y = to_categorical(train_set[1])

valid_x = valid_set[0].reshape((-1,28,28,1))
valid_y = to_categorical(valid_set[1])

test_x = test_set[0].reshape((-1,28,28,1))
test_y = to_categorical(test_set[1])

# 定义网络
model = Sequential()
# N x 28 x 28 x 1  ->  N x 24 x 24 x 32
# parameter : 5x5x32+32(bias) = 832
model.add(Conv2D(32, (5, 5), strides=(1,1), input_shape=(28,28,1), padding='valid', activation='relu', kernel_initializer='uniform'))

# N x 24 x 24 x 32  ->  N x 12 x 12 x 32
model.add(MaxPooling2D(pool_size=(2,2)))

# N x 12 x 12 x 32  ->  N x 8 x 8 x 64
model.add(Conv2D(64, (5,5), strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))

# N x 8 x 8 x 64   ->  N x 4 x 4 x 64
model.add(MaxPooling2D(pool_size=(2,2)))

# N x 4 x 4 x 64 ->  N x 1024
model.add(Flatten())

# N x 1024 ->  N x 100
model.add(Dense(100,activation='relu'))

# N x 100  ->  N x 10
model.add(Dense(10,activation='softmax'))

# 定义训练参数
# compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

model.fit(train_x, train_y, validation_data=(valid_x, valid_y), batch_size=200, epochs=2, verbose=1)
print(model.evaluate(test_x, test_y, batch_size=20, verbose=2))
#[0.05287418974854518, 0.9832999963760376]