#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

def build_model():
    model = Sequential()
    model.add(Conv2D(96,(7,7),strides=(2,2),input_shape=(224,224,3),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(2,2),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()    
    vgg_file_path = r"G:\Desktop\Classic-DNN-models\VGG\cat_and_dog_224x224.pickle"
    with open(vgg_file_path, "rb") as f:
        data = pickle.load(f)
    X_train, X_val_test, y_train, y_val_test = train_test_split(data["images"], to_categorical(data["labels"]) , test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    print(X_train.shape, X_val.shape, X_test.shape)
    model = build_model()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=10, epochs=1, verbose=1)
    print(model.evaluate(X_test, y_test, batch_size=10, verbose=2))