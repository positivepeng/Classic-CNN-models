#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D
import numpy as np
import pickle
from skimage import transform
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(7)

def build_model():
    model = Sequential()

    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    origin_file_path = r"../cat_and_dog_1000.pickle"
    vgg_file_path = r"cat_and_dog_224x224.pickle"
# =============================================================================
#     with open(origin_file_path, "rb") as f:
#         data = pickle.load(f)
#     data_vgg = {
#         "images" : transform.resize(data["images"], (data["images"].shape[0], 224,224,3)),
#         "labels" : data["labels"]
#     }
#     with open(vgg_file_path, "wb") as f:
#         pickle.dump(data_vgg, f)
# =============================================================================
    with open(vgg_file_path, "rb") as f:
        data = pickle.load(f)
    X_train, X_val_test, y_train, y_val_test = train_test_split(data["images"], to_categorical(data["labels"]) , test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    print(X_train.shape, X_val.shape, X_test.shape)
    model = build_model()
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=10, epochs=1, verbose=1)
    print(model.evaluate(X_test, y_test, batch_size=10, verbose=2))

