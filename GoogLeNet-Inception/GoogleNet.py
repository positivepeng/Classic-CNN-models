#coding=utf-8
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import to_categorical


np.random.seed(seed = 7)

def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Inception(x,nb_filter):
    branch1x1 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)

    branch3x3 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch3x3 = Conv2d_BN(branch3x3,nb_filter,(3,3), padding='same',strides=(1,1),name=None)

    branch5x5 = Conv2d_BN(x,nb_filter,(1,1), padding='same',strides=(1,1),name=None)
    branch5x5 = Conv2d_BN(branch5x5,nb_filter,(1,1), padding='same',strides=(1,1),name=None)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2d_BN(branchpool,nb_filter,(1,1),padding='same',strides=(1,1),name=None)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x

def build_model():
    inpt = Input(shape=(224,224,3))
    #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,64)#256
    x = Inception(x,120)#480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,128)#512
    x = Inception(x,128)
    x = Inception(x,128)
    x = Inception(x,132)#528
    x = Inception(x,208)#832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Inception(x,208)
    x = Inception(x,256)#1024
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(2,activation='softmax')(x)
    model = Model(inpt,x,name='inception')
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    model = build_model()
    # model.summary()
    vgg_file_path = r"G:\Desktop\Classic-DNN-models\VGG\cat_and_dog_224x224.pickle"
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
