#coding=utf-8
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
from keras.layers import add,Flatten
#from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

seed = 7
np.random.seed(seed)

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

def build_model():
    inpt = Input(shape=(224,224,3))
    x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    #(56,56,64)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    #(28,28,128)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    #(14,14,256)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    #(7,7,512)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(2,activation='softmax')(x)
    
    model = Model(inputs=inpt,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model

if __name__ == '__main__':
    vgg_file_path = r"G:\Desktop\Classic-DNN-models\VGG\cat_and_dog_224x224.pickle"
    with open(vgg_file_path, "rb") as f:
        data = pickle.load(f)
    X_train, X_val_test, y_train, y_val_test = train_test_split(data["images"], to_categorical(data["labels"]) , test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    print(X_train.shape, X_val.shape, X_test.shape)
    
    model = build_model()
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=10, epochs=1, verbose=1)
    print(model.evaluate(X_test, y_test, batch_size=10, verbose=2))
