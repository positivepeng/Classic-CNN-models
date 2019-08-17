# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 08:58:27 2019

@author: hp
"""

#coding=utf-8
from keras.models import Model
from keras.layers import Input,Dense,BatchNormalization,Conv2D,ReLU
from keras.layers import MaxPooling2D,ZeroPadding2D,GlobalAveragePooling2D
from keras.layers import add,Flatten,multiply,AveragePooling2D
from keras import backend as K



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

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False, with_SE_module = False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        # 在x后加入SE module，然后再和shortcut相加
        if with_SE_module:
            gamma = 16
            # print(x.shape)
            C = int(x.shape[-1])
            avg = GlobalAveragePooling2D()(x)
            # print(avg.shape)
            # print(C // gamma)
            relu_out = Dense(C // gamma, activation="relu")(avg)
            # print(relu_out.shape)
            sigmod_out = Dense(C, activation=K.sigmoid)(relu_out)
            # print(sigmod_out.shape)
            # x = x * sigmod_out      
            x = multiply([x,sigmod_out])
            # print(x.shape)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

class SE_ResNet34():
    def __init__(self):
        inpt = Input(shape=(224,224,3))
        x = ZeroPadding2D((3,3))(inpt)
        x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
        x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
        #(56,56,64)
        x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
        x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
        x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
        #(28,28,128)
        x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True,with_SE_module=True)
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
        self.model = model

if __name__ == '__main__':
    model = SE_ResNet34()
    # model.model.summary()