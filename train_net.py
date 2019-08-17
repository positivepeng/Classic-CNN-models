# -*- coding: utf-8 -*-
from AlexNet.AlexNet import AlexNet
from VGG.Vgg13 import VGG13
from VGG.Vgg19 import VGG19
from ZFNet.ZFNet import ZFNet
from GoogLeNet.GoogleNet import GoogLeNet
from ResNet.ResNet34 import ResNet34
from ResNet.ResNet50 import ResNet50
from DenseNet.DenseNet import DenseNet
from SENet.SE_ResNet import SE_ResNet34
from keras.datasets import cifar10


import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

def train_net(net, input_shape = "224x224"):
    # 训练网络，传入定义好的网络已经该网络对应的输入图片格式
    # 加载数据
    if input_shape in ["227x227", "224x224"]:
        data_file_path = "cat_and_dog_{}.pickle".format(input_shape)
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)
    elif input_shape in ["32x32"]:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data = {}
        data["images"] = x_train
        data["labels"] = y_train
    else:
        print("invalid input type")
        return None
    
    X_train, X_val_test, y_train, y_val_test = train_test_split(data["images"], to_categorical(data["labels"]) , test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    
    X_mean = np.mean(X_train, axis=0)
    X_train -= X_mean
    X_val -= X_mean
    X_test -= X_mean
    
    net.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=20, epochs=2, verbose=1)
    test_result = net.model.evaluate(X_test, y_test, batch_size=10, verbose=2)
    print("test loss:", test_result[0])
    print("test acc:", test_result[1])

if __name__ == "__main__":
    # train_net(AlexNet(), "227x227")

    # train_net(ZFNet(), "224x224")
    
    # train_net(GoogLeNet(), "224x224")
    
    # train_net(VGG13(), "224x224")
    
    # train_net(VGG19(), "224x224")
    
    # train_net(ResNet34(), "224x224")
    
    # train_net(ResNet50(), "224x224")
    
    # train_net(DenseNet(), "32x32")
    
    train_net(SE_ResNet34(), "224x224")
    
    
    
    
    
    
    
    
    
    
    
    