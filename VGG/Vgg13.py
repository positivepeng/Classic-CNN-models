from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Conv2D,MaxPooling2D

class VGG13:
    def __init__(self):
        model = Sequential()
        # 注意padding = same时，意味着卷积核移动时只要左边边界不出界，右边就进行填充，所以输出的大小为输入大小/步长 的上取整
        # 而padding = valid时，意味着卷积核移动中整个卷积核都不能出界
        model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(2,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        self.model = model
