import tensorflow as tf
import tensorflow.keras as kr

from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import ReLU, Lambda, Dropout, Add, GlobalAveragePooling2D, AveragePooling2D, Concatenate
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

NoNormalization = lambda: Lambda(lambda x: x)

AVAILABLE_ARCHS = ["SmolAlexNet", "AlexNet", "VGG16", "VGG19", "CifarResNet", "ResNet50", "DenseNet169", "MobileNetV2"]

def SmolAlexNet(input_tensor=None, classes=1000, Normalization=NoNormalization):
    x = Conv2D(96, 11, padding="valid", activation="relu")(input_tensor)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 5, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation="softmax")(x)

    return Model(inputs=input_tensor, outputs=x)

def AlexNet(input_tensor=None, classes=1000, Normalization=NoNormalization):
    x = Conv2D(96, 11, strides=4, padding="valid", activation="relu")(input_tensor)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, 5, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(384, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = Conv2D(256, 3, padding="same", activation="relu")(x)
    x = Normalization()(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(classes, activation="softmax")(x)

    return Model(inputs=input_tensor, outputs=x)

def ResBlockA(x, num, stride, norm_layer):
    if stride != 1:
       conv0 = Conv2D(256 * num, 1, strides=stride, padding="valid")(x)
       conv1 = Conv2D(64 * num, 1, strides=stride, padding="valid")(x)
    else:
       conv0 = Conv2D(256 * num, 1, strides=stride, padding="valid")(x)
       conv1 = Conv2D(64 * num, 1, strides=stride, padding="valid")(x)
        
    conv1 = norm_layer()(conv1)
    conv1 = ReLU()(conv1)
    
    conv1 = Conv2D(64 * num, 3, padding="same")(conv1)
    conv1 = norm_layer()(conv1)
    conv1 = ReLU()(conv1)

    conv1 = Conv2D(256 * num, 1, padding="same")(conv1)
    conv1 = norm_layer()(conv1)

    x = Add()([conv1, conv0])
    x = ReLU()(x)
    return x

def ResBlockB(x, num, norm_layer):
    conv1 = Conv2D(64 * num, 1, padding="same")(x)
    conv1 = norm_layer()(conv1)
    conv1 = ReLU()(conv1)

    conv1 = Conv2D(64 * num, 3, padding="same")(conv1)
    conv1 = norm_layer()(conv1)
    conv1 = ReLU()(conv1)

    conv1 = Conv2D(256 * num, 1, padding="same")(conv1)
    conv1 = norm_layer()(conv1)
    
    x = Add()([conv1, x])
    x = ReLU()(x)

    return x

def CifarResNetBlock(x, nfilters, norm_layer, strides=1):
    conv1 = Conv2D(nfilters, 3, strides=strides, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    conv1 = norm_layer()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv2D(nfilters, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(conv1)
    conv2 = norm_layer()(conv2)

    if strides == 2:
        x = Conv2D(nfilters, 1, strides=strides, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(x)
    
    x = Add()([conv2, x])
    x = ReLU()(x)
    return x
   
def CifarResNet(n, input_tensor=None, classes=1000, Normalization=BatchNormalization):
    x = Conv2D(16, 3, padding="same", kernel_regularizer=l2(1e-4))(input_tensor)
    x = Normalization()(x)
    x = ReLU()(x)

    for i in range(n):
        x = CifarResNetBlock(x, 16, Normalization)

    x = CifarResNetBlock(x, 32, Normalization, strides=2)
    for i in range(n - 1):
        x = CifarResNetBlock(x, 32, Normalization)

    x = CifarResNetBlock(x, 64, Normalization, strides=2)
    for i in range(n - 1):
        x = CifarResNetBlock(x, 64, Normalization)

    x = AveragePooling2D(pool_size=8)(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    return Model(inputs=input_tensor, outputs=x)
    
def SmolResNet50(input_tensor=None, classes=1000, Normalization=BatchNormalization):
    x = Conv2D(64, 7, padding="same")(input_tensor)
    x = Normalization()(x)
    x = ReLU()(x)

    x = ResBlockA(x, 1, 1, Normalization)
    for i in range(2):
        x = ResBlockB(x, 1, Normalization)

    x = ResBlockA(x, 2, 2, Normalization)
    for i in range(3):
        x = ResBlockB(x, 2, Normalization)

    x = ResBlockA(x, 4, 2, Normalization)
    for i in range(5):
        x = ResBlockB(x, 4, Normalization)

    x = ResBlockA(x, 8, 2, Normalization)
    for i in range(2):
        x = ResBlockB(x, 8, Normalization)

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation="softmax")(x)
    return Model(inputs=input_tensor, outputs=x)

# num_filters: k in the paper
# num_layers: l in the paper
def DenseBlock(x, num_filters, num_layers):
    temp = x
    for _ in range(num_layers):
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, (3, 3), use_bias=False, padding="same")(x)
        x = Dropout(0.2)(x)
        temp = Concatenate(axis=-1)([temp, x])
    return temp

def TransitionBlock(x, num_filters):
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (1, 1), use_bias=False)
    x = Dropout(0.2)(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    return x

# Terminology from paper: k "growth rate": num of filters
#                         l: number of layers per denseblock
def DenseNetCifar(input_tensor, classes, k, l):
    x = Conv2D(k, (3, 3), use_bias=False, padding="same")(input_tensor)
    # Three Dense-blocks with same num of layers
    x = DenseBlock(x, k, l)
    x = TransitionBlock(x, k)

    x = DenseBlock(x, k, l)
    x = TransitionBlock(x, k)

    x = DenseBlock(x, k, l)
    x = TransitionBlock(x, k)

    x = BatchNormalization()(x)
    x = Activarion("relu")(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax")(x)

    return Model(inputs=input_tensor, outputs=x)

def show_available():
    print("Available architectures:", ", ".join(AVAILABLE_ARCHS))

def get_arch(arg, input_shape, classes, **kwargs):
    input_tensor = Input(shape=input_shape)

    if "Normalization" in kwargs:
        if kwargs["Normalization"] == "BatchNormalization":
            kwargs["Normalization"] = BatchNormalization
        elif kwargs["Normalization"] == "LayerNormalization":
            kwargs["Normalization"] = LayerNormalization
        elif kwarfs["Normalization"] == "NoNormalization":
            kwargs["Normalization"] = NoNormalization
        # if its not a string assume its a normalization layer
        elif type(kwargs["Normalization"]) == str:
            print("Warning: couldn't understand your normalization")
            kwargs["Normalization"] = NoNormalization

    if arg == "AlexNet":
        return AlexNet(input_tensor=input_tensor, classes=classes, **kwargs)
    elif arg == "SmolAlexNet":
        return SmolAlexNet(input_tensor=input_tensor, classes=classes, **kwargs)
    elif arg == "VGG16":
        return VGG16(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "VGG19":
        return VGG19(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "ResNet50":
        return LocalResNet50(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "ResNet152":
        return ResNet152(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "CifarResNet":
        return CifarResNet(3, input_tensor=input_tensor, classes=classes)
    elif arg == "DenseNet169":
        return DenseNet169(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "MobileNetV2":
        return MobileNetV2(input_tensor=input_tensor, classes=classes, weights=None, **kwargs)
    elif arg == "DenseNetCifar":
        return DenseNetCifar(input_tensor, classes, 12, 12)
    else:
        show_available()
        raise Exception(arg + " not an available architecture")
