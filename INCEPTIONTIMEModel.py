from keras.layers import Conv1D, Dense, AveragePooling1D, concatenate, Input, Flatten
from keras.models import Model


def inception_module(inputs, num_filters):
    conv1 = Conv1D(filters=num_filters[0], kernel_size=1, padding='same', activation='relu')(inputs)
    conv3 = Conv1D(filters=num_filters[1], kernel_size=3, padding='same', activation='relu')(inputs)
    conv5 = Conv1D(filters=num_filters[2], kernel_size=5, padding='same', activation='relu')(inputs)
    conv1_pool = AveragePooling1D(pool_size=3, strides=1, padding='same')(inputs)
    conv1_pool = Conv1D(filters=num_filters[3], kernel_size=1, padding='same', activation='relu')(conv1_pool)
    output = concatenate([conv1, conv3, conv5, conv1_pool], axis=-1)
    return output


def InceptionTime(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inception_module(inputs, num_filters=[32, 32, 32, 32])
    x = inception_module(x, num_filters=[64, 64, 64, 64])
    x = inception_module(x, num_filters=[128, 128, 128, 128])
    x = inception_module(x, num_filters=[256, 256, 256, 256])
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model


def InceptionTime_soft(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inception_module(inputs, num_filters=[32, 32, 32, 32])
    x = inception_module(x, num_filters=[64, 64, 64, 64])
    x = inception_module(x, num_filters=[128, 128, 128, 128])
    x = inception_module(x, num_filters=[256, 256, 256, 256])
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
