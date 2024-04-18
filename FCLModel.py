from keras.models import Sequential
from keras.layers import Dense, Flatten


def fcl_model(inputShape):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(inputShape,)))
    # model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model


def fcl_model2(inputShape):
    model = Sequential()
    model.add(Flatten(input_shape=inputShape))
    # model.add(BatchNormalization())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model
