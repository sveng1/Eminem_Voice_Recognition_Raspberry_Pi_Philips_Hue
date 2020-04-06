from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, GRU, Permute, Reshape


# This model architecture is inspired by https://github.com/ZainNasrullah/music-artist-classification-crnn
def crnn(first_conv_size=64, other_convs_size=128, gru_size=32, dense_activation='softmax', input_shape=(128, 94, 1)):

    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    model = Sequential()

    # Normalize across frequency axis
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First Conv2D layer
    model.add(Conv2D(first_conv_size, (3, 3), padding='same', activation='elu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    # Second Conv2D layer
    model.add(Conv2D(other_convs_size, (3, 3), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(Dropout(0.1))

    # Third Conv2D layer
    model.add(Conv2D(other_convs_size, (3, 3), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(Dropout(0.1))

    # Fourth Conv2D layer
    model.add(Conv2D(other_convs_size, (3, 3), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(4, 2), strides=(4, 2)))
    model.add(Dropout(0.1))

    # Reshape from shape (frequency, time, channels) to (time, frequency, channels)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # Recurrent layer
    model.add(GRU(gru_size, return_sequences=True, activation='tanh'))
    model.add(GRU(gru_size, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.3))

    # Dense layer
    model.add(Dense(2))
    model.add(Activation(dense_activation))

    return model
