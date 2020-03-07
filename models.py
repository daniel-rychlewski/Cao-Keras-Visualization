import keras

def choose_model(model_name, input_shape, num_classes):
    if model_name == "cao":
        return cao(input_shape, num_classes)

def cao(input_shape, num_classes):
    kernel_size = 3
    lr = 0.001

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(300, (kernel_size), strides=(1, 1), padding='valid', activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(200, (kernel_size), strides=(1, 1), padding='valid', activation='relu'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(200))
    model.add(keras.layers.Dense(100))

    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

    return model, lr