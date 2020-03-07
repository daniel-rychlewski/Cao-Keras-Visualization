# Contains useful utility methods for all things CNN.
# The pruning related methods prune_model, get_total_channels, get_model_apoz are taken from kerassurgeon examples.
import time
import keras
import kerassurgeon
import math
import pandas as pd
import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from kerassurgeon.identify import get_apoz
from keras.models import load_model
from tqdm import tqdm

# For vector quantization, use tf-nightly-gpu (around 16 times faster than the CPU-based tf-nightly)
import tensorflow as tf

from time_history import time_history

def prune_model(model, apoz_df, n_channels_delete):
    # Identify 5% of channels with the highest APoZ in model
    sorted_apoz_df = apoz_df.sort_values('apoz', ascending=False)
    high_apoz_index = sorted_apoz_df.iloc[0:n_channels_delete, :]

    # Create the Surgeon and add a 'delete_channels' job for each layer
    # whose channels are to be deleted.
    surgeon = kerassurgeon.Surgeon(model, copy=True)
    for name in high_apoz_index.index.unique().values:
        channels = list(pd.Series(high_apoz_index.loc[name, 'index'],
                                  dtype=np.int64).values)
        surgeon.add_job('delete_channels', model.get_layer(name),
                        channels=channels)
    # Delete channels
    return surgeon.operate()


def get_total_channels(model):
    start = None
    end = None
    channels = 0
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            channels += layer.filters
    return channels


def get_model_apoz(model, generator):
    # APoZ = Average Percentage of Zeros
    start = None
    end = None
    apoz = []
    for layer in model.layers[start:end]:
        if layer.__class__.__name__ == 'Conv2D':
            print(layer.name)
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    return apoz_df

def prune(model, x_train, x_test, y_train, y_test, batch_size, pruning_epochs, pruning_start, pruning_increment, pruning_end):
    """
    APoZ-based iterative channel-based model pruning. The pruning percentages used here are slightly higher than the actual pruning percentages, which can be understood by taking a look at the number of remaining model parameters in the model summary.

    """

    total_channels = get_total_channels(model)
    # the int type cast below is the reason why small percentages, i.e. below 1%, won't work (because n_channels_delete will be zero, so nothing will be pruned)
    n_channels_delete = int(math.floor(pruning_increment / 100 * total_channels))

    # Set up data generators
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=batch_size)
    train_steps = train_generator.n // train_generator.batch_size

    test_datagen = ImageDataGenerator()

    validation_generator = test_datagen.flow(
        x_test,
        y_test,
        batch_size=batch_size)
    val_steps = validation_generator.n // validation_generator.batch_size

    # Incrementally prune the network, retraining it each time
    while pruning_start < pruning_end:
        # Prune the model
        apoz_df = get_model_apoz(model, validation_generator)
        pruning_start += pruning_increment
        print('pruning up to ', str(pruning_start),
              '% of the original model weights')
        model = prune_model(model, apoz_df, n_channels_delete)
        print(model.summary())

        # Clean up tensorflow session after pruning and re-load model
        model.save("trained_pruned_model_"+ str(pruning_start) + ".h5")

        del model
        K.clear_session()
        tf.reset_default_graph()

        model = load_model("trained_pruned_model_" + str(pruning_start) + '.h5')

        # Re-train the model
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])

        csv_logger = CSVLogger("trained_pruned_model" + str(pruning_start) + '.csv')

        time_callback = time_history()
        model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            epochs=pruning_epochs,
                            validation_data=validation_generator,
                            validation_steps=val_steps,
                            workers=4,
                            callbacks=[csv_logger, time_callback])

        # predict again for time measurement
        start = time.clock()
        model.predict(x_test)
        end = time.clock()
        print("Time per image (pruning up to "+str(pruning_start)+" percent): {} ".format((end - start) / len(x_test)))

        # output the time
        print("time measurement:")
        for onetime in time_callback.times:
            print(onetime)

    # Evaluate the final model performance
    loss = model.evaluate_generator(validation_generator,
                                    validation_generator.n //
                                    validation_generator.batch_size)
    print('pruned model loss: ', loss[0], ', acc: ', loss[1])


def calculate_tflite_parameters(model_path, x_train, x_test, y_train, y_test):
    """
    Reads and evaluates the tflite model.
    """
    K.clear_session()
    tf.reset_default_graph()

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tflite_inference(X=x_train, y=y_train, interpreter=interpreter, event="Training")
    tflite_inference(X=x_test, y=y_test, interpreter=interpreter, event="Testing")

def tflite_inference(X, y, interpreter, event):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    start = time.clock()
    for img, label in tqdm(zip(X, y)):
        interpreter.set_tensor(input_details[0]['index'], np.asarray([img], dtype=np.float32))

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if (np.argmax(output_data[0])==np.argmax(label)).all():
            correct += 1

    print('Tensorflow Lite Model - '+event+' accuracy:', correct * 1.0 / len(y))
    end = time.clock()
    print("Time per image (quantization process - "+event+" phase): {} ".format((end - start) / len(X)))

def quantize(x_train, x_test, y_train, y_test, model_name, model_path_folder, quantize_pruned_models, pruning_start, pruning_increment, pruning_end):
    """
    Quantizes the model, evaluates its accuracies and losses for the given x and y values and saves it to disk.
    """

    if quantize_pruned_models:
        while pruning_start < pruning_end:
            K.clear_session()
            tf.reset_default_graph()

            file_name = "trained_pruned_model_" + str(pruning_start) + ".h5"

            try:
                converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path_folder + file_name)
            except OSError:
                print("could not read file "+file_name+", continuing with the next iteration")
                pruning_start += pruning_increment
                continue

            converter.post_training_quantize = True
            tflite_quantized_model = converter.convert()

            # Write to disk
            open("quantized_pruned_model_" + str(pruning_start) + ".tflite", "wb").write(tflite_quantized_model)

            calculate_tflite_parameters("quantized_pruned_model_" + str(pruning_start) + ".tflite", x_train, x_test, y_train, y_test)

            # Next model, i.e. next pruning percentage
            pruning_start += pruning_increment

    else:
        converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path_folder + model_name)
        converter.post_training_quantize = True
        tflite_quantized_model = converter.convert()

        # Write to disk
        open("quantized_"+ model_name +".tflite", "wb").write(tflite_quantized_model)

        calculate_tflite_parameters("quantized_" + model_name + ".tflite", x_train, x_test, y_train, y_test)
