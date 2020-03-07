import argparse
import numpy as np
from spectral import spy_colors

import preprocessing as pp
import visualization as viz
from models import choose_model


def get_parser():
    parser = argparse.ArgumentParser(description="Generate deep learning models for hyperspectral classification with the options of pruning and quantization")

    image_generation = parser.add_argument_group("Image generation")
    image_generation.add_argument("--patch_size", metavar="PSIZE", default=5, help="size of the patch of hyperspectral image band considered", type=int)
    image_generation.add_argument("--classes_authorized", metavar="TAKECLASS", default=[2, 3, 5, 6, 10, 0, 11, 12, 14, 15], type=list, help="list of classes to take from the image band - the rest is ignored")
    image_generation.add_argument("--num_classes", metavar="TOTALCLASS", default=9, type=int)
    image_generation.add_argument("--show_images", action='store_true', help="if enabled, shows visualizations of the confusion matrix, the labeled image predicted by the model and the ground truth - this may take up to one minute")

    training_parameters = parser.add_argument_group("Training parameters")
    training_parameters.add_argument("model", help="the model to use", choices=["cao"])
    training_parameters.add_argument("--batch_size", metavar="BSIZE", default=128, type=int)
    training_parameters.add_argument("--epochs", default=10, type=int)

    image_compression = parser.add_argument_group("Image compression")
    image_compression.add_argument("--band_selection", metavar="BSEL", default=None, choices=[None, "PCA", "NMF"],
                        help="image extraction technique to apply to reduce the number of components")
    image_compression.add_argument("--components", metavar="COMP", default=100,
                        help="number of components for image extraction technique", type=int)

    model_compression = parser.add_argument_group("Model compression")
    model_compression.add_argument("--prune", action='store_true', help="if enabled, the channels of the neural network will be iteratively pruned based on the average percentage of zeros (APoZ) metric")
    model_compression.add_argument("--prune_epochs", metavar="PEPOCHS", default=10, type=int, help="number of epochs used to retrain neural network after each pruning step")
    model_compression.add_argument("--prune_start", metavar="PSTART", default=0, type=int, help="the starting pruning percentage value")
    model_compression.add_argument("--prune_increment", metavar="PINC", default=5, type=int, help="percentage by which to increment the pruning percentage")
    model_compression.add_argument("--prune_end", metavar="PEND", default=98, type=int, help="maximum pruning percentage (inclusive). Choices where the respective overall accuracy reference of the cao/he/hu/luo/santara model is maintained are 40/65/20/99/42 percent")

    model_compression.add_argument("--quantize", action='store_true', help="if enabled, the h5 model will be converted to a tflite model, becoming quantized in the process")
    model_compression.add_argument("--quantize_pruned_models", action='store_true', help="enable this to perform a quantization after this program has done pruning and saved the pruned model files")
    model_compression.add_argument("--quantize_folder", metavar="QFOLDER", default="", help="specify the /folder/of/your/model/files/ that should be quantized - the quantized models will be generated in the same folder")

    return parser

def pretty_print_count(train_data, test_data):
    unique_train = np.unique(train_data, return_counts=True)
    unique_test = np.unique(test_data, return_counts=True)
    for i in range(0, unique_train[0].shape[0]):
        print(unique_train[0][i], "=>", unique_train[1][i], "/", unique_test[1][i])
    print()

if __name__ == "__main__":
    args = get_parser().parse_args()

    # Globals and hyperparameters
    patch_size = args.patch_size
    classes_authorized = args.classes_authorized
    num_classes = args.num_classes
    show_images = args.show_images

    model_name = args.model
    batch_size = args.batch_size
    epochs = args.epochs

    compression_method = args.band_selection
    components = args.components

    pruning_enabled = args.prune
    pruning_epochs = args.prune_epochs
    pruning_start = args.prune_start
    pruning_increment = args.prune_increment
    pruning_end = args.prune_end

    quantization_enabled = args.quantize
    quantize_pruned_models = args.quantize_pruned_models
    models_folder_for_quantization = args.quantize_folder

    target_names = ['Corn-notill',
                    'Corn-mintill',
                    'Grass-pasture',
                    'Grass-trees',
                    'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Woods',
                    'Buildings-Grass-Trees-Drives', ]
    label_dictionary = {
        0: 'Unclassified',
        1: 'Corn-notill',
        2: 'Corn-mintill',
        3: 'Grass-pasture',
        4: 'Grass-trees',
        5: 'Soybean-notill',
        6: 'Soybean-mintill',
        7: 'Soybean-clean',
        8: 'Woods',
        9: 'Buildings-Grass-Trees-Drives',
    }

    X, X_train, X_test, y_train, y_test = pp.preprocess_dataset(classes_authorized, components, compression_method, patch_size)

    # Training
    input_shape = X_train[0].shape
    print(input_shape)

    model, lr = choose_model(model_name, input_shape, num_classes)
    model.summary()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)

    if not pruning_enabled and not quantization_enabled:
        model.save("trained_model.h5")

    if show_images:
        viz.predict(model, X, X_test, y_test, target_names, classes_authorized, spy_colors, label_dictionary)

    if pruning_enabled:
        print(model.summary())
        from compression import prune
        try:
            prune(model, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, batch_size=batch_size, pruning_epochs=pruning_epochs, pruning_start=pruning_start, pruning_increment=pruning_increment, pruning_end=pruning_end)
        except ValueError:
            pass # most likely, the pruning percentage has become too high: cannot prune anymore because there is not enough left

    if quantization_enabled:
        from compression import quantize
        quantize(x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, model_name="cao", model_path_folder=models_folder_for_quantization, quantize_pruned_models=quantize_pruned_models, pruning_start=pruning_start, pruning_increment=pruning_increment, pruning_end=pruning_end)