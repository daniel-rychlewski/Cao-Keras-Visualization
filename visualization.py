import itertools
import argparse
import keras
import matplotlib.pyplot as plt
import numpy as np
import spectral
from keras.engine.saving import load_model
from matplotlib import patches
from sklearn.metrics import classification_report, confusion_matrix

# our utils functions
import compression
import preprocessing
import preprocessing as pp
from deepvizkeras.integrated_gradients import IntegratedGradients
from deepvizkeras.saliency import GradientSaliency
from deepvizkeras.visual_backprop import VisualBackprop
from models import cao, choose_model

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def reports(model, X_test, y_test, target_names):
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    cm = confusion
    target = y_test
    target = np.argmax(target, axis=1)

    score = model.evaluate(X_test, y_test, batch_size=32)
    test_loss =  score[0]*100
    test_accuracy = score[1]*100

    # Compute global accuracy (overall accuracy)
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    # results["Accuracy"] = accuracy
    print("OA: "+str(accuracy))

    # Compute average accuracy: "the mean of the percentages of correctly classified pixels for each class"
    aa_sum = 0
    count = 0
    for i in range(len(cm)):
        if np.count_nonzero(target==i) != 0:
            aa_sum += 100 * cm[i,i] / np.count_nonzero(target==i)
            count = count + 1
        else:
            aa_sum += 0

    # results["Average Accuracy"] = aa_sum / count
    print("AA: " + str(aa_sum / count))

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    # results["Kappa"] = kappa
    print("Kappa: " + str(kappa))

    return classification, confusion, test_loss, test_accuracy


def patch(data, height_index, width_index, PATCH_SIZE):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

def create_predicted_image(X, y, model, patch_size, height, width):
    outputs = np.zeros((height,width)) # zeroed image
    index = 0
    for i in range(0, height - patch_size + 1):
        if i % 8 == 0 or index == (height - patch_size + 1) - 1:
            preprocessing.print_progress_bar(index + 1, (height - patch_size + 1))
        index += 1
        for j in range(0, width - patch_size + 1):
            target = int(y[int(i + patch_size / 2)][int(j + patch_size / 2)])
            if target == 0 :
                continue
            else :
                image_patch = patch(X, i, j, patch_size)
                #print (image_patch.shape)
                X_test_image = image_patch.reshape(1, image_patch.shape[0],image_patch.shape[1], image_patch.shape[2]).astype('float32')#.reshape(1,image_patch.shape[2],image_patch.shape[0],image_patch.shape[1]).astype('float32')
                prediction = (model.predict_classes(X_test_image))
                outputs[int(i + patch_size / 2)][int(j + patch_size / 2)] = prediction + 1
    return outputs

def predict(model, X, X_test, y_test, target_names, classes_authorized, spy_colors, label_dictionary):
    classification, confusion, test_loss, test_accuracy = reports(model, X_test, y_test, target_names)
    print(classification)

    plt.figure(figsize=(13, 10))
    plot_confusion_matrix(confusion, classes=target_names,
                              title='Confusion matrix, without normalization')

    X_garbage, train_data, test_data = pp.load_data()
    y = np.add(train_data, test_data)
    y = pp.delete_useless_classes(y, classes_authorized)

    outputs = create_predicted_image(X, y, model, 5, y.shape[0], y.shape[1])

    print("PREDICTED IMAGE:")
    predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(5, 5))
    label_patches = [patches.Patch(color=spy_colors[x] / 255.,
                                  label=label_dictionary[x]) for x in np.unique(y)]
    plt.legend(handles=label_patches, ncol=2, fontsize='medium',
               loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.show()

    ground_truth = spectral.imshow(classes=y, figsize=(5, 5))
    print("IDEAL IMAGE: ")

    label_patches = [patches.Patch(color=spy_colors[x] / 255.,
                                  label=label_dictionary[x]) for x in np.unique(y)]
    plt.legend(handles=label_patches, ncol=2, fontsize='medium',
               loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.show()

def activation_map(model, X_train, from_band, to_band, step_band):
    for layer_number in range(6): # 6 is out of range
        from vis.visualization import visualize_activation
        grads = visualize_activation(model, layer_idx=layer_number, filter_indices=None, seed_input=X_train, backprop_modifier=None,grad_modifier="absolute")
        grads = grads.reshape(grads.shape[2], grads.shape[0], grads.shape[1]) # bands as first dimension

        chosen_bands = list(range(from_band, to_band, step_band))
        fig, ax = plt.subplots(2, len(chosen_bands))
        for chosen_band, index in zip(chosen_bands, range(len(chosen_bands))):
            grads_reshaped = grads[chosen_band].reshape((5, 5))

            ax[0][index].imshow(grads_reshaped, cmap='gray')
            ax[1][index].imshow(grads_reshaped, cmap='jet')

            ax[0][index].get_xaxis().set_visible(False)
            ax[1][index].get_xaxis().set_visible(False)

            ax[0][index].get_yaxis().set_visible(False)
            ax[1][index].get_yaxis().set_visible(False)

        plt.savefig("activation_map_" + str(from_band) + "_to_" + str(to_band) + "_in_" + str(step_band) + "_layer" + str(
            layer_number) + ".png", bbox_inches='tight')

def guided_backpropagation(model, x_train, from_band, to_band, step_band, n_bands):
    x_train = x_train.reshape(x_train.shape[0], 5, 5, n_bands, 1).astype('float32')
    x_train = x_train / 255.0
    import matplotlib.pyplot as plt

    img = x_train[0]
    for trainable_weight in range(9):
        img=img.reshape((5,5,n_bands))
        print("trainable weight number: " + str(trainable_weight))
        from deepvizkeras.guided_backprop import GuidedBackprop
        single = GuidedBackprop(model, trainable_weight, n_bands=n_bands)
        grad = single.get_mask(input_image=img)

        filter_grad = (grad > 0.0).reshape((5,5,n_bands))
        img.reshape((5,5,n_bands))
        average_grad = single.get_smoothed_mask(img)
        filter_average_grad = (average_grad > 0.0).reshape((5,5,n_bands))

        # img.shape is 5,5,n_bands. now choose a band of the n_bands to visualize.
        img=img.reshape((n_bands,5,5))
        grad_filter_grad = (grad*filter_grad).reshape((n_bands,5,5))
        grad=grad.reshape((n_bands,5,5))
        average_grad_filter_average_grad = (average_grad*filter_average_grad).reshape((n_bands,5,5))
        average_grad=average_grad.reshape((n_bands,5,5))

        chosen_bands = list(range(from_band,to_band,step_band))
        fig, ax = plt.subplots(5, len(chosen_bands))
        for chosen_band, index in zip(chosen_bands, range(len(chosen_bands))):
            img_reshaped = img[chosen_band].reshape((5,5))
            grad_reshaped = grad[chosen_band].reshape((5, 5))
            grad_filter_grad_reshaped = grad_filter_grad[chosen_band].reshape((5,5))
            average_grad_filter_average_grad_reshaped = average_grad_filter_average_grad[chosen_band].reshape((5,5))
            average_grad_reshaped=average_grad[chosen_band].reshape((5,5))

            ax[0][index].imshow(img_reshaped, cmap='gray')
            ax[1][index].imshow(grad_filter_grad_reshaped, cmap='gray')
            ax[2][index].imshow(average_grad_filter_average_grad_reshaped, cmap='gray')
            ax[3][index].imshow(grad_reshaped, cmap='jet')
            ax[4][index].imshow(average_grad_reshaped, cmap='jet')

            ax[0][index].get_xaxis().set_visible(False)
            ax[1][index].get_xaxis().set_visible(False)
            ax[2][index].get_xaxis().set_visible(False)
            ax[3][index].get_xaxis().set_visible(False)
            ax[4][index].get_xaxis().set_visible(False)

            ax[0][index].get_yaxis().set_visible(False)
            ax[1][index].get_yaxis().set_visible(False)
            ax[2][index].get_yaxis().set_visible(False)
            ax[3][index].get_yaxis().set_visible(False)
            ax[4][index].get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig("guided_backpropagation_"+str(from_band)+"_to_"+str(to_band)+"_in_"+str(step_band)+"_trainable_weight" + str(trainable_weight)+".png", bbox_inches='tight')

def integrated_gradients(model, n_bands, x_train, from_band, to_band, step_band):
    x_train = x_train.reshape(x_train.shape[0], 5, 5, n_bands, 1).astype('float32')
    x_train = x_train / 255.0
    import matplotlib.pyplot as plt

    img = x_train[0]
    for trainable_weight in range(9):
        print("trainable weight number: " + str(trainable_weight))
        single = IntegratedGradients(model, trainable_weight, n_bands)
        grad = single.get_grad(img)
        filter_grad = (grad > 0.0).reshape((5, 5, n_bands))
        img.reshape((5, 5, n_bands))
        # img = img.squeeze(axis=3)

        # img.shape is 5,5,n_bands. now choose a band of the n_bands to visualize.
        img = img.reshape((n_bands, 5, 5))
        grad_filter_grad = (grad * filter_grad).reshape((n_bands, 5, 5))
        grad = grad.reshape((n_bands, 5, 5))

        chosen_bands = list(range(from_band, to_band, step_band))
        fig, ax = plt.subplots(3, len(chosen_bands))
        for chosen_band, index in zip(chosen_bands, range(len(chosen_bands))):
            img_reshaped = img[chosen_band].reshape((5, 5))
            grad_reshaped = grad[chosen_band].reshape((5, 5))
            grad_filter_grad_reshaped = grad_filter_grad[chosen_band].reshape((5, 5))

            ax[0][index].imshow(img_reshaped, cmap='gray')
            ax[1][index].imshow(grad_filter_grad_reshaped, cmap='gray')
            ax[2][index].imshow(grad_reshaped, cmap='jet')

            ax[0][index].get_xaxis().set_visible(False)
            ax[1][index].get_xaxis().set_visible(False)
            ax[2][index].get_xaxis().set_visible(False)

            ax[0][index].get_yaxis().set_visible(False)
            ax[1][index].get_yaxis().set_visible(False)
            ax[2][index].get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig("integrated_gradients_" + str(from_band) + "_to_" + str(to_band) + "_in_" + str(step_band) + "_trainable_weight" + str(
            trainable_weight) + ".png", bbox_inches='tight')

def visual_backpropagation(model, n_bands, x_train, from_band, to_band, step_band):
    x_train = x_train.reshape(x_train.shape[0], 5, 5, n_bands, 1).astype('float32')
    x_train = x_train / 255.0
    import matplotlib.pyplot as plt

    img = x_train[0]
    for trainable_weight in range(9):
        print("trainable weight number: " + str(trainable_weight))
        single = VisualBackprop(model, trainable_weight, n_bands)
        img.reshape((5, 5, n_bands))
        grad = single.get_mask(img)
        filter_grad = (grad > 0.0).reshape((5, 5, n_bands))
        # img = img.squeeze(axis=3)

        # img.shape is 5,5,n_bands. now choose a band of the n_bands to visualize.
        img = img.reshape((n_bands, 5, 5))
        grad_filter_grad = (grad * filter_grad).reshape((n_bands, 5, 5))
        grad = grad.reshape((n_bands, 5, 5))

        chosen_bands = list(range(from_band, to_band, step_band))
        fig, ax = plt.subplots(3, len(chosen_bands))
        for chosen_band, index in zip(chosen_bands, range(len(chosen_bands))):
            img_reshaped = img[chosen_band].reshape((5, 5))
            grad_reshaped = grad[chosen_band].reshape((5, 5))
            grad_filter_grad_reshaped = grad_filter_grad[chosen_band].reshape((5, 5))

            ax[0][index].imshow(img_reshaped, cmap='gray')
            ax[1][index].imshow(grad_filter_grad_reshaped, cmap='gray')
            ax[2][index].imshow(grad_reshaped, cmap='jet')

            ax[0][index].get_xaxis().set_visible(False)
            ax[1][index].get_xaxis().set_visible(False)
            ax[2][index].get_xaxis().set_visible(False)

            ax[0][index].get_yaxis().set_visible(False)
            ax[1][index].get_yaxis().set_visible(False)
            ax[2][index].get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig(
            "visual_backpropagation_" + str(from_band) + "_to_" + str(to_band) + "_in_" + str(step_band) + "_trainable_weight" + str(
                trainable_weight) + ".png", bbox_inches='tight')

def gradients(model, n_bands, x_train, x_test, from_band, to_band, step_band):
    x_train = x_train.reshape(x_train.shape[0], 5, 5, n_bands, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 5, 5, n_bands, 1).astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    import matplotlib.pyplot as plt

    img = x_train[0]
    for trainable_weight in range(9):
        print("trainable weight number: " + str(trainable_weight))
        single = GradientSaliency(model, trainable_weight, n_bands)
        grad = single.get_grad(img)
        filter_grad = (grad > 0.0).reshape((5,5,n_bands))
        img.reshape((5,5,n_bands))
        # img = img.squeeze(axis=3)
        average_grad = single.get_smoothed_mask(img)
        filter_average_grad = (average_grad > 0.0).reshape((5,5,n_bands))

        # img.shape is 5,5,n_bands. now choose a band of the n_bands to visualize.
        img=img.reshape((n_bands,5,5))
        grad_filter_grad = (grad*filter_grad).reshape((n_bands,5,5))
        grad=grad.reshape((n_bands,5,5))
        average_grad_filter_average_grad = (average_grad*filter_average_grad).reshape((n_bands,5,5))
        average_grad=average_grad.reshape((n_bands,5,5))

        chosen_bands = list(range(from_band,to_band,step_band))
        fig, ax = plt.subplots(5, len(chosen_bands))
        for chosen_band, index in zip(chosen_bands, range(len(chosen_bands))):
            img_reshaped = img[chosen_band].reshape((5,5))
            grad_reshaped = grad[chosen_band].reshape((5, 5))
            grad_filter_grad_reshaped = grad_filter_grad[chosen_band].reshape((5,5))
            average_grad_filter_average_grad_reshaped = average_grad_filter_average_grad[chosen_band].reshape((5,5))
            average_grad_reshaped=average_grad[chosen_band].reshape((5,5))

            ax[0][index].imshow(img_reshaped, cmap='gray')
            ax[1][index].imshow(grad_filter_grad_reshaped, cmap='gray')
            ax[2][index].imshow(average_grad_filter_average_grad_reshaped, cmap='gray')
            ax[3][index].imshow(grad_reshaped, cmap='jet')
            ax[4][index].imshow(average_grad_reshaped, cmap='jet')

            ax[0][index].get_xaxis().set_visible(False)
            ax[1][index].get_xaxis().set_visible(False)
            ax[2][index].get_xaxis().set_visible(False)
            ax[3][index].get_xaxis().set_visible(False)
            ax[4][index].get_xaxis().set_visible(False)

            ax[0][index].get_yaxis().set_visible(False)
            ax[1][index].get_yaxis().set_visible(False)
            ax[2][index].get_yaxis().set_visible(False)
            ax[3][index].get_yaxis().set_visible(False)
            ax[4][index].get_yaxis().set_visible(False)

        # plt.show()
        plt.savefig("gradient"+str(from_band)+"_to_"+str(to_band)+"_in_"+str(step_band)+"_trainable_weight" + str(trainable_weight)+".png", bbox_inches='tight')

def get_parser():
    parser = argparse.ArgumentParser(description="Visualization of deep learning models for hyperspectral classification")

    parser.add_argument("model", default="cao", help="the model to use", choices=["cao"])
    parser.add_argument("model_path", metavar="MODEL", help="path or file name of the model to be read for visualization purposes")

    image_generation = parser.add_argument_group("Image generation")
    image_generation.add_argument("--patch_size", metavar="PSIZE", default=5, help="size of the patch of hyperspectral image band considered", type=int)
    image_generation.add_argument("--classes_authorized", metavar="TAKECLASS", default=[2, 3, 5, 6, 10, 0, 11, 12, 14, 15], type=list, help="list of classes to take from the image band - the rest is ignored")
    image_generation.add_argument("--num_classes", metavar="TOTALCLASS", default=9, type=int)

    image_compression = parser.add_argument_group("Image compression")
    image_compression.add_argument("--band_selection", metavar="BSEL", default=None, choices=[None, "PCA", "NMF"],
                        help="image extraction technique to apply to reduce the number of components")
    image_compression.add_argument("--components", metavar="COMP", default=100,
                        help="number of components for image extraction technique", type=int)

    visualization = parser.add_argument_group("Visualization")
    visualization.add_argument("visualize", metavar="VIS", help="select the visualization method",
                        choices=["guided_backprop", "visual_backprop", "gradient", "integrated_gradient", "activation_map"])
    visualization.add_argument("--from_band", metavar="FBAND", type=int, default=0, help="image band (inclusive) at which to start showing visualizations")
    visualization.add_argument("--to_band", metavar="TBAND", type=int, default=100, help="image band (exclusive) at which to stop showing visualizations")
    visualization.add_argument("--step_band", metavar="SBAND", type=int, default=10, help="image band increment for the interval [from_band; to_band)")

    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    model_path = args.model_path

    patch_size = args.patch_size
    classes_authorized = args.classes_authorized
    num_classes = args.num_classes

    model_name = args.model
    compression_method = args.band_selection
    components = args.components

    visualize = args.visualize
    from_band = args.from_band
    to_band = args.to_band
    step_band = args.step_band

    X, X_train, X_test, y_train, y_test = pp.preprocess_dataset(classes_authorized, components, compression_method, patch_size)

    input_shape = X_train[0].shape

    # Visualization
    _, lr = choose_model(model_name, input_shape, num_classes)
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr), metrics=['accuracy'])

    if visualize == "guided_backprop":
        # might not work for pruned models: KeyError: "The name 'dense_3_1/Softmax:0' refers to a Tensor which does not exist. The operation, 'dense_3_1/Softmax', does not exist in the graph."
        guided_backpropagation(model, n_bands=X.shape[2], x_train=X_train, from_band=from_band, to_band=to_band, step_band=step_band)
    elif visualize == "gradient":
        gradients(model, n_bands=X.shape[2], x_train=X_train, x_test=X_test, from_band=from_band, to_band=to_band, step_band=step_band)
    elif visualize == "activation_map":
        activation_map(model, X_train, from_band=from_band, to_band=to_band, step_band=step_band)
    elif visualize == "integrated_gradient":
        integrated_gradients(model, n_bands=X.shape[2], x_train=X_train, from_band=from_band, to_band=to_band, step_band=step_band)
    elif visualize == "visual_backprop":
        visual_backpropagation(model, n_bands=X.shape[2], x_train=X_train, from_band=from_band, to_band=to_band, step_band=step_band)