# Cao-Keras-Visualization

A fork of [M2-DeepLearning](https://github.com/MeryllEssig/M2-DeepLearning) to visualize the computations of the [cao model](https://github.com/xiangyongcao/CNN_HSIC_MRF) on the IndianPines hyperspectral dataset using activation maps, guided / visual backpropagation and (integrated) gradient-based saliency maps.
This tool allows to generate compressed variants of the model using [average-percentage-of-zeros](https://arxiv.org/abs/1607.03250)\-based <i>parameter pruning</i> on channel level as well as post-training quantization of the Keras model (`.h5`) to Tensorflow Lite (`.tflite`), where [by default, only weights are quantized](https://www.tensorflow.org/lite/performance/post_training_quant) to 8-bit, leaving the rest in floating points. As such, it is possible to compare the visualizations of the original model with its pruned variant, hopefully gaining an insight into how the visible patterns change with an increased pruning percentage.
 
## My Results

For the cao model, I picked the pruning percentages 0%, 40%, 55%, 70% and 85% for visualization purposes because looking at an accuracy graph, these were the points where the behavior of the curve changed (cf. [pruning evaluation](https://github.com/daniel-rychlewski/hsi-toolbox/blob/master/DeepHyperX/outputs/Excel%20Evaluations/Evaluation%20Pruning.xlsx)).
All gradient-based saliency map visualizations without prior band selection have turned out to be black squares. However, using PCA or NMF for image compression, one can see a gradient picture which is becoming increasingly distorted the higher we set the pruning percentage. This allows for the assumption that <b>important neurons remain unimpaired by moderate pruning</b>. My NMF results vary less with increasing pruning percentage than the PCA results. These trends are valid for all trainable weights of all layers of the cao model, as the following picture illustrates (from left to right: <i>from_band</i>: 0, <i>to_band</i>: 100, <i>step_band</i>: 10)

![Visualization](https://github.com/daniel-rychlewski/Cao-Keras-Visualization-private/blob/master/outputs/Visualization.png)

As for the activation maps, they remain completely unchanged throughout all layers and all pruning percentages, but vary among the band compression techniques (the same image bands as above are chosen): 

![Activation Maps](https://github.com/daniel-rychlewski/Cao-Keras-Visualization-private/blob/master/outputs/Activation%20Maps.png)

For more visualization-related theory, please delve into my [master thesis](https://github.com/daniel-rychlewski/Cao-Keras-Visualization-private/blob/master/outputs/Master%20Thesis.pdf) and/or my [thesis defense](https://github.com/daniel-rychlewski/Cao-Keras-Visualization-private/blob/master/outputs/Thesis%20Defense.pptx).

## Architecture

For the sake of maximum flexibility in the visualization implementations, e.g., to be able to use integrated gradients instead of "normal" gradients, I have decided to use the methods of [deep-viz-keras](https://github.com/experiencor/deep-viz-keras) instead of [Keras-vis](https://github.com/raghakot/keras-vis).

![Architecture](https://github.com/daniel-rychlewski/Cao-Keras-Visualization-private/blob/master/outputs/Architecture.png)

The following are the core components of the application:

* `generate_model.py`: generates the desired model with the specified parameters or default ones, if none given. Can perform pruning and quantization (or both) after the training phase
* `visualization.py`: reads a model from a file so that one of the following visualization methods can be applied: (integrated) gradients, guided/visual backpropagation, activation maps
* `preprocessing.py`: loads and preprocesses the hyperspectral dataset
* `compression.py`: contains the model compression logic of pruning and quantization, both of which include inference time measurement
* `models.py`: a selection of models to choose from (to be expanded)

The program will access the IndianPines dataset from `Indian_pines_corrected.mat`. The train and test labels are given in the files `train_data.npy` and `test_data.npy` respectively.
The `output` folder contains templates of the cao model (baseline, PCA-100, NMF-100). 

## Getting Started

### Installation

Install the required dependencies using

`pip install -r requirements.txt`

I recommend an Anaconda Python 3.6 environment to run the program.

### Sample Commands 

There are two steps to execute the program:

<b>1\. To generate a model</b> in a model file, run `generate_model.py`, e.g.:

`generate_model.py cao --epochs 30 --band_selection NMF --components 150 --prune --prune_end 30 --quantize --quantize_pruned_models`

* generates the cao model, trains it with 30 epochs, applies the NMF band selection technique with 150 components; the model is pruned after training until 30 percent and the pruned models are quantized

`generate_model.py cao --epochs 20 --band_selection PCA --components 20 --show_images --prune --prune_increment 10 --prune_end 60`

* generates the cao model, trains it with 20 epochs, applies the PCA band selection technique with 20 components, visualizing the confusion matrix along with the predicted and ideal images (prior to pruning), pruning the model until 60 percent after training with 10 percent per step

`generate_model.py cao`

* minimum command to run the model generation script, in other words, you only need to specify the <b>model name</b> as a mandatory parameter, the defaults are taken for the rest. Please note that without using a band selection technique like PCA for accuracies at 100%, accuracies as low as 25% are absolutely possible. However, feel free to add a model to optimize it for usage without band selection: e.g., ReLU activation instead of Softmax can improve cao's accuracy to 50% (still bad, but a considerable improvement through just one change).
 
<b>2\. To visualize the model</b>, read in the generated model file from 1. and run `visualization.py`, specifying the desired visualization technique, e.g.:  

`visualization.py cao trained_pruned_model_20.h5 activation_map --band_selection NMF --components 100 --from_band 30 --to_band 90 --step_band 10`

* generates activation maps for the cao model read from the file `trained_pruned_model_20.h5`, using NMF with 100 components as band selection for the dataset, visualizing bands 30 to 80 (end is exclusive) in steps of 10

`visualization.py cao trained_model.h5 gradient`

* minimum command to run the visualization script, with the mandatory arguments being the <b>model name, model file and visualization type</b>

## Authors

* [Daniel Rychlewski](https://github.com/daniel-rychlewski) - sole contributor of this project

## Possible Expansions

* load (perhaps quantized) `.tflite` models to visualize them, just like Keras' `.h5` models can be read
* make guided backpropagation work for pruned models: currently, tensors are not found (sample output: `The name 'dense_3_1/Softmax:0' refers to a Tensor which does not exist. The operation, 'dense_3_1/Softmax', does not exist in the graph.`)
* additional arguments for the argument parser:
    * make the save paths of generated model and visualizations adjustable instead of saving everything in the same folder
    * allow for a custom naming of the visualization output files
    * do not hardcode the number of trainable weights / layers in the visualization methods
    * choose which visualizations shall be generated instead of hardcoding gray and jet visualization types for smoothed and non-smoothed masks, generating all of them every time though perhaps not desired
* try to implement further hyperspectral models in Keras - their [PyTorch implementations](https://github.com/daniel-rychlewski/hsi-toolbox) can be taken as the template
* support further hyperspectral datasets: [EHU GIC](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), [RS Lab](https://rslab.ut.ac.ir/data)
* include variants of the implemented pruning and quantization methods (vary the granularity of pruning and the bit numbers for quantization components: [hsi-toolbox](https://github.com/daniel-rychlewski/hsi-toolbox))
* add other compression methods like low-rank factorization, compact convolutional filters and knowledge distillation (cf. [survey](https://arxiv.org/abs/1710.09282))