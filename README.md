# Anomaly Detection Approach to Identify Early Cases in a Pandemic using Chest X-rays
If you use this code, please use the following citation - *Shehroz S. Khan, Faraz Khosbakhtian, Ahmad Bilal Ashraf, Anomaly Detection Approach to Identify Early Cases in aPandemic using Chest X-rays, 34th Canadian Conference on Artificial Intelligence, BC, 2021*

## Background 

COVID-19 continues to have a devestating effect on the health and the wellness
 of global population. Here, we are proposing a Convolutional
 Variational Autoencoder for Anomaly Detection of COVID-19 cases from CXR images. 

For a detailed description of COVIDx dataset and the methodology behind COVIDomaly please visit [here](https://arxiv.org/abs/2010.02814)
## Data

The data used for this network is a modified version of [COVIDx](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md). I have downsampled the images into  **224 * 224**.

You can download the modified dataset [here](https://drive.google.com/file/d/1HPfpYyCxeDQFj0y6fc1eAiCTA79PCtx_/view?usp=sharing)

Please make sure `resized_COVIDx.zip` is placed at the same directory as the repository's root directory (at the same place as the `main.py`). 



## Running an Experiment (Training/Testing)

Before trying to run the experiments, please make sure you:

1. Have **downloaded and extracted** the data from [here](https://drive.google.com/file/d/1OUvFc96sHbbzRbfQrRbyU4XCeWD6NAU3/view?usp=sharing)
2. You meet the library requirements listed in the **Requirements** section of this document 

You can choose the behaviour of the code and the hyper-parameters of network by command line arguments.
**To get some help run**: 

```
python main.py -h
```

To experiment only with *normal* cases in training:
```
python main.py -n normal
```

To experiment with *normal* cases and *non-COVID Pneumonia* in training:
```
python main.py -n normal pneumonia
```

You can also turn off the training, and just do the **testing**:
```
python main.py -n normal pneumonia --train=False
```

## Requirements

To use this project you will need:

```OpenCV, sklearn, Pandas, pyTorch, Tensorflow, TensorBoard, and matplotlib```

For ease of use, you can clone the environment used in development of the project. Make sure you 
are in the root directory of the project and run this command in terminal:
`You can clone my environment with command:
```
conda env create -f ./Z-conda-env/COVIDomaly.yml
conda activate COVIDomaly
````
