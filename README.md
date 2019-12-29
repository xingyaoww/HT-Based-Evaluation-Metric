# HTBased-MSE Evaluation Metric

## Dataset
Source: https://xingangpan.github.io/projects/CULane.html

This Project used **CULane** dataset which is a large scale challenging dataset for traffic lane detection. 

It consists of 133,235 total frames, including 88880 for trainning, 9675 for validation, and 34680 for test.

The resolution of individual image is : 1640x590.

## Model

The model used Down/Up Sampler, GCN, BR to implement lane recognition.

* Model Input Shape: (294, 820, 3)
* Epoch = 40
* Batchsize = 8

## Files
* `run.py`: start the trainning process.
* `HTBloss_core.py`: Detailed implementation for HTBloss Metric.
* `loadDataset.py`: Keras data generator for training. Also includes data pre-processing codes.
* `buildmodel.py`: Detailed implementation of Neural network.
 