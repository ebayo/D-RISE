# D-RISE implementation 

> We propose D-RISE, a method for generating visual explanations for the predictions of object detectors. Utilizing the proposed similarity metric that accounts for both localization and categorization aspects of object detection allows our method to produce saliency maps that show image areas that most affect the prediction. D-RISE can be considered "black-box" in the software testing sense, as it only needs access to the inputs and outputs of an object detector. Compared to gradient-based methods, D-RISE is more general and agnostic to the particular type of object detector being tested, and does not need knowledge of the inner workings of the model. We show that D-RISE can be easily applied to different object detectors including one-stage detectors such as YOLOv3 and two-stage detectors such as Faster-RCNN. We present a detailed analysis of the generated visual explanations to highlight the utilization of context and possible biases learned by object detectors. 

 - Using pytorch
 - Original [paper](https://arxiv.org/abs/2006.03204)
 - Third-party implementation on [github](https://github.com/hysts/pytorch_D-RISE)
   - [Notebook](https://github.com/hysts/pytorch_D-RISE/blob/main/demo.ipynb)

## Requirements:
   - OpenCV
   - numpy >= 21.0.4 (implementation of ```numpy.concatenate()```)
```
pip install opencv-python
pip install numpy>=21.0.4
```

## Implementation Notes

**Similarity metric**

Implemented both as described in the paper and in a simplified version.

The simplified version in [utils.metrics.similarity_metric_simplified](utils/metrics.py) takes into account the fact
that ```target``` is one-hot encoded and only confidence score of the target's class is taken into account.

**Mask Generation**

Removed random offset in mask generations. In Google Colab it was generating masks with the wrong size (it work fine on local)

## Usage

 - See [```drise-yolov5.py```](drise-yolov5.py) for a working example using YOLOv5 and cpu (as an installed package)
 - See [```D-Rise - Basic usage.ipynb```](https://colab.research.google.com/drive/1g81rLrN7peIuh961z3la-SNAkRea-VWd?usp=sharing) colab notebook for an example notebook (tested in Google Colab with GPU) and using YOLOv5 as detector.

***TODO***

# RISE (Randomized Input Sampling for Explanation of Black-box Models) implementation

Implementation of the [paper](https://arxiv.org/abs/1806.07421)

Based on the D-Rise implementation above.

## Implementation Notes

**Mask weights**

Multiplied by the original classification probability.

**Normalization**

***TODO*** How to scale masks so way less probable classes are shown as such

## Usage
   - See [```Rise - Basic usage.ipynb```](https://colab.research.google.com/drive/1SqpvF0OfEF_lIT9QbzE_r3QJKvbX5hUf?usp=sharing) colab notebook for an example notebook (tested in Google Colab with GPU) and using Mobilenet from pytorch model zoo as classifier.

