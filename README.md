# D-RISE and RISE implementation 

Implementation of both methods for visual inspection of classifiers and object detectors neural networks. 

## Requirements:
   - OpenCV
   - numpy >= 21.0.4 (implementation of ```numpy.concatenate()```)

```bash
pip install opencv-python
pip install numpy>=21.0.4
```

**Additionally**, deep learning libraries will be needed to perform detection and classification of the images, but both `DRise` and `Rise` classes are implemented to be independent  of them.

The example uses pytorch both with and without GPU. Go to [pytorch](https://pytorch.org/get-started/locally/) for installation instructions.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
(pytorch for Linux on CPU using pip)

```bash
pip install tqdm
pip install yolov5
```

## D-RISE

Abstract from the original D-RISE paper:
> We propose D-RISE, a method for generating visual explanations for the predictions of object detectors. Utilizing the proposed similarity metric that accounts for both localization and categorization aspects of object detection allows our method to produce saliency maps that show image areas that most affect the prediction. D-RISE can be considered "black-box" in the software testing sense, as it only needs access to the inputs and outputs of an object detector. Compared to gradient-based methods, D-RISE is more general and agnostic to the particular type of object detector being tested, and does not need knowledge of the inner workings of the model. We show that D-RISE can be easily applied to different object detectors including one-stage detectors such as YOLOv3 and two-stage detectors such as Faster-RCNN. We present a detailed analysis of the generated visual explanations to highlight the utilization of context and possible biases learned by object detectors. 

 
 - Original [paper](https://arxiv.org/abs/2006.03204)
 - Third-party implementation on [github](https://github.com/hysts/pytorch_D-RISE)
   - [Notebook](https://github.com/hysts/pytorch_D-RISE/blob/main/demo.ipynb)
 - Using pytorch and [YOLOv5](https://github.com/ultralytics/yolov5) (installed) to perform object detection

### Implementation Notes

**Similarity metric**

Implemented both as described in the paper and in a simplified version.

The simplified version in [utils.metrics.similarity_metric_simplified](utils/metrics.py) takes into account the fact
that ```target``` is one-hot encoded and only confidence score of the target's class is taken into account.

**Mask Generation**

Removed random offset in mask generations. In Google Colab it was generating masks with the wrong size (it worked fine on local)

### Usage

- See [```drise-yolov5.py```](drise-yolov5.py) for a working example using YOLOv5 and cpu (as an installed package)
  - Tested with an older version of the yolov5 package, it's not currently working as is.


## RISE (Randomized Input Sampling for Explanation of Black-box Models) implementation

> Deep neural networks are being used increasingly to automate data analysis and decision making, yet their decision-making process is largely unclear and is difficult to explain to the end users. In this paper, we address the problem of Explainable AI for deep neural networks that take images as input and output a class probability. We propose an approach called RISE that generates an importance map indicating how salient each pixel is for the model’s prediction. In contrast to white-box approaches that estimate pixel importance using gradients or other internal network state, RISE works on blackbox models. It estimates importance empirically by probing the model with randomly masked versions of the input image and obtaining the corresponding outputs. We compare our approach to state-of-the-art importance extraction methods using both an automatic deletion/insertion metric and a pointing metric based on human-annotated object segments. Extensive experiments on several benchmark datasets show that our approach matches or exceeds the performance of other methods, including white-box approaches

- Implementation of the [paper](https://arxiv.org/abs/1806.07421)
- Based on the D-Rise implementation

## Implementation Notes

**Mask weights**

Multiplied by the original classification probability.

**Normalization**

***TODO*** How to scale masks so way less probable classes are shown as such


