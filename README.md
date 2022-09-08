# COVID19_segmentation_classification: Classification of COVID-19 from chest X-ray images using Deep Convolution Neural Networks
## Introductions
-  Direct classification on chest X-ray images uses existing deep learning models such as [VGG19_bn](https://arxiv.org/pdf/1409.1556.pdf), [ResNet50](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf), and [EfficientnetB7](https://arxiv.org/pdf/1905.11946.pdf).
 - Segmentation of chest X-ray images uses [Feature Pyramid Network (FPN) model with DenseNet121 encoder](https://arxiv.org/pdf/1612.03144.pdf), generated ground-truth lung segmentation masks for the benchmark COVIDx CXR3 dataset. Then combining segmented image and original chest x-ray image for classification.
 - The experimental results on original Chest X-ray (CXR), Segmented Lung, CXR without Lung images with three models: ResNet50, VGG19_bn, EfficientNetB7 show **the VGG19_bn achieves 99.00% accuracy on Segmented Lung images** 
## Pipeline
![The pipeline of the segmentation-classification](https://github.com/loan1/COVID19-segmentation_classification/blob/main/images/pipeline.png)
## Datasets
- **Train Lung Segmentation**: [COVID_QU_Ex](https://www.sciencedirect.com/science/article/pii/S0010482521007964) contains of 33,920 CXR images including:
  - 11,956 COVID-19
  - 11,263 Non-COVID infections (Viral or Bacterial Pneumonia)
  - 10,701 Normal 
*Ground-truth lung segmentation masks are provided for the entire dataset.*
- **Prediction Lung Segmentation and classification Covid-19**: [COVIDx CXR3 dataset](https://www.nature.com/articles/s41598-020-76550-z) (update 06/02/2022) contains 29,986 CXR images
## Requirements
- Ubuntu
- GeForce RTX
- Python 3.7
- cuda 11.6
- cuDNN 8.3.20
- PyTorch 1.12.0
## Compiling
Segmentation script:
```bash
python source\segmentation\mainseg.py
```
Classification script:
```bash
python source\classification\mainclassification.py
```
## Results
| **Models**      | **Experiments**     | **Accuracy** |  **Recall** | **Precision** | **F1-score** |
|:-------|:---------|:---------:|:-------:|:---------:|:---------:|
| ResNet50        | Original CXR     | 98.75%     | 98.75%  | 98.78%    | 98.75%|
|               | Segmented Lung   | 96.50%     | 96.50%  | 96.62%    | 96.50%|
|              | CXR without Lung | 98.75%     | 98.75%  | 98.78%    | 98.75%|
| VGG19_bn        | Original CXR     | 98.00%     | 98.00%  | 98.08%    | 98.00%|
|                | **Segmented Lung**  | **99.00%**     | **99.00%**  | **99.02%**    | **99.00%**|
|                | CXR without Lung | 98.25%     | 98.25%  | 98.31%    | 98.25%|
| EfficientNetB7  | Original CXR     | 97.75%     | 97.75%  | 97.85%    | 97.75%|
|                | Segmented Lung   | 98.50%     | 98.50%  | 98.52%    | 98.50%|
|                | CXR without Lung | 96.75%     | 96.75%  | 96.95%    | 96.75%|
     
# COVID19-segmentation-classification
