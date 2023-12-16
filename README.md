# Skin Cancer Detection and Classification Using Machine Learning

<img src="https://miro.medium.com/v2/resize:fit:640/1*daa7-pWyR8mQtQSnjVLwFg.gif">

## Geoff

```
python3 -m venv myenv
source myenv/bin/activate
pip install tensorflow-macos
pip install tensorflow-metal  # for GPU support on M1/M2 Macs
pip install -r requirements.txt
pip install notebook
jupyter notebook
```

## Introduction
This project aims to apply `Artificial Intelligence` in the `detection` and `classification` of skin cancers.
Just like many other cancer types, skin cancer when detected early, can be cured through surgical interventions. However, early detection remains a challenge for the following reasons:

1. Clinical examinations are expensive and require a high level of training, and effort to operate the equipment.
2. There are 9 classes of skin cancer, and so, it is usually difficult for even experienced medical practitioners to accurately identify and classify them visually.

## Objectives
1. To train a machine learning model which can detect and classify skin cancer types.
2. To deploy our model as a web app that will enable doctors (and anyone else) to quickly diagnose skin cancer using their smartphones, instead of going to perform the experiments in the laboratory.



## About the Dataset
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed by the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, except for melanomas and moles, whose images are slightly dominant.
The data set contains the following diseases:
1. _Actinic keratosis_
2. _Basal cell carcinoma_
3. _Dermatofibroma_
4. _Melanoma_
5. _Nevus_
6. _Pigmented benign keratosis_
7. _Seborrheic keratosis_
8. _Squamous cell carcinoma_
9. _Vascular lesion_

The dataset can be found [here](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) on Kaggle.


## Model Summary
Because this is a multi-class classification problem, we used a Convolution Neural Network (CNN) consisting of 16 layers to train the model. We also used other Python libraries such as Tensorflow, Pandas, Matplotlib, etc., for image preprocessing, data visualizations, and other ancillary statistical analysis, and the resulting model takes about 45 minutes to train.

__For additional information, please read the project article on [Medium](https://medium.com/@kennethodoh/skin-cancer-classification-using-ai-45cdf70e808c).__

## Contributors

Volker Tachin

Omar Alqaysi

Kenneth Odoh

Promise Uzoagulu

Shreyansh Gupta

Emako Efatobor

Thandazi Mnisi

Mutholib Yusira

Sharon Prempeh