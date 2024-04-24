# MNIST Digit Localization and Classification

This repository contains code for localizing MNIST digits in an image and classifying images into MNIST buckets.

## Description

The MNIST dataset is a widely-used benchmark dataset in the field of computer vision. This repository provides functionality to localize digits within an image and classify them into their respective MNIST buckets using machine learning techniques.

## Files

- `main.py`: Main script for executing the digit localization and classification process.
- `image_localisation.py`: Module containing functions for localizing MNIST digits within an image.
- `image_classification.py`: Module containing functions for classifying images into MNIST buckets.

## Usage

To use the digit localization and classification functionalities provided by this repository, follow these steps:

1. Clone the repository:

```
git clone <repository-url>
cd mnist-digit-localization-classification
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the main script to execute the localization and classification process:

```
python main.py
```

## Approach

- **Digit Localization**: The `image_localisation.py` module implements algorithms to localize MNIST digits within an image. This involves techniques such as contour detection, bounding box extraction, and image segmentation.

- **Digit Classification**: The `image_classification.py` module provides functions to classify images into MNIST buckets. This typically involves training a machine learning model (e.g., convolutional neural network) on the MNIST dataset and then using it to predict the class of digits extracted from images.


[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
