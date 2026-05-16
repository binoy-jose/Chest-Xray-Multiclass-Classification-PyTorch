# Chest-Xray-Multiclass-Classification-PyTorch
Deep learning project for multi-class chest X-ray classification using PyTorch and transfer learning with ResNet and DenseNet architectures.
---

# 1. Introduction

The objective of this project is to perform multi-class classification of chest X-ray images using deep learning techniques. Chest X-rays are widely used in medical diagnosis to detect various thoracic diseases, making automated classification an important task for assisting healthcare professionals.

The dataset consists of labeled chest X-ray images belonging to multiple disease categories. Each image is associated with one of 20 classes, and the goal is to correctly classify each image into its respective category.

The main objective of this project is to develop a robust deep learning model that can accurately classify X-ray images while optimizing performance under the given evaluation metric.

---

# 2. Methodology

## Models Used

The following models were implemented and tested:

* ResNet18 (baseline model)
* ResNet34 (deeper architecture)
* DenseNet121 (dense connectivity network)

ResNet models were chosen due to their residual connections, which help in training deep networks efficiently. DenseNet was selected because of its ability to reuse features and improve gradient flow.

## Preprocessing

The following preprocessing steps were applied:

* Resizing all images to 224 × 224
* Converting images to tensors
* Normalization using ImageNet mean and standard deviation

## Data Augmentation

To improve generalization, the following augmentation techniques were used:

* Random horizontal flipping
* Random rotation
* Slight brightness and contrast adjustment (later removed due to poor perfromance )

## Training Strategy

* Transfer learning was used with pretrained models
* Final classification layer was modified for 20 classes
* CrossEntropyLoss with class weights was used to handle class imbalance
* Adam optimizer with learning rate 5e-5 was used (3e-5,4e-5 were used but not found as optimal)

## Prediction Strategy

* Model outputs were converted to class predictions using argmax(tried softmax also but submission file allowed only 0 and 1 )
* Predictions were converted to one-hot encoded vectors (0/1 format) as required by the competition

---

# 3. Experiments

The following experiments were conducted:	

### Models Used

- ResNet18 (baseline model)
- ResNet34 (deeper variant of ResNet)
- DenseNet121 (dense connectivity network)

All models were trained using transfer learning with pretrained ImageNet weights. The final classification layer was modified to output 20 classes, and the same preprocessing, augmentation, and training strategy were applied across models for fair comparison.


### Hyperparameter Variations

* Epochs: 5, 6, 10
* Learning rate adjustments (5e-5,3e-5,4e-5)
* Use of scheduler and label smoothing (later removed due to poor performance)

### Observations

- **ResNet18** provided the most stable and consistent performance across runs.
- **ResNet34**, despite being deeper, showed poorer performance due to possible undertraining and overfitting.
- **DenseNet121** achieved better validation metrics but failed to generalize well on the Kaggle test set.
- Increasing epochs beyond optimal caused overfitting
- Additional tuning (scheduler, label smoothing) reduced performance
- Simpler configurations performed better


This highlights that increasing model complexity does not always lead to better performance, especially under limited training time and computational constraints.
---

# 4. Results

## Evaluation Strategy

A custom evaluation metric was used in this project to measure model performance. The scoring function assigns:

- +1 for a correct prediction
- -6 for an incorrect prediction

The final score is computed as the average over all predictions.

Mathematically, the custom score is defined as:

Custom Score = (Sum of scores for all predictions) / (Total number of samples)

This scoring mechanism heavily penalizes incorrect predictions, making it important for the model to minimize misclassification rather than just maximizing accuracy.

In addition to the custom score, the weighted F1 score was also used as a validation metric to account for class imbalance.

| Model       | Custom Score | Weighted F1 Score | Kaggle Score |
| ----------- | ------------ | ----------------- | ------------ |
| ResNet18    | -4.279       | 0.3037            | -6.55        |
| DenseNet121 | -4.02        | 0.3454            | -10.68       |
| ResNet34    |              |                   | -8.33        |

---

## Key Insight

ResNet18 achieved the best and most stable performance. DenseNet and ResNet34 showed unstable behavior and poorer generalization under the given constraints.

---

# 5. Conclusion

In this project, multiple deep learning models were implemented and evaluated for chest X-ray classification. Among all models tested, ResNet18 provided the most stable and reliable performance.

It was observed that increasing model complexity or training duration did not necessarily improve performance. In fact, excessive training led to overfitting, which significantly reduced the evaluation score due to the harsh penalty for incorrect predictions.

The key learning from this project is that simpler models with proper tuning can outperform more complex architectures, especially under computational and time constraints.

---

# 6. References

* Rajpurkar et al.,"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"
* Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison"
* He et al., "Deep Residual Learning for Image Recognition"
* Huang et al., "Densely Connected Convolutional Networks"
* PyTorch Documentation: https://pytorch.org/docs/

---
