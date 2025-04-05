# Wine Quality Classification Report

> Author: Tien Ly  
> CS 271: Topics in Machine Learning - Spring 2025 at San Jose State University

## Introduction
This report documents the development and evaluation of a neural network model for classifying wine quality. The objective was to classify wine samples into 5 quality categories (Low, Medium-Low, Medium, Medium-High, and High) using various physicochemical properties as features. The project aimed to achieve a classification accuracy of 40-45%, which was successfully accomplished.

## Dataset Overview
The dataset contains multiple features related to wine composition, including:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target variable is "quality," which has 5 ordinal categories: Low, Medium-Low, Medium, Medium-High, and High.

## Data Preprocessing Methods

### Exploratory Data Analysis
Before preprocessing, exploratory data analysis was conducted to understand the dataset's characteristics:
- Examined the distribution of wine quality classes
- Analyzed descriptive statistics for all features
- Created correlation matrices to identify relationships between features

### Data Cleaning
- Checked for missing values in the dataset (none were found)
- Examined the class distribution, which showed some imbalance across quality categories

### Feature Transformation
1. Label Encoding: 
   - Applied ordinal encoding to the text-based quality labels
   - Mapping: {"Low": 0, "Medium-Low": 1, "Medium": 2, "Medium-High": 3, "High": 4}
   - This approach preserves the ordinal nature of the quality ratings.

2. Feature Normalization:
   - Applied StandardScaler to normalize all features
   - This ensured all features had mean = 0 and standard deviation = 1
   - Normalization helps prevent features with larger magnitudes from dominating the learning process and improves gradient descent convergence during model training.

### Data Splitting
- Split the dataset into 3 parts using stratified sampling to maintain class distribution:
  - Training set (80% of data)
  - Validation set (10% of data)
  - Test set (10% of data)
- Stratification ensured that each subset contained representative proportions of all quality classes.

### PyTorch Data Preparation
- Converted data to PyTorch tensors
- Created TensorDatasets for efficient data handling
- Implemented DataLoaders with batch processing (batch size=32)
- Used GPU acceleration when available to speed up training

## Model Architecture
The neural network architecture consists of:
- Input layer: Dimension matches the number of features
- First hidden layer: 64 neurons with ReLU activation
- Second hidden layer: 32 neurons with ReLU activation
- Dropout layer with 0.3 probability
- Output layer: 5 neurons (1 per quality class)

Key model parameters:
- Loss function: CrossEntropyLoss
- Optimizer: Adam with learning rate of 0.001
- Weight initialization: He initialization for hidden layers, Xavier initialization for output layer

## Strategies to Mitigate Overfitting
Several techniques were implemented to prevent the model from overfitting:

1. Dropout Regularization:
   - Applied dropout with a rate of 0.3 between the second hidden layer and output layer
   - Forces the network to learn more robust features that do not rely on specific neuron combinations

2. Weight Decay:
   - Added L2 regularization via weight decay parameter (1e-5) in the Adam optimizer
   - Penalizes large weight values to prevent the model from becoming too complex
   - Encourages the model to learn simpler, more generalizable patterns

3. Early Stopping:
   - Monitored validation accuracy and stopped training when no improvement was observed for 20 consecutive epochs
   - Saved the best-performing model based on validation accuracy
   - Prevented the model from overfitting to the training data by stopping before validation performance degraded

4. Learning Rate Scheduling:
   - Implemented ReduceLROnPlateau scheduler
   - Reduced learning rate by a factor of 0.5 when validation loss plateaued for 10 epochs
   - Allowed fine-tuning of weights in later training stages when approaching a solution

5. Data Stratification:
   - Used stratified sampling in the train-test split to ensure balanced class representation
   - Helped the model learn from all classes equally, preventing bias toward majority classes

6. Proper Weight Initialization:
   - Used He initialization for ReLU activation layers
   - Used Xavier initialization for the output layer
   - Proper initialization helps prevent vanishing/exploding gradients and accelerates convergence.

## Experimental Results
The final model achieved the following performance metrics:
- Training accuracy: Approximately 47-49%
- Validation accuracy: Approximately 45-47%
- Test accuracy: 45.40%

The model successfully met the target accuracy range of 40-45% for this multi-class classification task.

### Learning Curves
The training and validation loss/accuracy curves showed:
- Initial rapid improvement in both training and validation performance
- Gradual convergence with diminishing returns as training progressed

### Feature Importance Analysis
A simple feature importance analysis revealed that pH, fixed acidity, and total sulfur dioxide were among the most influential features for predicting wine quality.

## Conclusion
The implemented neural network successfully classified wines into 5 quality categories with 45% accuracy, meeting the project's objective. The preprocessing methods, particularly feature normalization and label encoding, played a crucial role in preparing the data for effective learning. Additionally, the various anti-overfitting strategies ensured the model generalized well to unseen data rather than simply memorizing the training examples.

Future work could explore more sophisticated feature selection techniques, ensemble methods, or different neural network architectures to potentially improve classification accuracy further.
