# Student Mental Health Classification

A machine learning classification project exploring how different supervised learning approaches perform when identifying patterns associated with depression and anxiety in college student survey data.

The project compares **Decision Trees**, **Random Forests**, and a **TensorFlow neural network**, with a focus on data preprocessing, categorical feature encoding, model experimentation, feature selection, and comparative performance evaluation.

> **Important:** This project was created for academic and educational purposes. The models are not intended to diagnose, screen, or provide medical advice regarding mental health conditions.

---

## Overview

Mental health challenges among college students present an interesting machine learning classification problem because student survey datasets contain a combination of demographic, academic, and categorical information.

The objective of this project was to explore whether supervised machine learning techniques could identify meaningful patterns within student survey data and determine how different models performed on the same general classification task.

The project investigates three approaches:

- Decision Tree classification
- Random Forest classification
- TensorFlow neural networks

Rather than focusing only on achieving the highest accuracy, the project also examines how preprocessing, feature selection, model architecture, and dataset limitations affect model performance.

---

## Project Goals

The primary goals of the project were to:

- Explore the use of machine learning for classification of student mental health survey data.
- Compare traditional machine learning models with a neural network.
- Experiment with feature selection and model configuration.
- Convert categorical survey information into machine-readable features.
- Evaluate how different modeling approaches respond to the same dataset.
- Identify limitations caused by small datasets and potentially weak predictive features.

---

## Technologies

### Languages

- Python

### Machine Learning & Data Analysis

- TensorFlow
- scikit-learn
- Pandas
- KNIME

### Techniques

- Decision Tree Classification
- Random Forest Classification
- Neural Networks
- One-Hot Encoding
- Train/Test Splitting
- Feature Selection
- Data Preprocessing
- Model Evaluation
- Hyperparameter Experimentation

### Development Environment

- Jupyter Notebook
- KNIME Analytics Platform

---

## Dataset

The project uses the **Student Mental Health** dataset published on Kaggle by Shariful Islam.

Dataset:

https://www.kaggle.com/datasets/shariful07/student-mental-health

The dataset contains survey responses from approximately **100 college students** and includes attributes such as:

- Age
- Academic major
- GPA-related information
- Marital status
- Depression indicators
- Anxiety indicators
- Other demographic and academic attributes

The original project primarily explored classification of:

**Depression**

and

**Anxiety**

based on the available student attributes.

### Dataset Limitations

The relatively small dataset is one of the most important limitations of the project.

A dataset of approximately 100 observations makes it difficult to determine whether patterns learned by a model will generalize to a broader population.

Because of this, the results should be interpreted as an **academic machine learning experiment rather than evidence of a clinically useful predictive model**.

---

# Machine Learning Pipeline

The overall experimentation process can be represented as:

```text
Student Survey Dataset
        │
        ▼
Data Exploration
        │
        ▼
Data Cleaning
        │
        ▼
Categorical Encoding
        │
        ▼
Feature Selection
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
Decision Tree      Random Forest    Neural Network
        │               │                │
        └───────────────┴────────────────┘
                        │
                        ▼
                Model Evaluation
                        │
                        ▼
               Results Comparison
```

Two separate modeling environments were used.

**KNIME** was used for the Decision Tree and Random Forest experiments.

**Python and TensorFlow** were used to create and evaluate the neural network.

---

# Decision Tree Classification

The first approach explored was a categorical **Decision Tree classifier**.

Decision trees were useful for this project because they allow relationships between individual student attributes and classification outcomes to be examined relatively easily.

The project experimented with different decision-tree settings while attempting to classify depression and anxiety.

## Depression Classification

Initial experiments produced approximately:

**63% accuracy**

After changing the tree's quality measure from **Gini Index** to **Gain Ratio**, the recorded accuracy increased to approximately:

**82% accuracy**

This demonstrated how algorithm configuration can significantly influence the resulting model.

It also provided an opportunity to explore which attributes were being used by the tree when making classifications.

---

## Anxiety Classification

The same approach was also used to classify anxiety.

The Decision Tree produced approximately:

**65% accuracy**

Changing the quality measure did not provide the same improvement observed during depression classification.

This suggested that the attributes available in the dataset may have contained weaker predictive information for anxiety than they did for depression.

---

# Random Forest Classification

The next model evaluated was a **Random Forest classifier**.

Random Forests combine predictions from multiple decision trees and generally provide better resistance to overfitting than a single tree.

The project experimented with both model configuration and feature selection.

## Depression Classification

The initial Random Forest produced performance similar to the Decision Tree.

One particularly interesting experiment involved removing the student's **academic major** from the feature set.

After removing this feature, the recorded depression-classification accuracy increased to approximately:

**85% accuracy**

This experiment demonstrated that adding more features does not automatically improve a model.

Some features may add noise rather than useful predictive information.

---

## Anxiety Classification

Random Forest was also tested against the anxiety classification task.

The recorded accuracy remained relatively low compared with depression classification, at approximately:

**62%**

Experimenting with Random Forest parameters did not produce a meaningful improvement.

This reinforced one of the project's major observations:

> A more powerful algorithm cannot compensate for features that contain limited predictive information.

---

# TensorFlow Neural Network

After experimenting with traditional machine learning classifiers, the project explored whether a neural network could produce better results.

The neural-network implementation was developed using:

- Python
- TensorFlow
- Pandas
- scikit-learn

The implementation is available in:

```text
MentalHealthWithTF.ipynb
```

---

## Data Preprocessing

One of the primary challenges when moving the dataset into TensorFlow was handling categorical information.

Many dataset attributes contained text values rather than numeric values.

Examples include information such as:

```text
Academic Major
Marital Status
Categorical Survey Responses
```

Machine learning models require these values to be converted into numerical representations.

The project used Pandas:

```python
pandas.get_dummies()
```

to perform **one-hot encoding**.

For example:

```text
Major
────────────
Computer Science
Engineering
Psychology
Business
```

could be transformed into features similar to:

```text
Major_ComputerScience
Major_Engineering
Major_Psychology
Major_Business
```

Each feature becomes a binary indicator.

---

## Feature Expansion

One challenge encountered during preprocessing was the rapid increase in the number of features.

Encoding the academic-major field alone generated approximately **40 additional columns**.

This demonstrated one of the disadvantages of one-hot encoding high-cardinality categorical variables.

It can significantly increase the dimensionality of a dataset while potentially introducing redundant information.

---

# Train/Test Split

The processed dataset was separated into training and testing subsets using:

```python
train_test_split()
```

from scikit-learn.

One experiment used:

```text
85% Training Data
15% Testing Data
```

Later experimentation also tested:

```text
90% Training Data
10% Testing Data
```

The purpose of the split was to evaluate the model against observations it had not used during training.

---

# Neural Network Architecture

According to the original project implementation, the network consisted of a **Flatten layer followed by three Dense layers**.

The model was compiled using:

```text
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Evaluation Metric: Accuracy
```

Additional experiments tested deeper networks containing additional layers.

Increasing the number of layers did not meaningfully improve performance on this dataset.

This was an important result because it demonstrated that increasing model complexity does not necessarily increase predictive accuracy.

---

# Neural Network Results

The initial TensorFlow model produced approximately:

**68.75% test accuracy**

After experimenting with the training and testing split, a configuration using approximately 90% training data and 10% testing data produced:

**72.72% accuracy**

The neural network therefore did not outperform the strongest traditional machine learning model used in the project.

---

# Model Comparison

| Model | Target | Approx. Recorded Accuracy |
|---|---|---:|
| Decision Tree | Depression | 82% |
| Random Forest | Depression | 85% |
| Neural Network | Depression | 68.75–72.72% |
| Decision Tree | Anxiety | ~65% |
| Random Forest | Anxiety | ~62% |

These values represent results recorded during the original experiments and should not be interpreted as directly comparable benchmark scores.

Different preprocessing decisions, feature selections, model configurations, and train/test splits were explored during the project.

The small dataset also means that relatively small changes in the test set can have a substantial effect on reported accuracy.

---

# Key Findings

Several useful machine learning lessons emerged from the project.

### More Complex Models Are Not Always Better

The neural network was more computationally complex than the Decision Tree and Random Forest models but did not produce better results.

For this dataset, the Random Forest achieved the strongest recorded depression-classification result.

---

### Feature Selection Matters

Removing academic major from one Random Forest experiment improved the recorded depression-classification accuracy.

This demonstrated that irrelevant or noisy features can reduce model performance.

---

### Dataset Quality Matters More Than Algorithm Complexity

Both Decision Tree and Random Forest models struggled more with anxiety classification.

Changing algorithms or adjusting model parameters provided limited improvement.

This suggested that the available features were not sufficiently informative for that particular classification task.

---

### Categorical Data Requires Careful Preprocessing

Student survey datasets contain significant amounts of categorical information.

One-hot encoding provided a straightforward method for converting those categories into numerical features, but it also substantially increased dimensionality.

---

### Small Datasets Limit Generalization

The dataset contains approximately 100 student responses.

With such a small sample, accuracy can vary substantially depending on how the data is divided between training and testing sets.

A larger dataset would be necessary before drawing meaningful conclusions about real-world predictive performance.

---

# Challenges

## Categorical Feature Encoding

The largest technical preprocessing challenge involved converting categorical survey attributes into numerical representations.

Using:

```python
pandas.get_dummies()
```

simplified this process, but categorical variables containing many possible values generated a large number of additional columns.

---

## Dataset Size

The limited number of observations significantly constrained experimentation.

Machine learning models generally benefit from substantially larger datasets, particularly neural networks.

With approximately 100 records, there is significant potential for:

- Overfitting
- High variance
- Unstable evaluation metrics
- Poor generalization

---

## Model Selection

Another major lesson from the project was determining when additional model complexity was useful.

Adding additional neural-network layers did not produce meaningful improvements.

The simpler ensemble model ultimately produced the strongest recorded result.

---

# Repository Structure

```text
MentalHealthMachineLearning/
│
├── MentalHealthClassifiers/
│   └── Classical machine learning experiment materials
│
├── MentalHealthWithTF.ipynb
│   └── TensorFlow neural network implementation and preprocessing
│
└── README.md
```

### `MentalHealthClassifiers`

Contains materials associated with the classical machine learning portion of the project, including the Decision Tree and Random Forest experiments performed using KNIME.

### `MentalHealthWithTF.ipynb`

Jupyter Notebook containing the Python/TensorFlow implementation used to preprocess the dataset, build the neural network, train the model, and evaluate its performance.

---

# Running the TensorFlow Notebook

Clone the repository:

```bash
git clone https://github.com/notfennecks/MentalHealthMachineLearning.git
```

Navigate into the project:

```bash
cd MentalHealthMachineLearning
```

A Python environment containing the primary libraries used by the project will be required.

Example:

```bash
pip install tensorflow pandas scikit-learn jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
MentalHealthWithTF.ipynb
```

> Dependency versions were not pinned in the original academic project, so newer library versions may require small compatibility changes.

---

# Skills Demonstrated

This project demonstrates experience with:

### Machine Learning

- Supervised learning
- Classification
- Decision Trees
- Random Forests
- Neural Networks
- Model comparison
- Model evaluation

### Data Science

- Dataset exploration
- Feature preprocessing
- Feature selection
- Categorical-variable handling
- One-hot encoding
- Train/test splitting
- Experimental analysis

### Python

- Pandas
- TensorFlow
- scikit-learn
- Jupyter Notebook

### Machine Learning Tools

- KNIME Analytics Platform
- TensorFlow

---

# Limitations

This project has several important limitations.

### Small Dataset

The dataset contains approximately 100 student records, which is far too small to establish reliable real-world predictive performance.

### Accuracy as the Primary Metric

The original project primarily evaluated models using accuracy.

For a classification problem involving potentially imbalanced data, future experiments should also consider:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### Limited Validation

The original project used train/test splits rather than more extensive validation strategies.

Future work should evaluate methods such as:

```text
Stratified K-Fold Cross Validation
```

to provide a more reliable estimate of model performance.

### Correlation Does Not Establish Causation

Relationships discovered within the dataset should not be interpreted as causal relationships or medical conclusions.

They represent statistical patterns within a small survey dataset.

---

# Future Improvements

If the project were expanded, several improvements could substantially strengthen it.

## Larger Dataset

Training and validating the models using a larger and more diverse dataset would be the most important improvement.

---

## Cross Validation

Implementing stratified cross-validation would produce more reliable model comparisons than relying on individual train/test splits.

---

## Expanded Evaluation Metrics

Future versions should compare models using:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrices
```

This would provide a more complete understanding of classification performance.

---

## Automated Machine Learning Pipeline

The preprocessing and modeling stages could be moved into a reusable scikit-learn pipeline:

```text
Raw Data
   ↓
Preprocessing
   ↓
Feature Encoding
   ↓
Feature Selection
   ↓
Model Training
   ↓
Evaluation
```

This would improve reproducibility and reduce the possibility of preprocessing inconsistencies.

---

## Hyperparameter Optimization

Models could be systematically optimized using techniques such as:

```text
GridSearchCV
RandomizedSearchCV
```

rather than relying primarily on manual experimentation.

---

## Explainable AI

Feature importance and explainability techniques could help determine which variables contribute most strongly to model predictions.

Potential approaches include:

- Random Forest feature importance
- Permutation importance
- SHAP values

This would be especially important for a project involving human-centered data.

---

## Model Reproducibility

A modernized version of the repository could also include:

```text
requirements.txt
```

or:

```text
pyproject.toml
```

along with fixed random seeds and a repeatable training pipeline.

---

# Ethical Considerations

Machine learning involving mental health data requires particular care.

The models developed in this project should **not** be interpreted as medical diagnostic systems.

Model predictions can be affected by:

- Dataset bias
- Missing variables
- Small sample size
- Demographic imbalance
- Survey design
- Label quality
- Model bias
- Overfitting

A false positive or false negative in a real healthcare environment could have serious consequences.

The purpose of this project is therefore to demonstrate and explore **machine learning classification techniques**, not to replace professional mental health assessment.

---

# Academic Context

This project was originally developed as an academic exploration of machine learning classification techniques.

The accompanying research explored:

1. Decision Tree classification
2. Random Forest classification
3. TensorFlow neural networks
4. Categorical feature preprocessing
5. Model configuration
6. Feature selection
7. Comparative model performance

One of the central conclusions of the project was that **increasing algorithmic complexity does not guarantee improved performance**, particularly when working with a small dataset containing limited predictive information.

---

# Dataset Credit

The dataset used in this project was obtained from Kaggle:

**Student Mental Health**

Created by **Shariful Islam**

https://www.kaggle.com/datasets/shariful07/student-mental-health

Credit for the original dataset belongs to its respective creator.

---

# Author

**Dylan Santiago**

Computer Science / Artificial Intelligence

Portfolio:  
https://dylansantiago.dev

GitHub:  
https://github.com/notfennecks

---

## Disclaimer

This repository is an educational machine learning project.

It is **not a medical device, diagnostic system, clinical screening tool, or substitute for evaluation by a qualified healthcare professional**.

Predictions and statistical relationships produced by these models should not be used to make decisions regarding an individual's mental health or medical treatment.
