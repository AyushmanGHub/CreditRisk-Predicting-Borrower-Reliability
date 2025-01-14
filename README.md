# CreditRisk: Predicting Borrower Reliability
🚀 **Explore the repository for code, analysis, and results!**
---

## 🔍 Introduction  

Credit card default is a critical issue in the financial sector, where banks and financial institutions need to assess the risk associated with lending. This project aims to develop a **machine learning model** that predicts whether a customer will default on their credit card payment.  

Using the **AmExpert CodeLab 2021 dataset**, we analyze various demographic, financial, and credit-related features to identify patterns in customer behavior. The goal is to **enhance risk assessment strategies**, helping financial institutions make informed lending decisions and reduce financial losses.  

In this project, we explore different machine learning techniques, including **ensemble models, deep learning, and feature engineering**, to improve predictive accuracy.  

## 📌 About the Dataset  
📂 **Dataset Link**: [Kaggle - AmExpert CodeLab 2021](https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021)  

The dataset consists of various features related to customer demographics, financial status, and credit behavior: `customer_id`, `age`,`gender`, `owns_car`, `owns_house`, `num_children`, `net_yearly_income`, `num_days_employed`, `occupation_type`, `total_family_members`, `migrant_worker`, `yearly_debt_payments`, `credit_limit`, `credit_limit_used (%)`, `credit_score`, `previous_defaults`, `default_in_last_6_months`, `credit_card_default 🔴 Target Variable`

### Distribution of `1 Will Default` & `0 Will not Default`
<img src="plots/01.png" alt="Alt Text" width="450" height ="300"/>


## 🌲 Extra Tree Classifier
The **Extra Trees Classifier (Extremely Randomized Trees)** is an ensemble learning method that builds multiple decision trees and aggregates their results to improve accuracy and reduce overfitting. Unlike Random Forest, it selects split points **completely at random**, making it more computationally efficient while maintaining strong predictive performance.  
### Feature Importance of `Extra Tree Classifier`
<img src="plots/13.png" alt="Alt Text" width="800" height ="280"/>


## 🔗 MLP Classifier (Neural Networks)  
The **MLP Classifier (Multi-Layer Perceptron)** is a type of **artificial neural network** that learns complex patterns from data through multiple layers of interconnected neurons. It uses **backpropagation** and **gradient descent** to adjust weights and minimize errors.  
### Feature Importance of `MLP Classifier (Neural Networks)`
<img src="plots/12.png" alt="Alt Text" width="800" height ="280"/>


## 🚀 Gradient Boosting Classifier  
The **Gradient Boosting Classifier** is an **ensemble learning algorithm** that builds a series of weak learners (typically decision trees) and combines them to create a **strong predictive model**. It minimizes errors by **sequentially correcting previous mistakes** using gradient descent.  
### Feature Importance of `Gradient Boosting Classifier`
<img src="plots/11.png" alt="Alt Text" width="800" height ="280"/>


## 🔹 Logistic Regression  
**Logistic Regression** is a **supervised learning algorithm** used for **binary classification** problems. It predicts the probability that a given input belongs to a particular class using the **logistic (sigmoid) function**.  
### Feature Importance of `Logistic Regression`
<img src="plots/10.png" alt="Alt Text" width="800" height ="280"/>


## 🚀 Adaptive Boosting (AdaBoost) Classifier  
The **Adaptive Boosting (AdaBoost) Classifier** is an **ensemble learning method** that combines multiple weak learners (typically decision trees) to create a **strong classifier**. It works by **iteratively adjusting the weights** of misclassified instances, making the next model focus more on difficult cases.  
### Feature Importance of `Adaptive Boosting Classifier`
<img src="plots/09.png" alt="Alt Text" width="800" height ="280"/>


## 🚀 Support Vector Machine (SVM)  
The **Support Vector Machine (SVM)** is a **powerful supervised learning algorithm** used for both **classification and regression tasks**. It works by finding the **optimal hyperplane** that best separates different classes in the feature space.  
### Feature Importance of `Support Vector Machine`
<img src="plots/08.png" alt="Alt Text" width="800" height ="280"/>


## 🚀 K-Nearest Neighbors (K-NN) Classifier  
The **K-Nearest Neighbors (K-NN)** classifier is a simple, yet powerful **instance-based learning algorithm** used for **classification and regression** tasks. It works by classifying a data point based on the majority class of its **K nearest neighbors** in the feature space.  
### Feature Importance of `K-Neighbours Classifier`
<img src="plots/07.png" alt="Alt Text" width="800" height ="280"/>

## 🌲 Random Forest Classifier 
The **Random Forest Classifier** is an ensemble learning algorithm that combines the predictions of multiple decision trees to improve classification accuracy. It builds multiple decision trees during training and merges them together to obtain a more accurate and stable prediction.
### Feature Importance of `Random Forest Classifier`
<img src="plots/06.png" alt="Alt Text" width="800" height ="280"/>

## 🌱 Decision Tree Classifier
The **Decision Tree Classifier** is a popular supervised machine learning algorithm used for classification tasks. It builds a tree-like model by recursively splitting the dataset based on feature values. Each internal node of the tree represents a decision based on a feature, and each leaf node corresponds to a class label. The model selects the feature that best separates the data at each step using criteria such as **Gini Impurity** or **Information Gain**.
### Feature Importance of `Decision Tree Classifier`
<img src="plots/05.png" alt="Alt Text" width="800" height ="280"/>


## 🐱 CatBoost Classifier
The **CatBoost Classifier** is an advanced gradient boosting algorithm developed by Yandex, designed for handling categorical data efficiently. It is based on **Gradient Boosting** techniques and is known for its speed, accuracy, and ability to handle categorical features without the need for manual encoding. The model builds an ensemble of decision trees, optimizing them to minimize loss and improve prediction accuracy iteratively.
### Feature Importance of `CatBoost Classifierr`
<img src="plots/04.png" alt="Alt Text" width="800" height ="280"/>

## 🔥 LightGBM Classifier

**LightGBM (Light Gradient Boosting Machine)** is a highly efficient, scalable gradient boosting framework that is designed for fast training with large datasets. It uses a histogram-based approach and supports **parallelization** and **GPU acceleration**. LightGBM is capable of handling both **categorical features** and large-scale data with minimal memory usage. It is widely used for classification and regression tasks due to its high performance and accuracy, especially in competitive machine learning environments.
### Feature Importance of `LightGBM Classifier`
<img src="plots/03.png" alt="Alt Text" width="800" height ="280"/>


## 🌟 XGBoost Classifier
## XGBoost Classifier
**XGBoost (Extreme Gradient Boosting)** is an optimized gradient boosting algorithm designed for speed and performance. It is widely known for its **accuracy** and **scalability**, particularly in classification and regression tasks. XGBoost employs advanced regularization techniques, such as **L1** and **L2** regularization, to prevent overfitting and improve model generalization. It supports **parallel processing**, **GPU acceleration**, and is capable of handling large datasets efficiently. XGBoost is often the go-to algorithm for data science competitions due to its outstanding performance and ability to handle diverse types of data.
### Feature Importance of `XGBoost Classifier`
<img src="plots/02.png" alt="Alt Text" width="800" height ="280"/>


## 13. Majority Classifier

### Figure 05. Significance of features for (a) Random Forest Model, (b) XGBoost Model
<img src="plots/06.png" alt="Alt Text" width="800" height ="250"/>


## Classification and Evaluation

## Model Performance Metrics
`1 Will Default` & `0 Will not Default`

|                Model                   | Accuracy | Precision(1) | Recall(1) | Precision(0) | Recall(0) |
|----------------------------------------|----------|--------------|-----------|--------------|-----------|
| `Extra Tree Classifier`                | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |
| `MLP Classifier (Neural Networks)`     | 0.973569 |    0.86      |   0.82    |     0.98     |   0.99    |
| `Gradient Boosting Classifier`         | 0.975983 |    0.92      |   0.79    |     0.98     |   0.99    |
| `Logistic Regression`                  | 0.978970 |    0.97      |   0.78    |     0.98     |   1.00    |
| `Adaptive Boosting Classifier`         | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |
| `Support Vector Machine`               | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |
| `K-Neighbours Classifier`              | 0.974259 |    0.96      |   0.73    |     0.98     |   1.00    |
| `Random Forest Classifier`             | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |
| `Decision Tree Classifier `            | 0.978626 |    0.99      |   0.75    |     0.98     |   1.00    |
| `CatBoost Classifier`                  | 0.977936 |    0.94      |   0.79    |     0.98     |   1.00    |
| `LightGBM Classifier`                  | 0.977706 |    0.94      |   0.79    |     0.98     |   1.00    |
| `XGBoost Classifier`                   | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |
|  -------                               |          |              |           |              |           |
| `Majority Classifier`                  | 0.978970 |    1.00      |   0.75    |     0.98     |   1.00    |


## Result and Conclusion



# ----------------------------------------------------------------
### * If want to know more about this project there are python Notebook file, Project report paper and all other resources included in same repository.
### * Feel free to reach out, I'm open to engaging in meaningful conversations and exchanging ideas on these areas. I welcome the chance to explore new insights, collaborate on projects, and contribute to ongoing discussions in these fields.







