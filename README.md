🚀 **Explore my repository for code, analysis, and results!**
---
# CreditRisk: Predicting Borrower Reliability
---

## 🔍 Introduction  

Credit card default is a critical issue in the financial sector, where banks and financial institutions need to assess the risk associated with lending. This project aims to develop a **machine learning model** that predicts whether a customer will default on their credit card payment.  

Using the **AmExpert CodeLab 2021 dataset**, I analyzed various demographic, financial, and credit-related features to identify patterns in customer behavior. The goal is to **enhance risk assessment strategies**, helping financial institutions make informed lending decisions and reduce financial losses.  

In this project, I used different machine learning techniques, including **ensemble models, deep learning, and feature engineering**, to improve predictive accuracy and show feature dependency of each model.  






## 🛠 Steps in Project  

### 1. **Data Preparation**  
   - **Data Loading**: Imported the dataset and explored its structure to understand the features and target variables.  
   - **Data Cleaning**:  Handled missing values and inconsistencies,  Removed unnecessary columns like `customer_id` and `name`.  
   - **Feature Selection**: Retained relevant features for model training and removed redundant data.  

### 2. **Feature Scaling** : Applied `StandardScaler` to normalize the feature values for models sensitive to scale, such as SVM and Neural Networks.  

### 3. **Data Balancing** :  Addressed class imbalance in the `credit_card_default` column using the `RandomUnderSampler` technique to achieve a 50-50 distribution of classes (1 and 0).  

### 4. **Data Splitting** :  Split the dataset into training and testing sets using an 80-20 split ratio to evaluate model performance effectively.  

### 5. **Model Training and Evaluation**  
   - **Models Used**: Extra Tree Classifier, MLP Classifier (Neural Networks), Gradient Boosting Classifier, Logistic Regression, Adaptive Boosting Classifier, Support Vector Machine, K-Nearest Neighbors, Random Forest Classifier, Decision Tree Classifier, CatBoost Classifier, LightGBM Classifier, XGBoost Classifier, Majority Classifier (Ensemble approach)
   - **Metrics Evaluated**: Accuracy, Precision(1), Precision(0), Recall (1), Recall (0)
   - `1 Will Default` & `0 Will not Default`

### 6. **Feature Importance Analysis**: Analyzed the feature significance for models capable of providing feature importance scores (e.g., Extra Trees, Gradient Boosting, Random Forest, CatBoost, LightGBM, XGBoost).  

### 7. **Metric Visualization**:  
   - Plotted various metrics (Accuracy, Precision, Recall) to compare the models’ performance.  
   - Created visualizations to represent the importance of features across different models.  


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
**XGBoost (Extreme Gradient Boosting)** is an optimized gradient boosting algorithm designed for speed and performance. It is widely known for its **accuracy** and **scalability**, particularly in classification and regression tasks. XGBoost employs advanced regularization techniques, such as **L1** and **L2** regularization, to prevent overfitting and improve model generalization. It supports **parallel processing**, **GPU acceleration**, and is capable of handling large datasets efficiently. XGBoost is often the go-to algorithm for data science competitions due to its outstanding performance and ability to handle diverse types of data.
### Feature Importance of `XGBoost Classifier`
<img src="plots/02.png" alt="Alt Text" width="800" height ="280"/>

## Majority Plot

### Table for Weight of features for each model
<img src="plots/15.png" alt="Alt Text" width="800" height ="350"/>

### Plot for Significance of features for each model
<img src="plots/14.png" alt="Alt Text" width="800" height ="350"/>


## Classification and Evaluation

## Model Performance Metrics
`1 Will Default` & `0 Will not Default`

|                Model                   | Accuracy | Precision(1) | Recall(1) | Precision(0) | Recall(0) |
|----------------------------------------|----------|--------------|-----------|--------------|-----------|
| `Extra Tree Classifier`                | 0.9708   | 0.7990       | 0.8769    | 0.9885       | 0.9795    |
| `MLP Classifier (Neural Networks)`     | 0.9712   | 0.8120       | 0.8593    | 0.9869       | 0.9815    |
| `Gradient Boosting Classifier`         | 0.9744   | 0.8707       | 0.8200    | 0.9834       | 0.9887    |
| `Logistic Regression`                  | 0.9635   | 0.7223       | 0.9256    | 0.9929       | 0.9670    |
| `Adaptive Boosting Classifier`         | 0.9577   | 0.6732       | 0.9756    | 0.9976       | 0.9560    |
| `Support Vector Machine`               | 0.9613   | 0.7068       | 0.9296    | 0.9933       | 0.9642    |
| `K-Neighbours Classifier`              | 0.9515   | 0.6636       | 0.8701    | 0.9876       | 0.9591    |
| `Random Forest Classifier`             | 0.9683   | 0.7592       | 0.9175    | 0.9922       | 0.9730    |
| `Decision Tree Classifier`             | 0.9661   | 0.7367       | 0.9350    | 0.9938       | 0.9690    |
| `CatBoost Classifier`                  | 0.9745   | 0.8585       | 0.8376    | 0.9850       | 0.9872    |
| `LightGBM Classifier`                  | 0.9747   | 0.8629       | 0.8349    | 0.9847       | 0.9877    |
| `XGBoost Classifier`                   | 0.9591   | 0.6840       | 0.9635    | 0.9965       | 0.9587    |
| -------------------------------------- |          |              |           |              |           |
| `Majority Classifier`                  | 0.9707   | 0.7902       | 0.8917    | 0.9898       | 0.9780    |

### 📌 Metric Definitions:
- **Accuracy**: Measures the overall correctness of the model’s predictions.
- **Precision (1)**: The proportion of correctly predicted default

### Plot of accuracy for each model
<img src="plots/18.png" alt="Alt Text" width="800" height ="300"/>

### Plot of accuracy for each model
<img src="plots/20.png" alt="Alt Text" width="800" height ="400"/>

### Evaluation metices of majority plot
<img src="plots/19.png" alt="Alt Text" width="800" height ="300"/>

## 📊 **Key Evaluation Metrics and Their Importance**  
- For Above project we need to consider **Recall (1) as most important metric** because- Identifying actual defaulters is critical to minimize financial risk. A high recall ensures fewer risky customers are overlooked, preventing losses and missing true defaulters (false negatives) can lead to serious financial consequences.  
- **Precision (0) & Recall (0) as next important**  since,It ensures non-defaulters are correctly identified, preventing rejection of good customers.It is mportant for maintaining customer satisfaction and business profitability.  
- **Accuracy** it is needed because it provides an overall measure of model performance and can be useful for comparing models, but should be interpreted cautiously due to class imbalance. High accuracy might still mislead if the minority class (defaulters) isn't well predicted.  
 


## **📊 Results Overview**  
### sample result for each model
<img src="plots/16.png" alt="Alt Text" width="800" height ="350"/>
<img src="plots/17.png" alt="Alt Text" width="800" height ="350"/>

The performance of various machine learning models was evaluated using key metrics such as **Accuracy, Precision (1), Recall (1), Precision (0), and Recall (0)**. Among all models, **Gradient Boosting Classifier, LightGBM, and CatBoost** showed the best trade-off between high recall and precision for defaulters, making them strong candidates for credit risk prediction.  

Key observations:  
- **Gradient Boosting Classifier** achieved the highest overall accuracy (**0.974374**) with a strong balance in Recall (1) and Precision (1).  
- **LightGBM and CatBoost** performed similarly, ensuring minimal false negatives and strong overall prediction capability.  
- **Logistic Regression and Adaptive Boosting** had **high recall but lower precision**, meaning they flagged more defaulters but also included some misclassifications.  
- **Majority Classifier** performed well in accuracy but lacked the adaptability of other models.  

## **✅ Conclusion**  
This project successfully built and compared multiple machine learning models for **credit default prediction**. Given the financial implications, **Recall (1) was prioritized** to ensure actual defaulters are correctly identified. The best-performing models (**Gradient Boosting, LightGBM, and CatBoost**) offer a **data-driven solution** to minimize financial risk while maintaining operational efficiency.  




# ----------------------------------------------------------------
### * If want to know more about this project there are python Notebook file, Project report paper and all other resources included in same repository.
### * 😊 Feel free to reach out, I'm open to engaging in meaningful conversations and exchanging ideas on these areas. I welcome the chance to explore new insights, collaborate on projects, and contribute to ongoing discussions in these fields.







