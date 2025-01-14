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


## 01. Extra Tree Classifier
### Feature Importance of `Extra Tree Classifier`
<img src="plots/13.png" alt="Alt Text" width="800" height ="280"/>

## 02. MLP Classifier (Neural Networks)
### Feature Importance of `MLP Classifier (Neural Networks)`
<img src="plots/12.png" alt="Alt Text" width="800" height ="280"/>

## 03. Gradient Boosting Classifier
### Feature Importance of `Gradient Boosting Classifier`
<img src="plots/11.png" alt="Alt Text" width="800" height ="280"/>

## 04. Logistic Regression
### Feature Importance of `Logistic Regression`
<img src="plots/10.png" alt="Alt Text" width="800" height ="280"/>

## 05. Adaptive Boosting Classifier
### Feature Importance of `Adaptive Boosting Classifier`
<img src="plots/09.png" alt="Alt Text" width="800" height ="280"/>

## 06. Support Vector Machine
### Feature Importance of `Support Vector Machine`
<img src="plots/08.png" alt="Alt Text" width="800" height ="280"/>

## 07. K-Neighbours Classifier
### Feature Importance of `K-Neighbours Classifier`
<img src="plots/07.png" alt="Alt Text" width="800" height ="280"/>

## 08. Random Forest Classifier
### Feature Importance of `Random Forest Classifier`
<img src="plots/06.png" alt="Alt Text" width="800" height ="280"/>

## 09. Decision Tree Classifier
### Feature Importance of `Decision Tree Classifier`
<img src="plots/05.png" alt="Alt Text" width="800" height ="280"/>

## 10. CatBoost Classifier
### Feature Importance of `CatBoost Classifierr`
<img src="plots/04.png" alt="Alt Text" width="800" height ="280"/>

## 11. LightGBM Classifier
### Feature Importance of `LightGBM Classifier`
<img src="plots/03.png" alt="Alt Text" width="800" height ="280"/>


## 12. XGBoost Classifier
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







