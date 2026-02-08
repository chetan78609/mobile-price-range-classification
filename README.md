# 📱 Mobile Price Range Classification using Machine Learning

## 1. Problem Statement
The objective of this project is to predict the **price range of a mobile phone** based on its technical specifications such as battery power, RAM, display resolution, camera features, and connectivity options.

This is a **multi-class classification problem**, where the target variable `price_range` represents four price categories:
- **0** – Low cost  
- **1** – Medium cost  
- **2** – High cost  
- **3** – Very high cost  

---

## 2. Dataset Description
- **Dataset Name:** Mobile Price Classification Dataset  
- **Source:** Kaggle  
- **Number of Instances:** 1599
- **Number of Features:** 20 independent variables  
- **Target Variable:** `price_range`

The dataset is **balanced across all four classes**, ensuring unbiased evaluation.

### Feature Categories
- **Hardware:** `battery_power`, `ram`, `int_memory`, `n_cores`
- **Display:** `px_height`, `px_width`, `sc_h`, `sc_w`
- **Connectivity:** `wifi`, `blue`, `three_g`, `four_g`, `dual_sim`
- **Camera & Others:** `fc`, `pc`, `mobile_wt`, `talk_time`

---

## 3. Data Preprocessing
- The dataset was split into **training and test sets** using an **80–20 stratified split**.
- **Feature scaling** was performed using `StandardScaler`.
- All preprocessing and model training steps were implemented using **Scikit-learn Pipelines** to ensure consistency between training and inference.
- Scaling was applied to all features (continuous, discrete, and binary) to support distance- and gradient-based models.

---

## 4. Models Implemented
The following machine learning models were trained and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest  
6. XGBoost  

All models were trained using the same preprocessing pipeline and evaluated on the same test dataset for fair comparison.

---

## 5. Evaluation Metrics
Model performance was evaluated on the **held-out test dataset** using the following metrics:

- Accuracy  
- Precision (Weighted)  
- Recall (Weighted)  
- F1 Score (Weighted)  
- Matthews Correlation Coefficient (MCC)  
- AUC Score (One-vs-Rest)  
- Confusion Matrix  

---

## 6. Results Summary (Test Dataset Performance)

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUC |
|------|----------|-----------|--------|----------|-----|-----|
| **Logistic Regression** | **0.955** | **0.956** | **0.955** | **0.955** | **0.940** | **0.997** |
| XGBoost | 0.890 | 0.891 | 0.890 | 0.890 | 0.854 | 0.984 |
| Random Forest | 0.878 | 0.878 | 0.878 | 0.878 | 0.837 | 0.980 |
| Decision Tree | 0.810 | 0.814 | 0.810 | 0.811 | 0.748 | 0.873 |
| Naive Bayes | 0.781 | 0.791 | 0.781 | 0.784 | 0.708 | 0.942 |
| KNN | 0.459 | 0.480 | 0.459 | 0.461 | 0.281 | 0.728 |

---

## 7. Observations
- **Logistic Regression** achieved the best overall performance, indicating strong linear separability among the features after scaling.
- **XGBoost** and **Random Forest** performed strongly, demonstrating the effectiveness of ensemble methods in capturing non-linear feature interactions.
- **Decision Tree** showed reasonable performance but was more prone to overfitting compared to ensemble models.
- **Naive Bayes** was limited by its assumption of feature independence.
- **KNN** performed the weakest due to the high-dimensional and mixed-type feature space, where distance-based similarity becomes less informative.

---

## 8. Streamlit Application
A **Streamlit web application** was developed to demonstrate the trained models interactively.

### Application Features
- Model selection via dropdown
- Downloadable **sample test dataset (`test.csv`)**
- CSV upload for prediction
- Automatic detection of labeled data for evaluation
- Display of evaluation metrics and confusion matrix
- Real-time prediction on uploaded datasets

### Evaluation Strategy
- Evaluation metrics and confusion matrix are computed when the uploaded dataset contains the target variable `price_range`.
- When `price_range` is absent, the application performs prediction-only inference.

---

## 9. Repository Structure
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── train.csv
│   └── test.csv
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl


## 10. How to Run the Project

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>


## 11. Deployment

The application is deployed on **Streamlit Community Cloud**, allowing users to access the trained models directly from the browser without any local setup.

🔗 **Live App:**  
https://mobile-price-range-classification-fbek4rgepyckgz3ysmt8vb.streamlit.app/

### Deployment Highlights
- Interactive web interface  
- No installation required  
- Upload your own CSV files for prediction  
- Automatic evaluation when `price_range` column is present  
- Download sample **test.csv** from the application  

This deployment makes the project easy to test, demonstrate, and share.

---

## 12. Conclusion

This project demonstrates a complete **end-to-end machine learning workflow**, covering:

- Data preprocessing  
- Feature scaling  
- Model training  
- Performance comparison  
- Deployment to the cloud  

### Key Insights
- **Logistic Regression** achieved the highest accuracy, indicating strong linear separability after scaling.
- **Random Forest** and **XGBoost** effectively captured non-linear relationships and delivered competitive performance.
- **Decision Tree** showed moderate results but was more prone to overfitting.
- **Naive Bayes** was constrained by independence assumptions.
- **KNN** underperformed in the high-dimensional feature space where distance measures become less meaningful.

Using **Scikit-learn Pipelines** ensured consistent transformations during both training and inference while preventing data leakage.

The Streamlit application enables real-time experimentation and practical usability of the trained models.

---

## 13. Author

**Chetan Mazumder**  
*M.Tech – Data Science & Engineering*
