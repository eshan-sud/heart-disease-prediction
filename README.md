# Heart Disease Risk Prediction using Ensemble Learning

### Dataset:

Dataset Link (Kaggle): <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset">https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset</a>

### Libraries used:

- os
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle
- joblib
- ipywidgets
- IPython.display
- warnings

### Problem Statement:

Heart disease is a major global health concern, making early detection essential for effective treatment. Conventional diagnostic techniques depend on medical tests and expert evaluation, which can be time-intensive and susceptible to human error. This project focuses on creating a machine learning model to predict the likelihood of heart disease based on key medical factors, including age, blood pressure, cholesterol levels, and other health indicators.

The primary challenge lies in the complexity and variability of heart disease symptoms, which differ among individuals. By leveraging machine learning algorithms, we can analyze patterns in large datasets and improve predictive accuracy. The proposed model will utilize supervised learning techniques, trained on historical medical data, to provide reliable risk assessments. This approach can enhance decision-making for healthcare professionals and enable proactive measures to prevent severe complications.

Furthermore, the integration of such predictive models into healthcare systems can reduce the burden on medical professionals by streamlining the diagnostic process. With the rising availability of electronic health records and advancements in computational power, machine learning-based heart disease prediction has the potential to significantly impact public health by facilitating early detection and personalized treatment strategies.

### Solution Approach:

The proposed approach leverages an ensemble learning method using a Voting Classifier that combines multiple base models to improve predictive accuracy and robustness. The key steps include:

- Data Preprocessing: Handling missing values and feature scaling.
- Model Selection: Using an ensemble of Logistic Regression, Random Forest (with and without Hyperparameter tuning), SVM, and XGBoost.
- Soft Voting Classifier: Combining predictions of base models using probability-weighted voting.
- Evaluation & Optimization: Assessing model performance using Accuracy, Precision, Recall, F1-score, and MCC, with hyperparameter tuning for optimal performance.

### Results:

- The highest accruacy of 85% was achieved with the hyperparameter tuned random forest model with a precision of 80%.
- An Ensemble model with an accuracy of 82% and precision of 76% has been achieved.
