# 🩺 Diabetes Prediction System

A machine learning–based web application that predicts whether a patient is diabetic based on clinical parameters.  
The system evaluates multiple machine learning models and **automatically selects the best-performing model** for prediction.

---

## 🚀 Features
- Trained and evaluated multiple ML models
- Automated selection of best model based on accuracy
- Flask-based backend API
- Clean and responsive Bootstrap frontend
- Real-time diabetes prediction
- End-to-end ML pipeline (training → evaluation → deployment)

---

## 🧠 Machine Learning Models Used
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes

The model with the **highest accuracy** is selected automatically at runtime for inference.

---

## 📊 Model Performance Comparison

The bar chart below shows the accuracy comparison of all trained models:

![Model Accuracy Comparison](screenshots/model_accuracy.png)
![ROC AUC CURVE](screenshots/roc_auc_curve.png)

---

## 🖥️ User Interface

### Prediction Result
![Prediction Result](screenshots/ui_result.png)

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Backend:** Flask  
- **Frontend:** HTML, Bootstrap, JavaScript  
- **Machine Learning:** Scikit-learn  
- **Data Handling:** NumPy, Pandas  

---

## Key Design Decision

- Model selection is handled entirely by the backend.
- Users are not required to choose a model, eliminating bias and improving usability.

---
## Future Improvements

- Differentiate between Type 1 and Type 2 diabetes
- Model explainability using feature importance
- Deployment on cloud (Render / AWS / Vercel)

---

## Author

Ananya Chatterjee
