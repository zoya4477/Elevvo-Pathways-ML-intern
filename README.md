# Elevvo-Pathways-ML-inten

This repository contains my **Machine Learning internship tasks**, covering **Regression, Classification, Clustering, Recommendation Systems, Time Series Forecasting, Audio Processing, and Computer Vision**.

Each task includes **data preprocessing, visualization, model building, and performance evaluation**.

---

## ✅ Task Details

### 1️⃣ Student Score Prediction
**Objective:** Predict students' exam scores based on study hours and other factors.  
**Approach:**
- Cleaned dataset and handled missing values.
- Performed exploratory data analysis (EDA) to understand score patterns.
- Trained a **Linear Regression** model (also tested Polynomial Regression).
- Evaluated performance using RMSE and R² score.  
**Tools:** Python, Pandas, Matplotlib, Scikit-learn.

---

### 2️⃣ Customer Segmentation
**Objective:** Group mall customers based on their annual income and spending score.  
**Approach:**
- Scaled features and visualized customer distribution.
- Applied **K-Means Clustering** and determined optimal clusters using the **Elbow Method**.
- Visualized clusters in a 2D scatter plot.  
**Tools:** Python, Pandas, Matplotlib, Scikit-learn.

---

### 3️⃣ Forest Cover Type Classification
**Objective:** Predict the type of forest cover using cartographic and environmental features.  
**Approach:**
- Cleaned and preprocessed data, handled categorical features.
- Trained **Random Forest** and **XGBoost** multi-class models.
- Compared accuracy and plotted feature importance.  
**Tools:** Python, Pandas, Scikit-learn, XGBoost.

---

### 4️⃣ Loan Approval Prediction
**Objective:** Predict whether a loan application will be approved.  
**Approach:**
- Handled missing values and encoded categorical variables.
- Trained **Logistic Regression** and **Decision Tree** models.
- Evaluated with **Precision, Recall, and F1-score** (due to imbalanced data).  
**Tools:** Python, Pandas, Scikit-learn.

---

### 5️⃣ Movie Recommendation System
**Objective:** Recommend movies to users based on their preferences.  
**Approach:**
- Built a **user-item rating matrix** from the dataset.
- Calculated similarity between users using cosine similarity.
- Recommended top-rated unseen movies for a given user.  
**Tools:** Python, Pandas, NumPy, Scikit-learn.

---

### 6️⃣ Music Genre Classification
**Objective:** Classify songs into genres using audio data.  
**Approach:**
- Extracted **MFCC** features from audio files using Librosa.
- Built two models:
  1. **Tabular data model** using Scikit-learn.
  2. **Image-based CNN** using spectrograms.
- Compared both approaches and reported accuracy.  
**Tools:** Python, Librosa, Scikit-learn, Keras.

---

### 7️⃣ Sales Forecasting
**Objective:** Predict future sales using historical data.  
**Approach:**
- Created time-based features (month, day, lag variables).
- Applied **Linear Regression** and **XGBoost** models.
- Visualized actual vs. predicted sales over time.  
**Tools:** Python, Pandas, Matplotlib, Scikit-learn.

---

### 8️⃣ Traffic Sign Recognition
**Objective:** Recognize traffic signs from images using deep learning.  
**Approach:**
- Preprocessed images (resizing, normalization).
- Trained a **CNN model** from scratch and compared with **MobileNet** (Transfer Learning).
- Used data augmentation to improve performance.
- Evaluated with accuracy score and confusion matrix.  
**Tools:** Python, TensorFlow/Keras, OpenCV.

---


