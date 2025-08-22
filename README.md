# Cancer Diagnosis Model: SHAP & LIME Interpretability  
## Project Overview  
This project focuses on building a **machine learning model for cancer diagnosis** and enhancing its **interpretability** using SHAP (SHapley Additive Explanations) and LIME (Local Interpretable Model-Agnostic Explanations).  

The goal is not only to achieve **high accuracy in cancer detection** but also to provide **explainable AI insights**, ensuring that predictions can be trusted and understood by medical professionals.  

## Dataset  
- **Source:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Samples:** 569  
- **Features:** 30 numerical features (e.g., radius, texture, smoothness, concavity, etc.)  
- **Target:** Diagnosis → `M = Malignant`, `B = Benign`
  
## Technologies Used  
- Python (NumPy, Pandas, Matplotlib, Seaborn)  
- Scikit-learn (Logistic Regression, Random Forest, SVM, KNN)  
- SHAP (model interpretability)  
- LIME (local explanations for predictions)  
- Jupyter Notebook / Google Colab  

## Exploratory Data Analysis (EDA)  
- Distribution plots of features (radius, texture, etc.)  
- Correlation heatmap for feature importance  
- Box plots for class separation  
- PCA visualization for dimensionality reduction  

## Machine Learning Models  
| Model                | Accuracy |  
|----------------------|----------|  
| Logistic Regression  | 96.49%   |  
| Support Vector Machine (SVM) | 97.36%   |  
| K-Nearest Neighbors (KNN) | 95.78%   |  
| Random Forest        | 98.24%   |  

Random Forest achieved the **highest accuracy**.  

## Explainability with SHAP & LIME  
- **SHAP:** Global and local explanations of feature importance, showing how features (e.g., radius mean, texture) influence predictions.  
- **LIME:** Case-based explanations to understand why the model classified a particular sample as malignant or benign.  

These tools ensure transparency in the ML model, making it suitable for healthcare applications.  

##  How to Run the Project  

### 1. Clone the Repository  
git clone https://github.com/your-username/cancer-diagnosis-shap-lime.git
cd cancer-diagnosis-shap-lime
### 2. Install Dependencies
pip install -r requirements.txt
### 3. Run the Notebook
Open **Jupyter Notebook / Google Colab** and run the cells to preprocess data, train models, and generate SHAP & LIME explanations.
### 4. Make Predictions
Run the notebook/script and input patient features to predict cancer diagnosis:
* Example Input:
  * Radius Mean: 17.9
  * Texture Mean: 10.4
  * Smoothness Mean: 0.118
* **Output:**
  * Logistic Regression: Malignant
  * SVM: Malignant
  * KNN: Benign
  * Random Forest: Malignant
## Results:
* Random Forest achieved the **highest accuracy (98.24%)**.
* SHAP explained the **most influential features** for classification.
* LIME provided **easy-to-understand explanations** for individual predictions.

## Future Improvements
* Deploy as a **Flask/Django web application** for real-time diagnosis.
* Integrate with medical datasets for broader generalization.
* Combine SHAP & LIME explanations in a unified dashboard.
* Explore deep learning models for enhanced accuracy.

