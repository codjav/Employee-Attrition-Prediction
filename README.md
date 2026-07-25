# Employee Attrition Prediction using Decision Tree and Random Forest

## Objective

Predict employee attrition using Decision Tree and Random Forest classifiers and compare their performance.

---

## Dataset

https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## Libraries Used

- Python
- Pandas
- Matplotlib
- Scikit-Learn

---

## Methodology

1. Load dataset
2. Explore data
3. Remove unnecessary columns
4. Encode categorical variables
5. Split data into training and testing sets
6. Train Decision Tree classifier
7. Train Random Forest classifier (100 estimators)
8. Evaluate using Accuracy, Precision, Recall and F1 Score
9. Generate confusion matrices
10. Plot feature importance for Random Forest

---

## Results

### Decision Tree

- Accuracy: **75.85%**
- Precision: **29.31%**
- Recall: **36.17%**
- F1 Score: **32.38%**

### Random Forest

- Accuracy: **83.33%**
- Precision: **42.86%**
- Recall: **12.77%**
- F1 Score: **19.67%**

---

## Model Comparison

- Random Forest achieved higher overall accuracy.
- Decision Tree identified more attrition cases (higher recall).
- Random Forest produced more stable predictions by combining multiple trees.
- Feature importance provides insights into influential employee attributes.

---

## Conclusion

This project compared Decision Tree and Random Forest classifiers for employee attrition prediction. The Random Forest model achieved higher overall accuracy (**83.33%**) due to its ensemble learning approach, which reduces overfitting and improves generalization. Decision Trees are simple to interpret but often overfit the training data, resulting in unstable predictions. Random Forest overcomes this limitation by averaging predictions from many trees, leading to better robustness. However, Random Forest models are more computationally expensive and less interpretable than a single Decision Tree. Feature importance analysis also helps identify the factors that contribute most to employee attrition, making the model valuable for HR analytics and decision-making.
