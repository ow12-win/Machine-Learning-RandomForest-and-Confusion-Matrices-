# Powerlifting Placement Prediction Using Random Forest

## Overview

This project uses a **Random Forest Classifier** to predict a powerlifter's competition placing using data from the OpenPowerlifting dataset.

The model predicts an athlete's placing based on three features:

- Age
- Bodyweight (kg)
- Total Weight Lifted (kg)

During development, several versions of the model were tested to compare their performance. These included training on the combined dataset, separating competitors by gender, and experimenting with different numbers of trees (`n_estimators`) in the Random Forest.

---

## Dataset

The dataset used is:

- `openpowerlifting.csv`

The data is preprocessed by:

- Removing rows with missing values.
- Selecting the required features.
- Preparing the data for machine learning.

---

## Features

### Input Features

- Age
- BodyweightKg
- TotalKg

### Target Variable

- Place

---

## Machine Learning Model

The project uses Scikit-learn's **Random Forest Classifier**.

### Model Parameters

- Algorithm: Random Forest
- Number of Trees (`n_estimators`): **10,000**
- Random State: **42**

Separate models are trained for male and female competitors.

---

## Libraries Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Installation

Install the required libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

## Running the Project

Run the program with:

```bash
python main.py
```

The program will:

1. Load the OpenPowerlifting dataset.
2. Remove rows containing missing values.
3. Prepare the selected features.
4. Train Random Forest models.
5. Generate predictions.
6. Display confusion matrices for each model.

---

## Results

Several versions of the model were developed and compared.

### Version 1 – Combined Male and Female Dataset

The first version trained a single model using both male and female competitors together.

Confusion Matrix:

```
77  0  0
1   5  0
0   0  1
```

This model achieved very high prediction accuracy on the dataset.

---

### Version 2 – Gender-Separated Model (`n_estimators = 1`)

The dataset was then separated into male and female competitors, with a separate model trained for each gender.

Using only one decision tree produced noticeably poorer results.

Confusion Matrix:

```
39  1  0
1   1  0
1   0  0
```

---

### Version 3 – Gender-Separated Model (`n_estimators = 100`)

Increasing the number of trees to 100 significantly improved the model's predictions.

Confusion Matrix:

```
40  0  0
0   2  0
0   0  1
```

---

### Version 4 – Gender-Separated Model (`n_estimators = 10000`)

The final version increased the number of trees to 10,000.

The confusion matrix remained almost identical to the 100-tree model, showing that increasing the forest size further did not provide any significant improvement.

Example confusion matrix:

```
37  0
1   3
```

This suggests that increasing the number of trees beyond 100 had very little impact on prediction accuracy while increasing training time.

---

## Project Structure

```
.
├── main.py
├── before_.py
├── openpowerlifting.csv
├── README.md
```

---

## Future Improvements

Possible improvements include:

- Correctly implementing separate training and testing datasets for each gender-specific model.
- Performing cross-validation to better evaluate model performance.
- Hyperparameter tuning using GridSearchCV.
- Adding more input features such as:
  - Weight Class
  - Equipment
  - Federation
  - Best Squat
  - Best Bench Press
  - Best Deadlift
- Comparing Random Forest against other machine learning algorithms such as Decision Trees, Support Vector Machines (SVM), K-Nearest Neighbours (KNN), or XGBoost.

---

## Conclusion

This project demonstrates how machine learning can be used to predict powerlifting competition placings using athlete data from the OpenPowerlifting dataset.

Different versions of the Random Forest model were explored throughout development. The experiments showed that:

- Training a single model on the combined dataset produced strong results.
- Separating competitors by gender allowed independent models to be trained.
- Using only one decision tree resulted in noticeably poorer performance.
- Increasing the number of trees from 100 to 10,000 produced almost identical results, indicating that a forest of 100 trees was already sufficient for this dataset.

Overall, the project provided practical experience with data preprocessing, supervised machine learning, Random Forest classifiers, and model evaluation using confusion matrices.
