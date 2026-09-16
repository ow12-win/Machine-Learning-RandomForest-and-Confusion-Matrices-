import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("openpowerlifting.csv")

# Preprocess the data
# Drop rows with missing values
data = data.dropna()
# Select relevant features and target variable
features = ["Age", "BodyweightKg", "TotalKg"]
target = "Place"

placement_models = RandomForestClassifier(n_estimators=1000, random_state=42)
placement_models.fit(data[features], data[target])


def confusion_matrix(model, X, y):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


print("Confusion Matrix for model:")
confusion_matrix(
    placement_models,
    data[features],
    data[target],
)
