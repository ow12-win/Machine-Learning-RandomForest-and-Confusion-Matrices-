import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("openpowerlifting.csv")

# Preprocess the data
# Drop rows with missing values
data = data.dropna()
# Select relevant features and target variable
features = ["Age", "BodyweightKg", "TotalKg"]
target = "Place"
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)
placement_models = {}

for sex in data["Sex"].unique():
    sex_data = data[data["Sex"] == sex]
    placement_model = RandomForestClassifier(n_estimators=10000, random_state=42)
    placement_model.fit(sex_data[features], sex_data[target])
    placement_models[sex] = placement_model

    print(f"{sex} model trained successfully!")
    print(placement_model.predict(sex_data[features]))


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


for sex, placement_model in placement_models.items():
    sex_data = data[data["Sex"] == sex]

    print(f"Confusion Matrix for {sex} Model:")
    confusion_matrix(
        placement_model,
        sex_data[features],
        sex_data[target],
    )
