
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv(
    'Social_media_impact_on_life.csv',
    encoding='latin-1'
)

print("Original Dataset:")
print(df.head(10))

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


df['Perceived_Stress_Score'] = df['Perceived_Stress_Score'].fillna(
    df['Perceived_Stress_Score'].mean()
)

df['Academic_Performance_GPA'] = df['Academic_Performance_GPA'].fillna(
    df['Academic_Performance_GPA'].mean()
)


df = df.drop('Student_ID', axis=1)

le = LabelEncoder()

categorical_columns = [
    'Gender',
    'Academic_Level',
    'Primary_Platform',
    'Device_Type',
    'Social_Comparison_Frequency'
]

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

target_encoder = LabelEncoder()

df['Overall_Impact'] = target_encoder.fit_transform(
    df['Overall_Impact']
)

print("\nTarget Classes:")

for number, name in enumerate(target_encoder.classes_):
    print(number, "=", name)

print("\nEncoded Dataset:")
print(df.head(10))

print("\nOverall Impact Distribution:")
print(df['Overall_Impact'].value_counts())


X = df.drop('Overall_Impact', axis=1)

y = df['Overall_Impact']


print("\nX shape:", X.shape)
print("y shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


logistic = LogisticRegression(
    max_iter=1000
)

decision_tree = DecisionTreeClassifier(
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

knn = KNeighborsClassifier(
    n_neighbors=5
)

models = {
    "Logistic Regression": logistic,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "KNN": knn
}

results = []


for name, model in models.items():

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Precision
    precision = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # F1 Score
    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    # Print metrics
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_encoder.classes_,
            zero_division=0
        )
    )

    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

results_df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df)

for name, model in models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_encoder.classes_,
        yticklabels=target_encoder.classes_
    )

    plt.title(name + " - Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()


results_df.set_index('Model')[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score']
].plot(
    kind='bar',
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend()

plt.tight_layout()

plt.show()