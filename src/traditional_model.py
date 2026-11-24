
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the processed dataset
df = pd.read_csv("data/processed/feedback_cleaned.csv")

# 2. Select input and output
X = df["cleaned_comment"]
y = df["sentiment"]

# 3. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create the TF-IDF and Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 5. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Predict sentiments for test data
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred,
    labels=["negative", "neutral", "positive"]
))

# 8. Save the trained model
Path("models").mkdir(parents=True, exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")

print("\nModel saved successfully!")
