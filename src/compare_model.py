
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load processed dataset
df = pd.read_csv("data/processed/feedback_cleaned.csv")

X = df["cleaned_comment"]
y = df["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define two traditional ML models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "Naive Bayes": MultinomialNB(alpha=0.5)
}

# Train and compare
for name, classifier in models.items():

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("Macro F1-score:", f1_score(
        y_test,
        predictions,
        average="macro"
    ))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
