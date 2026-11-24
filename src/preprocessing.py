
import pandas as pd
import re
from pathlib import Path

# 1. Load the original dataset
df = pd.read_csv(
    "data/raw/feedback_student.csv",
    encoding="cp1252"
)

print("Original dataset shape:", df.shape)

# 2. Remove duplicate records
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

# 3. Remove missing feedback and quality labels
df = df.dropna(subset=["comment", "quality"])

# 4. Convert quality labels into sentiment labels
sentiment_mapping = {
    "awesome": "positive",
    "good": "positive",
    "average": "neutral",
    "poor": "negative",
    "awful": "negative"
}

df["sentiment"] = df["quality"].str.lower().map(
    sentiment_mapping
)

# 5. Clean the feedback text
def clean_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_comment"] = df["comment"].apply(clean_text)

# 6. Remove empty cleaned comments
df = df[df["cleaned_comment"] != ""]

# 7. Create processed folder if it does not exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

# 8. Save the cleaned dataset
df.to_csv(
    "data/processed/feedback_cleaned.csv",
    index=False
)

# 9. Display results
print("\nDATA PREPROCESSING COMPLETED!")

print("\nFinal dataset shape:", df.shape)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nSample processed records:")
print(df[["comment", "cleaned_comment", "sentiment"]].head())

print("\nCleaned dataset saved successfully!")
