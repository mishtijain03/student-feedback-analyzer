
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load cleaned dataset
df = pd.read_csv("data/processed/feedback_cleaned.csv")

# Set visualization style
sns.set_theme(style="whitegrid")

# 1. Sentiment Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="sentiment",
    order=["positive", "negative", "neutral"]
)

plt.title("Student Feedback Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.show()


# 2. Sentiment Percentage
sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Sentiment Percentage")
plt.show()


# 3. Average Feedback Length
df["feedback_length"] = df["cleaned_comment"].apply(
    lambda text: len(text.split())
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sentiment",
    y="feedback_length",
    order=["positive", "negative", "neutral"]
)

plt.title("Average Feedback Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Average Number of Words")
plt.tight_layout()
plt.show()


# 4. Word Cloud
all_text = " ".join(df["cleaned_comment"].astype(str))

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Student Feedback")
plt.show()


print("EDA completed successfully!")
