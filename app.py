
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load trained model and processed dataset
model = joblib.load("models/sentiment_model.pkl")
df = pd.read_csv("data/processed/feedback_cleaned.csv")
# Store prediction history during the session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Page configuration
st.set_page_config(
    page_title="Smart Student Feedback Analyzer",
    page_icon="",
    layout="wide"
)

# Dashboard title
st.title("🎓 Smart Student Feedback Analyzer")
st.write(
    "Analyze student feedback and identify "
    "Positive, Negative, and Neutral sentiments."
)

st.divider()

# Section 1: Enter feedback
st.subheader("Analyze Student Feedback")

feedback = st.text_area(
    "Enter student feedback:",
    placeholder="Example: The teacher explains concepts clearly..."
)


if st.button("Analyze Feedback"):

    if feedback.strip():

        prediction = model.predict([feedback])[0]

        # Save prediction in history
        st.session_state.prediction_history.append({
            "Feedback": feedback,
            "Predicted Sentiment": prediction
        })

        st.subheader("Prediction Result")

        if prediction == "positive":
            st.success("Positive Sentiment ")

        elif prediction == "negative":
            st.error("Negative Sentiment ")

        else:
            st.info("Neutral Sentiment ")

    else:
        st.warning("Please enter some feedback first.")


# Section 2: Dataset statistics
st.subheader("Overall Feedback Analysis")

sentiment_counts = df["sentiment"].value_counts()

total_feedback = len(df)
positive = sentiment_counts.get("positive", 0)
negative = sentiment_counts.get("negative", 0)
neutral = sentiment_counts.get("neutral", 0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Feedback", total_feedback)
col2.metric("Positive", positive)
col3.metric("Negative", negative)
col4.metric("Neutral", neutral)

st.divider()


# Section 3: Sentiment distribution chart
st.subheader("Sentiment Distribution")

fig, ax = plt.subplots()

sentiment_counts = sentiment_counts.reindex(
    ["positive", "negative", "neutral"],
    fill_value=0
)

ax.bar(sentiment_counts.index, sentiment_counts.values)

ax.set_xlabel("Sentiment")
ax.set_ylabel("Number of Feedback Comments")
ax.set_title("Student Feedback Sentiment Distribution")

st.pyplot(fig)
# Section 4: Show sample feedback
st.subheader("Sample Student Feedback")

st.dataframe(
    df[["comment", "sentiment"]].head(10),
    use_container_width=True
)

# Section 5: Download Feedback Report

st.divider()

st.subheader("Download Feedback Report")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Feedback Report",
        data=csv_data,
        file_name="student_feedback_report.csv",
        mime="text/csv"
    )

else:
    st.info("Analyze some feedback first to download a report.")

# Section 6: Prediction History

st.divider()

st.subheader("Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    # Clear prediction history
    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info("No prediction history available yet.")
