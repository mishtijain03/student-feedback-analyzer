
import joblib

# Load the trained model
model = joblib.load("models/sentiment_model.pkl")

print("STUDENT FEEDBACK SENTIMENT ANALYZER")
print("-----------------------------------")

while True:

    feedback = input("\nEnter student feedback (or type 'exit' to stop): ")

    if feedback.lower() == "exit":
        print("Thank you!")
        break

    # Predict sentiment
    prediction = model.predict([feedback])[0]

    print("Predicted Sentiment:", prediction.upper())
