import streamlit as st
import joblib

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("SMS spam classifier")
st.write("Enter a message and let the model predict if it's **Spam** or **ham**")

user_input = st.text_area("Type Your Message here: ")

if st.button("Classify"):
    if user_input == "":
        st.warning("Please enter a message")
    else:
        vectorizer_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorizer_input)[0]
        prediction_prob = model.predict_proba(vectorizer_input).max()

        if prediction =="spam":
            st.error(f"Prediction: **SPAM** ({prediction_prob: .2%} confidence)")
        else:
            st.success(f"Prediction: **HAM* ({prediction_prob: .2%} confidence)")