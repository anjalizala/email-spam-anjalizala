import streamlit as st
import pickle

model=pickle.load(open('spam.pkl','rb'))
cv=pickle.load(open('vectorizer.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("This is a Machine Learning Application to classify emails as spam or ham.")
user_input = st.text_area("Enter an email to classify")

if st.button("Classify"):
    if user_input:
        data = [user_input]
        vect = cv.transform(data).toarray()
        pred = model.predict(vect)
        if pred[0]==0:
            st.write("This email is not spam")
            st.success("This email is not spam")
        else:
            st.error("This is spam email")
    else:
        print("Please type email")