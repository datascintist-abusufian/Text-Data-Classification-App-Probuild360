# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import altair as alt

# Display the GIF image using st.image
st.image("activation.gif", use_column_width=True)

# Title and description
st.title("Text Data Classification App-Probuild360")
st.write("This app demonstrates text classification into different classes using Streamlit.")

# Function to load data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.dropna(inplace=True)  # Drop missing values
        return df
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        if 'text' in df.columns and 'label' in df.columns:
            # Assuming 'text' is the column with statements and 'label' has categories like true, false, etc.
            X = df['text']
            y = df['label']

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Define and train tfidf_vectorizer
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
            X_test_tfidf = tfidf_vectorizer.transform(X_test)

            # Train a Multinomial Naive Bayes classifier
            classifier = MultinomialNB()
            classifier.fit(X_train_tfidf, y_train)

            # Make predictions on the test data
            y_pred = classifier.predict(X_test_tfidf)

            # Display accuracy and classification report
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))
            
            # Define the class labels
            class_labels = ["true", "false", "mostly-true", "half-true", "pants-fire", "mostly-false"]
            
            # Add a selection box for class labels
            st.title("Class Prediction")
            selected_class = st.selectbox("Select a class for prediction", class_labels, key='class_selection')
            
            # Button to generate random prediction
            if st.button("Show Random Prediction", key='random_prediction_button'):
                if selected_class:
                    # Generate random prediction based on the selected class
                    random_prediction = df[df['label'] == selected_class].sample(n=1)['text'].values[0]
                    st.write(f"Random '{selected_class}' statement: {random_prediction}")
                else:
                    st.error("Please select a class first.")

            # User input for text classification
            st.subheader("Try the Classifier")
            text_input = st.text_input("Enter text for classification:")
            if text_input:
                # Predict the class of the entered text
                text_input_tfidf = tfidf_vectorizer.transform([text_input])
                prediction = classifier.predict(text_input_tfidf)
                st.write("Prediction Result:")
                st.success(f"The predicted class for the entered text is: {prediction[0]}")
else:
    st.sidebar.markdown("Please upload a CSV file to begin.")
