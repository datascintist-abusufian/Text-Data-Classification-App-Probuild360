# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Display the GIF image using st.image
st.image ("Bayes_theorem_illustration.gif", use_column_width=True)

# Title and description
st.title("Text Data Classification App-Probuild360")
st.write("This app demonstrates text classification into different classes using Streamlit.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='color: green;'>
        <strong>Important Note:</strong> The accuracy score is based on an analysis using test data, 
        which is a subset typically used to evaluate the model's predictions and not for training. 
        Therefore, the scores may not reflect the model's potential accuracy with a fully trained dataset.
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar for file upload

st.sidebar.header("Upload Your Data")
st.sidebar.write("Upload a CSV file for data analysis.")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
        # Process the uploaded file
        df_uploaded = pd.read_csv(uploaded_file)
        st.sidebar.write("Uploaded Data:")
        st.sidebar.dataframe(df_uploaded.head())

# Existing functionality for loading data from GitHub
st.sidebar.header("Or Use Example Dataset")
if st.sidebar.button("Load Example Data"):
        df_example = load_data_from_github()
        if df_example is not None:
            st.write("Example Data:")
            st.dataframe(df_example.head())
                
@st.cache(show_spinner=False)
def load_data_from_github():
    url = 'https://raw.githubusercontent.com/datascintist-abusufian/Text-Data-Classification-App-for-Probuild360/main/test.csv'
    st.write("Loading data from:", url)  # Debug print
    try:
        df = pd.read_csv(url)
        # Debug print
        st.write("Data loaded successfully. Here's a preview:")
        st.write(df.head())
        # Selective dropna
        df = df.dropna(subset=['Statement', 'Truth Value'])
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the dataset
df = load_data_from_github()

# Check if the dataset is loaded
if df is not None and not df.empty:
    if 'Statement' in df.columns and 'Truth Value' in df.columns:
        # Display the DataFrame head
        st.write("First five rows of the loaded dataset:")
        st.write(df.head())
        
    if 'Statement' in df.columns and 'Truth Value' in df.columns:
        # Split the data into training and testing sets
        X = df['Statement']
        y = df['Truth Value']
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
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        # Define the class labels
        class_labels = ["true", "false", "mostly-true", "half-true", "pants-fire", "mostly-false"]

        # Section for Class Selection and Displaying Random Texts
        st.title("Class Selection and Random Texts")
        selected_class = st.selectbox("Select a class to see a random text", class_labels)
            
        if st.button("Show Random Text"):
            random_text = df[df['Truth Value'] == selected_class].sample(n=1)['Statement'].iloc[0]
            st.write(f"Random text from '{selected_class}': {random_text}")

        # Section for User Input Prediction
        st.subheader("Text Classification Prediction")
        text_input = st.text_input("Enter text for classification prediction:")
        if text_input:
            text_input_tfidf = tfidf_vectorizer.transform([text_input])
            prediction = classifier.predict(text_input_tfidf)
            st.write("Prediction Result:")
            st.success(f"The predicted class for the entered text is: {prediction[0]}")
else:
    st.error("Failed to load data. Please check the dataset URL or format")
