import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Data loading function with caching
@st.cache(allow_output_mutation=True)  # Allow st.write within the cached function
def load_data_from_github():
    url = 'https://raw.githubusercontent.com/datascintist-abusufian/Text-Data-Classification-App-for-Probuild360/main/test.csv'
    try:
        df = pd.read_csv(url)
        df = df.dropna(subset=['Statement', 'Truth Value'])
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

# Initialize session state variables if they don't exist
if 'load_example' not in st.session_state:
    st.session_state['load_example'] = False

if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False

# UI for the app
st.image("markov_chain.gif", use_column_width=True)
st.title("Text Data Classification App-Probuild360")
st.write("This app demonstrates text classification into different classes using Streamlit.")
st.markdown("<span style='color:blue'>Author Md Abu Sufian</span>", unsafe_allow_html=True)
st.markdown("""
    <div style='color: green;'>
        <strong>Important Note:</strong> The accuracy score is based on an analysis using test data,
        which is a subset typically used to evaluate the model's predictions and not for training.
        Therefore, the scores may not reflect the model's potential accuracy with a fully trained dataset.
    </div>
    """, unsafe_allow_html=True)

# Sidebar for file upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state['data_uploaded'] = True
    st.session_state['load_example'] = False
    df_uploaded = pd.read_csv(uploaded_file)
    st.sidebar.write("Uploaded Data:")
    st.sidebar.dataframe(df_uploaded.head())

if st.sidebar.button("Load Example Data"):
    st.session_state['load_example'] = True
    st.session_state['data_uploaded'] = False

df = pd.DataFrame()
# Decision logic for displaying data based on user interaction
if st.session_state['load_example']:
    df = load_data_from_github()
elif st.session_state['data_uploaded'] and 'df_uploaded' in locals():
    df = df_uploaded

if not df.empty:
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    if 'Statement' in df.columns and 'Truth Value' in df.columns:
        # Preprocessing and model training
        X = df['Statement']
        y = df['Truth Value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        classifier = MultinomialNB()
        classifier.fit(X_train_tfidf, y_train)
        y_pred = classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))
        
        # Class selection and displaying random texts
        st.title("Class Selection and Random Texts")
        selected_class = st.selectbox("Select a class to see a random text", df['Truth Value'].unique())
        
        if st.button("Show Random Text", key="show_random_text_1"):
            filtered_df = df[df['Truth Value'] == selected_class]
            if not filtered_df.empty:
                random_text = filtered_df.sample(n=1)['Statement'].iloc[0]
                st.write(f"Random text from '{selected_class}': {random_text}")
            else:
                st.write(f"No texts found for the selected class '{selected_class}'.")

        # User input prediction
        st.subheader("Text Classification Prediction")
        text_input = st.text_input("Enter text for classification prediction:")
        if text_input:
            text_input_tfidf = tfidf_vectorizer.transform([text_input])
            prediction = classifier.predict(text_input_tfidf)
            st.write("Prediction Result:")
            st.success(f"The predicted class for the entered text is: {prediction[0]}")
else:
    st.error("Failed to load or display data. Please check the dataset url, format, or your file upload.")
