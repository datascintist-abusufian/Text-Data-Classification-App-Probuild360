# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Data loading function with caching
@st.cache (allow_output_mutation=True)  # Allow st.write within the cached function
def load_data_from_github():
    url = 'https://raw.githubusercontent.com/datascintist-abusufian/Text-Data-Classification-App-for-Probuild360/main/test.csv'
    #st.write("Loading data from:", url)  # Debug print
    try:
        df = pd.read_csv(url)
        #st.write("Data loaded successfully. Here's a preview:")  # Debug print
        #st.write(df.head())
        df = df.dropna(subset=['Statement', 'Truth Value'])
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()

# Initialize session state variables
if 'load_example' not in st.session_state:
    st.session_state['load_example'] = False

if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False

# Display the GIF, Title, and Description...
st.image("markov_chain.gif", use_column_width=True)

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
    unsafe_allow_html=True)

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

# Decision logic for displaying data based on user interaction
if st.session_state['load_example']:
    df_example = load_data_from_github()
    if df_example is not None and not df_example.empty:
        st.write("Example Data:")
        st.dataframe(df_example.head())
            
# Load the dataset
df, error_message = load_data_from_github()

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

if st.button("Show Random Text", key="show_random_text_1"):
    # Filter the DataFrame based on the selected class
    filtered_df = df[df['Truth Value'] == selected_class]

    # Check if the filtered DataFrame is empty
    if not filtered_df.empty:
        random_text = filtered_df.sample(n=1)['Statement'].iloc[0]
        st.write(f"Random text from '{selected_class}': {random_text}")
    else:
        st.write(f"No texts found for the selected class '{selected_class}'.")

# Section for User Input Prediction
st.subheader("Text Classification Prediction")
text_input = st.text_input("Enter text for classification prediction:")
if text_input:
            text_input_tfidf = tfidf_vectorizer.transform([text_input])
            prediction = classifier.predict(text_input_tfidf)
            st.write("Prediction Result:")
            st.success(f"The predicted class for the entered text is: {prediction[0]}")
else:
    st.error("Failed to load data. Please check the dataset url or format. Error: " + error_message)

