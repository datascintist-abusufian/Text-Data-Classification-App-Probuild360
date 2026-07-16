import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NLTK DATA DOWNLOAD WITH ERROR HANDLING
# ============================================================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
        # Try to download all required resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
        
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
        
        return True
    except Exception as e:
        st.warning(f"⚠️ NLTK download warning: {str(e)}")
        return False

# Download NLTK data
nltk_available = download_nltk_data()

# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================
def preprocess_text(text):
    """Clean and preprocess text"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        # Advanced preprocessing if NLTK is available
        if nltk_available:
            try:
                # Tokenize
                tokens = nltk.word_tokenize(text)
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
                
                # Lemmatize
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
                
                return ' '.join(tokens)
            except:
                return text
        else:
            return text
            
    except Exception as e:
        return text

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Advanced Text Classification App",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5

initialize_session_state()

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess data from GitHub"""
    try:
        url = 'https://raw.githubusercontent.com/datascintist-abusufian/Text-Data-Classification-App-Probuild360/main/test.csv'
        df = pd.read_csv(url)
        
        # Check and rename columns
        if 'Statement' not in df.columns and 'Truth Value' not in df.columns:
            # Try to find appropriate columns
            text_col = next((col for col in df.columns if 'statement' in col.lower() or 'text' in col.lower()), df.columns[0])
            label_col = next((col for col in df.columns if 'truth' in col.lower() or 'label' in col.lower() or 'class' in col.lower()), df.columns[1])
            df = df.rename(columns={text_col: 'Statement', label_col: 'Truth Value'})
        
        # Drop rows with missing values
        df = df.dropna(subset=['Statement', 'Truth Value'])
        
        # Clean text
        df['Statement'] = df['Statement'].astype(str).apply(clean_text)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_model(df):
    """Train the text classification model"""
    try:
        # Prepare features and labels
        X = df['Statement'].apply(preprocess_text)
        y = df['Truth Value']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Transform data
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train classifier
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store in session state
        st.session_state.classifier = classifier
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
        st.session_state.model_trained = True
        st.session_state.model_results = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'class_names': label_encoder.classes_
        }
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return False

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_class_distribution(df):
    """Create class distribution visualization"""
    try:
        if df is None or df.empty:
            return None
            
        class_counts = df['Truth Value'].value_counts()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Class Distribution (Bar)', 'Class Distribution (Pie)'],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color='#667eea',
                text=class_counts.values,
                textposition='auto',
                name='Count'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.3,
                name='Distribution'
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=450, showlegend=False)
        return fig
        
    except Exception as e:
        st.warning(f"Could not create distribution chart: {str(e)}")
        return None

def create_confusion_matrix(y_test, y_pred, class_names):
    """Create confusion matrix visualization"""
    try:
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            x=class_names,
            y=class_names,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title="Confusion Matrix"
        )
        
        fig.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create confusion matrix: {str(e)}")
        return None

def create_feature_importance(vectorizer, classifier, label_encoder, top_n=20):
    """Create feature importance visualization"""
    try:
        feature_names = vectorizer.get_feature_names_out()
        log_probs = classifier.feature_log_prob_
        
        # Calculate importance for each class
        importance_data = []
        for i, class_name in enumerate(label_encoder.classes_):
            importance = np.exp(log_probs[i])
            top_indices = np.argsort(importance)[-top_n:]
            
            for idx in top_indices:
                importance_data.append({
                    'Class': class_name,
                    'Feature': feature_names[idx],
                    'Importance': importance[idx]
                })
        
        df_importance = pd.DataFrame(importance_data)
        
        fig = px.bar(
            df_importance,
            x='Feature',
            y='Importance',
            color='Class',
            title=f'Top {top_n} Features by Class',
            facet_col='Class',
            facet_col_wrap=2,
            height=500
        )
        
        fig.update_layout(showlegend=False)
        return fig
        
    except Exception as e:
        st.warning(f"Could not create feature importance: {str(e)}")
        return None

def create_prediction_metrics():
    """Create prediction metrics dashboard"""
    if not st.session_state.prediction_history:
        return None
    
    try:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        class_counts = df_history['prediction'].value_counts()
        
        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Prediction Distribution',
            hole=0.3
        )
        
        return fig
    except:
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_text(text):
    """Predict class for input text"""
    if not st.session_state.model_trained:
        st.warning("⚠️ Model not trained yet. Please load data first.")
        return None, None
    
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Transform
        text_tfidf = st.session_state.vectorizer.transform([processed_text])
        
        # Predict
        prediction_encoded = st.session_state.classifier.predict(text_tfidf)
        prediction = st.session_state.label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Get probabilities
        probabilities = st.session_state.classifier.predict_proba(text_tfidf)
        prob_dict = dict(zip(st.session_state.label_encoder.classes_, probabilities[0]))
        
        # Store in history
        st.session_state.prediction_history.append({
            'text': text[:100],  # Truncate for storage
            'prediction': prediction,
            'confidence': max(probabilities[0])
        })
        
        return prediction, prob_dict
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, None

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="gradient-header">
        <h1>📝 Advanced Text Classification App</h1>
        <p>Machine Learning powered text classification with comprehensive analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        # Model Training Section
        st.markdown("### 📊 Data Loading")
        if st.button("🔄 Load & Train Model", use_container_width=True):
            with st.spinner("Loading data and training model..."):
                df = load_data()
                if not df.empty:
                    st.session_state.df = df
                    success = train_model(df)
                    if success:
                        st.success(f"✅ Model trained with {st.session_state.model_results['accuracy']:.2%} accuracy!")
                    else:
                        st.error("❌ Model training failed")
                else:
                    st.error("❌ Failed to load data")
        
        if st.session_state.model_trained:
            st.markdown("---")
            st.markdown("### 📈 Model Performance")
            if st.session_state.model_results:
                st.metric("Accuracy", f"{st.session_state.model_results['accuracy']:.2%}")
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Settings")
            
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
        
        # Statistics
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        if st.session_state.model_trained and st.session_state.df is not None:
            st.metric("Total Samples", len(st.session_state.df))
            st.metric("Unique Classes", len(st.session_state.df['Truth Value'].unique()))
            st.metric("Predictions Made", len(st.session_state.prediction_history))
        
        # Clear history
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    
    # Main content
    if st.session_state.df is not None and not st.session_state.df.empty:
        df = st.session_state.df
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Explorer",
            "🎯 Single Prediction",
            "📈 Model Analysis",
            "📄 Batch Processing",
            "📝 History"
        ])
        
        with tab1:
            st.markdown("### 📊 Data Explorer")
            
            # Class distribution
            fig_dist = create_class_distribution(df)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.info("No data available for visualization")
            
            # Data preview
            st.markdown("### 📋 Data Preview")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(df.head(100), use_container_width=True)
            
            with col2:
                st.markdown("#### Quick Stats")
                st.write(f"Total samples: {len(df)}")
                st.write(f"Classes: {', '.join(df['Truth Value'].unique())}")
                
                class_counts = df['Truth Value'].value_counts()
                for class_name, count in class_counts.items():
                    st.write(f"• {class_name}: {count}")
            
            # Random text generator
            st.markdown("### 🎲 Random Text Generator")
            selected_class = st.selectbox("Select class for random text", df['Truth Value'].unique())
            
            if st.button("🎲 Show Random Text", use_container_width=True):
                filtered_df = df[df['Truth Value'] == selected_class]
                if not filtered_df.empty:
                    random_text = filtered_df.sample(n=1)['Statement'].iloc[0]
                    st.info(f"**Random text from '{selected_class}':**\n\n{random_text}")
                else:
                    st.warning(f"No texts found for class '{selected_class}'")
        
        with tab2:
            st.markdown("### 🎯 Text Classification Prediction")
            
            st.info("Enter text to classify. The model will predict the most likely class and show confidence scores for all classes.")
            
            # Text input
            text_input = st.text_area(
                "Enter text for classification:",
                placeholder="Enter your text here for classification...",
                height=150
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔍 Predict", type="primary", use_container_width=True):
                    if text_input and text_input.strip():
                        prediction, probabilities = predict_text(text_input)
                        
                        if prediction and probabilities:
                            # Display prediction
                            confidence = probabilities[prediction]
                            confidence_color = "#28a745" if confidence >= st.session_state.confidence_threshold else "#ffc107"
                            
                            st.markdown(f"""
                            <div class="prediction-card" style="border-left: 5px solid {confidence_color};">
                                <div style="font-size: 1rem; color: #666;">Predicted Class</div>
                                <div style="font-size: 2.5rem; color: {confidence_color}; margin: 0.5rem 0;">
                                    {prediction}
                                </div>
                                <div>Confidence: <strong>{confidence:.2%}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show all class probabilities
                            st.markdown("#### Class Probabilities")
                            
                            prob_df = pd.DataFrame({
                                'Class': list(probabilities.keys()),
                                'Probability': list(probabilities.values())
                            })
                            
                            # Create probability bar chart
                            fig = px.bar(
                                prob_df,
                                x='Class',
                                y='Probability',
                                color='Class',
                                title='Prediction Probabilities',
                                range_y=[0, 1]
                            )
                            
                            fig.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Please enter text to classify")
        
        with tab3:
            st.markdown("### 📈 Model Analysis")
            
            if st.session_state.model_results:
                results = st.session_state.model_results
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Accuracy", f"{results['accuracy']:.2%}")
                with col2:
                    st.metric("Test Samples", len(results['y_test']))
                with col3:
                    st.metric("Classes", len(results['class_names']))
                
                # Confusion Matrix
                st.markdown("#### Confusion Matrix")
                fig_cm = create_confusion_matrix(
                    results['y_test'],
                    results['y_pred'],
                    results['class_names']
                )
                if fig_cm:
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.info("Confusion matrix not available")
                
                # Classification Report
                st.markdown("#### Classification Report")
                report = classification_report(
                    results['y_test'],
                    results['y_pred'],
                    target_names=results['class_names'],
                    output_dict=True
                )
                
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
                
                # Feature Importance
                st.markdown("#### Feature Importance")
                fig_importance = create_feature_importance(
                    st.session_state.vectorizer,
                    st.session_state.classifier,
                    st.session_state.label_encoder
                )
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance not available")
            else:
                st.info("ℹ️ Train the model first to see analysis")
        
        with tab4:
            st.markdown("### 📄 Batch Processing")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file for batch prediction",
                type=['csv'],
                help="CSV file should contain a column named 'text' or 'statement'"
            )
            
            if uploaded_file:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    
                    # Find text column
                    text_col = next((col for col in batch_df.columns if 'text' in col.lower() or 'statement' in col.lower()), batch_df.columns[0])
                    
                    st.markdown(f"#### Batch Data Preview (using column: '{text_col}')")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("📊 Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            predictions = []
                            confidences = []
                            
                            for text in batch_df[text_col]:
                                if text and isinstance(text, str):
                                    pred, probs = predict_text(text)
                                    if pred and probs:
                                        predictions.append(pred)
                                        confidences.append(probs[pred] if probs else 0)
                                    else:
                                        predictions.append("Error")
                                        confidences.append(0)
                                else:
                                    predictions.append("Empty")
                                    confidences.append(0)
                            
                            # Add predictions to dataframe
                            batch_df['prediction'] = predictions
                            batch_df['confidence'] = confidences
                            
                            # Display results
                            st.markdown("#### Prediction Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.markdown("#### Summary Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Processed", len(batch_df))
                            with col2:
                                valid_preds = len([p for p in predictions if p != "Error" and p != "Empty"])
                                st.metric("Valid Predictions", valid_preds)
                            with col3:
                                valid_conf = [c for c in confidences if c > 0]
                                avg_conf = np.mean(valid_conf) if valid_conf else 0
                                st.metric("Average Confidence", f"{avg_conf:.2%}" if avg_conf > 0 else "N/A")
                
                except Exception as e:
                    st.error(f"❌ Error processing batch file: {str(e)}")
        
        with tab5:
            st.markdown("### 📝 Prediction History")
            
            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    unique_classes = len(history_df['prediction'].unique())
                    st.metric("Unique Classes Predicted", unique_classes)
                
                # Display history
                st.dataframe(history_df, use_container_width=True)
                
                # Prediction distribution
                fig = create_prediction_metrics()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Clear history
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()
            else:
                st.info("ℹ️ No predictions made yet. Start predicting in the 'Single Prediction' tab.")
    
    else:
        # No data loaded
        st.warning("⚠️ No data loaded. Click the 'Load & Train Model' button in the sidebar to get started.")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. Click **Load & Train Model** in the sidebar
        2. Wait for the model to train (this may take a few seconds)
        3. Start predicting text in the **Single Prediction** tab
        4. Explore data in the **Data Explorer** tab
        
        #### Features:
        - ✅ Text classification with TF-IDF and Naive Bayes
        - ✅ Interactive data exploration
        - ✅ Batch prediction support
        - ✅ Comprehensive analytics
        - ✅ Prediction history tracking
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📝 Advanced Text Classification App v2.0</p>
        <p style='font-size: 0.8rem;'>Author: Md Abu Sufian | For educational and research purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
