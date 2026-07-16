import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import base64
import os
from collections import Counter
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Try to import wordcloud, but handle gracefully if not available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    st.warning("⚠️ WordCloud library not available. Some visualizations will be limited.")

# ============================================================================
# NLTK DATA DOWNLOAD WITH ERROR HANDLING
# ============================================================================
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    try:
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
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        if nltk_available:
            try:
                tokens = nltk.word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                tokens = [token for token in tokens if token not in stop_words]
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
    .info-box {
        padding: 1rem;
        background: #d1ecf1;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        background: #d4edda;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
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
        
        if 'Statement' not in df.columns and 'Truth Value' not in df.columns:
            text_col = next((col for col in df.columns if 'statement' in col.lower() or 'text' in col.lower()), df.columns[0])
            label_col = next((col for col in df.columns if 'truth' in col.lower() or 'label' in col.lower() or 'class' in col.lower()), df.columns[1])
            df = df.rename(columns={text_col: 'Statement', label_col: 'Truth Value'})
        
        df = df.dropna(subset=['Statement', 'Truth Value'])
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
        X = df['Statement'].apply(preprocess_text)
        y = df['Truth Value']
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        classifier = MultinomialNB(alpha=0.1)
        classifier.fit(X_train_tfidf, y_train)
        
        y_pred = classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.session_state.classifier = classifier
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
        st.session_state.model_trained = True
        st.session_state.model_results = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'class_names': label_encoder.classes_,
            'X_train_tfidf': X_train_tfidf,
            'y_train': y_train
        }
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return False

# ============================================================================
# GIF DISPLAY FUNCTION
# ============================================================================
def display_gif():
    """Display the Markov chain GIF"""
    gif_paths = [
        "markov_chain.gif",
        "images/markov_chain.gif",
        "static/markov_chain.gif",
        "assets/markov_chain.gif"
    ]
    
    for path in gif_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    gif_bytes = f.read()
                gif_base64 = base64.b64encode(gif_bytes).decode('utf-8')
                st.markdown(f"""
                <div style="text-align: center; margin: 1rem 0;">
                    <img src="data:image/gif;base64,{gif_base64}" 
                         style="max-width: 100%; max-height: 300px; border-radius: 10px;" 
                         alt="Markov Chain Animation">
                </div>
                """, unsafe_allow_html=True)
                return True
            except:
                continue
    
    st.info("""
    💡 **Markov Chain Animation**  
    A Markov chain is a stochastic model describing a sequence of possible events 
    where the probability of each event depends only on the state attained in the previous event.
    """)
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
        
        # Create separate figures for bar and pie
        fig_bar = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title='Class Distribution (Bar)',
            labels={'x': 'Class', 'y': 'Count'},
            color=class_counts.index,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(showlegend=False)
        
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title='Class Distribution (Pie)',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Display both charts side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.warning(f"Could not create distribution chart: {str(e)}")
        return None

def create_spider_chart(model_results):
    """Create spider/radar chart for model performance metrics"""
    try:
        if not model_results:
            return None
        
        report = classification_report(
            model_results['y_test'],
            model_results['y_pred'],
            target_names=model_results['class_names'],
            output_dict=True
        )
        
        categories = ['Precision', 'Recall', 'F1-Score']
        
        fig = go.Figure()
        
        for class_name in model_results['class_names']:
            if class_name in report:
                values = [
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=class_name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Model Performance Radar Chart',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create spider chart: {str(e)}")
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

def create_word_cloud(df, class_name=None):
    """Create word cloud for text data"""
    if not WORDCLOUD_AVAILABLE:
        return None
        
    try:
        if class_name:
            texts = df[df['Truth Value'] == class_name]['Statement'].tolist()
            title = f"Word Cloud - {class_name}"
        else:
            texts = df['Statement'].tolist()
            title = "Word Cloud - All Classes"
        
        if not texts:
            return None
        
        text = ' '.join(texts)
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create word cloud: {str(e)}")
        return None

def create_text_length_distribution(df):
    """Create text length distribution visualization"""
    try:
        df['text_length'] = df['Statement'].apply(len)
        
        # Create separate charts
        fig_hist = px.histogram(
            df,
            x='text_length',
            nbins=30,
            title='Text Length Distribution',
            labels={'text_length': 'Text Length'},
            color_discrete_sequence=['#667eea']
        )
        
        fig_box = px.box(
            df,
            x='Truth Value',
            y='text_length',
            title='Length by Class',
            labels={'Truth Value': 'Class', 'text_length': 'Text Length'},
            color='Truth Value',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_box, use_container_width=True)
        
        return True
        
    except Exception as e:
        st.warning(f"Could not create length distribution: {str(e)}")
        return None

def create_tsne_visualization(model_results):
    """Create t-SNE visualization of text embeddings"""
    try:
        if not model_results or 'X_train_tfidf' not in model_results:
            return None
        
        X = model_results['X_train_tfidf'][:500]
        y = model_results['y_train'][:500]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X.toarray())
        
        df_tsne = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'class': [model_results['class_names'][i] for i in y]
        })
        
        fig = px.scatter(
            df_tsne,
            x='x',
            y='y',
            color='class',
            title='t-SNE Visualization of Text Embeddings',
            labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(height=500)
        return fig
        
    except Exception as e:
        st.warning(f"Could not create t-SNE visualization: {str(e)}")
        return None

def create_top_words_per_class(vectorizer, classifier, label_encoder, top_n=15):
    """Create visualization of top words per class"""
    try:
        feature_names = vectorizer.get_feature_names_out()
        log_probs = classifier.feature_log_prob_
        
        fig = make_subplots(
            rows=len(label_encoder.classes_), 
            cols=1,
            subplot_titles=[f"Top Words - {class_name}" for class_name in label_encoder.classes_],
            shared_xaxes=True
        )
        
        for i, class_name in enumerate(label_encoder.classes_):
            importance = np.exp(log_probs[i])
            top_indices = np.argsort(importance)[-top_n:]
            top_words = [feature_names[idx] for idx in top_indices]
            top_scores = [importance[idx] for idx in top_indices]
            
            fig.add_trace(
                go.Bar(
                    x=top_scores,
                    y=top_words,
                    orientation='h',
                    name=class_name,
                    marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(height=200 * len(label_encoder.classes_), showlegend=False)
        fig.update_xaxes(title_text="Importance Score")
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create top words visualization: {str(e)}")
        return None

def create_feature_importance(vectorizer, classifier, label_encoder, top_n=20):
    """Create feature importance visualization"""
    try:
        feature_names = vectorizer.get_feature_names_out()
        log_probs = classifier.feature_log_prob_
        
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
            height=500,
            color_discrete_sequence=px.colors.qualitative.Set2
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
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Set2
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
        processed_text = preprocess_text(text)
        text_tfidf = st.session_state.vectorizer.transform([processed_text])
        
        prediction_encoded = st.session_state.classifier.predict(text_tfidf)
        prediction = st.session_state.label_encoder.inverse_transform(prediction_encoded)[0]
        
        probabilities = st.session_state.classifier.predict_proba(text_tfidf)
        prob_dict = dict(zip(st.session_state.label_encoder.classes_, probabilities[0]))
        
        st.session_state.prediction_history.append({
            'text': text[:100],
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
    
    # Display GIF
    st.markdown("### 🔄 Markov Chain Visualization")
    display_gif()
    
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Explorer",
            "🎯 Single Prediction",
            "📈 Model Analysis",
            "📄 Batch Processing",
            "📝 History",
            "🔬 Advanced Analytics"
        ])
        
        with tab1:
            st.markdown("### 📊 Data Explorer")
            
            # Class distribution
            st.markdown("#### Class Distribution")
            create_class_distribution(df)
            
            # Text length distribution
            st.markdown("#### Text Length Analysis")
            create_text_length_distribution(df)
            
            # Word clouds
            if WORDCLOUD_AVAILABLE:
                st.markdown("### ☁️ Word Clouds")
                
                # Overall word cloud
                fig_wc = create_word_cloud(df)
                if fig_wc:
                    st.pyplot(fig_wc)
                    plt.close()
                
                # Word clouds per class
                st.markdown("#### Word Clouds by Class")
                cols = st.columns(min(3, len(df['Truth Value'].unique())))
                for idx, class_name in enumerate(df['Truth Value'].unique()[:3]):
                    with cols[idx % 3]:
                        fig_wc_class = create_word_cloud(df, class_name)
                        if fig_wc_class:
                            st.pyplot(fig_wc_class)
                            plt.close()
            else:
                st.info("💡 Install wordcloud library for word cloud visualizations")
            
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
                            
                            st.markdown("#### Class Probabilities")
                            
                            prob_df = pd.DataFrame({
                                'Class': list(probabilities.keys()),
                                'Probability': list(probabilities.values())
                            })
                            
                            fig = px.bar(
                                prob_df,
                                x='Class',
                                y='Probability',
                                color='Class',
                                title='Prediction Probabilities',
                                range_y=[0, 1],
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            
                            fig.update_layout(showlegend=False, height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Please enter text to classify")
        
        with tab3:
            st.markdown("### 📈 Model Analysis")
            
            if st.session_state.model_results:
                results = st.session_state.model_results
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Accuracy", f"{results['accuracy']:.2%}")
                with col2:
                    st.metric("Test Samples", len(results['y_test']))
                with col3:
                    st.metric("Classes", len(results['class_names']))
                with col4:
                    report = classification_report(
                        results['y_test'],
                        results['y_pred'],
                        target_names=results['class_names'],
                        output_dict=True
                    )
                    f1_macro = report['macro avg']['f1-score']
                    st.metric("Macro F1", f"{f1_macro:.3f}")
                
                # Radar Chart (Spider Graph)
                st.markdown("#### 🕸️ Performance Radar Chart")
                fig_radar = create_spider_chart(results)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)
                
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
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
                
                # Top Words per Class
                st.markdown("#### 🏆 Top Words per Class")
                fig_top_words = create_top_words_per_class(
                    st.session_state.vectorizer,
                    st.session_state.classifier,
                    st.session_state.label_encoder
                )
                if fig_top_words:
                    st.plotly_chart(fig_top_words, use_container_width=True)
                
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
                            
                            batch_df['prediction'] = predictions
                            batch_df['confidence'] = confidences
                            
                            st.markdown("#### Prediction Results")
                            st.dataframe(batch_df, use_container_width=True)
                            
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )
                            
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
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    unique_classes = len(history_df['prediction'].unique())
                    st.metric("Unique Classes Predicted", unique_classes)
                with col3:
                    avg_conf = history_df['confidence'].mean()
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                
                st.dataframe(history_df, use_container_width=True)
                
                # Prediction distribution
                fig = create_prediction_metrics()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()
            else:
                st.info("ℹ️ No predictions made yet. Start predicting in the 'Single Prediction' tab.")
        
        with tab6:
            st.markdown("### 🔬 Advanced Analytics")
            
            if st.session_state.model_trained:
                # t-SNE Visualization
                st.markdown("#### 🎯 t-SNE Visualization")
                fig_tsne = create_tsne_visualization(st.session_state.model_results)
                if fig_tsne:
                    st.plotly_chart(fig_tsne, use_container_width=True)
                else:
                    st.info("t-SNE visualization not available")
                
                # Text Statistics
                st.markdown("#### 📊 Text Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                char_counts = df['Statement'].apply(len)
                with col1:
                    st.metric("Average Length", f"{char_counts.mean():.1f} chars")
                with col2:
                    st.metric("Median Length", f"{char_counts.median():.0f} chars")
                with col3:
                    st.metric("Max Length", f"{char_counts.max():.0f} chars")
                
                # Vocabulary analysis
                all_words = ' '.join(df['Statement']).split()
                vocab_size = len(set(all_words))
                st.metric("Vocabulary Size", vocab_size)
                
                # Common words
                st.markdown("#### 🔤 Most Common Words")
                word_counts = Counter(all_words)
                common_words = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Count'])
                
                fig = px.bar(
                    common_words,
                    x='Word',
                    y='Count',
                    title='Top 20 Most Common Words',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Class-specific statistics
                st.markdown("#### 📈 Class-Specific Statistics")
                
                class_stats = []
                for class_name in df['Truth Value'].unique():
                    class_df = df[df['Truth Value'] == class_name]
                    lengths = class_df['Statement'].apply(len)
                    class_stats.append({
                        'Class': class_name,
                        'Count': len(class_df),
                        'Avg Length': lengths.mean(),
                        'Std Length': lengths.std(),
                        'Min Length': lengths.min(),
                        'Max Length': lengths.max()
                    })
                
                stats_df = pd.DataFrame(class_stats)
                st.dataframe(stats_df.round(2), use_container_width=True)
                
                # Co-occurrence analysis
                st.markdown("#### 🔗 Word Co-occurrence")
                
                top_words = [word for word, _ in word_counts.most_common(10)]
                co_occurrence = {}
                
                for word in top_words:
                    co_occurrence[word] = {}
                    for other in top_words:
                        if word != other:
                            count = 0
                            for text in df['Statement']:
                                if word in text and other in text:
                                    count += 1
                            co_occurrence[word][other] = count
                
                if co_occurrence:
                    co_df = pd.DataFrame(co_occurrence).fillna(0)
                    fig = px.imshow(
                        co_df.values,
                        x=co_df.columns,
                        y=co_df.index,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Blues',
                        title='Word Co-occurrence Matrix'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("ℹ️ Train the model first to see advanced analytics")
    
    else:
        st.warning("⚠️ No data loaded. Click the 'Load & Train Model' button in the sidebar to get started.")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. Click **Load & Train Model** in the sidebar
        2. Wait for the model to train (this may take a few seconds)
        3. Start predicting text in the **Single Prediction** tab
        4. Explore data in the **Data Explorer** tab
        5. Check **Advanced Analytics** for deeper insights
        
        #### Features:
        - ✅ Text classification with TF-IDF and Naive Bayes
        - ✅ Interactive data exploration
        - ✅ Batch prediction support
        - ✅ Comprehensive analytics
        - ✅ Prediction history tracking
        - ✅ Radar/Spider charts
        - ✅ Word clouds
        - ✅ t-SNE visualization
        - ✅ Text statistics and analysis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📝 Advanced Text Classification App v3.0</p>
        <p style='font-size: 0.8rem;'>Author: Md Abu Sufian | For educational and research purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
