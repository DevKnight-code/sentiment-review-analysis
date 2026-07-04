import os
os.environ.setdefault('MPLBACKEND', 'Agg')  # prevent matplotlib GUI crash on headless servers

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import pickle
import threading
import matplotlib
matplotlib.use('Agg')  # must be set before importing pyplot — prevents GUI crash on headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Set up static file serving for the React build
FRONTEND_BUILD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'build')

# Configure Flask for serving frontend
if os.path.exists(FRONTEND_BUILD_DIR):
    app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR, static_url_path='')
else:
    app = Flask(__name__)

# Configure CORS for both local development and production
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",           # Local development
            "https://*.onrender.com",          # Render domains
            "http://127.0.0.1:3000"            # Local testing
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Use path relative to this script so it works regardless of cwd
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# MongoDB — imported lazily so the app still starts without a configured URI
try:
    from . import database as db
except ImportError:
    import database as db

try:
    _mongo_available = db.is_connected()
    if _mongo_available:
        print("[MongoDB] Connection established — using MongoDB for persistence")
    else:
        print("[MongoDB] Not connected — using file-based persistence (set MONGO_URI in backend/.env)")
except Exception as _e:
    _mongo_available = False
    print(f"[MongoDB] Unavailable ({_e}) — using file-based persistence")

# Download required NLTK data
nltk_downloads = [
    ('punkt',         'tokenizers/punkt'),
    ('punkt_tab',     'tokenizers/punkt_tab'),
    ('stopwords',     'corpora/stopwords'),
]

for resource, resource_path in nltk_downloads:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[NLTK] Warning: Failed to download {resource}: {e}")

class SentimentAnalyzer:
    def __init__(self):
        # Enhanced TF-IDF vectorizer for better accuracy
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased features for better accuracy
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency to remove very common words
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.model = MultinomialNB(alpha=0.1)  # Add smoothing parameter
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        self.model_metrics = {}
        self.dataset = None
        self.user_feedback_data = []
        self.retrain_threshold = 10  # Retrain after 10 new samples
        self.total_predictions = 0
        self.analyzed_reviews = []  # Store analyzed reviews for dataset expansion
        self._retrain_lock = threading.Lock()   # prevent concurrent retrains
        self._retrain_status = 'idle'           # 'idle' | 'running' | 'done' | 'error'
        self._retrain_error  = None
        
    def preprocess_text(self, text):
        """Enhanced preprocessing to reduce bias and improve accuracy"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions and common abbreviations
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'m": " am", "'s": " is", "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs, emails, and phone numbers
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove excessive punctuation but keep some for sentiment
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple question marks
        text = re.sub(r'[.]{2,}', '.', text)  # Multiple periods
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Enhanced stopwords removal and stemming
        # Keep some sentiment-bearing words that are typically removed
        sentiment_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
        custom_stop_words = self.stop_words - sentiment_words
        
        processed_tokens = []
        for token in tokens:
            if (token not in custom_stop_words and 
                len(token) > 2 and 
                not token.isdigit() and
                token.isalpha()):
                processed_tokens.append(self.stemmer.stem(token))
        
        return ' '.join(processed_tokens)
    
    def create_sample_dataset(self):
        """Create a comprehensive dataset for better accuracy and reduced bias"""
        sample_reviews = [
            # Positive reviews (50 samples)
            ("This product is amazing! I love it so much.", "positive"),
            ("Excellent quality and fast delivery. Highly recommended!", "positive"),
            ("Perfect! Exactly what I was looking for.", "positive"),
            ("Great value for money. Will definitely buy again.", "positive"),
            ("Outstanding product with excellent features.", "positive"),
            ("Love this item! It exceeded my expectations.", "positive"),
            ("Fantastic quality and great customer service.", "positive"),
            ("Best purchase I've made in a long time!", "positive"),
            ("Superb product, very satisfied with my purchase.", "positive"),
            ("Wonderful experience, highly recommend to others.", "positive"),
            ("Absolutely brilliant! This is the best product ever.", "positive"),
            ("Incredible value and outstanding performance.", "positive"),
            ("Phenomenal quality, exceeded all my expectations.", "positive"),
            ("Magnificent product with superb features.", "positive"),
            ("Exceptional service and top-notch quality.", "positive"),
            ("Amazing product, works perfectly as described.", "positive"),
            ("Very happy with this purchase, great quality.", "positive"),
            ("Excellent product, would buy again.", "positive"),
            ("Perfect quality, fast shipping, highly satisfied.", "positive"),
            ("Great product, exceeded my expectations completely.", "positive"),
            ("Outstanding value, excellent build quality.", "positive"),
            ("Love it! Perfect for my needs.", "positive"),
            ("Fantastic product, very impressed.", "positive"),
            ("Excellent quality, great customer service.", "positive"),
            ("Perfect! Exactly what I needed.", "positive"),
            ("Amazing quality, very happy with purchase.", "positive"),
            ("Great product, works exactly as expected.", "positive"),
            ("Excellent value for money, highly recommend.", "positive"),
            ("Perfect quality, fast delivery, very satisfied.", "positive"),
            ("Outstanding product, exceeded expectations.", "positive"),
            ("Great quality, excellent service, very happy.", "positive"),
            ("Perfect product, works great, highly recommend.", "positive"),
            ("Excellent quality, great value, very satisfied.", "positive"),
            ("Amazing product, perfect quality, love it.", "positive"),
            ("Great purchase, excellent quality, very happy.", "positive"),
            ("Perfect product, great quality, highly recommend.", "positive"),
            ("Excellent value, great quality, very satisfied.", "positive"),
            ("Outstanding quality, perfect product, love it.", "positive"),
            ("Great product, excellent quality, very happy.", "positive"),
            ("Perfect quality, great value, highly recommend.", "positive"),
            ("Excellent product, amazing quality, very satisfied.", "positive"),
            ("Great quality, perfect product, love it.", "positive"),
            ("Outstanding product, excellent quality, very happy.", "positive"),
            ("Perfect quality, great product, highly recommend.", "positive"),
            ("Excellent value, perfect quality, very satisfied.", "positive"),
            ("Amazing quality, great product, love it.", "positive"),
            ("Great product, outstanding quality, very happy.", "positive"),
            ("Perfect quality, excellent product, highly recommend.", "positive"),
            ("Excellent quality, great value, very satisfied.", "positive"),
            ("Outstanding product, perfect quality, love it.", "positive"),
            
            # Negative reviews (50 samples)
            ("Terrible product. Complete waste of money.", "negative"),
            ("Poor quality and doesn't work as advertised.", "negative"),
            ("Very disappointed with this purchase.", "negative"),
            ("Awful customer service and slow delivery.", "negative"),
            ("Product broke after just one week of use.", "negative"),
            ("Not worth the price at all. Very poor quality.", "negative"),
            ("Horrible experience, would not recommend.", "negative"),
            ("Defective product, very frustrated.", "negative"),
            ("Waste of time and money. Avoid this product.", "negative"),
            ("Completely unsatisfied with this purchase.", "negative"),
            ("Completely useless and overpriced.", "negative"),
            ("Total rip-off, worst product I've ever bought.", "negative"),
            ("Extremely poor quality, avoid at all costs.", "negative"),
            ("Terrible experience, very upset with purchase.", "negative"),
            ("Awful product, complete disappointment.", "negative"),
            ("Worst purchase ever, complete waste of money.", "negative"),
            ("Terrible quality, doesn't work at all.", "negative"),
            ("Very poor product, not as described.", "negative"),
            ("Disappointed with quality, waste of money.", "negative"),
            ("Bad product, poor quality, avoid this.", "negative"),
            ("Terrible experience, would never buy again.", "negative"),
            ("Poor quality product, not worth the money.", "negative"),
            ("Awful product, complete waste of time.", "negative"),
            ("Very bad quality, disappointed with purchase.", "negative"),
            ("Terrible product, poor quality, avoid.", "negative"),
            ("Worst quality ever, complete disappointment.", "negative"),
            ("Bad product, terrible quality, waste of money.", "negative"),
            ("Poor quality, not as advertised, avoid.", "negative"),
            ("Terrible purchase, very disappointed.", "negative"),
            ("Awful quality, waste of money, avoid.", "negative"),
            ("Very poor product, terrible quality.", "negative"),
            ("Bad quality, disappointed, waste of money.", "negative"),
            ("Terrible product, poor quality, avoid.", "negative"),
            ("Worst quality, complete waste, avoid.", "negative"),
            ("Bad product, terrible quality, disappointed.", "negative"),
            ("Poor quality, waste of money, avoid.", "negative"),
            ("Terrible quality, bad product, avoid.", "negative"),
            ("Awful quality, waste of money, avoid.", "negative"),
            ("Very bad quality, terrible product, avoid.", "negative"),
            ("Poor quality, disappointed, waste of money.", "negative"),
            ("Terrible product, bad quality, avoid.", "negative"),
            ("Worst quality, terrible product, avoid.", "negative"),
            ("Bad quality, poor product, waste of money.", "negative"),
            ("Terrible quality, awful product, avoid.", "negative"),
            ("Poor quality, bad product, waste of money.", "negative"),
            ("Very bad quality, poor product, avoid.", "negative"),
            ("Terrible product, worst quality, avoid.", "negative"),
            ("Bad quality, terrible product, waste of money.", "negative"),
            ("Poor quality, awful product, avoid.", "negative"),
            ("Terrible quality, bad product, waste of money.", "negative"),
            
            # Neutral reviews (50 samples)
            ("The product is okay, nothing special.", "neutral"),
            ("Average quality, serves its purpose.", "neutral"),
            ("It's fine, meets basic expectations.", "neutral"),
            ("Standard product, neither good nor bad.", "neutral"),
            ("Acceptable quality for the price.", "neutral"),
            ("It works as expected, no complaints.", "neutral"),
            ("Average experience, nothing remarkable.", "neutral"),
            ("Decent product, could be better.", "neutral"),
            ("It's alright, does what it's supposed to do.", "neutral"),
            ("Moderate quality, acceptable for daily use.", "neutral"),
            ("It's decent enough for what I need.", "neutral"),
            ("Standard fare, nothing to write home about.", "neutral"),
            ("Average product with average performance.", "neutral"),
            ("It's functional, that's about it.", "neutral"),
            ("Mediocre quality, but it works.", "neutral"),
            ("The product is average, nothing exceptional.", "neutral"),
            ("It works fine, nothing special about it.", "neutral"),
            ("Standard quality, meets basic requirements.", "neutral"),
            ("It's okay, does the job adequately.", "neutral"),
            ("Average product, nothing to complain about.", "neutral"),
            ("It's decent, serves its purpose well.", "neutral"),
            ("Standard quality, works as expected.", "neutral"),
            ("It's fine, nothing remarkable but functional.", "neutral"),
            ("Average product, meets expectations.", "neutral"),
            ("It works okay, nothing special.", "neutral"),
            ("Standard quality, adequate for needs.", "neutral"),
            ("It's acceptable, does what it should.", "neutral"),
            ("Average product, nothing outstanding.", "neutral"),
            ("It's fine, works as described.", "neutral"),
            ("Standard quality, meets basic needs.", "neutral"),
            ("It's okay, nothing to write home about.", "neutral"),
            ("Average product, functional and adequate.", "neutral"),
            ("It works fine, standard quality.", "neutral"),
            ("It's decent, meets expectations.", "neutral"),
            ("Standard product, nothing exceptional.", "neutral"),
            ("It's okay, does the job.", "neutral"),
            ("Average quality, works adequately.", "neutral"),
            ("It's fine, nothing special but functional.", "neutral"),
            ("Standard product, meets basic requirements.", "neutral"),
            ("It's acceptable, works as expected.", "neutral"),
            ("Average quality, nothing remarkable.", "neutral"),
            ("It's okay, serves its purpose.", "neutral"),
            ("Standard product, adequate quality.", "neutral"),
            ("It's fine, meets basic needs.", "neutral"),
            ("Average quality, functional product.", "neutral"),
            ("It's decent, works as described.", "neutral"),
            ("Standard product, nothing outstanding.", "neutral"),
            ("It's okay, adequate for the purpose.", "neutral"),
            ("Average quality, meets expectations.", "neutral"),
            ("It's fine, standard product.", "neutral"),
            ("Standard quality, works fine.", "neutral"),
            ("It's okay, nothing special but adequate.", "neutral"),
        ]
        
        return pd.DataFrame(sample_reviews, columns=['review', 'sentiment'])
    
    def train_model(self, df=None, incremental=False):
        """Train the Naive Bayes model on the dataset"""
        if df is None:
            df = self.create_sample_dataset()
        
        # Store the dataset
        self.dataset = df.copy()
        
        # Preprocess the text
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['processed_review']
        y = df['sentiment']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text
        if incremental and self.is_trained:
            # For incremental learning, fit on new data and transform
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
        else:
            # For full retraining, fit the vectorizer
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics
        self.model_metrics = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_size': len(X_test),
            'train_size': len(X_train),
            'total_samples': len(df),
            'last_updated': pd.Timestamp.now().isoformat(),
            'precision': classification_rep['weighted avg']['precision'],
            'recall': classification_rep['weighted avg']['recall'],
            'f1_score': classification_rep['weighted avg']['f1-score']
        }
        
        self.is_trained = True
        
        # Save the model
        self.save_model()
        
        return self.model_metrics
    
    def predict_sentiment(self, text):
        """Enhanced sentiment prediction with confidence analysis"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        processed_text = self.preprocess_text(text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_tfidf)[0]
        probability = self.model.predict_proba(text_tfidf)[0]
        
        # Get class probabilities
        classes = self.model.classes_
        prob_dict = {classes[i]: float(probability[i]) for i in range(len(classes))}
        
        # Calculate confidence and determine reliability
        max_prob = float(max(probability))
        confidence_level = self._calculate_confidence_level(max_prob, probability)

        # Increment prediction counter
        self.total_predictions += 1

        # Persist prediction to MongoDB (or state file as fallback)
        if _mongo_available:
            try:
                db.save_prediction(text, prediction, max_prob, prob_dict)
            except Exception as e:
                print(f"[MongoDB] save_prediction failed: {e}")
        else:
            self._save_state()

        # Add analyzed review to dataset for continuous learning
        self._add_analyzed_review(text, prediction, max_prob)

        return {
            'prediction':       prediction,
            'probabilities':    prob_dict,
            'confidence':       max_prob,
            'confidence_level': confidence_level,
            'reliability': 'high' if max_prob > 0.7 else 'medium' if max_prob > 0.5 else 'low'
        }
    
    def _calculate_confidence_level(self, max_prob, probabilities):
        """Calculate confidence level based on probability distribution"""
        sorted_probs = sorted(probabilities, reverse=True)
        
        if max_prob > 0.8:
            return 'very_high'
        elif max_prob > 0.7:
            return 'high'
        elif max_prob > 0.6:
            return 'medium'
        elif max_prob > 0.5:
            return 'low'
        else:
            return 'very_low'
    
    def _add_analyzed_review(self, text, prediction, confidence):
        """Save every analyzed review to MongoDB (or in-memory list as fallback)."""
        if _mongo_available:
            try:
                # Save ALL reviews to MongoDB dataset collection immediately
                added = db.add_review_to_dataset(text, prediction, source='auto_analyzed')
                if added:
                    # Also update in-memory dataset so stats stay in sync
                    new_row = pd.DataFrame([{'review': text, 'sentiment': prediction}])
                    self.dataset = pd.concat([self.dataset, new_row], ignore_index=True) \
                                   if self.dataset is not None else new_row
            except Exception as e:
                print(f"[MongoDB] _add_analyzed_review failed: {e}")

            # Queue for retraining if high confidence — use DB count so restarts
            # don't reset the counter to zero
            if confidence > 0.7:
                self.analyzed_reviews.append({
                    'text': text, 'sentiment': prediction, 'confidence': confidence
                })
                # Check total unanalyzed count in DB, not just in-memory list
                try:
                    total_since_last = db.get_unretrained_count()
                except Exception:
                    total_since_last = len(self.analyzed_reviews)
                if total_since_last >= 20:
                    self._retrain_with_analyzed_reviews()
        else:
            # File fallback — queue high-confidence reviews for retraining
            if confidence > 0.7:
                self.analyzed_reviews.append({
                    'text': text, 'sentiment': prediction, 'confidence': confidence,
                    'timestamp': pd.Timestamp.now().isoformat(), 'source': 'auto_analyzed'
                })
                if len(self.analyzed_reviews) >= 20:
                    self._retrain_with_analyzed_reviews()

    def _retrain_with_analyzed_reviews(self):
        """Retrain the model with new reviews accumulated since last retrain."""
        if _mongo_available:
            try:
                # Load ALL dataset rows from MongoDB (the source of truth)
                rows = db.load_dataset()
                if not rows:
                    return
                full_df = pd.DataFrame(rows)
                print(f"[Retrain] Retraining on full MongoDB dataset: {len(full_df)} rows")
                self.train_model(full_df, incremental=False)
                # Mark all as trained so the counter resets
                db.mark_all_retrained()
            except Exception as e:
                print(f"[Retrain] MongoDB retrain failed: {e}")
            finally:
                self.analyzed_reviews = []
            return

        # File fallback
        if not self.analyzed_reviews:
            return

        analyzed_df = pd.DataFrame(self.analyzed_reviews)
        new_data = pd.DataFrame({
            'review':    analyzed_df['text'],
            'sentiment': analyzed_df['sentiment']
        })

        if self.dataset is not None:
            existing_texts = set(self.dataset['review'].str.lower())
            new_data_filtered = new_data[~new_data['review'].str.lower().isin(existing_texts)]
            if len(new_data_filtered) > 0:
                combined_df = pd.concat([self.dataset, new_data_filtered], ignore_index=True)
                print(f"[Retrain] Retraining with {len(combined_df)} samples "
                      f"({len(new_data_filtered)} new analyzed reviews)")
                self.train_model(combined_df, incremental=False)

        self.analyzed_reviews = []

    # ── Background-thread retrain helpers ─────────────────────────────────────

    def _do_retrain_bg(self, use_feedback: bool):
        """Actual retrain logic — runs inside a daemon thread."""
        try:
            if use_feedback:
                self.retrain_with_feedback()
            else:
                self._retrain_with_analyzed_reviews()
            self._retrain_status = 'done'
            self._retrain_error  = None
        except Exception as e:
            self._retrain_status = 'error'
            self._retrain_error  = str(e)
            print(f"[Retrain] Background retrain failed: {e}")

    def start_retrain_bg(self, use_feedback: bool = False):
        """
        Kick off retraining in a background thread so the HTTP request
        returns immediately and never times out.
        Returns True if started, False if one is already running.
        """
        if not self._retrain_lock.acquire(blocking=False):
            return False   # already running

        self._retrain_status = 'running'
        self._retrain_error  = None

        def run():
            try:
                self._do_retrain_bg(use_feedback)
            finally:
                self._retrain_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return True

    def save_model(self):
        """Save sklearn pkl to disk AND to MongoDB (so they survive Render restarts)."""
        import json
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Pickle to bytes in memory
        import io
        model_buf = io.BytesIO()
        pickle.dump(self.model, model_buf)
        model_bytes = model_buf.getvalue()

        vec_buf = io.BytesIO()
        pickle.dump(self.vectorizer, vec_buf)
        vec_bytes = vec_buf.getvalue()

        # Write to disk (fast path for same-process reloads)
        with open(os.path.join(MODELS_DIR, 'sentiment_model.pkl'), 'wb') as f:
            f.write(model_bytes)
        with open(os.path.join(MODELS_DIR, 'vectorizer.pkl'), 'wb') as f:
            f.write(vec_bytes)

        if _mongo_available:
            try:
                # Persist pkl blobs to MongoDB so they survive redeploys
                db.save_model_binaries(model_bytes, vec_bytes, np.__version__)
                if self.dataset is not None:
                    db.save_dataset(self.dataset.to_dict('records'))
                db.save_metrics(self.model_metrics)
                print(f"[MongoDB] Saved — dataset: {len(self.dataset) if self.dataset is not None else 0} rows, "
                      f"accuracy: {round(self.model_metrics.get('accuracy', 0)*100, 1)}%")
                return
            except Exception as e:
                print(f"[MongoDB] save_model failed: {e} — falling back to files")

        # File fallback
        if self.dataset is not None:
            self.dataset.to_csv(os.path.join(MODELS_DIR, 'dataset.csv'), index=False)
        state = {
            'model_metrics':      self.model_metrics,
            'total_predictions':  self.total_predictions,
            'user_feedback_data': self.user_feedback_data,
            'analyzed_reviews':   self.analyzed_reviews,
        }
        with open(os.path.join(MODELS_DIR, 'state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        print(f"[File] Saved — dataset: {len(self.dataset) if self.dataset is not None else 0} rows")

    def _save_state(self):
        """Lightweight save: persist counters/metrics without rewriting pkl files."""
        import json
        if _mongo_available:
            try:
                db.save_metrics(self.model_metrics)
                return
            except Exception as e:
                print(f"[MongoDB] _save_state failed: {e} — falling back to file")
        # File fallback
        os.makedirs(MODELS_DIR, exist_ok=True)
        state = {
            'model_metrics':      self.model_metrics,
            'total_predictions':  self.total_predictions,
            'user_feedback_data': self.user_feedback_data,
            'analyzed_reviews':   self.analyzed_reviews,
        }
        with open(os.path.join(MODELS_DIR, 'state.json'), 'w') as f:
            json.dump(state, f, indent=2)

    def load_model(self):
        """Load sklearn pkl from MongoDB (preferred) or disk, then restore all state."""
        import json
        model_path      = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
        vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')

        loaded = False

        # ── 1. Try loading pkl blobs from MongoDB ────────────────────────────
        if _mongo_available:
            try:
                model_bytes, vec_bytes, saved_numpy = db.load_model_binaries()
                if model_bytes and vec_bytes:
                    import io
                    self.model      = pickle.loads(model_bytes)
                    self.vectorizer = pickle.loads(vec_bytes)
                    self.is_trained = True
                    loaded = True
                    print(f"[MongoDB] Model binaries loaded (saved with numpy {saved_numpy})")
            except Exception as e:
                print(f"[MongoDB] load_model_binaries failed: {e} — will try disk")

        # ── 2. Fall back to disk pkl files ───────────────────────────────────
        if not loaded:
            if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
                return False
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self.is_trained = True
                loaded = True
                print("[File] Model loaded from disk pkl files")
            except (ModuleNotFoundError, AttributeError) as e:
                print(f"[Model] Warning: Could not load pickled model ({e}) — will retrain")
                return False
            except Exception as e:
                print(f"[Model] Unexpected error loading pkl: {e} — will retrain")
                return False

        if not loaded:
            return False

        # ── 3. Restore state / dataset ───────────────────────────────────────
        try:
            if _mongo_available:
                try:
                    rows = db.load_dataset()
                    if rows:
                        self.dataset = pd.DataFrame(rows)
                        print(f"[MongoDB] Dataset restored: {len(self.dataset)} rows")
                    else:
                        self.dataset = self.create_sample_dataset()
                        db.save_dataset(self.dataset.to_dict('records'))
                        print("[MongoDB] No dataset found — seeded from sample data")

                    self.model_metrics = db.load_latest_metrics()
                    self.model_metrics.pop('_id', None)
                    self.model_metrics.pop('created_at', None)

                    self.total_predictions  = db.get_total_predictions()
                    self.user_feedback_data = db.load_pending_feedback()

                    print(f"[MongoDB] State restored — accuracy: "
                          f"{round(self.model_metrics.get('accuracy', 0)*100, 1)}%, "
                          f"predictions: {self.total_predictions}, "
                          f"feedback pending: {len(self.user_feedback_data)}")

                    if not self.model_metrics:
                        print("[MongoDB] No metrics found — recomputing...")
                        self.train_model(self.dataset)
                    return True
                except Exception as e:
                    print(f"[MongoDB] load_model state restore failed: {e} — falling back to files")

            # File fallback for state
            dataset_path = os.path.join(MODELS_DIR, 'dataset.csv')
            if os.path.exists(dataset_path):
                self.dataset = pd.read_csv(dataset_path)
                print(f"[File] Dataset restored: {len(self.dataset)} rows")
            else:
                self.dataset = self.create_sample_dataset()
                print("[File] No saved dataset — initialised from sample data")

            state_path = os.path.join(MODELS_DIR, 'state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.model_metrics      = state.get('model_metrics', {})
                self.total_predictions  = state.get('total_predictions', 0)
                self.user_feedback_data = state.get('user_feedback_data', [])
                self.analyzed_reviews   = state.get('analyzed_reviews', [])
                print(f"[File] State restored — accuracy: "
                      f"{round(self.model_metrics.get('accuracy', 0)*100, 1)}%, "
                      f"predictions: {self.total_predictions}")
            else:
                print("[File] No saved state — recomputing metrics...")
                self.train_model(self.dataset)

            return True
        except Exception as e:
            print(f"Error restoring model state: {e}")
            return False

    def add_user_feedback(self, text, predicted_sentiment, actual_sentiment, confidence):
        """Add user feedback to improve the model."""
        feedback_entry = {
            'text':                text,
            'predicted_sentiment': predicted_sentiment,
            'actual_sentiment':    actual_sentiment,
            'confidence':          confidence,
            'timestamp':           pd.Timestamp.now().isoformat()
        }
        self.user_feedback_data.append(feedback_entry)

        # Persist to MongoDB immediately so feedback survives restarts
        if _mongo_available:
            try:
                db.save_feedback(text, predicted_sentiment, actual_sentiment, confidence)
            except Exception as e:
                print(f"[MongoDB] save_feedback failed: {e}")
        else:
            self._save_state()

        if len(self.user_feedback_data) >= self.retrain_threshold:
            self.retrain_with_feedback()

    def retrain_with_feedback(self):
        """Retrain the model using accumulated user feedback."""
        if not self.user_feedback_data and not (_mongo_available and db.get_feedback_count() > 0):
            return

        if _mongo_available:
            try:
                # Load full dataset + pending feedback from MongoDB
                rows = db.load_dataset()
                feedback_rows = db.load_pending_feedback()
                if not rows and not feedback_rows:
                    return
                base_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['review', 'sentiment'])
                fb_df = pd.DataFrame([{
                    'review': f['text'], 'sentiment': f['actual_sentiment']
                } for f in feedback_rows]) if feedback_rows else pd.DataFrame(columns=['review', 'sentiment'])
                combined_df = pd.concat([base_df, fb_df], ignore_index=True).drop_duplicates(subset='review')
                print(f"[Retrain] Feedback retrain on {len(combined_df)} rows "
                      f"({len(fb_df)} from feedback)")
                self.train_model(combined_df, incremental=False)
                db.mark_feedback_used()
                db.mark_all_retrained()
            except Exception as e:
                print(f"[MongoDB] retrain_with_feedback failed: {e}")
            finally:
                self.user_feedback_data = []
            return

        # File fallback
        if not self.user_feedback_data:
            return
        feedback_df = pd.DataFrame(self.user_feedback_data)
        new_data = pd.DataFrame({
            'review':    feedback_df['text'],
            'sentiment': feedback_df['actual_sentiment']
        })
        combined_df = pd.concat([self.dataset, new_data], ignore_index=True) \
                      if self.dataset is not None else new_data
        print(f"[Retrain] Retraining with {len(combined_df)} samples "
              f"({len(new_data)} from feedback)")
        self.train_model(combined_df, incremental=False)
        self.user_feedback_data = []

    
    def batch_predict(self, texts):
        """Predict sentiment for multiple texts"""
        if not self.is_trained:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        results = []
        for text in texts:
            try:
                result = self.predict_sentiment(text)
                results.append({
                    'text': text,
                    'sentiment': result['prediction'],
                    'probabilities': result['probabilities'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                results.append({
                    'text': text,
                    'error': str(e)
                })
        
        return results
    
    def get_dataset_stats(self):
        """Get comprehensive dataset statistics — pulls live counts from MongoDB when connected."""
        if _mongo_available:
            try:
                total          = db.get_dataset_count()
                sentiment_dist = db.get_sentiment_distribution()
                total_preds    = db.get_total_predictions()
                feedback_count = db.get_feedback_count()
                avg_len = self.dataset['review'].str.len().mean() \
                          if self.dataset is not None else 0
                return {
                    'total_reviews':            total,
                    'sentiment_distribution':   sentiment_dist,
                    'average_review_length':    avg_len,
                    'feedback_samples':         feedback_count,
                    'analyzed_reviews_pending': len(self.analyzed_reviews),
                    'last_retrain':             self.model_metrics.get('last_updated', 'Never'),
                    'model_accuracy':           self.model_metrics.get('accuracy', 0),
                    'total_predictions':        total_preds,
                    'precision':                self.model_metrics.get('precision', 0),
                    'recall':                   self.model_metrics.get('recall', 0),
                    'f1_score':                 self.model_metrics.get('f1_score', 0),
                    'storage':                  'mongodb',
                }
            except Exception as e:
                print(f"[MongoDB] get_dataset_stats failed: {e}")

        # File / in-memory fallback
        if self.dataset is None:
            return None
        return {
            'total_reviews':            len(self.dataset),
            'sentiment_distribution':   self.dataset['sentiment'].value_counts().to_dict(),
            'average_review_length':    self.dataset['review'].str.len().mean(),
            'feedback_samples':         len(self.user_feedback_data),
            'analyzed_reviews_pending': len(self.analyzed_reviews),
            'last_retrain':             self.model_metrics.get('last_updated', 'Never'),
            'model_accuracy':           self.model_metrics.get('accuracy', 0),
            'total_predictions':        self.total_predictions,
            'precision':                self.model_metrics.get('precision', 0),
            'recall':                   self.model_metrics.get('recall', 0),
            'f1_score':                 self.model_metrics.get('f1_score', 0),
            'storage':                  'file',
        }

# Initialize the sentiment analyzer
analyzer = SentimentAnalyzer()

# Try to load existing model, otherwise train a new one
if not analyzer.load_model():
    print("Training new model...")
    analyzer.train_model()

@app.route('/', methods=['GET'])
def serve_root():
    """Serve the React frontend index.html"""
    index_path = os.path.join(FRONTEND_BUILD_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_BUILD_DIR, 'index.html')
    return jsonify({
        'status': 'ok',
        'service': 'sentiment-review-analysis',
        'message': 'Backend is running',
        'note': 'Frontend build not found. Run: cd frontend && npm run build'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': analyzer.is_trained,
        'mongodb': 'connected' if _mongo_available else 'not connected (using file storage)'
    })

@app.route('/api/db-status', methods=['GET'])
def db_status():
    """Check MongoDB connection status and collection counts."""
    if not _mongo_available:
        return jsonify({
            'connected': False,
            'message': 'MongoDB not configured. Set MONGO_URI in backend/.env'
        })
    try:
        return jsonify({
            'connected':        True,
            'database':         db.MONGO_DB_NAME,
            'dataset_count':    db.get_dataset_count(),
            'prediction_count': db.get_total_predictions(),
            'feedback_count':   db.get_feedback_count(),
            'sentiment_dist':   db.get_sentiment_distribution(),
        })
    except Exception as e:
        return jsonify({'connected': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train or retrain the model"""
    try:
        data = request.get_json()
        df = None
        
        if data and 'reviews' in data:
            # Use provided dataset
            reviews_data = data['reviews']
            df = pd.DataFrame(reviews_data)
        
        metrics = analyzer.train_model(df)
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_sentiment():
    """Predict sentiment of a review"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        result = analyzer.predict_sentiment(text)
        return jsonify({
            'success': True,
            'text': text,
            'sentiment': result['prediction'],
            'probabilities': result['probabilities'],
            'confidence': result['confidence'],
            'reliability': result.get('reliability', 'medium'),
            'confidence_level': result.get('confidence_level', 'medium')
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    if not analyzer.is_trained:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    return jsonify({
        'success': True,
        'metrics': analyzer.model_metrics
    })

@app.route('/api/sample-dataset', methods=['GET'])
def get_sample_dataset():
    """Get current dataset (including sample data + added reviews)"""
    try:
        if analyzer.dataset is None:
            return jsonify({'error': 'No dataset available'}), 400
        
        # Return the current dataset (which includes sample data + added reviews)
        dataset = analyzer.dataset.to_dict('records')
        return jsonify({
            'dataset': dataset,
            'total_reviews': len(dataset),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/original-sample-dataset', methods=['GET'])
def get_original_sample_dataset():
    """Get original sample dataset for reference"""
    try:
        df = analyzer.create_sample_dataset()
        return jsonify({
            'success': True,
            'dataset': df.to_dict('records'),
            'total_reviews': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for model improvement"""
    try:
        data = request.get_json()
        if not data or not all(key in data for key in ['text', 'predicted_sentiment', 'actual_sentiment', 'confidence']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        analyzer.add_user_feedback(
            data['text'],
            data['predicted_sentiment'],
            data['actual_sentiment'],
            data['confidence']
        )
        
        return jsonify({
            'success': True,
            'message': 'Feedback submitted successfully',
            'feedback_count': len(analyzer.user_feedback_data),
            'retrain_threshold': analyzer.retrain_threshold
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict sentiment for multiple texts"""
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Texts must be a non-empty array'}), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({'error': 'Maximum 100 texts allowed per batch'}), 400
        
        results = analyzer.batch_predict(texts)
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Batch prediction failed'}), 500

@app.route('/api/dataset-stats', methods=['GET'])
def get_dataset_stats():
    """Get comprehensive dataset statistics"""
    try:
        # Ensure dataset is initialized
        if analyzer.dataset is None:
            analyzer.dataset = analyzer.create_sample_dataset()
            print("Dataset initialized for stats")
        
        stats = analyzer.get_dataset_stats()
        if stats is None:
            return jsonify({'error': 'No dataset available'}), 400
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def force_retrain():
    """
    Start model retraining in a background thread.
    Returns 202 Accepted immediately — poll /api/retrain-status to check progress.
    """
    try:
        has_feedback = bool(analyzer.user_feedback_data)
        has_analyzed = bool(analyzer.analyzed_reviews)

        # Check MongoDB directly — in-memory lists reset on restart
        if _mongo_available:
            try:
                has_feedback = has_feedback or db.get_feedback_count() > 0
                has_analyzed = has_analyzed or db.get_dataset_count() > 152
            except Exception:
                pass

        if not has_feedback and not has_analyzed:
            return jsonify({'error': 'No new data available for retraining. '
                                     'Submit some reviews or feedback first.'}), 400

        started = analyzer.start_retrain_bg(use_feedback=has_feedback)
        if not started:
            return jsonify({
                'success': True,
                'status':  'running',
                'message': 'Retraining is already in progress.'
            }), 202

        return jsonify({
            'success': True,
            'status':  'running',
            'message': 'Retraining started in background. Poll /api/retrain-status for updates.'
        }), 202

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain-status', methods=['GET'])
def retrain_status():
    """Poll retraining progress."""
    status = analyzer._retrain_status   # 'idle' | 'running' | 'done' | 'error'
    resp = {
        'success': True,
        'status':  status,
    }
    if status == 'done':
        resp['metrics'] = analyzer.model_metrics
        analyzer._retrain_status = 'idle'   # reset so next poll is clean
    elif status == 'error':
        resp['error'] = analyzer._retrain_error
        analyzer._retrain_status = 'idle'
    return jsonify(resp)

@app.route('/api/retrain-analyzed', methods=['POST'])
def retrain_with_analyzed():
    """Force retraining with analyzed reviews"""
    try:
        if not analyzer.analyzed_reviews:
            return jsonify({'error': 'No analyzed reviews available for retraining'}), 400
        
        analyzer._retrain_with_analyzed_reviews()
        return jsonify({
            'success': True,
            'message': 'Model retrained with analyzed reviews successfully',
            'metrics': analyzer.model_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-dataset', methods=['GET'])
def export_dataset():
    """Export current dataset as CSV"""
    try:
        if analyzer.dataset is None:
            return jsonify({'error': 'No dataset available'}), 400
        
        # Create CSV string
        csv_data = analyzer.dataset.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'total_records': len(analyzer.dataset)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Catch-all route to serve index.html for React Router
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend files or fallback to index.html for React Router"""
    # Don't handle API routes here - they're handled by explicit routes above
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    
    file_path = os.path.join(FRONTEND_BUILD_DIR, path)
    
    # If the file exists in the build directory, serve it
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(FRONTEND_BUILD_DIR, path)
    
    # Otherwise, serve index.html for React Router to handle
    index_path = os.path.join(FRONTEND_BUILD_DIR, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_BUILD_DIR, 'index.html')
    
    # If build doesn't exist, return 404
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
