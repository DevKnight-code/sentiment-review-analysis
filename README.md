
# Sentiment Analysis of Product Reviews using Naive Bayes

This project implements a comprehensive sentiment analysis system for product reviews using Natural Language Processing (NLP) techniques and the Naive Bayes machine learning algorithm. The system consists of a Python Flask backend for the AI model and a React.js frontend for the user interface.

## 🚀 Features

- **Sentiment Analysis**: Classify product reviews as positive, negative, or neutral
- **NLP Preprocessing**: Tokenization, stopword removal, stemming, and TF-IDF vectorization
- **Naive Bayes Classifier**: Machine learning model trained on product review data
- **Real-time Prediction**: Live sentiment analysis with confidence scores
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and confusion matrix
- **Modern UI**: Beautiful and responsive React.js interface
- **Sample Dataset**: Built-in dataset for demonstration and training

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework for API
- **scikit-learn**: Machine learning library for Naive Bayes
- **NLTK**: Natural language processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Frontend
- **React.js 18**: Modern UI framework
- **React Router**: Navigation
- **Axios**: HTTP client for API calls
- **Lucide React**: Beautiful icons
- **CSS3**: Modern styling with animations

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn package manager

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd PROJECT-PRODUCTREVIEW
```

### 2. Backend Setup

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### Run the Backend Server
```bash
cd backend
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Frontend Setup

#### Install Node Dependencies
```bash
cd frontend
npm install
```

#### Start the Frontend Development Server
```bash
npm start
```

The frontend will start on `http://localhost:3000`

## 🎯 Usage

### 1. Sentiment Analysis
- Navigate to the home page (`/`)
- Enter a product review in the text area
- Click "Analyze Sentiment" to get the prediction
- View the sentiment classification with confidence scores

### 2. Model Metrics
- Go to the Metrics page (`/metrics`)
- View model performance statistics
- Check accuracy, precision, recall, and F1-scores
- Examine the confusion matrix

### 3. Dataset Viewer
- Visit the Dataset page (`/dataset`)
- Browse the sample dataset used for training
- View sentiment distribution and statistics

## 🔬 Technical Details

### NLP Pipeline
1. **Text Preprocessing**:
   - Convert to lowercase
   - Remove special characters and digits
   - Tokenization using NLTK
   - Stopword removal
   - Porter stemming

2. **Feature Extraction**:
   - TF-IDF vectorization with 5000 features
   - English stopwords filtering

3. **Model Training**:
   - Multinomial Naive Bayes classifier
   - 80/20 train-test split
   - Stratified sampling for balanced classes

### API Endpoints

- `GET /api/health` - Health check
- `POST /api/predict` - Predict sentiment of text
- `GET /api/metrics` - Get model performance metrics
- `GET /api/sample-dataset` - Get sample dataset
- `POST /api/train` - Train/retrain the model

### Model Performance
- **Accuracy**: Typically 85-90% on test data
- **Classes**: Positive, Negative, Neutral
- **Features**: TF-IDF vectors (5000 dimensions)
- **Algorithm**: Multinomial Naive Bayes

## 📊 Sample Data

The system includes a sample dataset with:
- 50+ product reviews
- Balanced distribution across sentiment classes
- Real-world review examples
- Pre-labeled sentiment annotations

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern Interface**: Clean and intuitive design
- **Real-time Analysis**: Instant sentiment prediction
- **Visual Feedback**: Progress bars, animations, and icons
- **Comprehensive Metrics**: Detailed model performance visualization

## 🔧 Customization

### Adding Custom Dataset
1. Prepare your dataset in CSV format with columns: `review`, `sentiment`
2. Modify the `create_sample_dataset()` method in `backend/app.py`
3. Retrain the model using the `/api/train` endpoint

### Adjusting Model Parameters
- Modify `max_features` in TF-IDF vectorizer
- Adjust train-test split ratio
- Experiment with different preprocessing techniques

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**:
   - Check Python version (3.8+ required)
   - Install all dependencies: `pip install -r requirements.txt`
   - Ensure port 5000 is available

2. **Frontend not connecting to backend**:
   - Verify backend is running on port 5000
   - Check CORS settings
   - Ensure API_BASE_URL is correct

3. **Model training issues**:
   - Check NLTK data downloads
   - Verify dataset format
   - Ensure sufficient training data

## 📈 Future Enhancements

- [ ] Support for multiple languages
- [ ] Real-time model retraining
- [ ] Batch processing of multiple reviews
- [ ] Advanced visualization charts
- [ ] Model comparison tools
- [ ] Export functionality for results
- [ ] User authentication and history
- [ ] Integration with external APIs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- NLTK community for excellent NLP tools
- scikit-learn team for machine learning algorithms
- React.js community for the amazing frontend framework
- Flask community for the lightweight web framework

---

**Note**: This project is for educational purposes and demonstrates the application of machine learning and natural language processing techniques for sentiment analysis.

