import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Brain, BarChart3, MessageSquare, Upload, TrendingUp, Users, FileText, Download, RefreshCw, ThumbsUp, ThumbsDown } from 'lucide-react';
import axios from 'axios';
import './App.css';

// API configuration
const API_BASE_URL = 'http://localhost:5000/api';

// Global state for metrics refresh
let globalMetricsRefresh = 0;
const triggerMetricsRefresh = () => {
  globalMetricsRefresh += 1;
  // Dispatch custom event to notify components
  window.dispatchEvent(new CustomEvent('metricsRefresh', { detail: globalMetricsRefresh }));
};

// Components
const Navbar = () => {
  const location = useLocation();
  
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <Link to="/" className="navbar-brand">
          <Brain size={24} style={{ marginRight: '8px' }} />
          Sentiment Analysis
        </Link>
        <div className="navbar-nav">
          <Link 
            to="/" 
            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
          >
            <MessageSquare size={16} />
            Analyze
          </Link>
          <Link 
            to="/batch" 
            className={`nav-link ${location.pathname === '/batch' ? 'active' : ''}`}
          >
            <FileText size={16} />
            Batch
          </Link>
          <Link 
            to="/metrics" 
            className={`nav-link ${location.pathname === '/metrics' ? 'active' : ''}`}
          >
            <BarChart3 size={16} />
            Metrics
          </Link>
          <Link 
            to="/dataset" 
            className={`nav-link ${location.pathname === '/dataset' ? 'active' : ''}`}
          >
            <Upload size={16} />
            Dataset
          </Link>
        </div>
      </div>
    </nav>
  );
};

const SentimentAnalyzer = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackSentiment, setFeedbackSentiment] = useState('');
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        text: text.trim()
      });
      
      setResult(response.data);
      // Trigger global metrics update
      triggerMetricsRefresh();
      
      // Show notification if review was added to dataset
      if (response.data.reliability === 'high') {
        setTimeout(() => {
          alert('This high-confidence review has been automatically added to the training dataset!');
        }, 1000);
      }
    } catch (err) {
      setError('Failed to analyze sentiment. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'sentiment-positive';
      case 'negative': return 'sentiment-negative';
      case 'neutral': return 'sentiment-neutral';
      default: return '';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '😊';
      case 'negative': return '😞';
      case 'neutral': return '😐';
      default: return '';
    }
  };

  const submitFeedback = async () => {
    if (!result) {
      setError('No prediction to provide feedback on');
      return;
    }

    // If feedbackSentiment is empty, it means user clicked "Wrong Prediction"
    if (!feedbackSentiment) {
      setError('Please select the correct sentiment for wrong predictions');
      return;
    }

    setFeedbackLoading(true);
    try {
      await axios.post(`${API_BASE_URL}/feedback`, {
        text: text,
        predicted_sentiment: result.sentiment,
        actual_sentiment: feedbackSentiment,
        confidence: result.confidence
      });
      
      setShowFeedback(false);
      setFeedbackSentiment('');
      setError('');
      // Trigger metrics update after feedback
      triggerMetricsRefresh();
      alert('Thank you for your feedback! This will help improve the model.');
    } catch (err) {
      setError('Failed to submit feedback');
    } finally {
      setFeedbackLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Product Review Sentiment Analysis</h2>
        <p style={{ color: '#6c757d', marginBottom: '20px' }}>
          Enter a product review below to analyze its sentiment using our trained Naive Bayes model.
        </p>
        
        <div className="input-group">
          <label htmlFor="review-text">Review Text</label>
          <textarea
            id="review-text"
            rows="6"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter your product review here..."
            style={{ resize: 'vertical' }}
          />
        </div>

        <button 
          className="btn btn-primary" 
          onClick={analyzeSentiment}
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner"></div>
              Analyzing...
            </>
          ) : (
            <>
              <TrendingUp size={16} />
              Analyze Sentiment
            </>
          )}
        </button>

        {error && (
          <div style={{ 
            background: '#f8d7da', 
            color: '#721c24', 
            padding: '12px', 
            borderRadius: '6px', 
            marginTop: '15px' 
          }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{ marginTop: '30px' }}>
            <h3>Analysis Result</h3>
            <div style={{ 
              background: '#f8f9fa', 
              padding: '20px', 
              borderRadius: '8px', 
              marginTop: '15px' 
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                <span style={{ fontSize: '1.5rem' }}>
                  {getSentimentIcon(result.sentiment)}
                </span>
                <span className={getSentimentClass(result.sentiment)}>
                  {result.sentiment.toUpperCase()}
                </span>
                <span style={{ color: '#6c757d' }}>
                  (Confidence: {(result.confidence * 100).toFixed(1)}%)
                </span>
                {result.reliability && (
                  <span style={{ 
                    padding: '2px 8px', 
                    borderRadius: '12px', 
                    fontSize: '0.8rem',
                    fontWeight: '600',
                    background: result.reliability === 'high' ? '#d4edda' : 
                               result.reliability === 'medium' ? '#fff3cd' : '#f8d7da',
                    color: result.reliability === 'high' ? '#155724' : 
                           result.reliability === 'medium' ? '#856404' : '#721c24'
                  }}>
                    {result.reliability.toUpperCase()} RELIABILITY
                  </span>
                )}
              </div>
              
              <div>
                <h4>Confidence Breakdown:</h4>
                <div style={{ display: 'flex', gap: '15px', marginTop: '10px' }}>
                  {Object.entries(result.probabilities).map(([sentiment, prob]) => (
                    <div key={sentiment} style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                        <span className={getSentimentClass(sentiment)}>
                          {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                        </span>
                        <span>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill" 
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Feedback Section */}
              <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid #dee2e6' }}>
                <h4>Was this prediction correct?</h4>
                <p style={{ color: '#6c757d', fontSize: '0.9rem', marginBottom: '15px' }}>
                  Help improve the model by providing feedback on this prediction.
                </p>
                
                {!showFeedback ? (
                  <button 
                    className="btn btn-secondary" 
                    onClick={() => setShowFeedback(true)}
                    style={{ marginRight: '10px' }}
                  >
                    <ThumbsUp size={16} />
                    Provide Feedback
                  </button>
                ) : (
                  <div>
                    <div style={{ marginBottom: '15px' }}>
                      <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600' }}>
                        What is the correct sentiment?
                      </label>
                      <div style={{ display: 'flex', gap: '10px' }}>
                        {['positive', 'negative', 'neutral'].map(sentiment => (
                          <button
                            key={sentiment}
                            className={`btn ${feedbackSentiment === sentiment ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setFeedbackSentiment(sentiment)}
                            style={{ flex: 1 }}
                          >
                            {getSentimentIcon(sentiment)} {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
                          </button>
                        ))}
                      </div>
                      
                      {/* Quick feedback buttons */}
                      <div style={{ marginTop: '10px' }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '600' }}>
                          Quick Feedback:
                        </label>
                        <div style={{ display: 'flex', gap: '10px' }}>
                          <button
                            className="btn btn-success"
                            onClick={() => setFeedbackSentiment(result.sentiment)}
                            style={{ flex: 1 }}
                          >
                            <ThumbsUp size={16} />
                            Correct Prediction
                          </button>
                          <button
                            className="btn btn-danger"
                            onClick={() => setFeedbackSentiment('')}
                            style={{ flex: 1 }}
                          >
                            <ThumbsDown size={16} />
                            Wrong Prediction
                          </button>
                        </div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '10px' }}>
                      <button 
                        className="btn btn-success" 
                        onClick={submitFeedback}
                        disabled={feedbackLoading || !feedbackSentiment}
                      >
                        {feedbackLoading ? (
                          <>
                            <div className="spinner"></div>
                            Submitting...
                          </>
                        ) : (
                          <>
                            <ThumbsUp size={16} />
                            Submit Feedback
                          </>
                        )}
                      </button>
                      <button 
                        className="btn btn-secondary" 
                        onClick={() => {
                          setShowFeedback(false);
                          setFeedbackSentiment('');
                        }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const BatchAnalyzer = () => {
  const [texts, setTexts] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeBatch = async () => {
    if (!texts.trim()) {
      setError('Please enter some texts to analyze');
      return;
    }

    const textArray = texts.split('\n').filter(text => text.trim());
    if (textArray.length === 0) {
      setError('Please enter at least one text to analyze');
      return;
    }

    if (textArray.length > 100) {
      setError('Maximum 100 texts allowed per batch');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/batch-predict`, {
        texts: textArray
      });
      
      setResults(response.data.results);
    } catch (err) {
      setError('Failed to analyze texts. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'sentiment-positive';
      case 'negative': return 'sentiment-negative';
      case 'neutral': return 'sentiment-neutral';
      default: return '';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive': return '😊';
      case 'negative': return '😞';
      case 'neutral': return '😐';
      default: return '';
    }
  };

  const exportResults = () => {
    if (results.length === 0) return;
    
    const csvContent = [
      'Text,Sentiment,Confidence,Positive%,Negative%,Neutral%',
      ...results.map(result => 
        `"${result.text.replace(/"/g, '""')}",${result.sentiment},${(result.confidence * 100).toFixed(1)}%,${(result.probabilities.positive * 100).toFixed(1)}%,${(result.probabilities.negative * 100).toFixed(1)}%,${(result.probabilities.neutral * 100).toFixed(1)}%`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentiment_analysis_results.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Batch Sentiment Analysis</h2>
        <p style={{ color: '#6c757d', marginBottom: '20px' }}>
          Analyze multiple product reviews at once. Enter each review on a new line.
        </p>
        
        <div className="input-group">
          <label htmlFor="batch-texts">Reviews (one per line)</label>
          <textarea
            id="batch-texts"
            rows="10"
            value={texts}
            onChange={(e) => setTexts(e.target.value)}
            placeholder="Enter your product reviews here, one per line...&#10;&#10;Example:&#10;This product is amazing!&#10;Terrible quality, don't buy.&#10;It's okay, nothing special."
            style={{ resize: 'vertical' }}
          />
        </div>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <button 
            className="btn btn-primary" 
            onClick={analyzeBatch}
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp size={16} />
                Analyze Batch
              </>
            )}
          </button>
          
          {results.length > 0 && (
            <button 
              className="btn btn-secondary" 
              onClick={exportResults}
            >
              <Download size={16} />
              Export Results
            </button>
          )}
        </div>

        {error && (
          <div style={{ 
            background: '#f8d7da', 
            color: '#721c24', 
            padding: '12px', 
            borderRadius: '6px', 
            marginBottom: '20px' 
          }}>
            {error}
          </div>
        )}

        {results.length > 0 && (
          <div>
            <h3>Analysis Results ({results.length} reviews)</h3>
            <div style={{ 
              background: '#f8f9fa', 
              padding: '20px', 
              borderRadius: '8px', 
              marginTop: '15px',
              maxHeight: '600px',
              overflow: 'auto'
            }}>
              <div style={{ display: 'grid', gap: '15px' }}>
                {results.map((result, index) => (
                  <div 
                    key={index}
                    style={{ 
                      background: 'white', 
                      padding: '15px', 
                      borderRadius: '6px',
                      border: '1px solid #dee2e6'
                    }}
                  >
                    <div style={{ marginBottom: '10px' }}>
                      <span className={getSentimentClass(result.sentiment)}>
                        {getSentimentIcon(result.sentiment)} {result.sentiment.toUpperCase()}
                      </span>
                      <span style={{ color: '#6c757d', marginLeft: '10px' }}>
                        (Confidence: {(result.confidence * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div style={{ fontSize: '0.95rem', lineHeight: '1.5', marginBottom: '10px' }}>
                      "{result.text}"
                    </div>
                    <div style={{ display: 'flex', gap: '10px', fontSize: '0.85rem' }}>
                      {Object.entries(result.probabilities).map(([sentiment, prob]) => (
                        <span key={sentiment} style={{ color: '#6c757d' }}>
                          {sentiment}: {(prob * 100).toFixed(1)}%
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ModelMetrics = () => {
  const [metrics, setMetrics] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retraining, setRetraining] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    fetchMetrics();
    fetchDatasetStats();
  }, [refreshTrigger]);

  useEffect(() => {
    const handleMetricsRefresh = (event) => {
      setRefreshTrigger(event.detail);
    };

    window.addEventListener('metricsRefresh', handleMetricsRefresh);
    return () => window.removeEventListener('metricsRefresh', handleMetricsRefresh);
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/metrics`);
      setMetrics(response.data.metrics);
    } catch (err) {
      setError('Failed to load metrics');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchDatasetStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/dataset-stats`);
      setDatasetStats(response.data.stats);
    } catch (err) {
      console.error('Error fetching dataset stats:', err);
    }
  };

  const forceRetrain = async () => {
    setRetraining(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/retrain`);
      setMetrics(response.data.metrics);
      await fetchDatasetStats();
      alert('Model retrained successfully!');
    } catch (err) {
      alert('Failed to retrain model: ' + (err.response?.data?.error || err.message));
    } finally {
      setRetraining(false);
    }
  };

  const exportDataset = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/export-dataset`);
      const blob = new Blob([response.data.csv_data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'sentiment_dataset.csv';
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to export dataset');
    }
  };

  if (loading) {
    return (
      <div className="container">
        <div className="card">
          <div className="loading">
            <div className="spinner"></div>
            Loading metrics...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="card">
          <div style={{ color: '#dc3545' }}>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h2>Model Performance Metrics</h2>
            <p style={{ color: '#6c757d', margin: '0' }}>
              Performance evaluation of the Naive Bayes sentiment analysis model.
            </p>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              className="btn btn-secondary" 
              onClick={exportDataset}
              disabled={!datasetStats}
            >
              <Download size={16} />
              Export Dataset
            </button>
            <button 
              className="btn btn-primary" 
              onClick={forceRetrain}
              disabled={retraining || (!datasetStats?.feedback_samples && !datasetStats?.analyzed_reviews_pending)}
            >
              {retraining ? (
                <>
                  <div className="spinner"></div>
                  Retraining...
                </>
              ) : (
                <>
                  <RefreshCw size={16} />
                  Retrain Model
                </>
              )}
            </button>
          </div>
        </div>

        {/* Dataset Statistics */}
        {datasetStats && (
          <div style={{ marginBottom: '30px' }}>
            <h3>Dataset Statistics</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-value">{datasetStats.total_reviews}</div>
                <div className="metric-label">Total Reviews</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-value">{datasetStats.feedback_samples}</div>
                <div className="metric-label">Feedback Samples</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-value">{datasetStats.average_review_length?.toFixed(0)}</div>
                <div className="metric-label">Avg Review Length</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-value">{(datasetStats.model_accuracy * 100).toFixed(1)}%</div>
                <div className="metric-label">Current Accuracy</div>
              </div>

              <div className="metric-card">
                <div className="metric-value">{(datasetStats.precision * 100).toFixed(1)}%</div>
                <div className="metric-label">Precision</div>
              </div>

              <div className="metric-card">
                <div className="metric-value">{(datasetStats.recall * 100).toFixed(1)}%</div>
                <div className="metric-label">Recall</div>
              </div>

              <div className="metric-card">
                <div className="metric-value">{(datasetStats.f1_score * 100).toFixed(1)}%</div>
                <div className="metric-label">F1-Score</div>
              </div>

              <div className="metric-card">
                <div className="metric-value">{datasetStats.total_predictions || 0}</div>
                <div className="metric-label">Total Predictions</div>
              </div>

              <div className="metric-card">
                <div className="metric-value">{datasetStats.analyzed_reviews_pending || 0}</div>
                <div className="metric-label">Reviews Pending Addition</div>
              </div>
            </div>

            {/* Sentiment Distribution */}
            <div style={{ marginTop: '20px' }}>
              <h4>Sentiment Distribution</h4>
              <div style={{ display: 'flex', gap: '20px', marginTop: '10px' }}>
                {Object.entries(datasetStats.sentiment_distribution).map(([sentiment, count]) => (
                  <div key={sentiment} style={{ flex: 1, textAlign: 'center' }}>
                    <div style={{ 
                      background: sentiment === 'positive' ? '#d4edda' : 
                                 sentiment === 'negative' ? '#f8d7da' : '#e2e3e5',
                      padding: '15px',
                      borderRadius: '8px',
                      border: `2px solid ${sentiment === 'positive' ? '#28a745' : 
                                               sentiment === 'negative' ? '#dc3545' : '#6c757d'}`
                    }}>
                      <div style={{ 
                        fontSize: '1.5rem', 
                        fontWeight: 'bold',
                        color: sentiment === 'positive' ? '#28a745' : 
                               sentiment === 'negative' ? '#dc3545' : '#6c757d'
                      }}>
                        {count}
                      </div>
                      <div style={{ 
                        textTransform: 'capitalize',
                        fontWeight: '600',
                        color: sentiment === 'positive' ? '#28a745' : 
                               sentiment === 'negative' ? '#dc3545' : '#6c757d'
                      }}>
                        {sentiment}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {datasetStats.last_retrain && (
              <div style={{ 
                marginTop: '15px', 
                padding: '10px', 
                background: '#e7f3ff', 
                borderRadius: '6px',
                fontSize: '0.9rem',
                color: '#004085'
              }}>
                <strong>Last Retrain:</strong> {new Date(datasetStats.last_retrain).toLocaleString()}
              </div>
            )}
          </div>
        )}

        {/* Model Performance Metrics */}
        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-value">
              {(metrics.accuracy * 100).toFixed(1)}%
            </div>
            <div className="metric-label">Accuracy</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-value">{metrics.train_size}</div>
            <div className="metric-label">Training Samples</div>
          </div>
          
          <div className="metric-card">
            <div className="metric-value">{metrics.test_size}</div>
            <div className="metric-label">Test Samples</div>
          </div>

          <div className="metric-card">
            <div className="metric-value">{metrics.total_samples}</div>
            <div className="metric-label">Total Samples</div>
          </div>
        </div>

        {metrics.classification_report && (
          <div style={{ marginTop: '30px' }}>
            <h3>Detailed Classification Report</h3>
            <div style={{ 
              background: '#f8f9fa', 
              padding: '20px', 
              borderRadius: '8px', 
              marginTop: '15px',
              overflow: 'auto'
            }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#e9ecef' }}>
                    <th style={{ padding: '10px', textAlign: 'left' }}>Class</th>
                    <th style={{ padding: '10px', textAlign: 'right' }}>Precision</th>
                    <th style={{ padding: '10px', textAlign: 'right' }}>Recall</th>
                    <th style={{ padding: '10px', textAlign: 'right' }}>F1-Score</th>
                    <th style={{ padding: '10px', textAlign: 'right' }}>Support</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.classification_report).map(([key, value]) => {
                    if (key === 'accuracy' || key === 'macro avg' || key === 'weighted avg') return null;
                    return (
                      <tr key={key} style={{ borderBottom: '1px solid #dee2e6' }}>
                        <td style={{ padding: '10px', fontWeight: '600' }}>
                          {key.charAt(0).toUpperCase() + key.slice(1)}
                        </td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>
                          {typeof value === 'object' ? value.precision?.toFixed(3) : 'N/A'}
                        </td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>
                          {typeof value === 'object' ? value.recall?.toFixed(3) : 'N/A'}
                        </td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>
                          {typeof value === 'object' ? value['f1-score']?.toFixed(3) : 'N/A'}
                        </td>
                        <td style={{ padding: '10px', textAlign: 'right' }}>
                          {typeof value === 'object' ? value.support : 'N/A'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {metrics.confusion_matrix && (
          <div style={{ marginTop: '30px' }}>
            <h3>Confusion Matrix</h3>
            <div style={{ 
              background: '#f8f9fa', 
              padding: '20px', 
              borderRadius: '8px', 
              marginTop: '15px',
              display: 'flex',
              justifyContent: 'center'
            }}>
              <div style={{ textAlign: 'center' }}>
                <table style={{ borderCollapse: 'collapse' }}>
                  <tbody>
                    {metrics.confusion_matrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => (
                          <td 
                            key={j}
                            style={{
                              padding: '15px',
                              border: '1px solid #dee2e6',
                              background: i === j ? '#d4edda' : '#f8d7da',
                              fontWeight: 'bold',
                              fontSize: '1.2rem'
                            }}
                          >
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ marginTop: '10px', fontSize: '0.9rem', color: '#6c757d' }}>
                  Rows: Actual, Columns: Predicted
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const DatasetViewer = () => {
  const [dataset, setDataset] = useState([]);
  const [originalDataset, setOriginalDataset] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [datasetStats, setDatasetStats] = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);

  useEffect(() => {
    fetchDataset();
  }, [refreshTrigger]); // Re-fetch when refreshTrigger changes

  useEffect(() => {
    const handleMetricsRefresh = (event) => {
      setRefreshTrigger(event.detail);
    };

    window.addEventListener('metricsRefresh', handleMetricsRefresh);
    return () => window.removeEventListener('metricsRefresh', handleMetricsRefresh);
  }, []); // Only attach event listener once

  const fetchDataset = async () => {
    try {
      setLoading(true);
      // Fetch current dataset, original sample dataset, and current dataset stats
      const [currentResponse, originalResponse, statsResponse] = await Promise.all([
        axios.get(`${API_BASE_URL}/sample-dataset`), // This now returns current dataset
        axios.get(`${API_BASE_URL}/original-sample-dataset`), // This returns original sample
        axios.get(`${API_BASE_URL}/dataset-stats`)
      ]);
      
      // Set the current dataset (includes added reviews)
      setDataset(currentResponse.data.dataset);
      // Set the original sample dataset for reference
      setOriginalDataset(originalResponse.data.dataset);
      // Store the current dataset statistics
      setDatasetStats(statsResponse.data.stats);
      setError('');
    } catch (err) {
      setError('Failed to load dataset');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentClass = (sentiment) => {
    switch (sentiment) {
      case 'positive': return 'sentiment-positive';
      case 'negative': return 'sentiment-negative';
      case 'neutral': return 'sentiment-neutral';
      default: return '';
    }
  };

  if (loading) {
    return (
      <div className="container">
        <div className="card">
          <div className="loading">
            <div className="spinner"></div>
            Loading dataset...
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="card">
          <div style={{ color: '#dc3545' }}>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h2>Dataset Viewer</h2>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '5px', alignItems: 'center' }}>
              <label style={{ fontSize: '14px', fontWeight: '500' }}>
                <input 
                  type="radio" 
                  name="datasetView" 
                  checked={!showOriginal}
                  onChange={() => setShowOriginal(false)}
                  style={{ marginRight: '5px' }}
                />
                Current Dataset
              </label>
              <label style={{ fontSize: '14px', fontWeight: '500' }}>
                <input 
                  type="radio" 
                  name="datasetView" 
                  checked={showOriginal}
                  onChange={() => setShowOriginal(true)}
                  style={{ marginRight: '5px' }}
                />
                Original Sample
              </label>
            </div>
            <button 
              className="btn btn-secondary" 
              onClick={fetchDataset}
              style={{ padding: '8px 16px' }}
            >
              🔄 Refresh
            </button>
          </div>
        </div>
        <p style={{ color: '#6c757d', marginBottom: '20px' }}>
          {showOriginal ? (
            <>
              This shows the <strong>original sample dataset</strong> used to initially train the sentiment analysis model. 
              This dataset contains the base product reviews with their corresponding sentiment labels.
            </>
          ) : (
            <>
              This shows the <strong>current dataset</strong> including the original sample data plus all reviews that have been 
              automatically added through analysis and user feedback. This is the actual dataset the model is currently using.
            </>
          )}
        </p>

        <div style={{ 
          background: '#e3f2fd', 
          padding: '15px', 
          borderRadius: '8px',
          marginBottom: '20px',
          border: '1px solid #bbdefb'
        }}>
          <strong>📊 {showOriginal ? 'Original Sample' : 'Current'} Dataset Statistics:</strong> {showOriginal ? originalDataset.length : (datasetStats ? datasetStats.total_reviews : dataset.length)} total reviews
          {showOriginal ? (
            <span style={{ marginLeft: '20px' }}>
              • Positive: {originalDataset.filter(item => item.sentiment === 'positive').length} 
              • Negative: {originalDataset.filter(item => item.sentiment === 'negative').length} 
              • Neutral: {originalDataset.filter(item => item.sentiment === 'neutral').length}
            </span>
          ) : datasetStats ? (
            <span style={{ marginLeft: '20px' }}>
              • Positive: {datasetStats.sentiment_distribution.positive} 
              • Negative: {datasetStats.sentiment_distribution.negative} 
              • Neutral: {datasetStats.sentiment_distribution.neutral}
              {datasetStats.analyzed_reviews_pending > 0 && (
                <span style={{ color: '#ff9800', fontWeight: 'bold' }}>
                  • Pending Addition: {datasetStats.analyzed_reviews_pending}
                </span>
              )}
            </span>
          ) : dataset.length > 0 && (
            <span style={{ marginLeft: '20px' }}>
              • Positive: {dataset.filter(item => item.sentiment === 'positive').length} 
              • Negative: {dataset.filter(item => item.sentiment === 'negative').length} 
              • Neutral: {dataset.filter(item => item.sentiment === 'neutral').length}
            </span>
          )}
        </div>

        <div style={{ 
          background: '#f8f9fa', 
          padding: '20px', 
          borderRadius: '8px',
          maxHeight: '600px',
          overflow: 'auto'
        }}>
          <div style={{ display: 'grid', gap: '15px' }}>
            {(showOriginal ? originalDataset : dataset).map((item, index) => (
              <div 
                key={index}
                style={{ 
                  background: 'white', 
                  padding: '15px', 
                  borderRadius: '6px',
                  border: '1px solid #dee2e6'
                }}
              >
                <div style={{ marginBottom: '10px' }}>
                  <span className={getSentimentClass(item.sentiment)}>
                    {item.sentiment.toUpperCase()}
                  </span>
                </div>
                <div style={{ fontSize: '0.95rem', lineHeight: '1.5' }}>
                  "{item.review}"
                </div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          background: '#e7f3ff', 
          borderRadius: '6px',
          border: '1px solid #b8daff'
        }}>
          <strong>Dataset Statistics:</strong>
          <ul style={{ marginTop: '10px', paddingLeft: '20px' }}>
            <li>Total Reviews: {dataset.length}</li>
            <li>Positive Reviews: {dataset.filter(item => item.sentiment === 'positive').length}</li>
            <li>Negative Reviews: {dataset.filter(item => item.sentiment === 'negative').length}</li>
            <li>Neutral Reviews: {dataset.filter(item => item.sentiment === 'neutral').length}</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

const App = () => {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<SentimentAnalyzer />} />
          <Route path="/batch" element={<BatchAnalyzer />} />
          <Route path="/metrics" element={<ModelMetrics />} />
          <Route path="/dataset" element={<DatasetViewer />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
