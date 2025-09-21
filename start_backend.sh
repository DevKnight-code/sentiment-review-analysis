#!/bin/bash
echo "Starting Sentiment Analysis Backend..."
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo ""
echo "Starting Flask server..."
cd backend
python app.py
