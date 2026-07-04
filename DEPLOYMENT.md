# Deployment Guide - Render

This project is configured for deployment on [Render](https://render.com), a modern cloud platform.

## Quick Start Deployment

### Prerequisites
- GitHub account with repository pushed
- Render account (free tier available)

### Step 1: Connect GitHub to Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Select your repository

### Step 2: Configure Services

Render will automatically detect `render.yaml` and set up both services:
- **Backend**: Python Flask API on `https://sentiment-backend.onrender.com`
- **Frontend**: React static site on `https://sentiment-frontend.onrender.com`

### Step 3: Deploy

1. Render will automatically build and deploy both services
2. Monitor deployment progress in the Render dashboard
3. Once complete, your app is live!

## Environment Configuration

### Backend Environment Variables
- `FLASK_ENV`: Set to `production` (configured in render.yaml)
- `PYTHON_VERSION`: Python 3.11 (configured in render.yaml)
- `MONGO_URI` (optional): MongoDB connection string for persistence

### Adding MongoDB (Optional)

To enable database persistence:

1. Create a MongoDB cluster (free tier at [mongodb.com/cloud](https://mongodb.com/cloud))
2. Get connection string
3. In Render dashboard:
   - Go to your backend service
   - Environment tab
   - Add variable: `MONGO_URI=<your-connection-string>`
   - Redeploy service

## Local Development

### Development Mode
```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2: Frontend
cd frontend
npm install
npm start
```

The frontend proxy is set to `http://localhost:5000` in development.

### Production Mode (Simulate Render locally)
```bash
# Backend with gunicorn
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:5000

# Frontend production build
npm run build
npm install -g serve
serve -s frontend/build -l 3000
```

## Troubleshooting

### Backend shows "Internal Server Error"
- Check backend logs in Render dashboard
- Verify NLTK data is downloading correctly
- Ensure all dependencies in requirements.txt are correct

### Frontend can't connect to API
- Check `REACT_APP_API_URL` environment variable
- Ensure backend URL is correct in frontend service config
- Check browser console for CORS errors

### Free tier auto-sleep
- Render free tier instances auto-sleep after 15 minutes of inactivity
- Upgrade to paid plan for 24/7 uptime
- Set up monitoring to keep instance active

## Deployment Logs

View logs in Render dashboard:
1. Select service
2. Click "Logs" tab
3. Check for errors

## Performance Optimization

- Backend uses 4 gunicorn workers for better concurrency
- Frontend build is optimized and minified
- Static files served efficiently from CDN

## Security

- CORS configured to accept only frontend domain
- Environment variables kept secure in Render dashboard
- HTTPS enforced automatically

## Next Steps

1. **Custom Domain**: Add your domain in Render → Settings → Domains
2. **Auto-deployment**: Push to GitHub to auto-redeploy
3. **Monitoring**: Set up alerts for deployment failures
4. **Backups**: If using MongoDB, enable automated backups

---

Need help? Check the README.md for feature documentation.
