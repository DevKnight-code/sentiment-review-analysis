# Deployment Guide - Render

This project is configured for deployment on [Render](https://render.com), a modern cloud platform.

## Quick Start Deployment

### Prerequisites
- GitHub account with repository pushed
- Render account (free tier available at render.com)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Blueprint"**
3. Select **"Connect a Git repository"** and choose your repo
4. Click **"Connect"**

### Step 3: Automatic Deployment

Render will automatically:
- Read `render.yaml` configuration
- Deploy **Backend** (Python Flask API) → `https://sentiment-backend.onrender.com`
- Deploy **Frontend** (React static site) → `https://sentiment-frontend.onrender.com`
- Build and start both services

## Environment Configuration

### Backend Environment Variables (Auto-configured)
- `FLASK_ENV=production`
- `PYTHON_VERSION=3.11`
- `PYTHONUNBUFFERED=1`
- `MONGO_URI` (optional - set in Render dashboard if using MongoDB)

### Adding MongoDB (Optional)

To enable persistent database storage:

1. Create MongoDB cluster at [mongodb.com/cloud](https://mongodb.com/cloud) (Atlas free tier)
2. Get connection string (mongodb+srv://...)
3. In Render Dashboard:
   - Select **sentiment-backend** service
   - Go to **Environment**
   - Add variable: `MONGO_URI=<your-mongodb-connection-string>`
   - Redeploy service

## Accessing Your App

After deployment:
- **API Backend**: `https://sentiment-backend.onrender.com/api`
- **Frontend**: `https://sentiment-frontend.onrender.com`
- **Health Check**: `https://sentiment-backend.onrender.com/api/health`

## Frontend API Configuration

The frontend automatically connects to the backend API using relative paths:
```javascript
// App.js uses:
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
```

On Render, the frontend makes API calls to the backend using the deployed URL.

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

The frontend will be at `http://localhost:3000` and connects to backend at `http://localhost:5000`.

### Production Mode (Simulate Render)
```bash
# Backend
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:5000

# Frontend (in separate terminal)
cd frontend
npm run build
npm install -g serve
serve -s build -l 3000
```

## Troubleshooting

### Backend shows "Module not found" error
1. Check backend build logs in Render dashboard
2. Verify all dependencies in `requirements.txt` are correct
3. Check NLTK data is downloading (check logs for `[NLTK]` messages)
4. Ensure `runtime.txt` specifies Python 3.11

### Frontend shows "Cannot GET /" error
1. Verify `npm run build` succeeded
2. Check `staticPublishPath` in `render.yaml` is set to `frontend/build`
3. Ensure `package.json` scripts are correct

### Frontend can't connect to API
1. Check browser console for CORS errors
2. Verify backend service is running (check backend logs)
3. Ensure backend CORS is configured correctly
4. Backend URL should be fully qualified (https://sentiment-backend.onrender.com/api)

### "Cannot find module" during deployment
1. All dependencies listed in `requirements.txt` and `package.json`
2. Clear build cache: Delete service and redeploy
3. Check for typos in import statements

## Free Tier Limitations

- **Auto-sleep**: Free tier services auto-sleep after 15 minutes of inactivity
- To prevent sleep, upgrade to paid tier or use [Koyeb](https://koyeb.com) or [Railway](https://railway.app)
- First request after sleep may be slow (30+ seconds)

## Performance Tips

- Backend uses 4 gunicorn workers for concurrency
- Frontend is minified and optimized
- Use `runtime.txt` to lock Python version
- Monitor service resource usage in Render dashboard

## Security Considerations

- ✅ CORS properly configured to accept only frontend domain
- ✅ HTTPS enforced automatically by Render
- ✅ Environment variables secure in dashboard (not in code)
- ✅ Add `MONGO_URI` only to backend environment variables

## Logs & Debugging

View logs in Render dashboard:
1. Select service → **Logs** tab
2. Look for errors starting with `[ERROR]` or `Traceback`
3. Check for `[MongoDB]` connection messages
4. Watch for `[NLTK]` download warnings

## Redeployment

Changes automatically deploy when pushed to main branch:
```bash
git add .
git commit -m "Fix: Update endpoint"
git push origin main
```

Render will auto-build and deploy within 1-2 minutes.

## Monitoring & Updates

- Check service status: Dashboard → Service → Events
- View resource usage: Dashboard → Metrics
- Update dependencies regularly to fix security issues

---

Need help? Check [Render Docs](https://render.com/docs) or review the README.md for feature documentation.
