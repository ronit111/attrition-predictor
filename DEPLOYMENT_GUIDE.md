# 🚀 Deployment Guide

## Quick Deploy to Streamlit Cloud (Free Hosting)

### Step 1: Create GitHub Repository

1. **Go to GitHub** and create a new repository:
   - Visit https://github.com/new
   - Repository name: `attrition-predictor` (or your preferred name)
   - Description: "AI-powered employee attrition risk predictor"
   - Set to **Public** (required for free Streamlit Cloud hosting)
   - **Do NOT** initialize with README (we already have one)
   - Click "Create repository"

2. **Push your local code to GitHub**:
   ```bash
   cd ~/Projects/attrition-predictor
   git remote add origin https://github.com/YOUR_USERNAME/attrition-predictor.git
   git branch -M main
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io
   - Click "Sign in" and authenticate with your GitHub account

2. **Create new app**:
   - Click "New app" button
   - Choose your repository: `YOUR_USERNAME/attrition-predictor`
   - Set branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy!"

3. **Wait for deployment** (2-3 minutes):
   - Streamlit Cloud will install dependencies
   - The first deployment includes model training (this is normal)
   - You'll see the build logs in real-time

4. **Get your shareable link**:
   - Once deployed, you'll get a URL like: `https://attrition-predictor-yourname.streamlit.app`
   - This link is ready to share with your network!

### Step 3: Custom Domain (Optional)

If you want a custom subdomain:

1. Go to app settings in Streamlit Cloud
2. Click on "Settings" → "General"
3. Set custom URL: `your-preferred-name.streamlit.app`

## 📱 Sharing Your App

Once deployed, you can:

- **Share the URL** directly with colleagues and network
- **Embed** in presentations or documentation
- **Post on social media** (LinkedIn, Twitter, etc.)
- **Add to your portfolio** or resume

Example share text:
```
Check out my new Employee Attrition Risk Predictor!
It uses AI to identify employees at risk of leaving,
with explainable insights and actionable recommendations.

🔗 https://your-app.streamlit.app

Built with #Python #MachineLearning #Streamlit #XGBoost
```

## 🔧 Troubleshooting

### Issue: Model not found error on Streamlit Cloud

**Solution**: The models need to be trained after deployment. Add this to your app.py before loading the model:

```python
import os
if not os.path.exists('models/attrition_model.pkl'):
    import subprocess
    subprocess.run(['python', 'train_model.py'])
```

### Issue: Memory error during deployment

**Solution**: Streamlit Cloud free tier has memory limits. To optimize:

1. Reduce SHAP sample size in train_model.py:
   ```python
   shap_values = model.compute_shap_values(X_test.head(50))  # Reduced from 100
   ```

2. Use lighter dependencies in requirements.txt if needed

### Issue: App is slow to load

**Solution**:
- First load always takes longer (model loading)
- Subsequent loads are cached and much faster
- Consider upgrading to Streamlit Cloud Pro for better performance

## 🎨 Customization

### Change theme colors

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"  # Change this
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f9fafb"
```

### Update app title and icon

In `app.py`:
```python
st.set_page_config(
    page_title="Your Custom Title",
    page_icon="🎯",  # Change icon
    layout="wide"
)
```

## 📊 Analytics (Optional)

To track usage, add Google Analytics:

1. Create a `.streamlit/secrets.toml` file (local only, not in git):
   ```toml
   [google_analytics]
   tracking_id = "G-XXXXXXXXXX"
   ```

2. Add to app.py:
   ```python
   import streamlit.components.v1 as components

   # Google Analytics
   GA_TRACKING_ID = st.secrets.get("google_analytics", {}).get("tracking_id", "")
   if GA_TRACKING_ID:
       components.html(f"""
       <!-- Google Analytics -->
       <script async src="https://www.googletagmanager.com/gtag/js?id={GA_TRACKING_ID}"></script>
       <script>
           window.dataLayer = window.dataLayer || [];
           function gtag(){{dataLayer.push(arguments);}}
           gtag('js', new Date());
           gtag('config', '{GA_TRACKING_ID}');
       </script>
       """, height=0)
   ```

## 🔄 Updating Your App

After making changes locally:

```bash
cd ~/Projects/attrition-predictor
git add .
git commit -m "Description of your changes"
git push
```

Streamlit Cloud will automatically detect the push and redeploy your app!

## 💡 Pro Tips

1. **Test locally first**: Always test changes with `streamlit run app.py` before pushing

2. **Use branches**: Create a `dev` branch for testing:
   ```bash
   git checkout -b dev
   # Make changes
   git push origin dev
   # Deploy dev branch separately on Streamlit Cloud for testing
   ```

3. **Monitor logs**: Check Streamlit Cloud logs regularly for errors

4. **Backup your work**: Keep local copies and regular git commits

## 📈 Scaling Options

If you need more resources:

1. **Streamlit Cloud Pro**: $20/month for better resources
2. **Heroku**: Free tier or paid plans
3. **AWS/GCP**: Full control but requires more setup
4. **Docker + Cloud Run**: Containerized deployment

## 🎉 You're Done!

Congratulations! Your app is now live and shareable.

Next steps:
- Share with your network
- Gather feedback
- Iterate and improve
- Add to your portfolio

---

Need help? Create an issue on GitHub or reach out!
