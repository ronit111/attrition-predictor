# 👔 Employee Attrition Risk Predictor

A premium, AI-powered web application for predicting employee attrition risk with explainable insights. Built with Streamlit, XGBoost, and SHAP.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **🎯 Accurate Predictions**: XGBoost model with 85% accuracy and 0.789 ROC-AUC score
- **🔍 Explainable AI**: SHAP values explain which factors drive each prediction
- **💎 Premium UI**: Modern, responsive design that works beautifully on all devices
- **📊 Interactive Visualizations**: Plotly charts and animated risk gauges
- **🚀 Easy to Use**: Simple form interface with sample data for quick testing
- **☁️ Cloud-Ready**: Deployable to Streamlit Cloud in minutes

## 🖼️ Screenshots

### Home Page
Clean, welcoming interface with clear call-to-actions

### Prediction Page
Interactive form with real-time risk assessment and visual feedback

### Risk Analysis
Animated risk gauge with detailed factor breakdown

### Model Insights
Feature importance and performance metrics dashboard

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/attrition-predictor.git
cd attrition-predictor
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model** (Required for first-time setup)
```bash
python train_model.py
```

This will:
- Load and preprocess the IBM HR Analytics dataset
- Train the XGBoost model with SMOTE for class balancing
- Generate SHAP explainability values
- Save the trained model to `models/` directory

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
attrition-predictor/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── data/                       # Dataset storage
│   └── HR-Employee-Attrition.csv
│
├── models/                     # Trained models (generated after training)
│   ├── attrition_model.pkl
│   └── data_processor.pkl
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_processing.py      # Data preprocessing and feature engineering
│   └── model.py                # Model training and prediction
│
├── assets/                     # Frontend assets
│   ├── style.css               # Custom CSS styling
│   └── images/                 # Screenshots and images
│
└── .streamlit/                 # Streamlit configuration
    └── config.toml             # App theme and settings
```

## 🎯 How It Works

### 1. Data Processing
- Loads IBM HR Analytics dataset with 1,470 employees and 35 features
- Engineers new features (tenure groups, work-life balance scores, etc.)
- Handles categorical variables with label encoding
- Scales numerical features using StandardScaler

### 2. Model Training
- Uses XGBoost Classifier for high performance
- Applies SMOTE to handle class imbalance (16% attrition rate)
- Cross-validates to prevent overfitting
- Achieves 85% accuracy and 0.789 ROC-AUC

### 3. Prediction & Explanation
- Takes employee data as input
- Generates risk probability (0-100%)
- Uses SHAP to explain which factors increase/decrease risk
- Provides actionable recommendations

## 🎨 Key Features Explained

### Premium UI/UX
- **Modern Design**: Glassmorphism effects, smooth gradients, professional typography
- **Responsive**: Mobile-first design that adapts to tablets and desktops
- **Intuitive**: Progressive disclosure - show advanced options only when needed
- **Delightful**: Smooth animations, clear visual hierarchy, generous whitespace

### Machine Learning
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Features**: 30+ employee attributes including demographics, job details, satisfaction scores
- **Explainability**: SHAP (SHapley Additive exPlanations) for interpretable predictions
- **Performance**: Balanced precision and recall with F1-score optimization

### Risk Assessment
- **Low Risk (0-30%)**: Green indicator, maintenance recommendations
- **Medium Risk (30-60%)**: Yellow indicator, preventive measures
- **High Risk (60-100%)**: Red indicator, immediate action items

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 85% |
| ROC-AUC | 0.789 |
| Precision | 60% |
| Recall | 26% |

### Top Predictive Features
1. Job Level
2. Stock Option Level
3. Tenure Group
4. Overtime
5. Marital Status
6. Department
7. Performance Rating
8. Work-Life Balance

## 🌐 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. **Push your code to GitHub**
```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/attrition-predictor.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Share your app**
   - You'll get a URL like: `attrition-predictor-yourname.streamlit.app`
   - Share this link with your network!

### Alternative Deployment Options
- **Heroku**: Use the included `Procfile` and `setup.sh`
- **Docker**: Create a Dockerfile for containerized deployment
- **AWS/GCP**: Deploy on cloud platforms with custom VMs

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit, HTML, CSS |
| **Backend** | Python 3.9+ |
| **ML Framework** | XGBoost, scikit-learn |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Data Processing** | pandas, NumPy |
| **Class Balancing** | imbalanced-learn (SMOTE) |

## 📖 Usage Guide

### For HR Professionals

1. **Navigate to the Predict Risk page**
2. **Enter employee information**
   - Fill in basic details (age, department, income, etc.)
   - Optionally expand "Additional Details" for more accuracy
3. **Click "Predict Attrition Risk"**
4. **Review results**
   - Check the risk gauge (Low/Medium/High)
   - Examine top contributing factors
   - Read tailored recommendations
5. **Take action**
   - Schedule meetings with high-risk employees
   - Implement suggested interventions
   - Monitor progress over time

### For Data Scientists

1. **Retrain the model** with new data:
```python
from src.data_processing import DataProcessor
from src.model import AttritionModel

processor = DataProcessor('path/to/your/data.csv')
X_train, X_test, y_train, y_test, df = processor.prepare_for_training()

model = AttritionModel()
model.train(X_train, y_train)
model.save_model('models/attrition_model.pkl')
```

2. **Experiment with hyperparameters** in `src/model.py`
3. **Add new features** in `src/data_processing.py`
4. **Evaluate on custom metrics** using `model.evaluate()`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **IBM HR Analytics Dataset**: Thank you to IBM for providing the sample dataset
- **Streamlit**: For the amazing framework that makes data apps so easy to build
- **SHAP**: For bringing interpretability to machine learning
- **XGBoost**: For the powerful gradient boosting algorithm

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- Create an issue on GitHub
- Connect with me on LinkedIn

## 🚀 What's Next?

Future enhancements:
- [ ] Batch prediction from CSV upload
- [ ] Historical trend analysis
- [ ] Integration with HRIS systems
- [ ] Custom model training interface
- [ ] A/B testing framework for interventions
- [ ] Email alerts for high-risk employees
- [ ] Multi-language support

---

**Made with ❤️ using Streamlit and XGBoost**

⭐ Star this repo if you find it useful!
