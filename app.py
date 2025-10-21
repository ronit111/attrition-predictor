"""
Employee Attrition Risk Predictor
A premium web application for predicting employee attrition risk
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Attrition Risk Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path("assets/style.css")
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Load model and processor
@st.cache_resource
def load_model_and_processor():
    """Load model and processor, train if needed"""
    import os
    import subprocess

    model_path = 'models/attrition_model.pkl'
    processor_path = 'models/data_processor.pkl'

    # Check if models exist
    if not os.path.exists(model_path) or not os.path.exists(processor_path):
        st.warning("🔄 Models not found. Training new model... (this will take ~1 minute)")
        with st.spinner("Training in progress..."):
            try:
                import sys
                python_exec = sys.executable
                result = subprocess.run([python_exec, 'train_model_simple.py'],
                                      check=True, capture_output=True, text=True)
                st.code(result.stdout)
                st.success("✅ Model trained successfully!")
            except subprocess.CalledProcessError as e:
                st.error(f"Training failed: {e.stderr}")
                raise
            except Exception as e:
                st.error(f"Failed to train model: {str(e)}")
                raise

    # Try to load models
    try:
        model_data = joblib.load(model_path)
        processor = joblib.load(processor_path)
        return model_data, processor
    except Exception as e:
        # If loading fails (compatibility issue), retrain
        st.warning(f"⚠️ Model compatibility issue detected")
        st.info(f"Error: {str(e)}")
        st.info("Retraining model now... (~1 minute). This is normal on first deployment.")

        with st.spinner("Training in progress..."):
            try:
                import sys
                python_exec = sys.executable
                st.write(f"Using Python: {python_exec}")
                result = subprocess.run([python_exec, 'train_model_simple.py'],
                                      check=True, capture_output=True, text=True,
                                      cwd=os.getcwd())
                st.code(result.stdout)

                # Reload models
                model_data = joblib.load(model_path)
                processor = joblib.load(processor_path)
                st.success("✅ Model retrained and loaded successfully!")
                st.info("Refreshing app...")
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Training failed!")
                st.error(f"Return code: {e.returncode}")
                st.error(f"STDOUT: {e.stdout}")
                st.error(f"STDERR: {e.stderr}")
                raise
            except Exception as retrain_error:
                st.error(f"Failed to retrain: {str(retrain_error)}")
                import traceback
                st.code(traceback.format_exc())
                raise

# Sample employee data for demo
def get_sample_employee():
    return {
        'Age': 35,
        'BusinessTravel': 'Travel_Rarely',
        'DailyRate': 800,
        'Department': 'Research & Development',
        'DistanceFromHome': 10,
        'Education': 3,
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 3,
        'Gender': 'Male',
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': 2,
        'JobRole': 'Research Scientist',
        'JobSatisfaction': 4,
        'MaritalStatus': 'Married',
        'MonthlyIncome': 5000,
        'MonthlyRate': 14000,
        'NumCompaniesWorked': 1,
        'OverTime': 'No',
        'PercentSalaryHike': 13,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': 10,
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': 3,
        'YearsAtCompany': 8,
        'YearsInCurrentRole': 7,
        'YearsSinceLastPromotion': 1,
        'YearsWithCurrManager': 7
    }

# Create risk gauge chart
def create_risk_gauge(risk_score):
    """Create an animated risk gauge using Plotly"""
    risk_score_percent = risk_score * 100

    # Determine color based on risk
    if risk_score < 0.3:
        color = "#10b981"  # Green
        risk_level = "LOW RISK"
    elif risk_score < 0.6:
        color = "#f59e0b"  # Orange
        risk_level = "MEDIUM RISK"
    else:
        color = "#ef4444"  # Red
        risk_level = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': risk_level, 'font': {'size': 24, 'color': color}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score_percent
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif"}
    )

    return fig

# Create SHAP waterfall plot
def create_shap_plot(shap_values, feature_names):
    """Create a SHAP waterfall plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    return fig

# Create feature importance chart
def create_feature_importance_chart(feature_importance_df, top_n=10):
    """Create an interactive feature importance chart"""
    top_features = feature_importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        font={'family': "Inter, sans-serif"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig

# Main app
def main():
    # Hero section
    st.markdown("""
    <div class='hero-section'>
        <h1>👔 Employee Attrition Risk Predictor</h1>
        <p>Predict employee turnover risk with AI-powered insights. Make data-driven retention decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        model_data, processor = load_model_and_processor()
        model_obj = model_data['model']
        feature_importance = model_data['feature_importance']
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run `python train_model.py` first to train the model.")
        return

    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "",
            ["🏠 Home", "🔮 Predict Risk", "📊 Model Insights"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses machine learning to predict employee attrition risk based on various factors.

        **Model:** XGBoost Classifier
        **Accuracy:** ~85%
        **ROC-AUC:** 0.789
        """)

    # Home page
    if page == "🏠 Home":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("## Welcome! 👋")
            st.markdown("""
            ### What is this tool?
            This application helps HR professionals and managers identify employees at risk of leaving the organization.
            Using advanced machine learning algorithms, it analyzes various employee factors to predict attrition probability.

            ### How it works
            1. **Input employee data** - Provide information about the employee
            2. **Get instant prediction** - Our AI model calculates the risk score
            3. **Understand the factors** - See which factors contribute most to the risk

            ### Why use it?
            - 🎯 **Early Detection** - Identify at-risk employees before they leave
            - 💰 **Cost Savings** - Reduce recruitment and training costs
            - 📈 **Better Retention** - Make data-driven retention decisions
            - 🤝 **Improved Culture** - Address issues proactively
            """)

            st.markdown("### 🚀 Quick Start")
            if st.button("Try with Sample Data", type="primary", use_container_width=True):
                st.session_state['use_sample'] = True
                st.session_state['page'] = "🔮 Predict Risk"
                st.rerun()

        with col2:
            st.markdown("## 📊 Quick Stats")

            # Display model performance metrics
            st.metric("Model Accuracy", "85%", help="Overall prediction accuracy")
            st.metric("ROC-AUC Score", "0.789", help="Area under the ROC curve")
            st.metric("Features Analyzed", "30+", help="Number of employee attributes considered")

            st.markdown("---")
            st.markdown("### 🎨 Key Features")
            st.markdown("""
            ✓ Real-time predictions
            ✓ Risk factor analysis
            ✓ Interactive visualizations
            ✓ Explainable AI insights
            ✓ Mobile-friendly design
            """)

    # Prediction page
    elif page == "🔮 Predict Risk":
        st.markdown("## 🔮 Predict Attrition Risk")

        # Check if sample data should be used
        use_sample = st.session_state.get('use_sample', False)

        if use_sample:
            st.info("📝 Using sample employee data. You can modify the values below.")
            st.session_state['use_sample'] = False

        sample_data = get_sample_employee() if use_sample else {}

        # Create form
        with st.form("prediction_form"):
            st.markdown("### Employee Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 👤 Personal")
                age = st.number_input("Age", min_value=18, max_value=70, value=sample_data.get('Age', 30))
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if sample_data.get('Gender', 'Male') == 'Male' else 1)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"],
                                              index=["Single", "Married", "Divorced"].index(sample_data.get('MaritalStatus', 'Single')))
                distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=100,
                                                      value=sample_data.get('DistanceFromHome', 10))

            with col2:
                st.markdown("#### 💼 Job Details")
                department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"],
                                          index=["Research & Development", "Sales", "Human Resources"].index(
                                              sample_data.get('Department', 'Research & Development')))
                job_role = st.selectbox("Job Role", ["Research Scientist", "Sales Executive", "Manager", "Laboratory Technician",
                                                      "Manufacturing Director", "Healthcare Representative", "Research Director"],
                                        index=0)
                job_level = st.slider("Job Level", 1, 5, sample_data.get('JobLevel', 2))
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000,
                                                 value=sample_data.get('MonthlyIncome', 5000), step=100)

            with col3:
                st.markdown("#### 📈 Experience")
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40,
                                                      value=sample_data.get('TotalWorkingYears', 10))
                years_at_company = st.number_input("Years at Company", min_value=0, max_value=40,
                                                   value=sample_data.get('YearsAtCompany', 5))
                years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=40,
                                                        value=sample_data.get('YearsInCurrentRole', 3))
                years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15,
                                                             value=sample_data.get('YearsSinceLastPromotion', 1))

            with st.expander("⚙️ Additional Details (Optional)", expanded=False):
                col4, col5, col6 = st.columns(3)

                with col4:
                    education = st.slider("Education Level", 1, 5, sample_data.get('Education', 3),
                                         help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
                    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"],
                                                   index=0)
                    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
                                                   index=0)

                with col5:
                    job_satisfaction = st.slider("Job Satisfaction", 1, 4, sample_data.get('JobSatisfaction', 3))
                    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, sample_data.get('EnvironmentSatisfaction', 3))
                    work_life_balance = st.slider("Work Life Balance", 1, 4, sample_data.get('WorkLifeBalance', 3))

                with col6:
                    overtime = st.selectbox("Works Overtime", ["Yes", "No"], index=1)
                    stock_option_level = st.slider("Stock Option Level", 0, 3, sample_data.get('StockOptionLevel', 1))
                    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10,
                                                           value=sample_data.get('NumCompaniesWorked', 1))

            submit_button = st.form_submit_button("🔮 Predict Attrition Risk", type="primary", use_container_width=True)

        if submit_button:
            # Prepare input data
            input_data = {
                'Age': age,
                'BusinessTravel': business_travel,
                'DailyRate': 800,  # Default values for fields not in form
                'Department': department,
                'DistanceFromHome': distance_from_home,
                'Education': education,
                'EducationField': education_field,
                'EnvironmentSatisfaction': environment_satisfaction,
                'Gender': gender,
                'HourlyRate': 65,
                'JobInvolvement': 3,
                'JobLevel': job_level,
                'JobRole': job_role,
                'JobSatisfaction': job_satisfaction,
                'MaritalStatus': marital_status,
                'MonthlyIncome': monthly_income,
                'MonthlyRate': monthly_income * 12 / 12,
                'NumCompaniesWorked': num_companies_worked,
                'OverTime': overtime,
                'PercentSalaryHike': 13,
                'PerformanceRating': 3,
                'RelationshipSatisfaction': 3,
                'StockOptionLevel': stock_option_level,
                'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': 3,
                'WorkLifeBalance': work_life_balance,
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_current_role,
                'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': years_in_current_role
            }

            # Make prediction
            try:
                # Prepare data
                X = processor.prepare_single_prediction(input_data)

                # Predict
                prediction = model_obj.predict(X)[0]
                probability = model_obj.predict_proba(X)[0][1]

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")

                # Risk gauge
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = create_risk_gauge(probability)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 🎯 Key Metrics")
                    st.metric("Attrition Risk", f"{probability * 100:.1f}%")
                    st.metric("Confidence", f"{max(1 - probability, probability) * 100:.1f}%")

                    if probability < 0.3:
                        st.success("✅ Low risk of attrition")
                    elif probability < 0.6:
                        st.warning("⚠️ Medium risk of attrition")
                    else:
                        st.error("🚨 High risk of attrition")

                # SHAP explanation
                st.markdown("### 🔍 Risk Factor Analysis")

                # Compute SHAP values
                from src.model import AttritionModel
                model_wrapper = AttritionModel()
                model_wrapper.model = model_obj
                shap_values = model_wrapper.get_prediction_explanation(X)

                # Get top risk factors
                risk_factors = model_wrapper.get_top_risk_factors(shap_values, X.columns, top_n=5)

                st.markdown("#### Top 5 Contributing Factors")

                for idx, row in risk_factors.iterrows():
                    impact = row['impact']
                    feature = row['feature']

                    if impact > 0:
                        st.markdown(f"🔴 **{feature}**: Increases risk (+{abs(impact):.3f})")
                    else:
                        st.markdown(f"🟢 **{feature}**: Decreases risk ({impact:.3f})")

                # SHAP waterfall plot
                with st.expander("📈 Detailed SHAP Explanation", expanded=False):
                    st.markdown("This chart shows how each feature contributes to the prediction:")
                    fig_shap = create_shap_plot(shap_values, X.columns)
                    st.pyplot(fig_shap)

                # Recommendations
                st.markdown("### 💡 Recommendations")

                if probability >= 0.6:
                    st.markdown("""
                    **Immediate Actions:**
                    - 🎯 Schedule a one-on-one meeting to discuss career goals
                    - 💰 Review compensation and benefits package
                    - 📈 Create a personalized development plan
                    - 🤝 Improve work-life balance initiatives
                    - 🏆 Recognize and reward contributions
                    """)
                elif probability >= 0.3:
                    st.markdown("""
                    **Preventive Measures:**
                    - 💬 Regular check-ins to understand satisfaction levels
                    - 📊 Monitor workload and provide support
                    - 🎓 Offer training and development opportunities
                    - 🌟 Ensure clear career progression path
                    """)
                else:
                    st.markdown("""
                    **Maintenance Actions:**
                    - ✅ Continue current engagement practices
                    - 📋 Periodic satisfaction surveys
                    - 🎉 Celebrate achievements and milestones
                    - 💪 Support continuous professional growth
                    """)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)

    # Insights page
    elif page == "📊 Model Insights":
        st.markdown("## 📊 Model Performance & Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Model Metrics")
            st.metric("Accuracy", "85%", help="Percentage of correct predictions")
            st.metric("ROC-AUC Score", "0.789", help="Model's ability to distinguish between classes")
            st.metric("Precision", "60%", help="Accuracy of positive predictions")
            st.metric("Recall", "26%", help="Ability to find all positive cases")

        with col2:
            st.markdown("### 📈 Model Information")
            st.markdown("""
            **Algorithm:** XGBoost Classifier
            **Training Data:** IBM HR Analytics Dataset
            **Features:** 30+ employee attributes
            **Class Balance:** SMOTE oversampling
            **Explainability:** SHAP values
            """)

        # Feature importance
        st.markdown("### 🔑 Feature Importance")
        st.markdown("These are the most important factors in predicting employee attrition:")

        fig = create_feature_importance_chart(feature_importance, top_n=15)
        st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.markdown("### 💡 Key Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Factors that Increase Attrition Risk:
            - 🔴 Low job level
            - 🔴 Frequent overtime
            - 🔴 Low stock options
            - 🔴 Recent hire (< 2 years)
            - 🔴 Low job satisfaction
            """)

        with col2:
            st.markdown("""
            #### Factors that Decrease Attrition Risk:
            - 🟢 High job level
            - 🟢 Good work-life balance
            - 🟢 Stock options available
            - 🟢 Long tenure
            - 🟢 High job satisfaction
            """)

if __name__ == "__main__":
    main()
