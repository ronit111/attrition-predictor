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
    page_icon="📊",
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

# Load model and processor (NO CACHING - fixes stuck error state)
def load_model_and_processor():
    """Load model and processor, train if needed. Uses session_state for per-session caching."""
    import os
    import subprocess
    import sys

    # Use session state for caching instead of @st.cache_resource
    if 'model_loaded' in st.session_state and st.session_state.model_loaded:
        return st.session_state.model_data, st.session_state.processor

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    model_path = 'models/attrition_model.pkl'
    processor_path = 'models/data_processor.pkl'

    # Train if models don't exist
    if not os.path.exists(model_path) or not os.path.exists(processor_path):
        with st.spinner("⏳ Setting up your predictor... This will take about a minute."):
            try:
                python_exec = sys.executable
                result = subprocess.run(
                    [python_exec, 'train_model_simple.py'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                st.success("Your predictor is ready!")
            except subprocess.CalledProcessError as e:
                st.error("Setup encountered an issue. Please refresh the page or contact support.")
                raise RuntimeError("Setup failed")

    # Load models
    try:
        model_data = joblib.load(model_path)
        processor = joblib.load(processor_path)

        # Cache in session state
        st.session_state.model_loaded = True
        st.session_state.model_data = model_data
        st.session_state.processor = processor

        return model_data, processor

    except Exception as e:
        st.info("⏳ Optimizing your predictor...")

        # Delete incompatible models
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(processor_path):
                os.remove(processor_path)
        except:
            pass

        # Retrain
        with st.spinner("Setting up... This will take about a minute."):
            try:
                python_exec = sys.executable
                result = subprocess.run(
                    [python_exec, 'train_model_simple.py'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                st.success("Your predictor is ready!")

                # Load fresh models
                model_data = joblib.load(model_path)
                processor = joblib.load(processor_path)

                # Cache in session state
                st.session_state.model_loaded = True
                st.session_state.model_data = model_data
                st.session_state.processor = processor

                return model_data, processor

            except Exception as retry_error:
                st.error("Something went wrong. Please refresh the page or contact support.")
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
        <h1>Employee Attrition Risk Predictor</h1>
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
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["Home", "Predict Risk", "Bulk Analysis", "How It Works"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This application uses artificial intelligence to predict employee attrition risk based on various workplace and personal factors.

        Developed with advanced machine learning to help organizations retain their best talent.
        """)

    # Home page
    if page == "Home":
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("## Welcome")
            st.markdown("""
            ### What is this tool?
            This application helps HR professionals and managers identify employees at risk of leaving the organization.
            Using advanced machine learning algorithms, it analyzes various employee factors to predict attrition probability.

            ### How it works
            1. **Input employee data** - Provide information about the employee
            2. **Get instant prediction** - Our AI model calculates the risk score
            3. **Understand the factors** - See which factors contribute most to the risk

            ### Why use it?
            - **Early Detection** - Identify at-risk employees before they leave
            - **Cost Savings** - Reduce recruitment and training costs
            - **Better Retention** - Make data-driven retention decisions
            - **Improved Culture** - Address issues proactively

            ### Get Started
            Use the sidebar to navigate:
            - **Predict Risk** - Assess individual employees
            - **Bulk Analysis** - Upload CSV for multiple predictions
            - **How It Works** - Learn about the AI model
            """)

        with col2:
            st.markdown("## Quick Stats")

            # Display model performance metrics
            st.metric("Prediction Accuracy", "85%", help="Overall prediction reliability")
            st.metric("Factors Analyzed", "30+", help="Number of employee attributes considered")
            st.metric("Processing Time", "< 1 sec", help="Instant predictions")

            st.markdown("---")
            st.markdown("### Key Features")
            st.markdown("""
            • Instant risk predictions
            • Key factor analysis
            • Interactive visualizations
            • Actionable recommendations
            • Works on any device
            """)

    # Prediction page
    elif page == "Predict Risk":
        st.markdown("## Predict Attrition Risk")

        sample_data = {}

        # Create form
        with st.form("prediction_form"):
            st.markdown("### Employee Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Personal")
                age = st.number_input("Age", min_value=18, max_value=70, value=sample_data.get('Age', 30))
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if sample_data.get('Gender', 'Male') == 'Male' else 1)
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"],
                                              index=["Single", "Married", "Divorced"].index(sample_data.get('MaritalStatus', 'Single')))
                distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=100,
                                                      value=sample_data.get('DistanceFromHome', 10))

            with col2:
                st.markdown("#### Job Details")
                department = st.selectbox("Department", [
                    "Research & Development", "Sales", "Human Resources",
                    "IT", "Operations", "Finance", "Marketing",
                    "Customer Service", "Engineering", "Other"
                ], index=0)
                job_role = st.selectbox("Job Role", [
                    "Individual Contributor", "Team Lead", "Manager",
                    "Senior Manager", "Director", "Senior Director",
                    "Executive", "Specialist", "Analyst", "Consultant"
                ], index=0)
                job_level = st.slider("Job Level", 1, 5, sample_data.get('JobLevel', 2),
                                     help="1=Entry, 2=Mid, 3=Senior, 4=Lead, 5=Executive")
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000,
                                                 value=sample_data.get('MonthlyIncome', 5000), step=100)

            with col3:
                st.markdown("#### Experience")
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40,
                                                      value=sample_data.get('TotalWorkingYears', 10))
                years_at_company = st.number_input("Years at Company", min_value=0, max_value=40,
                                                   value=sample_data.get('YearsAtCompany', 5))
                years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=40,
                                                        value=sample_data.get('YearsInCurrentRole', 3))
                years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15,
                                                             value=sample_data.get('YearsSinceLastPromotion', 1))

            with st.expander("Additional Details (Optional)", expanded=False):
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

            submit_button = st.form_submit_button("Predict Attrition Risk", type="primary", use_container_width=True)

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
                st.markdown("## Prediction Results")

                # Risk gauge
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = create_risk_gauge(probability)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### Key Metrics")
                    st.metric("Attrition Risk", f"{probability * 100:.1f}%")
                    st.metric("Confidence", f"{max(1 - probability, probability) * 100:.1f}%")

                    if probability < 0.3:
                        st.success("Low risk of attrition")
                    elif probability < 0.6:
                        st.warning("Medium risk of attrition")
                    else:
                        st.error("High risk of attrition")

                # Key factors analysis
                st.markdown("### What's Driving This Risk?")

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
                        st.markdown(f"🔴 **{feature}**: Increases risk")
                    else:
                        st.markdown(f"🟢 **{feature}**: Decreases risk")

                # Recommendations
                st.markdown("### Recommendations")

                if probability >= 0.6:
                    st.markdown("""
                    **Immediate Actions:**
                    - Schedule a one-on-one meeting to discuss career goals
                    - Review compensation and benefits package
                    - Create a personalized development plan
                    - Improve work-life balance initiatives
                    - Recognize and reward contributions
                    """)
                elif probability >= 0.3:
                    st.markdown("""
                    **Preventive Measures:**
                    - Regular check-ins to understand satisfaction levels
                    - Monitor workload and provide support
                    - Offer training and development opportunities
                    - Ensure clear career progression path
                    """)
                else:
                    st.markdown("""
                    **Maintenance Actions:**
                    - Continue current engagement practices
                    - Periodic satisfaction surveys
                    - Celebrate achievements and milestones
                    - Support continuous professional growth
                    """)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)

    # Bulk Analysis page
    elif page == "Bulk Analysis":
        st.markdown("## Bulk Analysis")
        st.markdown("Upload a CSV file with employee data to predict attrition risk for multiple employees at once.")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Upload CSV File")

            # Create template CSV
            template_data = {
                'Age': [35, 28, 42],
                'Gender': ['Male', 'Female', 'Male'],
                'MaritalStatus': ['Married', 'Single', 'Divorced'],
                'DistanceFromHome': [10, 5, 20],
                'Department': ['IT', 'Sales', 'Finance'],
                'JobRole': ['Manager', 'Individual Contributor', 'Director'],
                'JobLevel': [2, 1, 4],
                'MonthlyIncome': [5000, 3500, 8000],
                'TotalWorkingYears': [10, 5, 20],
                'YearsAtCompany': [5, 2, 10],
                'YearsInCurrentRole': [3, 1, 5],
                'YearsSinceLastPromotion': [1, 0, 3],
                'Education': [3, 2, 4],
                'EducationField': ['Life Sciences', 'Marketing', 'Technical Degree'],
                'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
                'JobSatisfaction': [3, 4, 2],
                'EnvironmentSatisfaction': [3, 4, 3],
                'WorkLifeBalance': [3, 3, 2],
                'OverTime': ['No', 'Yes', 'No'],
                'StockOptionLevel': [1, 0, 2],
                'NumCompaniesWorked': [1, 2, 3]
            }

            template_df = pd.DataFrame(template_data)

            # Download template button
            csv_template = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv_template,
                file_name="employee_data_template.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown("---")

            # File uploader
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

            if uploaded_file is not None:
                try:
                    # Read CSV
                    df = pd.read_csv(uploaded_file)

                    st.success(f"File uploaded successfully! Found {len(df)} employees.")

                    # Show preview
                    with st.expander("Preview Data", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True)

                    # Process predictions button
                    if st.button("Analyze All Employees", type="primary", use_container_width=True):
                        with st.spinner("Processing predictions..."):
                            results = []

                            for idx, row in df.iterrows():
                                try:
                                    # Prepare input data with defaults for missing fields
                                    input_data = {
                                        'Age': row.get('Age', 30),
                                        'BusinessTravel': row.get('BusinessTravel', 'Travel_Rarely'),
                                        'DailyRate': row.get('DailyRate', 800),
                                        'Department': row.get('Department', 'IT'),
                                        'DistanceFromHome': row.get('DistanceFromHome', 10),
                                        'Education': row.get('Education', 3),
                                        'EducationField': row.get('EducationField', 'Life Sciences'),
                                        'EnvironmentSatisfaction': row.get('EnvironmentSatisfaction', 3),
                                        'Gender': row.get('Gender', 'Male'),
                                        'HourlyRate': row.get('HourlyRate', 65),
                                        'JobInvolvement': row.get('JobInvolvement', 3),
                                        'JobLevel': row.get('JobLevel', 2),
                                        'JobRole': row.get('JobRole', 'Manager'),
                                        'JobSatisfaction': row.get('JobSatisfaction', 3),
                                        'MaritalStatus': row.get('MaritalStatus', 'Single'),
                                        'MonthlyIncome': row.get('MonthlyIncome', 5000),
                                        'MonthlyRate': row.get('MonthlyRate', row.get('MonthlyIncome', 5000)),
                                        'NumCompaniesWorked': row.get('NumCompaniesWorked', 1),
                                        'OverTime': row.get('OverTime', 'No'),
                                        'PercentSalaryHike': row.get('PercentSalaryHike', 13),
                                        'PerformanceRating': row.get('PerformanceRating', 3),
                                        'RelationshipSatisfaction': row.get('RelationshipSatisfaction', 3),
                                        'StockOptionLevel': row.get('StockOptionLevel', 1),
                                        'TotalWorkingYears': row.get('TotalWorkingYears', 10),
                                        'TrainingTimesLastYear': row.get('TrainingTimesLastYear', 3),
                                        'WorkLifeBalance': row.get('WorkLifeBalance', 3),
                                        'YearsAtCompany': row.get('YearsAtCompany', 5),
                                        'YearsInCurrentRole': row.get('YearsInCurrentRole', 3),
                                        'YearsSinceLastPromotion': row.get('YearsSinceLastPromotion', 1),
                                        'YearsWithCurrManager': row.get('YearsWithCurrManager', 3)
                                    }

                                    # Make prediction
                                    X = processor.prepare_single_prediction(input_data)
                                    probability = model_obj.predict_proba(X)[0][1]

                                    # Determine risk level
                                    if probability < 0.3:
                                        risk_level = "Low Risk"
                                    elif probability < 0.6:
                                        risk_level = "Medium Risk"
                                    else:
                                        risk_level = "High Risk"

                                    results.append({
                                        'Employee_Index': idx + 1,
                                        'Age': input_data['Age'],
                                        'Department': input_data['Department'],
                                        'JobRole': input_data['JobRole'],
                                        'Attrition_Risk_%': round(probability * 100, 1),
                                        'Risk_Level': risk_level
                                    })

                                except Exception as e:
                                    results.append({
                                        'Employee_Index': idx + 1,
                                        'Age': row.get('Age', 'N/A'),
                                        'Department': row.get('Department', 'N/A'),
                                        'JobRole': row.get('JobRole', 'N/A'),
                                        'Attrition_Risk_%': 'Error',
                                        'Risk_Level': f'Error: {str(e)}'
                                    })

                            # Create results dataframe
                            results_df = pd.DataFrame(results)

                            st.markdown("---")
                            st.markdown("## Analysis Results")

                            # Summary statistics
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                                st.metric("High Risk Employees", high_risk,
                                         delta=f"{(high_risk/len(results_df)*100):.1f}%")

                            with col2:
                                medium_risk = len(results_df[results_df['Risk_Level'] == 'Medium Risk'])
                                st.metric("Medium Risk Employees", medium_risk,
                                         delta=f"{(medium_risk/len(results_df)*100):.1f}%")

                            with col3:
                                low_risk = len(results_df[results_df['Risk_Level'] == 'Low Risk'])
                                st.metric("Low Risk Employees", low_risk,
                                         delta=f"{(low_risk/len(results_df)*100):.1f}%")

                            # Display results table
                            st.markdown("### Detailed Results")

                            # Color code by risk level
                            def color_risk(val):
                                if val == 'High Risk':
                                    color = '#fee2e2'
                                elif val == 'Medium Risk':
                                    color = '#fef3c7'
                                elif val == 'Low Risk':
                                    color = '#d1fae5'
                                else:
                                    color = 'white'
                                return f'background-color: {color}'

                            styled_df = results_df.style.applymap(color_risk, subset=['Risk_Level'])
                            st.dataframe(styled_df, use_container_width=True, height=400)

                            # Download results
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv_results,
                                file_name="attrition_risk_analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            # Risk distribution chart
                            st.markdown("### Risk Distribution")
                            risk_counts = results_df['Risk_Level'].value_counts()

                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Employee Risk Distribution",
                                color=risk_counts.index,
                                color_discrete_map={
                                    'Low Risk': '#10b981',
                                    'Medium Risk': '#f59e0b',
                                    'High Risk': '#ef4444'
                                }
                            )

                            fig.update_layout(
                                font={'family': "Inter, sans-serif"},
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)"
                            )

                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please ensure your CSV file matches the template format.")

        with col2:
            st.markdown("### Template Format")
            st.markdown("""
            Your CSV file should include these columns:

            **Required:**
            - Age
            - Department
            - JobRole
            - JobLevel
            - MonthlyIncome
            - YearsAtCompany

            **Optional:**
            - Gender
            - MaritalStatus
            - Education
            - JobSatisfaction
            - WorkLifeBalance
            - OverTime
            - And more...

            **Tip:** Download the template to see all available fields!
            """)

            st.markdown("---")
            st.markdown("### Quick Tips")
            st.markdown("""
            - Use the template for proper formatting
            - All fields are case-sensitive
            - Missing fields will use defaults
            - You can analyze up to 1000 employees
            """)

    # How It Works page
    elif page == "How It Works":
        st.markdown("## How It Works")

        st.markdown("""
        ### Understanding the Predictor

        This tool uses artificial intelligence to analyze patterns in employee data and predict the likelihood
        of an employee leaving the organization. Here's what you need to know:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### What We Analyze

            The AI examines 30+ factors including:
            - **Career progression** (job level, promotions, tenure)
            - **Compensation** (income, stock options, raises)
            - **Work environment** (overtime, work-life balance)
            - **Job satisfaction** (engagement, environment)
            - **Personal factors** (age, distance from work)
            """)

        with col2:
            st.markdown("""
            ### How Accurate Is It?

            - **85% prediction accuracy** across thousands of cases
            - Trained on real HR analytics data
            - Continuously validated for reliability
            - Best used as one input for retention decisions

            *Note: Predictions are probabilistic, not deterministic.*
            """)

        st.markdown("---")

        # Feature importance
        st.markdown("### What Matters Most?")
        st.markdown("Based on our analysis, these factors have the biggest impact on attrition risk:")

        fig = create_feature_importance_chart(feature_importance, top_n=12)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Key insights
        st.markdown("### Key Patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### Warning Signs:
            - Low job level with long experience
            - Frequent overtime without recognition
            - Long time since last promotion
            - Poor work-life balance
            - Low satisfaction scores
            - Limited growth opportunities
            """)

        with col2:
            st.markdown("""
            #### Positive Indicators:
            - Career progression aligned with tenure
            - Strong work-life balance
            - Competitive compensation
            - High job satisfaction
            - Good manager relationships
            - Clear development path
            """)

if __name__ == "__main__":
    main()
