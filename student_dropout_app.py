import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import your main class
# from student_dropout_predictor import StudentDropoutPredictor

class StreamlitDropoutPredictor:
    def __init__(self):
        self.predictor = None
        self.df = None
        
    def generate_data(self, n_students=1000):
        """Generate synthetic data for demo"""
        np.random.seed(42)
        
        video_views = np.random.poisson(25, n_students)
        quiz_attempts = np.random.poisson(15, n_students)
        login_frequency = np.random.poisson(20, n_students)
        time_spent_hours = np.random.gamma(2, 10, n_students)
        assignment_submissions = np.random.poisson(8, n_students)
        
        # Add realistic constraints
        video_views = np.clip(video_views, 1, 100)
        quiz_attempts = np.clip(quiz_attempts, 0, 50)
        login_frequency = np.clip(login_frequency, 1, 60)
        time_spent_hours = np.clip(time_spent_hours, 1, 100)
        assignment_submissions = np.clip(assignment_submissions, 0, 20)
        
        # Create engagement score
        engagement_score = (
            video_views * 0.3 + 
            quiz_attempts * 0.25 + 
            login_frequency * 0.25 + 
            time_spent_hours * 0.1 + 
            assignment_submissions * 0.1
        )
        
        # Generate dropout
        dropout_probability = 1 / (1 + np.exp(0.1 * (engagement_score - 20)))
        dropout = np.random.binomial(1, dropout_probability)
        
        df = pd.DataFrame({
            'student_id': range(1, n_students + 1),
            'video_views': video_views,
            'quiz_attempts': quiz_attempts,
            'login_frequency': login_frequency,
            'time_spent_hours': time_spent_hours,
            'assignment_submissions': assignment_submissions,
            'engagement_score': engagement_score,
            'dropout': dropout
        })
        
        return df

def main():
    st.set_page_config(page_title="Student Dropout Prediction Dashboard", 
                       page_icon="🎓", layout="wide")
    
    # Custom CSS for mango background - only for main content area
    st.markdown("""
    <style>
        /* Main content area background */
        .main .block-container {
            background: linear-gradient(135deg, #FFB347, #FF9933, #FFCC5C) !important;
            padding: 2rem !important;
            border-radius: 15px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        }
        
        /* Make text more readable on mango background */
        .main .block-container h1, 
        .main .block-container h2, 
        .main .block-container h3,
        .main .block-container h4,
        .main .block-container h5,
        .main .block-container h6 {
            color: #2E3440 !important;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.3) !important;
        }
        
        /* Style metric containers */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(255, 153, 51, 0.3) !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        }
        
        /* Style info/success/warning/error boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 10px !important;
        }
        
        /* Style buttons */
        .stButton > button {
            background: linear-gradient(45deg, #FF6B35, #FF9933) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: bold !important;
            box-shadow: 0 3px 15px rgba(255, 107, 53, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 20px rgba(255, 107, 53, 0.4) !important;
        }
        
        /* Style input boxes */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 8px !important;
        }
        
        /* Style dataframes/tables */
        .dataframe {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 10px !important;
        }
        
        /* File uploader styling */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        /* Keep sidebar original styling */
        .sidebar .sidebar-content {
            background: var(--background-color) !important;
        }
        
        /* Make sure text is readable */
        .main .block-container p,
        .main .block-container li,
        .main .block-container span {
            color: #2E3440 !important;
        }
        
        /* Footer styling */
        .footer {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 10px !important;
            margin-top: 2rem !important;
            padding: 1rem !important;
            text-align: center;
            color: #2E3440 !important;
            font-size: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎓 Student Dropout Prediction Dashboard")
    st.markdown("**Analyze student behavior to predict potential dropouts**")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitDropoutPredictor()
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Navigation
    page = st.sidebar.selectbox("Choose a page:", 
                               ["Home", "Data Analysis", "Model Training", "Predictions", "Insights"])
    
    if page == "Home":
        st.header("🏠 Welcome to Student Dropout Prediction System")
        
        st.subheader("📤 Upload Your CSV File OR Generate Sample Data")
        
        # Upload option
        uploaded_file = st.file_uploader("Upload a CSV file with student data", type=["csv"])
        
        # Or generate button
        generate_button = st.button("🚀 Generate Sample Data")
        
        # Decision: Upload or Generate
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅ CSV data uploaded successfully!")
        
        elif generate_button:
            with st.spinner("Generating synthetic student data..."):
                df = st.session_state.predictor.generate_data(1000)
                st.session_state.df = df
                st.success("✅ Sample data generated successfully!")
        
        # Show stats if data available
        if 'df' in st.session_state:
            df = st.session_state.df
            st.subheader("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(df))
            with col2:
                st.metric("Dropout Rate", f"{df['dropout'].mean():.1%}")
            with col3:
                st.metric("Avg Video Views", f"{df['video_views'].mean():.1f}")
            with col4:
                st.metric("Avg Quiz Attempts", f"{df['quiz_attempts'].mean():.1f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Project Overview")
                st.write("""
                This dashboard helps educational platforms:
                - **Analyze** student engagement patterns
                - **Predict** potential dropouts early
                - **Implement** timely interventions
                - **Improve** student retention rates
                """)
            with col2:
                st.subheader("🔧 Features")
                st.write("""
                - Real-time student risk assessment
                - Interactive data visualizations
                - Multiple ML model comparisons
                - Actionable insights and recommendations
                """)
    
    elif page == "Data Analysis":
        st.header("📊 Data Analysis")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please generate sample data first from the Home page.")
            return
        
        df = st.session_state.df
        
        # Add data analysis visualizations here
        st.subheader("Data Distribution")
        
        # Display basic statistics
        st.write("### Dataset Statistics")
        st.write(df.describe())
        
        # Correlation matrix
        st.write("### Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please generate sample data first from the Home page.")
            return
        
        df = st.session_state.df
        
        # Prepare data for training
        feature_columns = ['video_views', 'quiz_attempts', 'login_frequency', 
                          'time_spent_hours', 'assignment_submissions']
        X = df[feature_columns]
        y = df['dropout']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training Random Forest model..."):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model in session state
                st.session_state.model = model
                st.session_state.accuracy = accuracy
                
                st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
                
                # Show classification report
                st.subheader("📊 Model Performance")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
    
    elif page == "Predictions":
        st.header("🔮 Student Risk Prediction")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train a model first from the Model Training page.")
            return
        
        st.subheader("📝 Enter Student Data")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            video_views = st.number_input("Video Views", min_value=0, max_value=100, value=25)
            quiz_attempts = st.number_input("Quiz Attempts", min_value=0, max_value=50, value=15)
            login_frequency = st.number_input("Login Frequency", min_value=0, max_value=60, value=20)
        
        with col2:
            time_spent = st.number_input("Time Spent (Hours)", min_value=0.0, max_value=100.0, value=30.0)
            assignments = st.number_input("Assignment Submissions", min_value=0, max_value=20, value=8)
        
        if st.button("🎯 Predict Risk"):
            # Prepare input
            input_data = pd.DataFrame({
                'video_views': [video_views],
                'quiz_attempts': [quiz_attempts],
                'login_frequency': [login_frequency],
                'time_spent_hours': [time_spent],
                'assignment_submissions': [assignments]
            })
            
            # Make prediction
            model = st.session_state.model
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction[0] == 1:
                    st.error("⚠️ High Risk Student")
                else:
                    st.success("✅ Low Risk Student")
            
            with col2:
                st.metric("Dropout Probability", f"{probability[1]:.2%}")
            
            with col3:
                risk_level = "High" if probability[1] > 0.7 else "Medium" if probability[1] > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Recommendations
            if probability[1] > 0.5:
                st.subheader("💡 Recommendations")
                st.write("""
                - **Immediate intervention required**
                - Send personalized encouragement messages
                - Offer additional tutoring support
                - Schedule one-on-one mentoring sessions
                - Provide flexible learning schedules
                """)
    
    elif page == "Insights":
        st.header("💡 Business Insights")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please generate sample data first from the Home page.")
            return
        
        df = st.session_state.df
        
        # Key metrics
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Dropout Rate", f"{df['dropout'].mean():.1%}")
        with col3:
            high_risk = df[df['engagement_score'] < 15]
            st.metric("High Risk Students", len(high_risk))
        with col4:
            intervention_needed = len(high_risk)
            st.metric("Interventions Needed", intervention_needed)
        
        # Insights
        st.subheader("🔍 Key Insights")
        
        high_risk_students = df[df['dropout'] == 1]
        low_risk_students = df[df['dropout'] == 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**High-Risk Student Patterns:**")
            st.write(f"• Average video views: {high_risk_students['video_views'].mean():.1f}")
            st.write(f"• Average quiz attempts: {high_risk_students['quiz_attempts'].mean():.1f}")
            st.write(f"• Average login frequency: {high_risk_students['login_frequency'].mean():.1f}")
            st.write(f"• Average time spent: {high_risk_students['time_spent_hours'].mean():.1f} hours")
        
        with col2:
            st.write("**Action Items:**")
            st.write("• Monitor students with < 15 video views")
            st.write("• Alert for students with < 10 quiz attempts")
            st.write("• Intervene when logins < 12 per month")
            st.write("• Support students with < 20 study hours")
        
        # ROI Calculation
        st.subheader("💰 Return on Investment")
        
        total_students = len(df)
        potential_dropouts = len(high_risk_students)
        intervention_cost = 50  # USD per student
        retention_value = 200  # USD per student
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Intervention Cost", f"${intervention_cost * potential_dropouts:,}")
        with col2:
            st.metric("Potential Savings", f"${retention_value * potential_dropouts:,}")
        
        roi = ((retention_value - intervention_cost) * potential_dropouts) / (intervention_cost * potential_dropouts) * 100
        st.metric("ROI", f"{roi:.1f}%")

    # Footer with mango theme
    st.markdown("""
        <div class="footer">
            Developed by <strong>Group (Ali Nadeem ❤️ Ali Hassan ❤️ Mubeen Ahmad)</strong> &copy; 2025
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
