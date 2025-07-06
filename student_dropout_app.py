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
        
        # Generate sample data
        if st.button("🚀 Generate Sample Data"):
            with st.spinner("Generating synthetic student data..."):
                df = st.session_state.predictor.generate_data(1000)
                st.session_state.df = df
                st.success("✅ Sample data generated successfully!")
                
                # Show basic stats
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
    
    elif page == "Data Analysis":
        st.header("📊 Data Analysis")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please generate sample data first from the Home page.")
            return
        
        df = st.session_state.df
        
        # Data overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**First 10 Records:**")
            st.dataframe(df.head(10))
        
        with col2:
            st.write("**Statistical Summary:**")
            st.dataframe(df.describe())
        
        # Visualizations
        st.subheader("📈 Data Visualizations")
        
        # Engagement distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='engagement_score', color='dropout', 
                             title='Engagement Score Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dropout rate by engagement level
            df['engagement_level'] = pd.cut(df['engagement_score'], 
                                           bins=[0, 15, 30, 100], 
                                           labels=['Low', 'Medium', 'High'])
            engagement_dropout = df.groupby('engagement_level')['dropout'].mean()
            
            fig = px.bar(x=engagement_dropout.index, y=engagement_dropout.values,
                        title='Dropout Rate by Engagement Level')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation
        st.subheader("🔗 Feature Correlations")
        features = ['video_views', 'quiz_attempts', 'login_frequency', 
                   'time_spent_hours', 'assignment_submissions', 'dropout']
        
        corr_matrix = df[features].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        if 'df' not in st.session_state:
            st.warning("⚠️ Please generate sample data first from the Home page.")
            return
        
        df = st.session_state.df
        
        # Model selection
        st.subheader("⚙️ Model Configuration")
        model_type = st.selectbox("Select Model:", 
                                 ["Random Forest", "Logistic Regression", "Decision Tree"])
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                # Prepare data
                features = ['video_views', 'quiz_attempts', 'login_frequency', 
                           'time_spent_hours', 'assignment_submissions']
                X = df[features]
                y = df['dropout']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train model
                if model_type == "Random Forest":
                    model = RandomForestClassifier(random_state=42, n_estimators=100)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Store model
                st.session_state.model = model
                st.session_state.features = features
                st.session_state.scaler = StandardScaler()
                
                # Show results
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"✅ Model trained successfully!")
                st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='importance', y='feature',
                                title='Feature Importance')
                    st.plotly_chart(fig, use_container_width=True)
    
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

if __name__ == "__main__":
    main()
