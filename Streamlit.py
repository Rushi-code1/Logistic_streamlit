import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .survived {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .not-survived {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', 
    unsafe_allow_html=True
)
st.markdown("---")


# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained model and scaler from the saved file"""
    try:
        with open('titanic_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("""
        Model file not found! Please run 'python train_model.py' first 
        to train and save the model.
        """)
        st.stop()


# Load the model
model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
metrics = model_data['metrics']

# Sidebar for model information
st.sidebar.header("📊 Model Information")
st.sidebar.info(f"""
**Model**: Logistic Regression  
**Dataset**: Titanic Passenger Data (Real Dataset)  
**Features**: {len(feature_names)} features  
**Accuracy**: {metrics['accuracy']:.1%}  
**Precision**: {metrics['precision']:.1%}  
**Recall**: {metrics['recall']:.1%}  
**F1 Score**: {metrics['f1']:.3f}  
""")

# Show feature names in sidebar
with st.sidebar.expander("📝 Model Features"):
    for feature in feature_names:
        st.write(f"• {feature}")

# Main prediction interface
st.markdown(
    '<h2 class="sub-header">🎯 Make a Prediction</h2>', 
    unsafe_allow_html=True
)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Passenger Details")
    
    # Passenger class
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: (f"Class {x} - "
                              f"{'First' if x==1 else 'Second' if x==2 else 'Third'}"),
        help="The passenger's ticket class (1st, 2nd, or 3rd)"
    )
    
    # Age
    age = st.slider(
        "Age",
        min_value=0,
        max_value=80,
        value=30,
        help="Passenger's age in years"
    )
    
    # Sex
    sex = st.selectbox(
        "Sex",
        options=["Female", "Male"],
        help="Passenger's gender"
    )
    
    # Fare
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=500.0,
        value=32.0,
        step=0.1,
        help="Ticket fare paid by the passenger"
    )

with col2:
    st.subheader("👨‍👩‍👧‍👦 Family Information")
    
    # Siblings/Spouses
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        help="Number of siblings or spouses aboard the Titanic"
    )
    
    # Parents/Children
    parch = st.number_input(
        "Number of Parents/Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        help="Number of parents or children aboard the Titanic"
    )
    
    # Embarked
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["Southampton", "Cherbourg", "Queenstown"],
        help="Port where the passenger boarded the ship"
    )
    
    # Display calculated family size
    family_size = sibsp + parch + 1
    st.info(f"**Family Size:** {family_size}")


# Prepare input data for prediction using exact same format as training
def prepare_input_data(pclass, age, sex, fare, sibsp, parch, 
                      embarked, family_size):
    """Prepare input data for model prediction using exact feature format"""
    # Create input data matching the exact training features
    input_data = {
        'Pclass': pclass,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'FamilySize': family_size,
        'Sex_male': 1 if sex == "Male" else 0,
        'Embarked_Q': 1 if embarked == "Queenstown" else 0,
        'Embarked_S': 1 if embarked == "Southampton" else 0,
    }
    
    # Convert to DataFrame to maintain feature order
    input_df = pd.DataFrame([input_data])
    
    # Reorder columns to match training feature order
    input_df = input_df[feature_names]
    
    return input_df


# Prediction button
if st.button("🔮 Predict Survival", type="primary"):
    # Prepare input data
    input_df = prepare_input_data(
        pclass, age, sex, fare, sibsp, parch, embarked, family_size
    )
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.markdown(
        '<h2 class="sub-header">📋 Prediction Results</h2>', 
        unsafe_allow_html=True
    )
    
    if prediction == 1:
        st.markdown(f'''
        <div class="prediction-box survived">
            <h3>✅ SURVIVED</h3>
            <p><strong>Survival Probability:</strong> {probability[1]:.1%}</p>
            <p>The model predicts this passenger would have survived 
            the Titanic disaster.</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="prediction-box not-survived">
            <h3>❌ DID NOT SURVIVE</h3>
            <p><strong>Survival Probability:</strong> {probability[1]:.1%}</p>
            <p>The model predicts this passenger would not have survived 
            the Titanic disaster.</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Show probability breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Survival Probability", f"{probability[1]:.1%}")
    with col2:
        st.metric("Non-Survival Probability", f"{probability[0]:.1%}")
    
    # Show input data used for prediction
    with st.expander("🔍 View Input Data Used for Prediction"):
        st.write("**Processed Features:**")
        st.dataframe(input_df, use_container_width=True)

# Model performance section
st.markdown("---")
st.markdown(
    '<h2 class="sub-header">📈 Model Performance</h2>', 
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
with col2:
    st.metric("Precision", f"{metrics['precision']:.1%}")
with col3:
    st.metric("Recall", f"{metrics['recall']:.1%}")
with col4:
    st.metric("F1 Score", f"{metrics['f1']:.3f}")

# Feature importance (approximate based on logistic regression coefficients)
st.markdown(
    '<h2 class="sub-header">🔍 Feature Importance</h2>', 
    unsafe_allow_html=True
)

# Get feature importance from model coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=True)

st.bar_chart(feature_importance.set_index('Feature'))

# About section
st.markdown("---")
st.markdown(
    '<h2 class="sub-header">ℹ️ About This App</h2>', 
    unsafe_allow_html=True
)

st.markdown(f"""
This Streamlit application uses the **exact same Logistic Regression model** 
from your Jupyter notebook to predict passenger survival on the Titanic. 

**Model Details:**
- **Training Data**: Real Titanic dataset ({len(feature_names)} features)
- **Preprocessing**: Same as notebook (missing value imputation, feature engineering)
- **Model Performance**: Accuracy {metrics['accuracy']:.1%}, Precision {metrics['precision']:.1%}, Recall {metrics['recall']:.1%}

**Features Used:**
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Age**: Passenger's age
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard  
- **Fare**: Ticket fare
- **FamilySize**: Total family members (SibSp + Parch + 1)
- **Sex_male**: Gender encoding (1=Male, 0=Female)
- **Embarked_Q/S**: Port of embarkation encoding

This model provides the **same predictions** as your Jupyter notebook model!
""")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Trained on Real Titanic Data")
