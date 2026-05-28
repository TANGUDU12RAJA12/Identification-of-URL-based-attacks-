"""
Cyber Attack Detection Web Application
Simple interface for file upload and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Cyber Attack Detection System")
st.markdown("Upload your network traffic data (CSV or TXT) to detect cyber attacks")

# Load models
@st.cache_resource
def load_models():
    try:
        binary_model = joblib.load('models/binary_rf_model.pkl')
        multi_model = joblib.load('models/multi_xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return binary_model, multi_model, scaler, label_encoder, feature_names
    except:
        st.error("❌ Models not found. Please run 'python train_models.py' first.")
        return None, None, None, None, None

# Load models
binary_model, multi_model, scaler, label_encoder, feature_names = load_models()

# File upload
uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'txt'],
    help="Upload CSV or TXT file with network traffic data"
)

if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
        
        st.success(f"✅ File loaded successfully! Shape: {df.shape}")
        
        # Show data preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10))
        
        # Prediction button
        if st.button("🔍 Run Prediction", type="primary"):
            with st.spinner("Analyzing network traffic..."):
                try:
                    # Check required features
                    available_features = [f for f in feature_names if f in df.columns]
                    missing_features = [f for f in feature_names if f not in df.columns]
                    
                    if missing_features:
                        st.warning(f"Missing features: {missing_features[:5]}... Using available features")
                    
                    # Prepare features
                    X = df[available_features].copy()
                    
                    # Preprocess
                    for col in X.columns:
                        if X[col].dtype in ['float64', 'int64']:
                            X[col] = X[col].clip(lower=0)
                            X[col] = X[col].fillna(0)
                    
                    # Log transform
                    for col in ['source_ip_freq', 'destination_ip_freq']:
                        if col in X.columns:
                            X[col] = np.log1p(X[col].clip(lower=0))
                    
                    # Scale
                    X_scaled = scaler.transform(X)
                    
                    # Binary prediction
                    binary_pred = binary_model.predict(X_scaled)
                    binary_proba = binary_model.predict_proba(X_scaled)
                    
                    # Multi-class prediction
                    multi_pred_encoded = multi_model.predict(X_scaled)
                    multi_proba = multi_model.predict_proba(X_scaled)
                    multi_pred = label_encoder.inverse_transform(multi_pred_encoded)
                    
                    # Add predictions to dataframe
                    df_results = df.copy()
                    df_results['Binary_Prediction'] = binary_pred
                    df_results['Binary_Confidence'] = np.max(binary_proba, axis=1)
                    df_results['Binary_Result'] = df_results['Binary_Prediction'].apply(
                        lambda x: '⚠️ ATTACK' if x == 1 else '✅ NORMAL'
                    )
                    df_results['Attack_Type'] = multi_pred
                    df_results['Attack_Confidence'] = np.max(multi_proba, axis=1)
                    
                    # ============================================
                    # DISPLAY PREDICTIONS FIRST
                    # ============================================
                    st.markdown("---")
                    st.subheader("🎯 PREDICTION RESULTS")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    attack_count = (binary_pred == 1).sum()
                    normal_count = (binary_pred == 0).sum()
                    
                    with col1:
                        st.metric("Total Samples", len(df_results))
                    with col2:
                        st.metric("⚠️ Attacks Detected", attack_count)
                    with col3:
                        st.metric("✅ Normal Traffic", normal_count)
                    with col4:
                        st.metric("Attack Rate", f"{attack_count/len(df_results)*100:.1f}%")
                    
                    # Display predictions table
                    st.subheader("📋 Prediction Results")
                    display_cols = ['Binary_Result', 'Attack_Type', 'Binary_Confidence', 'Attack_Confidence']
                    
                    # Style the dataframe
                    def highlight_attack(row):
                        if row['Binary_Result'] == '⚠️ ATTACK':
                            return ['background-color: #ffcccc'] * len(row)
                        return ['background-color: #ccffcc'] * len(row)
                    
                    styled_df = df_results[display_cols].style.apply(highlight_attack, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download button
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # ============================================
                    # VISUALIZATION OF INPUT DATASET
                    # ============================================
                    st.markdown("---")
                    st.subheader("📊 Visualization of Input Dataset")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["Attack Distribution", "Feature Distribution", "Confidence Analysis"])
                    
                    with tab1:
                        # Attack type distribution
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            attack_types = df_results[df_results['Binary_Prediction'] == 1]['Attack_Type'].value_counts()
                            if len(attack_types) > 0:
                                attack_types.plot(kind='bar', ax=ax, color='red', alpha=0.7)
                                ax.set_title('Detected Attack Types', fontsize=14)
                                ax.set_xlabel('Attack Type')
                                ax.set_ylabel('Count')
                                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("No attacks detected in the data")
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            labels = ['Normal', 'Attack']
                            sizes = [normal_count, attack_count]
                            colors = ['#90ee90', '#ff6b6b']
                            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                            ax.set_title('Binary Classification Distribution', fontsize=14)
                            st.pyplot(fig)
                    
                    with tab2:
                        # Feature distributions
                        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for idx, col in enumerate(numeric_cols):
                            if idx < 6:
                                axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                                axes[idx].set_title(f'Distribution of {col}', fontsize=10)
                                axes[idx].set_xlabel(col)
                                axes[idx].set_ylabel('Frequency')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    with tab3:
                        # Confidence analysis
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            correct_conf = df_results[df_results['Binary_Prediction'] == 1]['Binary_Confidence']
                            if len(correct_conf) > 0:
                                ax.hist(correct_conf, bins=20, color='green', alpha=0.7, label='Attack Samples')
                            normal_conf = df_results[df_results['Binary_Prediction'] == 0]['Binary_Confidence']
                            if len(normal_conf) > 0:
                                ax.hist(normal_conf, bins=20, color='blue', alpha=0.7, label='Normal Samples')
                            ax.set_title('Binary Classification Confidence Distribution', fontsize=12)
                            ax.set_xlabel('Confidence Score')
                            ax.set_ylabel('Frequency')
                            ax.legend()
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            attack_conf = df_results[df_results['Binary_Prediction'] == 1]['Attack_Confidence']
                            if len(attack_conf) > 0:
                                ax.hist(attack_conf, bins=20, color='orange', alpha=0.7, edgecolor='black')
                                ax.set_title('Attack Type Classification Confidence', fontsize=12)
                                ax.set_xlabel('Confidence Score')
                                ax.set_ylabel('Frequency')
                            else:
                                ax.text(0.5, 0.5, 'No attacks detected', ha='center', va='center')
                            st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please ensure your file has the required features.")
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please upload a valid CSV or TXT file.")

# Footer
st.markdown("---")
st.markdown("🛡️ Cyber Attack Detection System | Powered by Machine Learning")