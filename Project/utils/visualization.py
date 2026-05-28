"""
Visualization utilities for the cyber attack detection app
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def create_confusion_matrix_plot(cm, class_names, title="Confusion Matrix"):
    """
    Create a confusion matrix plot using matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax)
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    return fig

def create_feature_importance_plot(feature_names, importances, top_n=15):
    """
    Create a feature importance plot
    """
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(importance_df)), importance_df['importance'].values)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Most Important Features')
    ax.invert_yaxis()
    return fig

def create_radar_chart(metrics_dict, model_names):
    """
    Create a radar chart comparing model metrics
    """
    categories = list(metrics_dict.keys())
    N = len(categories)
    
    fig = go.Figure()
    
    for model_name in model_names:
        values = [metrics_dict[cat][model_name] for cat in categories]
        values += values[:1]  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Model Performance Comparison",
        showlegend=True
    )
    
    return fig

def create_class_distribution_plot(class_distribution):
    """
    Create a pie chart for class distribution
    """
    fig = px.pie(
        values=list(class_distribution.values()),
        names=list(class_distribution.keys()),
        title="Attack Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig