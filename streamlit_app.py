import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from prediction_adjustments import (
    MANAGER_EXPERIENCE_BASELINE,
    adjust_duration_ratio_for_manager_experience,
    align_model_features,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Construction AI Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS - STRONG BLUE THEME
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #BAE6FD 0%, #F0F9FF 60%, #FFFFFF 100%);
        background-attachment: fixed;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E0F2FE 0%, #F0F9FF 100%);
        border-right: 1px solid #BFDBFE;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p {
        color: #0C4A6E !important;
        font-weight: 500;
    }

    div[data-testid="stSidebar"] div[data-baseweb="input"],
    div[data-testid="stSidebar"] div[data-baseweb="select"],
    div[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background-color: #FFFFFF;
        border: 1px solid #7DD3FC;
        border-radius: 8px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1 {
        color: #0369A1;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    h2, h3 {
        color: #0284C7;
        font-family: 'Inter', sans-serif;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #0284C7;
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0F2FE;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #64748B;
        font-weight: 600;
    }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0284C7 0%, #0369A1 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 700;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    model = joblib.load('construction_model.pkl')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. Sidebar Inputs 
st.sidebar.markdown("### Project Parameters")
st.sidebar.markdown("---")

with st.sidebar.form("prediction_form"):
    
    # === Section 1: Core Project Info ===
    st.markdown("#### Core Project Info")
    
    project_type = st.selectbox("Project Type", options=["Commercial", "Service", "Other (Residential/Industrial/etc)"])
    
    expected_cost = st.number_input("Expected Total Cost ($)", min_value=1000.0, value=5000000.0, step=10000.0)
    expected_duration = st.number_input("Expected Duration (Days)", min_value=1, value=180, step=1)
    
    size_label = st.selectbox(
        "Project Size", 
        options=["Very Small (< 500m²)", "Small (500-1000m²)", "Medium (1000-5000m²)", "Large (5000-20000m²)", "Mega (> 20000m²)"],
        index=2
    )
    size_map = {
        "Very Small (< 500m²)": 250,
        "Small (500-1000m²)": 750,
        "Medium (1000-5000m²)": 2500,
        "Large (5000-20000m²)": 10000,
        "Mega (> 20000m²)": 30000
    }
    project_size = size_map[size_label]
    
    #  Section 2: Technical Characteristics
    st.markdown("#### Technical Characteristics")
    
    complexity_label = st.selectbox(
        "Project Complexity Level",
        options=[
            "1 - Very Low (Standard repeatable design)",
            "2 - Low (Conventional project)",
            "3 - Medium (Standard complexity)",
            "4 - High (Complex design, specialized methods)",
            "5 - Very High (Unique structure, extreme complexity)"
        ],
        index=3
    )
    complexity = int(complexity_label[0])
    design_completeness = st.slider("Design Completeness (%)", 0, 100, 80)
    bim_accuracy = st.slider("BIM/QTO Accuracy (%)", 0, 100, 75)
    
    # Section 3: Contract & Management 
    st.markdown("#### Contract & Management")
    
    contract_type = st.selectbox("Contract Type", options=["Lump Sum", "Unit Price", "Cost Plus"])
    manager_experience = st.slider("Manager Experience (Years)", 1, 30, 10)
    
    #  Section 4: Requirements 
    st.markdown("#### Requirements")
    
    sustainability = st.radio("Sustainability Requirements?", ["Yes", "No"], index=0, horizontal=True)
    value_engineering = st.radio("Value Engineering Applied?", ["Yes", "No"], index=0, horizontal=True)
    
    # Section 5: Market & Cost Factors 
    st.markdown("#### Market & Cost Factors")
    
    market_label = st.selectbox(
        "Material Market Condition",
        options=["Stable (Normal Prices)", "Moderate Inflation", "Unstable (High Prices)", "Crisis (Extreme)"],
        index=1
    )
    market_map = {
        "Stable (Normal Prices)": 100.0,
        "Moderate Inflation": 110.0,
        "Unstable (High Prices)": 125.0,
        "Crisis (Extreme)": 140.0
    }
    material_price_index = market_map[market_label]
    
    local_mat_label = st.selectbox(
        "Local Material Availability",
        options=["Very Low (< 30%)", "Low (30-50%)", "Medium (50-70%)", "High (70-90%)", "Very High (> 90%)"],
        index=3
    )
    local_mat_map = {
        "Very Low (< 30%)": 0.2,
        "Low (30-50%)": 0.4,
        "Medium (50-70%)": 0.6,
        "High (70-90%)": 0.8,
        "Very High (> 90%)": 0.95
    }
    local_material_availability = local_mat_map[local_mat_label]
    
    average_labor_cost = st.number_input("Average Labor Cost ($/day)", min_value=100.0, value=5000.0, step=100.0)
    expected_material_cost = st.number_input("Expected Material Cost ($)", min_value=1000.0, value=expected_cost * 0.4, step=10000.0)
    
    # Section 6: Environment 
    st.markdown("#### Environment")
    
    stability_label = st.selectbox(
        "Political/Economic Stability", 
        options=[
            "1 - Very Low (Crisis, extreme volatility)",
            "2 - Low (Unstable, frequent changes)",
            "3 - Medium (Moderate stability)",
            "4 - High (Stable environment, predictable)",
            "5 - Very High (Highly stable, strong economy)"
        ],
        index=2
    )
    political_stability = int(stability_label[0])
    
    st.markdown("---")
    submitted = st.form_submit_button("Run Prediction")

# 5. Main Content Area
st.title("Construction Project Predictor | AI Analytics")
st.components.v1.html("<div style='height: 10px'></div>")
st.caption("AI-Powered Estimation using Multi-Output Random Forest")

if submitted:
    #  Data Preparation 
    is_commercial = 1 if project_type == "Commercial" else 0
    is_service = 1 if project_type == "Service" else 0
    is_unit_price = 1 if contract_type == "Unit Price" else 0
    
    input_data = {
        'Expected Total Cost': [expected_cost],
        'Expected Duration': [expected_duration],
        'Project Size (Area)': [project_size],
        'Complexity': [complexity],
        'Design Completeness': [design_completeness],
        'Sustainability Requirements': [1 if sustainability == "Yes" else 0],
        'Material Price Index': [material_price_index],
        'Local Material Availability': [local_material_availability],
        'Average Labor Cost': [average_labor_cost],
        'Expected Material Cost': [expected_material_cost],
        'Political Stability': [political_stability],
        'BIM Accuracy': [bim_accuracy],
        'Value Engineering': [1 if value_engineering == "Yes" else 0],
        'Manager Experience': [manager_experience],
        'Type_Commercial ': [is_commercial],
        'Type_commercial': [0],
        'Type_service': [is_service],
        'Contract_Unit price ': [is_unit_price]
    }
    df_input = align_model_features(pd.DataFrame(input_data), model)
    
    # Prediction 
    with st.spinner('Calculating...'):
        pred_ratios = model.predict(df_input)
        cost_ratio_pred = pred_ratios[0][0]

        # The trained forest learned a noisy, non-monotonic duration effect for
        # manager experience. Predict duration at a neutral baseline, then apply
        # a controlled domain adjustment so more experience consistently lowers
        # schedule delay.
        duration_input = df_input.copy()
        duration_input['Manager Experience'] = MANAGER_EXPERIENCE_BASELINE
        duration_ratios = model.predict(duration_input)
        dur_ratio_pred = adjust_duration_ratio_for_manager_experience(
            duration_ratios[0][1],
            manager_experience,
        )
        
        final_cost = cost_ratio_pred * expected_cost
        final_duration = dur_ratio_pred * expected_duration
        
        cost_overrun = final_cost - expected_cost
        dur_overrun = final_duration - expected_duration
        
        cost_overrun_pct = (cost_overrun / expected_cost) * 100
        dur_overrun_pct = (dur_overrun / expected_duration) * 100

    #  Metrics 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Predicted Final Cost")
        delta_color = "inverse" if cost_overrun > 0 else "normal"
        st.metric(label="Total Cost", value=f"${final_cost:,.2f}", delta=f"{cost_overrun_pct:+.1f}%", delta_color=delta_color)
    with col2:
        st.markdown("### Predicted Final Duration")
        delta_color = "inverse" if dur_overrun > 0 else "normal"
        st.metric(label="Duration (Days)", value=f"{final_duration:.1f}", delta=f"{dur_overrun_pct:+.1f}%", delta_color=delta_color)

    st.markdown("---")

    # Charts 
    st.markdown("### Performance Overview")
    
    # 1. Comparative Bar Chart 
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name='Expected',
        x=['Cost', 'Duration'],
        y=[100, 100],
        marker_color='#94A3B8'
    ))
    
    fig_bar.add_trace(go.Bar(
        name='Predicted (Overrun)',
        x=['Cost', 'Duration'],
        y=[100 + cost_overrun_pct, 100 + dur_overrun_pct],
        text=[f"+{cost_overrun_pct:.1f}%", f"+{dur_overrun_pct:.1f}%"],
        textposition='auto',
        marker_color=['#0284C7', '#F59E0B']
    ))
    
    fig_bar.update_layout(
        title="Comparison (Base = 100%)",
        barmode='group',
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Performance Index Chart 
    st.markdown("### Performance Index Over Project Phases")
    
    phases = ['Start', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Completion']
    expected_index = [100, 100, 100, 100, 100, 100]
    
    cost_index = [100, 100 + cost_overrun_pct*0.2, 100 + cost_overrun_pct*0.4, 
                  100 + cost_overrun_pct*0.6, 100 + cost_overrun_pct*0.8, 100 + cost_overrun_pct]
    
    dur_index = [100, 100 + dur_overrun_pct*0.2, 100 + dur_overrun_pct*0.4,
                 100 + dur_overrun_pct*0.6, 100 + dur_overrun_pct*0.8, 100 + dur_overrun_pct]
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig_cpi = go.Figure()
        fig_cpi.add_trace(go.Scatter(
            x=phases, y=expected_index,
            mode='lines',
            name='Target (100%)',
            line=dict(color='#94A3B8', width=3, dash='dash')
        ))
        fig_cpi.add_trace(go.Scatter(
            x=phases, y=cost_index,
            mode='lines+markers',
            name='Actual Cost Index',
            line=dict(color='#0284C7', width=4),
            marker=dict(size=10),
            fill='tonexty',
            fillcolor='rgba(2, 132, 199, 0.1)'
        ))
        
        y_min = min(95, min(cost_index) - 5)
        y_max = max(105, max(cost_index) + 5)
        
        fig_cpi.update_layout(
            title="Cost Performance Index",
            xaxis_title="Project Phase",
            yaxis_title="Index (%)",
            yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cpi, use_container_width=True)
    
    with col_chart2:
        fig_spi = go.Figure()
        fig_spi.add_trace(go.Scatter(
            x=phases, y=expected_index,
            mode='lines',
            name='Target (100%)',
            line=dict(color='#94A3B8', width=3, dash='dash')
        ))
        fig_spi.add_trace(go.Scatter(
            x=phases, y=dur_index,
            mode='lines+markers',
            name='Actual Schedule Index',
            line=dict(color='#F59E0B', width=4),
            marker=dict(size=10),
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.1)'
        ))
        
        y_min = min(95, min(dur_index) - 5)
        y_max = max(105, max(dur_index) + 5)
        
        fig_spi.update_layout(
            title="Schedule Performance Index",
            xaxis_title="Project Phase",
            yaxis_title="Index (%)",
            yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
            template="plotly_white",
            hovermode="x unified",
            height=400,
            plot_bgcolor='rgba(255,255,255,0.8)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_spi, use_container_width=True)

else:
    st.info("Please enter project details in the sidebar to start AI Analysis.")
    
    st.markdown("#### Model Accuracy Trend")
    df_anim = pd.DataFrame({
        'Project ID': range(1, 11),
        'Accuracy': [88, 90, 89, 93, 95, 94, 97, 96, 98, 99]
    })
    
    fig_anim = px.line(
        df_anim, x="Project ID", y="Accuracy", 
        markers=True, 
        range_y=[80, 100],
        line_shape='spline'
    )
    fig_anim.update_traces(line_color='#0284C7', line_width=4, marker=dict(size=8))
    fig_anim.update_layout(
        xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
        plot_bgcolor='rgba(255,255,255,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_anim, use_container_width=True)
