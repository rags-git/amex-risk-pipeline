import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

BACKEND_API_URL = "http://localhost:8000/api/v1/predict-risk"

st.set_page_config(
    layout="wide", 
    page_title="CREDIT RISK INTELLIGENCE CONSOLE", 
    page_icon=None
)

# Strict Institutional Corporate Theme CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');
    
    /* Global Container Structure Override */
    html, body, [class*="css"], .stApp {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        background-color: #000B1A !important;
        color: #F3F4F6 !important;
    }
    
    /* Strict Corporate Header Layout */
    .amex-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.05rem;
        color: #FFFFFF;
        text-transform: uppercase;
        margin-bottom: 2px;
        border-left: 4px solid #4A90E2;
        padding-left: 12px;
    }
    .amex-subtitle {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.12rem;
        margin-top: 0px;
        margin-bottom: 35px;
        padding-left: 16px;
    }
    
    /* Integrated Corporate Header Text */
    .amex-section-header {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
        color: #4A90E2;
        padding-bottom: 8px;
        margin-top: 10px;
        margin-bottom: 18px;
        border-bottom: 1px solid #1E293B;
    }
    
    /* Rigid Data Outputs Fields */
    .mono-value {
        font-family: 'Roboto Mono', monospace !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.05rem !important;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        border-radius: 0px;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    .status-approved {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .status-rejected {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid #EF4444;
    }
    
    /* Override Streamlit Inputs for Corporate Conformity */
    div[data-baseweb="input"] {
        background-color: #010B14 !important;
        border: 1px solid #334155 !important;
        border-radius: 0px !important;
    }
    input {
        color: #F3F4F6 !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    /* Institutional Primary Action Command Block */
    div.stButton > button:first-child {
        background-color: #4A90E2 !important;
        border: 1px solid #4A90E2 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08rem !important;
        padding: 14px 28px !important;
        border-radius: 0px !important;
        width: 100% !important;
        transition: background-color 0.15s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #357ABD !important;
        border-color: #357ABD !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Terminal Branding Header
st.markdown('<div class="amex-title">RISK INTELLIGENCE AND PORTFOLIO UNDERWRITING ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="amex-subtitle">REAL-TIME INFERENCE GATEWAY & SYSTEMS CALIBRATION FRAMEWORK</div>', unsafe_allow_html=True)

# Main Grid Processing Matrix
col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    # Systems Configuration Parameters
    st.sidebar.markdown('<p style="font-size:0.8rem; font-weight:700; color:#9CA3AF; letter-spacing:0.1rem; text-transform:uppercase; margin-bottom:15px;">SYSTEM PARAMETERS</p>', unsafe_allow_html=True)
    policy_threshold = st.sidebar.slider(
        "RISK CEILING DECISION THRESHOLD", 
        min_value=0.01, max_value=1.00, value=0.15, step=0.01
    )
    
    st.markdown('<div class="amex-section-header">ACCOUNT ASSESSMENT PIPELINE</div>', unsafe_allow_html=True)
    
    c_id = st.text_input("CUSTOMER IDENTIFICATION HASH", "0000099db563072c3d5e7")
    
    st.markdown("<br><p style='font-size:0.75rem; font-weight:700; color:#6B7280; letter-spacing:0.08rem; text-transform:uppercase; margin-bottom:10px;'>High-Dimensional Feature Vectors</p>", unsafe_allow_html=True)
    p_2 = st.slider("PAYMENT FACTOR METRIC (P_2 MEAN)", 0.0, 1.0, 0.75)
    d_39 = st.slider("DELINQUENCY METRIC (D_39 MAX)", 0.0, 1.0, 0.05)
    b_1 = st.slider("BALANCE VELOCITY METRIC (B_1 LAST)", 0.0, 1.0, 0.12)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RUN UNDERWRITING INFERENCE MATRIX"):
        payload = {"customer_ID": c_id, "P_2_mean": p_2, "D_39_max": d_39, "B_1_last": b_1}
        try:
            res = requests.post(BACKEND_API_URL, json=payload)
            if res.status_code == 200:
                data = res.json()
                prob = data["default_probability"]
                simulated_decision = "APPROVED" if prob <= policy_threshold else "REJECTED"
                
                badge_class = "status-approved" if simulated_decision == "APPROVED" else "status-rejected"
                
                st.markdown(f"""
                    <div style="border-top: 1px solid #1E293B; margin-top: 25px; padding-top: 20px;">
                        <div style="font-size:0.75rem; font-weight:700; color:#6B7280; letter-spacing:0.05rem; text-transform:uppercase;">MODEL OUTPUT DEFAULT PROBABILITY</div>
                        <div class="mono-value" style="color: #FFFFFF; margin-bottom:12px;">{prob:.4f}</div>
                        <div style="margin-bottom:20px;">
                            <span class="status-badge {badge_class}">DECISION STATUS: {simulated_decision}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                grid_col1, grid_col2 = st.columns(2)
                with grid_col1:
                    st.markdown(f"""
                        <div style="background-color:#010B14; border: 1px solid #1E293B; padding:14px;">
                            <div style="font-size:0.7rem; font-weight:700; color:#6B7280; letter-spacing:0.04rem; text-transform:uppercase; margin-bottom:4px;">RISK CLASSIFICATION</div>
                            <div style="font-size:1rem; font-weight:700; color:#F3F4F6; text-transform:uppercase;">{data["risk_tier"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                with grid_col2:
                    st.markdown(f"""
                        <div style="background-color:#010B14; border: 1px solid #1E293B; padding:14px;">
                            <div style="font-size:0.7rem; font-weight:700; color:#6B7280; letter-spacing:0.04rem; text-transform:uppercase; margin-bottom:4px;">RECOMMENDED CREDIT LIMIT</div>
                            <div style="font-size:1rem; font-weight:700; color:#4A90E2; font-family:'Roboto Mono', monospace;">USD {data['recommended_credit_limit']:,}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("SYSTEM LOG: RAW COMPLIANCE PAYLOAD JSON"):
                    st.json(data)
            else:
                st.error("SYSTEM ERROR: BACKEND SCHEMA MISMATCH.")
        except Exception as e:
            st.error(f"SYSTEM ERROR: SERVICE LINK UNREACHABLE: {e}")

with col_right:
    # Portfolio Trajectory Analytics Window
    st.markdown('<div class="amex-section-header">BEHAVIORAL HISTORICAL TREND DATA SUMMARY</div>', unsafe_allow_html=True)
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    spend_curve = np.array([0.24, 0.58, 0.45, 0.41, 0.15, 0.18])
    payment_curve = np.array([0.29, 0.65, 0.48, 0.43, 0.12, 0.21])
    
    fig = go.Figure()
    
    # Amex Style Time Series Representation: Direct Linear Coordinates with Square Grid Backing
    fig.add_trace(go.Scatter(
        x=months, y=spend_curve,
        mode='lines+markers',
        name='SPEND METRIC VARIANCE (S_*)',
        line=dict(color='#4A90E2', width=2, shape='linear'),
        marker=dict(size=6, symbol='square', color='#4A90E2')
    ))
    
    fig.add_trace(go.Scatter(
        x=months, y=payment_curve,
        mode='lines+markers',
        name='PAYMENT METRIC VARIANCE (P_*)',
        line=dict(color='#9CA3AF', width=2, shape='linear', dash='dash'),
        marker=dict(size=6, symbol='circle', color='#9CA3AF')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=10, b=30),
        font=dict(family="Roboto Mono, monospace", size=11),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#1E293B', 
            tickfont=dict(color='#9CA3AF'),
            showline=True,
            linecolor='#1E293B'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#1E293B', 
            title=dict(
                text="AGGREGATE METRIC RATIO VALUE",
                font=dict(color='#6B7280', size=10, family="Roboto Mono, monospace")
            ),
            tickfont=dict(color='#9CA3AF'),
            showline=True,
            linecolor='#1E293B'
        ),
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.05, 
            xanchor="left", 
            x=0,
            font=dict(size=10, color='#9CA3AF')
        )
    )
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    # Policy System Impact Log Block
    st.markdown('<div class="amex-section-header">MONTE CARLO SIMULATION ANALYSIS</div>', unsafe_allow_html=True)
    
    mock_portfolio_probs = np.random.beta(a=2, b=8, size=1000)
    approved_pct = (mock_portfolio_probs <= policy_threshold).mean()
    
    st.markdown(f"""
        <p style="margin:0; font-size: 0.85rem; color: #9CA3AF; line-height: 1.6; font-family: 'Roboto Mono', monospace;">
            [PARAM CALIBRATION] CRITERIA LIMIT SET TO: <span style="color:#FFFFFF; font-weight:700;">{policy_threshold:.2f}</span><br>
            [SIMULATION RES] APPROVAL RUN EXTRACTS AN AUTOMATED VELOCITY CONVERGENCE OF: 
            <span style="color: #4A90E2; font-weight: 700;">{approved_pct:.2%}</span> ACROSS TARGET SAMPLES.
        </p>
    """, unsafe_allow_html=True)