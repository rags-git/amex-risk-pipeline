import streamlit as st
import requests
import json
import plotly.graph_objects as ob

# Core Configuration Settings
st.set_page_config(
    page_title="Credit Risk Intelligence Console",
    layout="wide"
)

# Core Gateway Configuration Variables
BACKEND_BASE = "http://backend-service:8000"
predict_url = f"{BACKEND_BASE}/api/v1/predict-risk"
batch_url = f"{BACKEND_BASE}/api/v1/predict-batch"

# Clean UI Action Button Custom Styling Matrix
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0077b6; 
        color: white; 
        border-radius: 4px; 
        font-weight: bold; 
        width: 100%; 
        border: none; 
        padding: 0.6rem;
    }
    div.stButton > button:first-child:hover { 
        background-color: #0096c7; 
        color: white; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("RISK INTELLIGENCE AND PORTFOLIO UNDERWRITING ENGINE")
st.caption("REAL-TIME INFERENCE GATEWAY & SYSTEMS CALIBRATION FRAMEWORK")

# Sidebar Configuration Parameters
with st.sidebar:
    st.header("SYSTEM PARAMETERS")
    risk_threshold = st.slider("RISK CEILING DECISION THRESHOLD", 0.0, 1.0, 0.15)

# Interface Tab Workspace Split Selection Architecture
tab1, tab2, tab3 = st.tabs(["INDIVIDUAL EVALUATION", "BATCH PROCESSING OVERVIEW", "EXPLAINABLE AI (XAI) MATRIX"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("ACCOUNT ASSESSMENT PIPELINE")
        customer_id = st.text_input("CUSTOMER IDENTIFICATION HASH", "0000099db563072c3d5e7")
        st.write("---")
        st.write("**HIGH-DIMENSIONAL FEATURE VECTORS**")
        p2_mean = st.slider("PAYMENT FACTOR METRIC (P_2 MEAN)", 0.0, 1.0, 0.75)
        d39_max = st.slider("DELINQUENCY METRIC (D_39 MAX)", 0.0, 1.0, 0.05)
        b1_last = st.slider("BALANCE VELOCITY METRIC (B_1 LAST)", 0.0, 1.0, 0.12)
        execute_inference = st.button("RUN UNDERWRITING INFERENCE MATRIX")

    with col2:
        st.subheader("BEHAVIORAL HISTORICAL TREND DATA SUMMARY")
        fig = ob.Figure()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        fig.add_trace(ob.Scatter(x=months, y=[0.24, 0.58, 0.45, 0.41, 0.15, 0.18], name="SPEND METRIC VARIANCE (S_*)", line=dict(color='#0077b6')))
        fig.add_trace(ob.Scatter(x=months, y=[0.29, 0.66, 0.48, 0.43, 0.12, 0.21], name="PAYMENT METRIC VARIANCE (P_*)", line=dict(color='#ffb703', dash='dash')))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig, use_container_width=True)

    if execute_inference:
        payload = {
            "customer_ID": customer_id,
            "PAYMENT FACTOR METRIC (P_2 MEAN)": p2_mean,
            "DELINQUENCY METRIC (D_39 MAX)": d39_max,
            "BALANCE VELOCITY METRIC (B_1 LAST)": b1_last
        }
        try:
            response = requests.post(predict_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success("INFERENCE MATRIX PROTOCOL COMPLETED SUCCESSFULLY")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("DEFAULT PROBABILITY SCORE", f"{result['default_probability'] * 100:.2f}%")
                    st.metric("PORTFOLIO RISK CLASSIFICATION", result['risk_tier'])
                with res_col2:
                    st.metric("RECOMMENDED CREDIT BOUNDARY", f"${result['recommended_credit_limit']:.2f}")
                    
                    # Safe multi-line check blocks to guarantee truncation immunity
                    if result['default_probability'] <= risk_threshold:
                        status_eval = "APPROVED"
                    else:
                        status_eval = "REJECTED"
                        
                    st.metric("DECISION MATRIX CONCLUSION", status_eval)
            else:
                st.error("SYSTEM ERROR: BACKEND SCHEMA MISMATCH OR EXCEPTION.")
        except Exception as e:
            st.error(f"CONNECTION TIMEOUT GATEWAY FAULT: {str(e)}")

with tab2:
    st.subheader("BATCH PROCESSING OVERVIEW INTERFACE TERMINAL")
    st.markdown("Upload bulk pipeline batch array blocks here formatted as standard JSON arrays containing evaluation profile maps:")
    
    mock_template = [
        {"customer_ID": "CUST_SAMPLE_A", "PAYMENT FACTOR METRIC (P_2 MEAN)": 0.85, "DELINQUENCY METRIC (D_39 MAX)": 0.02, "BALANCE VELOCITY METRIC (B_1 LAST)": 0.05},
        {"customer_ID": "CUST_SAMPLE_B", "PAYMENT FACTOR METRIC (P_2 MEAN)": 0.21, "DELINQUENCY METRIC (D_39 MAX)": 0.68, "BALANCE VELOCITY METRIC (B_1 LAST)": 0.45}
    ]
    st.download_button("Download Template Sample Profile Schema", data=json.dumps(mock_template, indent=2), file_name="amex_batch_template.json")
    
    uploaded_batch = st.file_uploader("Submit Account Assessment Batch Package File", type=["json"])
    if uploaded_batch is not None:
        if st.button("Trigger Async Pipeline Engine Execution"):
            try:
                files = {"file": (uploaded_batch.name, uploaded_batch.getvalue(), "application/json")}
                resp = requests.post(batch_url, files=files, timeout=10)
                if resp.status_code == 202:
                    st.info(f"Batch queued into tracking registry context stack frame target: {resp.json()['batch_id']}")
                else:
                    st.error(resp.text)
            except Exception as e:
                st.error(str(e))
                
    st.write("---")
    st.write("**POLL PROCESSING METRIC QUEUE STATUS**")
    check_id = st.text_input("Enter Active Tracking Task Pointer Key", "BATCH_1")
    if st.button("Query Queue Processing Context"):
        try:
            status_resp = requests.get(f"{BACKEND_BASE}/api/v1/batch-status/{check_id}")
            if status_resp.status_code == 200:
                st.json(status_resp.json())
            else:
                st.error("Tracking ID allocation index reference mapping target context mismatch.")
        except Exception as e:
            st.error(str(e))

with tab3:
    st.subheader("EXPLAINABLE AI (XAI) ATTRIBUTION ENGINE")
    st.markdown("Renders local attribution metrics tracking feature impact vectors via Shapley charts.")
    
    search_id = st.text_input("Target Profiling Verification Index Tag (Customer ID)", "0000099db563072c3d5e7")
    if st.button("Compute Local Shapley Optimization Vector"):
        try:
            explain_resp = requests.get(f"{BACKEND_BASE}/api/v1/explain/{search_id}")
            if explain_resp.status_code == 200:
                xai_data = explain_resp.json()
                
                st.write(f"**Baseline Prediction Matrix Base Value:** `{xai_data['base_expected_value']}`")
                st.write(f"**Final Evaluated Target Risk Probability Output Value:** `{xai_data['final_predicted_probability']}`")
                
                features = [item["feature"] for item in xai_data["shap_attributions"]]
                values = [item["shap_value"] for item in xai_data["shap_attributions"]]
                
                # Render horizontal attribution map inside Plotly workspace
                shap_fig = ob.Figure(ob.Bar(
                    x=values, y=features, orientation='h',
                    marker=dict(color=['#2a9d8f' if val < 0 else '#e63946' for val in values])
                ))
                shap_fig.update_layout(title="SHAP Feature Attribution Factor Influence Maps", template="plotly_dark")
                st.plotly_chart(shap_fig, use_container_width=True)
            else:
                st.warning("Ensure that an individual inference run under Tab 1 has been executed for this requested customer ID first.")
        except Exception as e:
            st.error(str(e))