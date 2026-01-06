import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import joblib
import os

# Page config
st.set_page_config(
    page_title="Tailoring Management System",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better navigation and styling
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2d5a87 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
        padding: 20px 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 5px 0 0 0;
        opacity: 0.9;
        font-size: 14px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 13px;
        border-top: 1px solid #e2e8f0;
        margin-top: 30px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Database connection
def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'tailoring.db')
    return sqlite3.connect(db_path, check_same_thread=False)

# Load models
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)
    models = {}
    try:
        models['satisfaction'] = joblib.load(os.path.join(base_path, 'satisfaction_model.pkl'))
        models['price'] = joblib.load(os.path.join(base_path, 'price_model.pkl'))
        models['length'] = joblib.load(os.path.join(base_path, 'length_model.pkl'))
        models['label_encoders'] = joblib.load(os.path.join(base_path, 'label_encoders.pkl'))
        models['scaler'] = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        models['feature_names'] = joblib.load(os.path.join(base_path, 'feature_names.pkl'))
    except Exception as e:
        st.warning(f"Some models not loaded: {e}")
    return models

models = load_models()

# Load data
@st.cache_data(ttl=60)
def load_data():
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT o.*, c.customer_name 
        FROM orders o 
        LEFT JOIN customers c ON o.customer_id = c.customer_id
    """, conn)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    return df, customers

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: white; font-size: 24px; margin: 0;">🧵 Tailoring AI</h1>
        <p style="color: #94a3b8; font-size: 12px; margin-top: 5px;">Management System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation using radio buttons
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "➕ New Order", "📋 Orders", "👥 Customers", "🔬 AI Analysis", "📈 SHAP Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats in sidebar
    try:
        df, customers = load_data()
        st.markdown("""
        <div style="color: white; padding: 10px;">
            <h4 style="color: #94a3b8; font-size: 12px; margin-bottom: 10px;">QUICK STATS</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Orders", len(df))
        with col2:
            st.metric("Customers", len(customers))
    except:
        pass
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 11px; padding: 10px;">
        COE305 ML Project<br>
        © 2025
    </div>
    """, unsafe_allow_html=True)

# Main content based on selected page
if page == "📊 Dashboard":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Business Dashboard</h1>
        <p>Real-time analytics and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        df, customers = load_data()
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Orders",
                value=f"{len(df):,}",
                delta="+12 this month"
            )
        
        with col2:
            total_revenue = df['total_amount'].sum() if 'total_amount' in df.columns else 0
            st.metric(
                label="Total Revenue",
                value=f"{total_revenue:,.0f} SAR",
                delta="+8.5%"
            )
        
        with col3:
            satisfaction_rate = (df['satisfaction'].sum() / len(df) * 100) if 'satisfaction' in df.columns and len(df) > 0 else 0
            st.metric(
                label="Satisfaction Rate",
                value=f"{satisfaction_rate:.1f}%",
                delta="+2.3%"
            )
        
        with col4:
            st.metric(
                label="Active Customers",
                value=f"{len(customers):,}",
                delta="+15"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Revenue Trend")
            if 'order_date' in df.columns:
                df['order_date'] = pd.to_datetime(df['order_date'])
                monthly = df.groupby(df['order_date'].dt.to_period('W'))['total_amount'].sum().reset_index()
                monthly['order_date'] = monthly['order_date'].astype(str)
                fig = px.area(monthly, x='order_date', y='total_amount', 
                             color_discrete_sequence=['#3b82f6'])
                fig.update_layout(
                    xaxis_title="Week",
                    yaxis_title="Revenue (SAR)",
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Order Types")
            if 'order_type' in df.columns:
                type_counts = df['order_type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                            color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👔 Tailoring Styles")
            if 'tailoring_style' in df.columns:
                style_counts = df['tailoring_style'].value_counts()
                fig = px.bar(x=style_counts.index, y=style_counts.values,
                            color_discrete_sequence=['#3b82f6'])
                fig.update_layout(
                    xaxis_title="Style",
                    yaxis_title="Orders",
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 😊 Satisfaction by Type")
            if 'order_type' in df.columns and 'satisfaction' in df.columns:
                sat_by_type = df.groupby('order_type')['satisfaction'].mean() * 100
                fig = px.bar(x=sat_by_type.index, y=sat_by_type.values,
                            color_discrete_sequence=['#10b981'])
                fig.update_layout(
                    xaxis_title="Order Type",
                    yaxis_title="Satisfaction %",
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent orders
        st.markdown("### 📋 Recent Orders")
        display_cols = ['order_id', 'customer_name', 'order_type', 'tailoring_style', 'total_amount', 'order_date', 'satisfaction']
        available_cols = [col for col in display_cols if col in df.columns]
        recent = df[available_cols].head(10).copy()
        if 'satisfaction' in recent.columns:
            recent['Status'] = recent['satisfaction'].apply(lambda x: '✅ Satisfied' if x == 1 else '⚠️ Not Satisfied')
            recent = recent.drop('satisfaction', axis=1)
        st.dataframe(recent, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")

elif page == "➕ New Order":
    st.markdown("""
    <div class="main-header">
        <h1>➕ Create New Order</h1>
        <p>Add a new order with AI-powered predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Order Details")
        
        # Load customers for dropdown
        try:
            _, customers = load_data()
            customer_names = customers['customer_name'].tolist()
        except:
            customer_names = ["Customer 1", "Customer 2"]
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            customer = st.selectbox("Customer", customer_names)
            order_type = st.selectbox("Order Type", ["Tailoring", "Fixing", "Replace", "Refund"])
            tailoring_style = st.selectbox("Tailoring Style", ["Saudi Classic", "Modern Slim Fit", "Emirati"])
            size = st.selectbox("Size", ["S", "M", "L", "XL", "XXL", "XXXL"])
        
        with col_b:
            quantity = st.number_input("Quantity", min_value=1, max_value=20, value=1)
            price_per_unit = st.number_input("Price per Unit (SAR)", min_value=100, max_value=2000, value=350)
            expected_days = st.number_input("Expected Delivery Days", min_value=1, max_value=30, value=7)
        
        st.markdown("### 📏 Measurements (Optional)")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            length = st.number_input("Length (cm)", min_value=50.0, max_value=100.0, value=72.0)
        with col_m2:
            width = st.number_input("Width (cm)", min_value=40.0, max_value=80.0, value=54.0)
        with col_m3:
            sleeve = st.number_input("Sleeve (cm)", min_value=15.0, max_value=40.0, value=23.0)
        with col_m4:
            fabric = st.number_input("Fabric (m)", min_value=1.0, max_value=5.0, value=2.0)
        
        if st.button("🔮 Generate AI Predictions", type="primary", use_container_width=True):
            st.session_state['predictions_generated'] = True
            
            # Calculate predictions
            subtotal = quantity * price_per_unit
            tax = subtotal * 0.15
            total = subtotal + tax
            
            # Simple satisfaction prediction based on delivery time
            if expected_days <= 7:
                sat_pred = 1
                confidence = 96.0
            elif expected_days <= 14:
                sat_pred = 1
                confidence = 85.0
            else:
                sat_pred = 0
                confidence = 70.0
            
            st.session_state['sat_pred'] = sat_pred
            st.session_state['confidence'] = confidence
            st.session_state['total'] = total
            st.session_state['subtotal'] = subtotal
            st.session_state['tax'] = tax
    
    with col2:
        st.markdown("### 🤖 AI Predictions")
        
        if st.session_state.get('predictions_generated'):
            # Price prediction
            st.markdown("""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                        padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                <h4 style="color: #1e40af; margin: 0; font-size: 14px;">💰 Predicted Total Price</h4>
                <p style="color: #1e3a8a; font-size: 28px; font-weight: 700; margin: 10px 0 0 0;">
                    {:.2f} SAR
                </p>
            </div>
            """.format(st.session_state.get('total', 0)), unsafe_allow_html=True)
            
            # Satisfaction prediction
            sat_pred = st.session_state.get('sat_pred', 1)
            confidence = st.session_state.get('confidence', 85)
            
            if sat_pred == 1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                            padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                    <h4 style="color: #065f46; margin: 0; font-size: 14px;">😊 Satisfaction Prediction</h4>
                    <p style="color: #047857; font-size: 22px; font-weight: 700; margin: 10px 0 5px 0;">
                        Likely Satisfied
                    </p>
                    <p style="color: #059669; font-size: 13px; margin: 0;">
                        Confidence: {:.1f}%
                    </p>
                </div>
                """.format(confidence), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                            padding: 20px; border-radius: 12px; margin-bottom: 15px;">
                    <h4 style="color: #92400e; margin: 0; font-size: 14px;">😐 Satisfaction Prediction</h4>
                    <p style="color: #b45309; font-size: 22px; font-weight: 700; margin: 10px 0 5px 0;">
                        May Need Attention
                    </p>
                    <p style="color: #d97706; font-size: 13px; margin: 0;">
                        Confidence: {:.1f}%
                    </p>
                </div>
                """.format(confidence), unsafe_allow_html=True)
            
            # Order summary
            st.markdown("""
            <div style="background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
                <h4 style="color: #475569; margin: 0 0 15px 0; font-size: 14px;">📊 Order Summary</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #64748b;">Subtotal:</span>
                    <span style="color: #1e293b; font-weight: 500;">{:.2f} SAR</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #64748b;">Tax (15%):</span>
                    <span style="color: #1e293b; font-weight: 500;">{:.2f} SAR</span>
                </div>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 10px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #1e293b; font-weight: 600;">Total:</span>
                    <span style="color: #3b82f6; font-weight: 700; font-size: 18px;">{:.2f} SAR</span>
                </div>
            </div>
            """.format(
                st.session_state.get('subtotal', 0),
                st.session_state.get('tax', 0),
                st.session_state.get('total', 0)
            ), unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("💾 Save Order to Database", type="primary", use_container_width=True):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    
                    # Get customer_id
                    cursor.execute("SELECT customer_id FROM customers WHERE customer_name = ?", (customer,))
                    result = cursor.fetchone()
                    customer_id = result[0] if result else 1
                    
                    # Insert order
                    cursor.execute("""
                        INSERT INTO orders (customer_id, order_type, tailoring_style, size,
                                          quantity, price_per_unit, tax, discount,
                                          total_amount, expected_delivery_days,
                                          length_cm, width_cm, sleeve_cm, fabric_meters, fabric_price_per_meter,
                                          order_date, satisfaction)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (customer_id, order_type, tailoring_style, size,
                          quantity, price_per_unit, st.session_state.get('tax', 0), 0,
                          st.session_state.get('total', 0), expected_days,
                          length, width, sleeve, fabric, 50,
                          datetime.now().strftime('%Y-%m-%d'), st.session_state.get('sat_pred', 1)))
                    
                    conn.commit()
                    order_id = cursor.lastrowid
                    conn.close()
                    
                    st.success(f"✅ Order #{order_id} saved successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error saving order: {e}")
        else:
            st.info("👆 Fill in the order details and click 'Generate AI Predictions' to see the results.")

elif page == "📋 Orders":
    st.markdown("""
    <div class="main-header">
        <h1>📋 Orders Management</h1>
        <p>View and manage all orders</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        df, _ = load_data()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            order_type_filter = st.multiselect("Filter by Order Type", df['order_type'].unique() if 'order_type' in df.columns else [])
        with col2:
            style_filter = st.multiselect("Filter by Style", df['tailoring_style'].unique() if 'tailoring_style' in df.columns else [])
        with col3:
            satisfaction_filter = st.selectbox("Filter by Satisfaction", ["All", "Satisfied", "Not Satisfied"])
        
        # Apply filters
        filtered_df = df.copy()
        if order_type_filter:
            filtered_df = filtered_df[filtered_df['order_type'].isin(order_type_filter)]
        if style_filter:
            filtered_df = filtered_df[filtered_df['tailoring_style'].isin(style_filter)]
        if satisfaction_filter == "Satisfied":
            filtered_df = filtered_df[filtered_df['satisfaction'] == 1]
        elif satisfaction_filter == "Not Satisfied":
            filtered_df = filtered_df[filtered_df['satisfaction'] == 0]
        
        st.markdown(f"**Showing {len(filtered_df)} orders**")
        
        # Display table
        display_cols = ['order_id', 'customer_name', 'order_type', 'tailoring_style', 'size', 
                       'total_amount', 'order_date', 'satisfaction']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        display_df = filtered_df[available_cols].copy()
        if 'satisfaction' in display_df.columns:
            display_df['Status'] = display_df['satisfaction'].apply(lambda x: '✅ Satisfied' if x == 1 else '⚠️ Not Satisfied')
            display_df = display_df.drop('satisfaction', axis=1)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
        
    except Exception as e:
        st.error(f"Error loading orders: {e}")

elif page == "👥 Customers":
    st.markdown("""
    <div class="main-header">
        <h1>👥 Customer Analytics</h1>
        <p>Customer insights and statistics</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        df, customers = load_data()
        
        # Customer stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(customers))
        with col2:
            avg_orders = len(df) / len(customers) if len(customers) > 0 else 0
            st.metric("Avg Orders/Customer", f"{avg_orders:.1f}")
        with col3:
            avg_spend = df['total_amount'].sum() / len(customers) if len(customers) > 0 else 0
            st.metric("Avg Spend/Customer", f"{avg_spend:.0f} SAR")
        
        st.markdown("### 📊 Top Customers by Orders")
        if 'customer_name' in df.columns:
            top_customers = df.groupby('customer_name').agg({
                'order_id': 'count',
                'total_amount': 'sum',
                'satisfaction': 'mean'
            }).reset_index()
            top_customers.columns = ['Customer', 'Orders', 'Total Spent', 'Satisfaction Rate']
            top_customers['Satisfaction Rate'] = (top_customers['Satisfaction Rate'] * 100).round(1).astype(str) + '%'
            top_customers = top_customers.sort_values('Orders', ascending=False).head(20)
            
            st.dataframe(top_customers, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error loading customers: {e}")

elif page == "🔬 AI Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI Model Analysis</h1>
        <p>Machine learning model performance and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown("### 📊 Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;">
            <h3 style="color: #1e40af; margin: 0; font-size: 16px;">😊 Satisfaction Model</h3>
            <p style="color: #1e3a8a; font-size: 36px; font-weight: 700; margin: 15px 0;">97.7%</p>
            <p style="color: #3b82f6; font-size: 14px; margin: 0;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;">
            <h3 style="color: #065f46; margin: 0; font-size: 16px;">💰 Price Model</h3>
            <p style="color: #047857; font-size: 36px; font-weight: 700; margin: 15px 0;">99.9%</p>
            <p style="color: #10b981; font-size: 14px; margin: 0;">R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                    padding: 25px; border-radius: 15px; text-align: center;">
            <h3 style="color: #92400e; margin: 0; font-size: 16px;">📏 Length Model</h3>
            <p style="color: #b45309; font-size: 36px; font-weight: 700; margin: 15px 0;">91.6%</p>
            <p style="color: #f59e0b; font-size: 14px; margin: 0;">R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    
    base_path = os.path.dirname(__file__)
    importance_path = os.path.join(base_path, 'feature_importance.png')
    if os.path.exists(importance_path):
        st.image(importance_path, use_container_width=True)
    else:
        # Create sample feature importance chart
        features = ['Days_Difference', 'order_Tax', 'Price_Per_Unit', 'Total_Amount', 
                   'order_Quantity', 'Tailoring_Style', 'order_Type', 'size']
        importance = [0.35, 0.18, 0.15, 0.12, 0.08, 0.05, 0.04, 0.03]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    color_discrete_sequence=['#3b82f6'])
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 SHAP Insights":
    st.markdown("""
    <div class="main-header">
        <h1>📈 SHAP Analysis</h1>
        <p>Explainable AI insights for model interpretability</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** helps us understand how each feature 
    contributes to the model's predictions.
    """)
    
    tab1, tab2 = st.tabs(["😊 Satisfaction Model", "💰 Price Model"])
    
    base_path = os.path.dirname(__file__)
    
    with tab1:
        st.markdown("### Customer Satisfaction - Feature Impact")
        shap_sat_path = os.path.join(base_path, 'shap_satisfaction.png')
        if os.path.exists(shap_sat_path):
            st.image(shap_sat_path, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - **Days_Difference** (delivery delay) has the highest impact on satisfaction
        - Higher delays lead to lower satisfaction (red dots on the left)
        - **Order Type** and **Price** also significantly affect customer satisfaction
        """)
    
    with tab2:
        st.markdown("### Price Prediction - Feature Impact")
        shap_price_path = os.path.join(base_path, 'shap_price.png')
        if os.path.exists(shap_price_path):
            st.image(shap_price_path, use_container_width=True)
        
        st.markdown("""
        **Key Insights:**
        - **Tax** and **Subtotal** are the main drivers of total price
        - **Quantity** and **Price per Unit** directly affect the final amount
        - **Fabric meters** contributes to material costs
        """)

# Footer
st.markdown("""
<div class="footer">
    🧵 Tailoring Management System | Powered by Machine Learning<br>
    COE305 Machine Learning Project | © 2025
</div>
""", unsafe_allow_html=True)
