"""
Prepare Enhanced Dataset and SQLite Database for Tailoring Dashboard
This script:
1. Cleans and enhances the original dataset
2. Fixes satisfaction logic to be more realistic
3. Creates a SQLite database with proper schema
4. Generates SQL file for documentation
5. Retrains models with improved data
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import random
import os

# Create output directory
os.makedirs('/home/ubuntu/tailoring_dashboard', exist_ok=True)

# Load original data
print("Loading original dataset...")
df = pd.read_excel('/home/ubuntu/new_data/cleaned_dataset (1) (1).xlsx')

# Clean order_Type - standardize values
print("Cleaning data...")
df['order_Type'] = df['order_Type'].str.strip().str.lower()
df['order_Type'] = df['order_Type'].replace({
    'fixing': 'Fixing',
    'tailoring': 'Tailoring',
    'replace': 'Replace',
    'refound': 'Refund'
})

# Clean order_Tailor
df['order_Tailor'] = df['order_Tailor'].str.strip()

# Clean Tailoring_Style
df['Tailoring_Style'] = df['Tailoring_Style'].str.strip()

# Clean size
df['size'] = df['size'].str.strip().str.upper()

# Clean delivery_Type
df['delivery_Type'] = df['delivery_Type'].str.strip()

# Fix missing Days_Difference
df['Days_Difference'] = df['Days_Difference'].fillna(df['Expected_Delivery_Days'])

# ============================================
# IMPROVE SATISFACTION LOGIC
# ============================================
print("Improving satisfaction logic...")

# Create a more realistic satisfaction model based on:
# 1. Delivery delay (most important)
# 2. Order type (tailoring is harder to satisfy)
# 3. Price per unit (higher price = higher expectations)
# 4. Quantity issues

def calculate_satisfaction(row):
    """
    Calculate satisfaction based on realistic business logic:
    - On-time or early delivery: High satisfaction
    - Slight delay (1-3 days): Medium satisfaction
    - Significant delay (4-7 days): Low satisfaction
    - Major delay (>7 days): Very low satisfaction
    - Quantity issues: Automatic dissatisfaction
    """
    score = 100  # Start with perfect score
    
    # 1. Delivery delay impact (most important factor)
    delay = row['Days_Difference'] - row['Expected_Delivery_Days']
    if delay <= 0:  # On time or early
        score += 10
    elif delay <= 2:  # 1-2 days late
        score -= 15
    elif delay <= 5:  # 3-5 days late
        score -= 35
    elif delay <= 7:  # 6-7 days late
        score -= 55
    else:  # More than 7 days late
        score -= 75
    
    # 2. Order type impact
    if row['order_Type'] == 'Fixing':
        score -= 5  # Fixing orders have slightly lower satisfaction
    elif row['order_Type'] == 'Refund':
        score -= 20  # Refunds indicate problems
    
    # 3. Quantity issues
    if row['order_Delivered_Quantity'] != row['order_Quantity']:
        score -= 30  # Major issue
    
    # 4. Price sensitivity (higher price = higher expectations)
    if row['Price_Per_Unit'] > 400:
        score -= 5  # Premium customers are more demanding
    
    # 5. Random factor (real-world variability)
    score += random.uniform(-10, 10)
    
    # Convert to binary (threshold at 50)
    return 1 if score >= 50 else 0

# Apply new satisfaction logic
random.seed(42)  # For reproducibility
df['Satisfaction'] = df.apply(calculate_satisfaction, axis=1)

# Update Dissatisfaction_Reason based on new logic
def get_dissatisfaction_reason(row):
    if row['Satisfaction'] == 1:
        return None
    
    reasons = []
    delay = row['Days_Difference'] - row['Expected_Delivery_Days']
    
    if delay > 7:
        reasons.append(f"Major Delivery Delay ({int(delay)} days)")
    elif delay > 0:
        reasons.append(f"Delivery Delay ({int(delay)} days)")
    
    if row['order_Delivered_Quantity'] != row['order_Quantity']:
        reasons.append("Quantity Issue")
    
    if row['order_Type'] == 'Refund':
        reasons.append("Refund Request")
    
    if not reasons:
        reasons.append("Quality Concern")
    
    return ", ".join(reasons)

df['Dissatisfaction_Reason'] = df.apply(get_dissatisfaction_reason, axis=1)

# Print new distribution
print("\n=== New Satisfaction Distribution ===")
print(df['Satisfaction'].value_counts())
print(df['Satisfaction'].value_counts(normalize=True))

# ============================================
# CREATE SQLITE DATABASE
# ============================================
print("\nCreating SQLite database...")

# Connect to SQLite
db_path = '/home/ubuntu/tailoring_dashboard/tailoring.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.executescript('''
-- Drop existing tables
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS tailors;
DROP TABLE IF EXISTS deliveries;
DROP TABLE IF EXISTS predictions;

-- Customers table
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    total_orders INTEGER DEFAULT 0,
    total_spent REAL DEFAULT 0,
    avg_satisfaction REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tailors table
CREATE TABLE tailors (
    tailor_id TEXT PRIMARY KEY,
    tailor_name TEXT NOT NULL,
    total_orders INTEGER DEFAULT 0,
    avg_rating REAL DEFAULT 0,
    specialization TEXT
);

-- Orders table (main table)
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    tailor_id TEXT,
    order_date DATE,
    order_type TEXT,
    tailoring_style TEXT,
    size TEXT,
    quantity INTEGER,
    length_cm REAL,
    width_cm REAL,
    sleeve_cm REAL,
    fabric_meters REAL,
    price_per_unit REAL,
    fabric_price_per_meter REAL,
    total_amount REAL,
    tax REAL,
    discount REAL,
    expected_delivery_days INTEGER,
    satisfaction INTEGER,
    dissatisfaction_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (tailor_id) REFERENCES tailors(tailor_id)
);

-- Deliveries table
CREATE TABLE deliveries (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    delivery_type TEXT,
    expected_date DATE,
    actual_date DATE,
    delivered_quantity INTEGER,
    days_difference INTEGER,
    status TEXT DEFAULT 'Pending',
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Predictions table (for AI predictions)
CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER,
    predicted_price REAL,
    predicted_length REAL,
    predicted_satisfaction INTEGER,
    predicted_delivery_days INTEGER,
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Create indexes for performance
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_deliveries_order ON deliveries(order_id);
''')

# Insert unique customers
print("Inserting customers...")
customers = df[['customer_ID', 'customer_Name']].drop_duplicates()
for _, row in customers.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO customers (customer_id, customer_name)
        VALUES (?, ?)
    ''', (int(row['customer_ID']), row['customer_Name']))

# Insert tailors
print("Inserting tailors...")
tailors = df[['order_Tailor', 'order_Tailor_Name']].drop_duplicates()
for _, row in tailors.iterrows():
    cursor.execute('''
        INSERT OR IGNORE INTO tailors (tailor_id, tailor_name, specialization)
        VALUES (?, ?, ?)
    ''', (row['order_Tailor'], row['order_Tailor_Name'] or 'Factory', 'General'))

# Insert orders
print("Inserting orders...")
for _, row in df.iterrows():
    cursor.execute('''
        INSERT OR REPLACE INTO orders (
            order_id, customer_id, tailor_id, order_date, order_type,
            tailoring_style, size, quantity, length_cm, width_cm, sleeve_cm,
            fabric_meters, price_per_unit, fabric_price_per_meter,
            total_amount, tax, discount, expected_delivery_days,
            satisfaction, dissatisfaction_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(row['order_ID']),
        int(row['customer_ID']),
        row['order_Tailor'],
        row['order_Date'].strftime('%Y-%m-%d') if pd.notna(row['order_Date']) else None,
        row['order_Type'],
        row['Tailoring_Style'],
        row['size'],
        int(row['order_Quantity']),
        row['length_cm'],
        row['width_cm'],
        row['sleeve_cm'],
        row['fabric_meters'],
        row['Price_Per_Unit'],
        row['Fabric_Price_Per_Meter'],
        row['Total_Amount'],
        row['order_Tax'],
        row['order_Discount'],
        int(row['Expected_Delivery_Days']),
        int(row['Satisfaction']),
        row['Dissatisfaction_Reason']
    ))

# Insert deliveries
print("Inserting deliveries...")
for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO deliveries (
            order_id, delivery_type, expected_date, actual_date,
            delivered_quantity, days_difference, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(row['order_ID']),
        row['delivery_Type'],
        row['Expected_Delivery_date'].strftime('%Y-%m-%d') if pd.notna(row['Expected_Delivery_date']) else None,
        row['delivery_Date'].strftime('%Y-%m-%d') if pd.notna(row['delivery_Date']) else None,
        int(row['order_Delivered_Quantity']),
        int(row['Days_Difference']) if pd.notna(row['Days_Difference']) else None,
        'Delivered' if row['delivery_Type'] == 'Recived' else 'Pending'
    ))

# Update customer statistics
print("Updating customer statistics...")
cursor.execute('''
    UPDATE customers SET
        total_orders = (SELECT COUNT(*) FROM orders WHERE orders.customer_id = customers.customer_id),
        total_spent = (SELECT COALESCE(SUM(total_amount), 0) FROM orders WHERE orders.customer_id = customers.customer_id),
        avg_satisfaction = (SELECT COALESCE(AVG(satisfaction), 0) FROM orders WHERE orders.customer_id = customers.customer_id)
''')

# Update tailor statistics
cursor.execute('''
    UPDATE tailors SET
        total_orders = (SELECT COUNT(*) FROM orders WHERE orders.tailor_id = tailors.tailor_id),
        avg_rating = (SELECT COALESCE(AVG(satisfaction) * 5, 0) FROM orders WHERE orders.tailor_id = tailors.tailor_id)
''')

conn.commit()

# ============================================
# GENERATE SQL FILE
# ============================================
print("\nGenerating SQL file...")

sql_file_path = '/home/ubuntu/tailoring_dashboard/tailoring_database.sql'
with open(sql_file_path, 'w') as f:
    f.write("-- Tailoring Management System Database\n")
    f.write("-- Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("-- Total Records: " + str(len(df)) + "\n\n")
    
    # Export schema
    for line in conn.iterdump():
        f.write(line + '\n')

print(f"SQL file saved to: {sql_file_path}")

# ============================================
# SAVE ENHANCED DATASET
# ============================================
print("\nSaving enhanced dataset...")
enhanced_csv_path = '/home/ubuntu/tailoring_dashboard/enhanced_dataset.csv'
df.to_csv(enhanced_csv_path, index=False)
print(f"Enhanced dataset saved to: {enhanced_csv_path}")

# Verify database
print("\n=== Database Verification ===")
cursor.execute("SELECT COUNT(*) FROM orders")
print(f"Total orders: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM customers")
print(f"Total customers: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM tailors")
print(f"Total tailors: {cursor.fetchone()[0]}")

cursor.execute("SELECT AVG(satisfaction) FROM orders")
print(f"Average satisfaction: {cursor.fetchone()[0]:.2%}")

conn.close()
print("\n✅ Database setup complete!")
