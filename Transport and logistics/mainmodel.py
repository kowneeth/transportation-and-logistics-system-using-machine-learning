import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
import joblib

# ---------------------------------------------------------
# 1. LOAD DATA FROM ALL SHEETS
# ---------------------------------------------------------
DATA_PATH = r"C:\Users\kowne\Downloads\transport\Transportation and Logistics Tracking Dataset.xlsx"

# Load Primary data sheet
df_primary = pd.read_excel(DATA_PATH, sheet_name="Primary data")

# Load Refined sheet
df_refined = pd.read_excel(DATA_PATH, sheet_name="Refined")

# Load Q1-Q10 sheets
q_sheets = [f"Q{i}" for i in range(1, 11)]
df_q = []
for sheet in q_sheets:
    try:
        df_q.append(pd.read_excel(DATA_PATH, sheet_name=sheet))
    except:
        pass

print(f"Primary data shape: {df_primary.shape}")
print(f"Refined shape: {df_refined.shape}")
print(f"Q-sheets loaded: {len(df_q)}")

# Merge Primary sheet data with Refined sheet for additional fields
# Select relevant columns from Primary sheet: Driver_Name, Current_Location, supplierNameCode, vehicle_no
primary_cols = ['BookingID', 'Driver_Name', 'Current_Location', 'supplierNameCode', 'vehicle_no']
df_primary_subset = df_primary[primary_cols].drop_duplicates().reset_index(drop=True)

# Rename booking ID to delivery_id to match
df_primary_subset.rename(columns={'BookingID': 'delivery_id'}, inplace=True)

print(f"\nPrimary subset shape: {df_primary_subset.shape}")

# ---------------------------------------------------------
# 2. CLEAN + RENAME COLUMNS
# ---------------------------------------------------------

# Use only the refined columns from the original refined sheet
df = df_refined.copy()

df.columns = [
    "delivery_id", "origin_location", "destination_location", "region",
    "created_at", "actual_delivery_time", "on_time_delivery", "customer_rating",
    "condition_text", "fixed_costs", "maintenance", "difference",
    "area", "delivery_time"
]

# Merge with Primary sheet data to add driver_name, current_location, supplier_name
df = df.merge(df_primary_subset, on='delivery_id', how='left')

print(f"\nDataframe shape after merge: {df.shape}")

# Remove duplicates
df = df.loc[:, ~df.columns.duplicated()]
print(f"\nFinal dataset shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------
# 3. MAP CONDITIONS TO GOOD/BAD/WORST & ADD WEATHER
# ---------------------------------------------------------

def map_cargo_condition(condition):
    """Map weather conditions to cargo condition rating (Good/Bad/Worst)"""
    if pd.isna(condition):
        return 'Good'
    
    condition_str = str(condition).lower()
    
    # Good conditions
    if any(word in condition_str for word in ['sunny', 'clear', 'partly cloudy']):
        return 'Good'
    # Bad conditions
    elif any(word in condition_str for word in ['cloudy', 'overcast', 'mist']):
        return 'Bad'
    # Worst conditions
    elif any(word in condition_str for word in ['rainy', 'rain', 'patchy', 'drizzle', 'stormy']):
        return 'Worst'
    else:
        return 'Good'  # Default

# Add new columns for cargo condition category and weather
df['cargo_condition'] = df['condition_text'].apply(map_cargo_condition)
df['weather'] = df['condition_text'].astype(str)  # Keep original weather text

print("\nCargo condition mapping applied")
print(f"Good conditions: {(df['cargo_condition'] == 'Good').sum()}")
print(f"Bad conditions: {(df['cargo_condition'] == 'Bad').sum()}")
print(f"Worst conditions: {(df['cargo_condition'] == 'Worst').sum()}")

# Update df_refined with new columns
df_refined = df.copy()

# ---------------------------------------------------------
# 4. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------
cat_cols = [
    "origin_location", "destination_location",
    "region", "condition_text", "area", "cargo_condition"
]

encoders = {}

for col in cat_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col].astype(str))
    encoders[col] = enc

print("\nEncoders created for categorical columns")

# ---------------------------------------------------------
# 5. FEATURE SELECTION & MODEL TRAINING
# ---------------------------------------------------------

features = [
    "origin_location", "destination_location", "region",
    "customer_rating", "fixed_costs", "maintenance",
    "condition_text", "cargo_condition", "area", "delivery_time"
]

X = df[features].copy()
y = df["delivery_time"].copy()  # Use delivery_time instead of actual_delivery_time

# Convert to numeric if needed
y = pd.to_numeric(y, errors='coerce')

# Handle missing values - fill with median
X = X.fillna(X.median(numeric_only=True))
y = y.fillna(y.median())

# Remove outliers in target variable
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
y = y[(y >= (Q1 - 1.5 * IQR)) & (y <= (Q3 + 1.5 * IQR))]
X = X.loc[y.index]

print(f"Features after outlier removal: {X.shape}")
print(f"Target stats - Mean: {y.mean():.2f}, Std: {y.std():.2f}, Min: {y.min():.2f}, Max: {y.max():.2f}")

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features shape: {X_scaled.shape}")
print(f"Target shape: {y.shape}")

# ---------------------------------------------------------
# 5. TRAIN MULTIPLE MODELS FOR BETTER ACCURACY
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Random Forest Model with better hyperparameters
model = RandomForestRegressor(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training R² Score: {train_score:.4f}")
print(f"Testing R² Score: {test_score:.4f}")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"Root Mean Squared Error: {rmse:.2f} minutes")
print(f"Average Delivery Time: {y.mean():.2f} minutes")
print("="*60)

# ---------------------------------------------------------
# 6. BEST ROUTE & SUPPLIER RECOMMENDATION FUNCTION
# ---------------------------------------------------------

def get_best_suppliers_for_route(origin, destination, top_n=5):
    """
    Get best suppliers/transporters for a specific route based on:
    - Customer rating
    - On-time delivery rate
    - Average delivery time
    - Cargo condition
    """
    
    # Filter deliveries for this route
    route_deliveries = df_refined[
        (df_refined['origin_location'] == origin) & 
        (df_refined['destination_location'] == destination)
    ].copy()
    
    if len(route_deliveries) == 0:
        # Try reverse route
        route_deliveries = df_refined[
            (df_refined['origin_location'] == destination) & 
            (df_refined['destination_location'] == origin)
        ].copy()
        if len(route_deliveries) == 0:
            return None
    
    # Group by supplier (using delivery_id as proxy for supplier)
    # Calculate aggregated metrics
    suppliers = []
    
    # Since we don't have explicit supplier names, we'll use delivery groups
    for idx, delivery in route_deliveries.iterrows():
        supplier_data = {
            'delivery_id': delivery['delivery_id'],
            'driver_name': delivery.get('Driver_Name', 'Not specified'),
            'supplier_name': delivery.get('supplierNameCode', 'Not specified'),
            'vehicle_no': delivery.get('vehicle_no', 'Not specified'),
            'current_location': delivery.get('Current_Location', 'Not specified'),
            'origin': delivery['origin_location'],
            'destination': delivery['destination_location'],
            'rating': delivery['customer_rating'],
            'delivery_time': delivery['delivery_time'],
            'condition': delivery['condition_text'],
            'cargo_condition': delivery['cargo_condition'],
            'weather': delivery['weather'],
            'on_time': delivery['on_time_delivery'],
            'fixed_cost': delivery['fixed_costs'],
            'maintenance': delivery['maintenance'],
            'area': delivery['area']
        }
        suppliers.append(supplier_data)
    
    # Sort by rating (descending)
    suppliers_df = pd.DataFrame(suppliers)
    suppliers_df = suppliers_df.sort_values('rating', ascending=False)
    
    # Calculate route statistics
    route_stats = {
        'total_deliveries': len(route_deliveries),
        'avg_rating': route_deliveries['customer_rating'].mean(),
        'avg_delivery_time': route_deliveries['delivery_time'].mean(),
        'on_time_rate': (route_deliveries['on_time_delivery'].sum() / len(route_deliveries)) * 100,
        'best_weather': route_deliveries['weather'].mode().values[0] if len(route_deliveries) > 0 else 'Unknown',
        'best_cargo_condition': route_deliveries['cargo_condition'].mode().values[0] if len(route_deliveries) > 0 else 'Good',
        'avg_cost': route_deliveries['fixed_costs'].mean(),
        'avg_maintenance': route_deliveries['maintenance'].mean()
    }
    
    return {
        'suppliers': suppliers_df.head(top_n).to_dict('records'),
        'route_stats': route_stats,
        'total_suppliers': len(suppliers_df)
    }

def display_best_suppliers(origin, destination, supplier_info):
    """Display best suppliers for a route"""
    print("\n" + "="*80)
    print("BEST TRANSPORTERS/SUPPLIERS FOR YOUR ROUTE")
    print("="*80)
    
    if supplier_info is None:
        print("\nNo historical data available for this route.")
        print("System will use ML prediction for cost and time estimates.")
        return
    
    route_stats = supplier_info['route_stats']
    
    print(f"\nROUTE INFORMATION:")
    print(f"  From: {origin}")
    print(f"  To: {destination}")
    print(f"  Total Historical Deliveries: {route_stats['total_deliveries']}")
    print(f"  Average Cost: Rs {route_stats['avg_cost']:.0f}")
    print(f"  Average Maintenance: Rs {route_stats['avg_maintenance']:.0f}")
    
    print(f"\nROUTE PERFORMANCE METRICS:")
    print(f"  Average Rating: {route_stats['avg_rating']:.2f}/5.0 stars")
    print(f"  Average Delivery Time: {route_stats['avg_delivery_time']:.1f} minutes ({route_stats['avg_delivery_time']/60:.2f} hours)")
    print(f"  On-Time Delivery Rate: {route_stats['on_time_rate']:.1f}%")
    print(f"  Best Weather Condition: {route_stats['best_weather']}")
    print(f"  Best Cargo Condition: {route_stats['best_cargo_condition']}")
    
    print(f"\nTOP {min(5, len(supplier_info['suppliers']))} BEST TRANSPORTERS:")
    print("-" * 80)
    
    for idx, supplier in enumerate(supplier_info['suppliers'], 1):
        print(f"\nTransporter #{idx}:")
        print(f"  Service ID: {supplier['delivery_id']}")
        print(f"  Driver Name: {supplier['driver_name']}")
        print(f"  Supplier Name: {supplier['supplier_name']}")
        print(f"  Vehicle Number: {supplier['vehicle_no']}")
        print(f"  Current Location: {supplier['current_location']}")
        print(f"  Customer Rating: {supplier['rating']:.2f}/5.0 stars")
        
        # Handle delivery time NaN values
        if pd.notna(supplier['delivery_time']):
            delivery_time_str = f"{supplier['delivery_time']:.0f} minutes ({supplier['delivery_time']/60:.2f} hours)"
        else:
            delivery_time_str = "Not recorded"
        print(f"  Delivery Time: {delivery_time_str}")
        
        print(f"  On-Time Delivery: {'Yes' if supplier['on_time'] else 'No'}")
        print(f"  Weather: {supplier['weather']}")
        print(f"  Cargo Condition: {supplier['cargo_condition']}")
        print(f"  Fixed Cost: Rs {supplier['fixed_cost']:.0f}")
        print(f"  Maintenance Cost: Rs {supplier['maintenance']:.0f}")
        print(f"  Total Cost: Rs {supplier['fixed_cost'] + supplier['maintenance']:.0f}")
        
        # Handle area NaN values
        if pd.notna(supplier['area']):
            print(f"  Service Area: {supplier['area']}")
        else:
            print(f"  Service Area: Not specified")
    
    print("\n" + "="*80)

def recommend_best_service(origin, destination, condition_text="Sunny", customer_rating=4.5,
                          fixed_costs=5000, maintenance=1200, area="Rural"):
    """Recommend best vehicle/service for transportation"""
    
    # Check route history
    route_info = analyze_route(origin, destination)
    
    if route_info:
        print(f"\nRoute History Analysis:")
        print(f"  Total Deliveries: {route_info['total_deliveries']}")
        print(f"  Average Delivery Time: {route_info['avg_delivery_time']:.1f} minutes")
        print(f"  Average Rating: {route_info['avg_rating']:.2f}/5.0")
        print(f"  On-Time Success Rate: {route_info['success_rate']:.1f}%")
    
    # Get region from destination
    dest_region = df_refined[df_refined['destination_location'] == destination]['region'].iloc[0] if destination in df_refined['destination_location'].values else "Tamil Nadu"
    
    # Create input for prediction
    try:
        input_data = pd.DataFrame([[
            encoders["origin_location"].transform([origin])[0],
            encoders["destination_location"].transform([destination])[0],
            encoders["region"].transform([dest_region])[0],
            customer_rating,
            fixed_costs,
            maintenance,
            encoders["condition_text"].transform([condition_text])[0],
            encoders["area"].transform([area])[0],
            route_info['avg_delivery_time'] if route_info else 45  # Use actual average delivery time
        ]], columns=features)
        
        input_scaled = scaler.transform(input_data)
        predicted_time = model.predict(input_scaled)[0]
        
        # Clamp to realistic values
        predicted_time = max(20, min(predicted_time, 500))
        
        # Recommendation logic based on predicted time
        if predicted_time < 60:
            vehicle_type = "Express Delivery (2-wheeler/Auto)"
            cost_estimate = 2000
            speed = "Fast"
        elif predicted_time < 120:
            vehicle_type = "Standard Van (4-wheeler)"
            cost_estimate = 5000
            speed = "Normal"
        else:
            vehicle_type = "Heavy Truck (Multi-axle)"
            cost_estimate = 8000
            speed = "Slower (more capacity)"
        
        return {
            'origin': origin,
            'destination': destination,
            'recommended_vehicle': vehicle_type,
            'estimated_delivery_time_mins': float(predicted_time),
            'estimated_delivery_time_hours': float(predicted_time / 60),
            'estimated_cost': cost_estimate,
            'speed_category': speed,
            'condition': condition_text,
            'route_found': True if route_info else False
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# 7. SAVE TRAINED MODEL
# ---------------------------------------------------------

# Save model data for Flask app
model_data = {
    'model': model,
    'encoders': encoders,
    'scaler': scaler,
    'df_refined': df_refined
}

os.makedirs('models', exist_ok=True)
joblib.dump(model_data, 'models/transportation_model.pkl')

print("\n" + "="*70)
print("MODEL SAVED SUCCESSFULLY")
print("="*70)
print(f"Model Type: Random Forest Regressor")
print(f"Training Accuracy (R²): {train_score:.4f}")
print(f"Testing Accuracy (R²): {test_score:.4f}")
print(f"Mean Absolute Error: {mae:.2f} minutes")
print(f"Dataset Size: {len(df_refined)} records")
print(f"Model saved to: models/transportation_model.pkl")
print("="*70)
