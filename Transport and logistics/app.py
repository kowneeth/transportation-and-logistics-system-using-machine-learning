from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename

# Import your existing code
import sys
sys.path.append('.')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
df_refined = None
model = None
encoders = None
scaler = None
origins = []
destinations = []

def load_data_and_model():
    """Load data and trained model"""
    global df_refined, model, encoders, scaler, origins, destinations
    
    try:
        # Load your existing trained model
        model_data = joblib.load('models/transportation_model.pkl')
        model = model_data['model']
        encoders = model_data['encoders']
        scaler = model_data['scaler']
        df_refined = model_data['df_refined']
        
        # Get available locations
        origins = df_refined['origin_location'].unique().tolist()
        destinations = df_refined['destination_location'].unique().tolist()
        
        print("Model and data loaded successfully!")
    except:
        print("Model not found. Please train the model first.")
        # You can call your training function here if needed

# Initialize on startup
load_data_and_model()

# Helper functions from your code
def map_cargo_condition(condition):
    """Map weather conditions to cargo condition rating"""
    if pd.isna(condition):
        return 'Good'
    
    condition_str = str(condition).lower()
    
    if any(word in condition_str for word in ['sunny', 'clear', 'partly cloudy']):
        return 'Good'
    elif any(word in condition_str for word in ['cloudy', 'overcast', 'mist']):
        return 'Bad'
    elif any(word in condition_str for word in ['rainy', 'rain', 'patchy', 'drizzle', 'stormy']):
        return 'Worst'
    else:
        return 'Good'

def get_best_suppliers_for_route(origin, destination, top_n=5):
    """Get best suppliers for a specific route"""
    global df_refined
    
    route_deliveries = df_refined[
        (df_refined['origin_location'] == origin) &
        (df_refined['destination_location'] == destination)
    ].copy()
    
    if len(route_deliveries) == 0:
        route_deliveries = df_refined[
            (df_refined['origin_location'] == destination) &
            (df_refined['destination_location'] == origin)
        ].copy()
        if len(route_deliveries) == 0:
            return None
    
    suppliers = []
    for idx, delivery in route_deliveries.iterrows():
        supplier_data = {
            'delivery_id': delivery['delivery_id'],
            'driver_name': delivery.get('Driver_Name', 'Not specified'),
            'supplier_name': delivery.get('supplierNameCode', 'Not specified'),
            'vehicle_no': delivery.get('vehicle_no', 'Not specified'),
            'current_location': delivery.get('Current_Location', 'Not specified'),
            'origin': delivery['origin_location'],
            'destination': delivery['destination_location'],
            'rating': float(delivery['customer_rating']) if pd.notna(delivery['customer_rating']) else 0,
            'delivery_time': float(delivery['delivery_time']) if pd.notna(delivery['delivery_time']) else 0,
            'condition': delivery['condition_text'],
            'cargo_condition': delivery['cargo_condition'],
            'weather': delivery['weather'],
            'on_time': bool(delivery['on_time_delivery']),
            'fixed_cost': float(delivery['fixed_costs']) if pd.notna(delivery['fixed_costs']) else 0,
            'maintenance': float(delivery['maintenance']) if pd.notna(delivery['maintenance']) else 0,
            'area': delivery['area']
        }
        suppliers.append(supplier_data)
    
    suppliers_df = pd.DataFrame(suppliers)
    suppliers_df = suppliers_df.sort_values('rating', ascending=False)
    
    # Calculate route statistics
    route_stats = {
        'total_deliveries': len(route_deliveries),
        'avg_rating': float(route_deliveries['customer_rating'].mean()),
        'avg_delivery_time': float(route_deliveries['delivery_time'].mean()),
        'on_time_rate': float((route_deliveries['on_time_delivery'].sum() / len(route_deliveries)) * 100) if len(route_deliveries) > 0 else 0,
        'best_weather': route_deliveries['weather'].mode().values[0] if len(route_deliveries) > 0 else 'Unknown',
        'best_cargo_condition': route_deliveries['cargo_condition'].mode().values[0] if len(route_deliveries) > 0 else 'Good',
        'avg_cost': float(route_deliveries['fixed_costs'].mean()),
        'avg_maintenance': float(route_deliveries['maintenance'].mean())
    }
    
    return {
        'suppliers': suppliers_df.head(top_n).to_dict('records'),
        'route_stats': route_stats,
        'total_suppliers': len(suppliers_df)
    }

def predict_delivery_time(origin, destination, condition="Sunny", rating=4.5, fixed_cost=5000, maintenance=1200, area="Rural"):
    """Predict delivery time using trained model"""
    global model, encoders, scaler, df_refined
    
    try:
        # Get region from destination
        dest_region = "Tamil Nadu"  # Default
        if not df_refined.empty and destination in df_refined['destination_location'].values:
            dest_region = df_refined[df_refined['destination_location'] == destination]['region'].iloc[0]
        
        # Encode inputs
        features = [
            encoders["origin_location"].transform([origin])[0],
            encoders["destination_location"].transform([destination])[0],
            encoders["region"].transform([dest_region])[0],
            float(rating),
            float(fixed_cost),
            float(maintenance),
            encoders["condition_text"].transform([condition])[0],
            encoders["cargo_condition"].transform([map_cargo_condition(condition)])[0],
            encoders["area"].transform([area])[0],
            45  # Default delivery time for prediction
        ]
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        predicted_time = model.predict(features_scaled)[0]
        
        # Clamp to realistic values
        predicted_time = max(20, min(predicted_time, 500))
        
        return float(predicted_time)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 120  # Default fallback

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         origins=origins[:20], 
                         destinations=destinations[:20],
                         total_routes=len(df_refined) if df_refined is not None else 0)

@app.route('/book', methods=['GET', 'POST'])
def book_shipment():
    """Book a new shipment"""
    if request.method == 'POST':
        # Get form data
        sender_name = request.form['sender_name']
        sender_email = request.form['sender_email']
        sender_phone = request.form['sender_phone']
        sender_address = request.form['sender_address']
        
        receiver_name = request.form['receiver_name']
        receiver_email = request.form['receiver_email']
        receiver_phone = request.form['receiver_phone']
        receiver_address = request.form['receiver_address']
        
        origin = request.form['origin']
        destination = request.form['destination']
        package_type = request.form['package_type']
        package_weight = float(request.form['package_weight'])
        package_value = float(request.form['package_value'])
        priority = request.form['priority']
        insurance = 'insurance' in request.form
        
        # Generate tracking ID
        tracking_id = f"TRK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate estimated delivery time
        estimated_time = predict_delivery_time(origin, destination)
        
        # Calculate cost
        base_cost = 500
        weight_cost = package_weight * 10  # ₹10 per kg
        distance_factor = 1.5  # Simplified distance factor
        priority_multiplier = 1.2 if priority == 'express' else 1.0
        insurance_cost = package_value * 0.01 if insurance else 0
        
        total_cost = (base_cost + weight_cost) * distance_factor * priority_multiplier + insurance_cost
        
        # Create shipment record
        shipment = {
            'tracking_id': tracking_id,
            'sender_name': sender_name,
            'sender_email': sender_email,
            'sender_phone': sender_phone,
            'receiver_name': receiver_name,
            'receiver_email': receiver_email,
            'receiver_phone': receiver_phone,
            'origin': origin,
            'destination': destination,
            'package_type': package_type,
            'package_weight': package_weight,
            'package_value': package_value,
            'priority': priority,
            'insurance': insurance,
            'estimated_delivery_time': estimated_time,
            'total_cost': total_cost,
            'booking_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'Booked',
            'current_location': origin,
            'assigned_supplier': None
        }
        
        # Store in session (in real app, store in database)
        if 'shipments' not in session:
            session['shipments'] = {}
        session['shipments'][tracking_id] = shipment
        session.modified = True
        
        return render_template('booking_confirmation.html', 
                             shipment=shipment,
                             tracking_id=tracking_id)
    
    return render_template('book.html', origins=origins, destinations=destinations)

@app.route('/track', methods=['GET', 'POST'])
def track_shipment():
    """Track shipment status"""
    if request.method == 'POST':
        tracking_id = request.form['tracking_id']
        
        # Get shipment from session (in real app, query database)
        shipments = session.get('shipments', {})
        shipment = shipments.get(tracking_id)
        
        if shipment:
            # Simulate progress
            status_updates = [
                {'status': 'Booked', 'location': shipment['origin'], 'time': shipment['booking_date']},
                {'status': 'Picked Up', 'location': shipment['origin'], 'time': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')},
                {'status': 'In Transit', 'location': 'Halfway Point', 'time': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')},
                {'status': 'Out for Delivery', 'location': shipment['destination'], 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            ]
            
            return render_template('track_result.html', 
                                 shipment=shipment,
                                 status_updates=status_updates)
        else:
            return render_template('track.html', error="Tracking ID not found")
    
    return render_template('track.html')

@app.route('/suppliers')
def find_suppliers():
    """Find best suppliers for a route"""
    origin = request.args.get('origin', '')
    destination = request.args.get('destination', '')
    
    supplier_info = None
    if origin and destination:
        supplier_info = get_best_suppliers_for_route(origin, destination)
    
    return render_template('suppliers.html',
                         origins=origins,
                         destinations=destinations,
                         supplier_info=supplier_info,
                         origin=origin,
                         destination=destination)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for delivery time prediction"""
    data = request.json
    
    origin = data.get('origin')
    destination = data.get('destination')
    condition = data.get('condition', 'Sunny')
    rating = data.get('rating', 4.5)
    
    if not origin or not destination:
        return jsonify({'error': 'Origin and destination required'}), 400
    
    predicted_time = predict_delivery_time(origin, destination, condition, rating)
    
    return jsonify({
        'origin': origin,
        'destination': destination,
        'predicted_delivery_time_minutes': predicted_time,
        'predicted_delivery_time_hours': round(predicted_time / 60, 2),
        'condition': condition
    })

@app.route('/api/suppliers', methods=['GET'])
def api_suppliers():
    """API endpoint for supplier recommendations"""
    origin = request.args.get('origin')
    destination = request.args.get('destination')
    
    if not origin or not destination:
        return jsonify({'error': 'Origin and destination required'}), 400
    
    supplier_info = get_best_suppliers_for_route(origin, destination)
    
    if not supplier_info:
        return jsonify({'error': 'No data available for this route'}), 404
    
    return jsonify(supplier_info)

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    # Get all shipments from session
    shipments = session.get('shipments', {})
    
    # Calculate statistics
    total_shipments = len(shipments)
    total_revenue = sum(s.get('total_cost', 0) for s in shipments.values())
    
    # Get popular routes
    route_counts = {}
    for shipment in shipments.values():
        route = f"{shipment.get('origin', '')} → {shipment.get('destination', '')}"
        route_counts[route] = route_counts.get(route, 0) + 1
    
    popular_routes = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return render_template('dashboard.html',
                         total_shipments=total_shipments,
                         total_revenue=total_revenue,
                         popular_routes=popular_routes,
                         shipments=list(shipments.values())[-10:])  # Last 10 shipments

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)