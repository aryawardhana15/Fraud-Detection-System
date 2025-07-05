#!/usr/bin/env python3
"""
FinShield Link - Custom ML Models for Fraud Detection
Defines LSTM and other custom models for transaction risk assessment
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pymongo import MongoClient
import statistics
from collections import Counter
from datetime import datetime

class FraudDetectionLSTM(nn.Module):
    """
    LSTM-based fraud detection model
    Analyzes sequences of transaction features to detect anomalous patterns
    """
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(FraudDetectionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output from the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        return output

class TransactionFeatureExtractor:
    """
    Extracts and preprocesses features from transaction data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, transaction):
        """
        Extract numerical features from a transaction dictionary
        
        Args:
            transaction (dict): Transaction data
            
        Returns:
            np.array: Extracted features
        """
        features = []
        
        # Amount-based features
        amount = transaction.get('amount', 0)
        features.append(amount)
        features.append(np.log1p(amount))  # Log-scaled amount
        
        # Time-based features (hour of day as normalized value)
        try:
            from datetime import datetime
            timestamp = datetime.fromisoformat(transaction.get('timestamp', '').replace('Z', '+00:00'))
            hour_of_day = timestamp.hour / 24.0  # Normalize to 0-1
            day_of_week = timestamp.weekday() / 6.0  # Normalize to 0-1
            features.extend([hour_of_day, day_of_week])
        except:
            features.extend([0.5, 0.5])  # Default values if timestamp parsing fails
        
        # Merchant risk (simple heuristic)
        merchant = transaction.get('merchant', '').lower()
        high_risk_merchants = ['unknown', 'cash advance', 'atm withdrawal', 'money transfer']
        merchant_risk = 1.0 if any(risk_merchant in merchant for risk_merchant in high_risk_merchants) else 0.2
        features.append(merchant_risk)
        
        # Location risk (simple heuristic)
        location = transaction.get('location', '').lower()
        high_risk_locations = ['nigeria', 'russia', 'unknown']
        location_risk = 1.0 if any(risk_loc in location for risk_loc in high_risk_locations) else 0.1
        features.append(location_risk)
        
        # Card type encoding
        card_type = transaction.get('card_type', 'credit')
        card_risk = {'credit': 0.1, 'debit': 0.2, 'prepaid': 0.8}.get(card_type, 0.5)
        features.append(card_risk)
        
        # Include any pre-computed features
        if 'features' in transaction and isinstance(transaction['features'], list):
            features.extend(transaction['features'])
            
        return np.array(features, dtype=np.float32)
    
    def fit_scaler(self, feature_arrays):
        """Fit the scaler on training data"""
        all_features = np.vstack(feature_arrays)
        self.scaler.fit(all_features)
        self.is_fitted = True
        
    def transform_features(self, features):
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            # If not fitted, return normalized features
            return (features - np.mean(features)) / (np.std(features) + 1e-8)
        return self.scaler.transform(features.reshape(1, -1)).flatten()
    
    def save_scaler(self, filepath):
        """Save the fitted scaler"""
        joblib.dump(self.scaler, filepath)
        
    def load_scaler(self, filepath):
        """Load a fitted scaler"""
        if os.path.exists(filepath):
            self.scaler = joblib.load(filepath)
            self.is_fitted = True
            return True
        return False

class HybridFraudDetector:
    """
    Combines custom LSTM model with external AI service predictions
    """
    
    def __init__(self, model_path=None):
        self.lstm_model = None
        self.feature_extractor = TransactionFeatureExtractor()
        self.sequence_length = 10  # Look at last 10 transactions
        self.transaction_history = []
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained LSTM model"""
        try:
            self.lstm_model = FraudDetectionLSTM()
            self.lstm_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.lstm_model.eval()
            
            # Load scaler
            scaler_path = model_path.replace('.pth', '_scaler.pkl')
            self.feature_extractor.load_scaler(scaler_path)
            
            print(f"Loaded LSTM model from {model_path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def predict_custom_model(self, transaction):
        """
        Get fraud probability from custom LSTM model
        
        Args:
            transaction (dict): Transaction data
            
        Returns:
            float: Fraud probability (0-1)
        """
        if self.lstm_model is None:
            # Return random score if model not loaded
            return np.random.uniform(0.1, 0.9)
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(transaction)
            features = self.feature_extractor.transform_features(features)
            
            # Add to history
            self.transaction_history.append(features)
            if len(self.transaction_history) > self.sequence_length:
                self.transaction_history.pop(0)
            
            # If we don't have enough history, pad with zeros
            if len(self.transaction_history) < self.sequence_length:
                padding_needed = self.sequence_length - len(self.transaction_history)
                padded_history = [np.zeros_like(features) for _ in range(padding_needed)]
                padded_history.extend(self.transaction_history)
                sequence = np.array(padded_history)
            else:
                sequence = np.array(self.transaction_history[-self.sequence_length:])
            
            # Convert to tensor and predict
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                prediction = self.lstm_model(sequence_tensor)
                fraud_probability = prediction.item()
            
            # Tambahkan logika jam transaksi (2-4 pagi UTC = jam 2, 3, 4)
            from datetime import datetime
            try:
                timestamp = transaction.get('timestamp', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                if 2 <= dt.hour <= 4:
                    fraud_probability = min(fraud_probability + 0.25, 1.0)  # Naikkan skor fraud
            except Exception:
                pass
            
            return fraud_probability
            
        except Exception as e:
            print(f"Error in custom model prediction: {e}")
            return 0.5  # Return neutral score on error
    
    def combine_scores(self, aws_score, custom_score, aws_weight=0.6, custom_weight=0.4):
        """
        Combine AWS Fraud Detector score with custom model score
        
        Args:
            aws_score (float): Score from AWS Fraud Detector
            custom_score (float): Score from custom LSTM model
            aws_weight (float): Weight for AWS score
            custom_weight (float): Weight for custom score
            
        Returns:
            float: Combined risk score
        """
        combined_score = (aws_weight * aws_score) + (custom_weight * custom_score)
        
        # Apply ensemble rules
        # If both models agree on high risk, boost the score
        if aws_score > 0.7 and custom_score > 0.7:
            combined_score = min(combined_score * 1.2, 1.0)
        
        # If scores disagree significantly, be more conservative
        if abs(aws_score - custom_score) > 0.4:
            combined_score = max(aws_score, custom_score) * 0.8
            
        return combined_score

def save_alert_to_mongo(alert_data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['finsheild']
    alerts = db['alerts']
    alerts.insert_one(alert_data)

def get_user_history(user_id, limit=50):
    """Ambil riwayat transaksi user dari MongoDB (maksimal limit terakhir)"""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['finsheild']
    alerts = db['alerts']
    history = list(alerts.find({'user_id': user_id}).sort('timestamp', -1).limit(limit))
    return history

def compute_user_profile(user_history):
    """Hitung statistik profil user dari riwayat transaksi"""
    if not user_history:
        return None
    amounts = [t.get('amount', 0) for t in user_history if 'amount' in t]
    hours = []
    locations = []
    for t in user_history:
        ts = t.get('timestamp')
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                hours.append(dt.hour)
            except:
                pass
        loc = t.get('location')
        if loc:
            locations.append(loc)
    profile = {
        'mean_amount': statistics.mean(amounts) if amounts else 0,
        'std_amount': statistics.stdev(amounts) if len(amounts) > 1 else 1,
        'fav_hour': Counter(hours).most_common(1)[0][0] if hours else None,
        'fav_location': Counter(locations).most_common(1)[0][0] if locations else None,
        'locations': set(locations)
    }
    return profile

def compute_behavior_score(transaction, user_profile):
    """Hitung skor penyimpangan perilaku user (semakin tinggi = semakin menyimpang)"""
    if not user_profile:
        return 0.5  # Netral jika tidak ada data
    score = 0.0
    # Outlier amount
    amount = transaction.get('amount', 0)
    mean = user_profile['mean_amount']
    std = user_profile['std_amount']
    if std < 1: std = 1  # Hindari div 0
    z = abs(amount - mean) / std
    if z > 2:
        score += 0.4  # Outlier amount
    elif z > 1:
        score += 0.2
    # Outlier hour
    ts = transaction.get('timestamp')
    if ts and user_profile['fav_hour'] is not None:
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            hour = dt.hour
            if abs(hour - user_profile['fav_hour']) > 4:
                score += 0.2
        except:
            pass
    # Outlier location
    loc = transaction.get('location')
    if loc and user_profile['fav_location']:
        if loc != user_profile['fav_location']:
            score += 0.2
        if loc not in user_profile['locations']:
            score += 0.2
    return min(score, 1.0)

def process_transaction(transaction, detector, threshold=0.7):
    # Prediksi fraud score
    fraud_score = detector.predict_custom_model(transaction)
    # Behavior profiling
    user_id = transaction.get('user_id')
    user_history = get_user_history(user_id, limit=50)
    user_profile = compute_user_profile(user_history)
    user_behavior_score = compute_behavior_score(transaction, user_profile)
    # Jika skor di atas threshold, simpan ke MongoDB
    if fraud_score >= threshold:
        alert_data = transaction.copy()
        alert_data['fraud_score'] = fraud_score
        alert_data['user_behavior_score'] = user_behavior_score
        save_alert_to_mongo(alert_data)
        print(f"[ALERT] Fraud detected! Saved to MongoDB: {alert_data['transaction_id']} (score={fraud_score:.2f}, behavior={user_behavior_score:.2f})")
    else:
        print(f"[OK] Transaction normal: {transaction['transaction_id']} (score={fraud_score:.2f}, behavior={user_behavior_score:.2f})")

if __name__ == "__main__":
    # Contoh main loop sederhana
    # Anda bisa sesuaikan dengan consumer Kafka Anda
    print("=== FinShield Link - Real-Time Fraud Detection ===")
    print("Connecting to Kafka and MongoDB...")
    import json
    from kafka import KafkaConsumer
    import time

    # Inisialisasi detektor
    detector = HybridFraudDetector(model_path="model.pth")

    # Kafka consumer
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='fraud-detector'
    )

    print("Listening for transactions...")
    for message in consumer:
        transaction = message.value
        process_transaction(transaction, detector, threshold=0.7)
        # Tambahkan delay kecil agar log lebih mudah dibaca
        time.sleep(0.1)