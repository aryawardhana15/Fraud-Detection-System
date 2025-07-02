#!/usr/bin/env python3
"""
FinShield Link - Model Training Script
Trains the custom LSTM model on sample transaction data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os
from risk_model_py import FraudDetectionLSTM, TransactionFeatureExtractor

def generate_sample_data(num_samples=10000):
    """
    Generate sample transaction data for training
    In a real scenario, this would be replaced with historical transaction data
    """
    print("Generating sample training data...")
    
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    for i in range(num_samples):
        # 10% fraud, 90% legitimate
        is_fraud = np.random.random() < 0.1
        
        if is_fraud:
            # Fraudulent transaction patterns
            transaction = {
                'transaction_id': f'train_txn_{i}',
                'user_id': f'user_{np.random.randint(1, 1000)}',
                'amount': np.random.exponential(1000) + 100,  # Higher amounts
                'merchant': np.random.choice(['Unknown Merchant', 'Cash Advance', 'ATM']),
                'timestamp': '2024-01-01T' + f"{np.random.randint(0, 24):02d}:00:00Z",
                'location': np.random.choice(['Unknown', 'Foreign Country']),
                'card_type': np.random.choice(['credit', 'debit', 'prepaid']),
                'features': [
                    np.random.uniform(0.6, 1.0),  # High frequency
                    np.random.uniform(0.7, 1.0),  # Unusual location
                    np.random.uniform(0.5, 1.0),  # High amount
                    np.random.uniform(0.6, 1.0),  # Unusual time
                    np.random.uniform(0.4, 1.0)   # Merchant risk
                ],
                'is_fraud': 1
            }
        else:
            # Legitimate transaction patterns
            transaction = {
                'transaction_id': f'train_txn_{i}',
                'user_id': f'user_{np.random.randint(1, 1000)}',
                'amount': np.random.gamma(2, 50),  # Normal distribution of amounts
                'merchant': np.random.choice(['Amazon', 'Walmart', 'Starbucks', 'Target']),
                'timestamp': '2024-01-01T' + f"{np.random.randint(8, 20):02d}:00:00Z",
                'location': np.random.choice(['New York, NY', 'Los Angeles, CA', 'Chicago, IL']),
                'card_type': np.random.choice(['credit', 'debit']),
                'features': [
                    np.random.uniform(0.0, 0.3),  # Normal frequency
                    np.random.uniform(0.0, 0.2),  # Known location
                    np.random.uniform(0.0, 0.4),  # Normal amount
                    np.random.uniform(0.0, 0.3),  # Normal time
                    np.random.uniform(0.0, 0.2)   # Trusted merchant
                ],
                'is_fraud': 0
            }
        
        data.append(transaction)
    
    return data

def prepare_sequences(transactions, feature_extractor, sequence_length=10):
    """
    Prepare sequential data for LSTM training
    
    Args:
        transactions (list): List of transaction dictionaries
        feature_extractor: Feature extraction utility
        sequence_length (int): Length of sequences to create
        
    Returns:
        tuple: (X_sequences, y_labels)
    """
    print("Preparing sequences for LSTM training...")
    
    # Group transactions by user
    user_transactions = {}
    for txn in transactions:
        user_id = txn['user_id']
        if user_id not in user_transactions:
            user_transactions[user_id] = []
        user_transactions[user_id].append(txn)
    
    # Sort transactions by timestamp for each user
    for user_id in user_transactions:
        user_transactions[user_id].sort(key=lambda x: x['timestamp'])
    
    sequences = []
    labels = []
    
    # Create sequences for each user
    for user_id, user_txns in user_transactions.items():
        if len(user_txns) < sequence_length:
            continue
            
        # Extract features for all transactions
        features_list = []
        for txn in user_txns:
            features = feature_extractor.extract_features(txn)
            features_list.append(features)
        
        # Create sliding window sequences
        for i in range(len(features_list) - sequence_length + 1):
            sequence = features_list[i:i + sequence_length]
            label = user_txns[i + sequence_length - 1]['is_fraud']  # Label for the last transaction
            
            sequences.append(sequence)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

def train_model():
    """Main training function"""
    print("=== FinShield Link - Model Training ===")
    
    # Generate sample data
    sample_data = generate_sample_data(10000)
    
    # Initialize feature extractor
    feature_extractor = TransactionFeatureExtractor()
    
    # Extract features for all transactions to fit scaler
    all_features = []
    for txn in sample_data:
        features = feature_extractor.extract_features(txn)
        all_features.append(features)
    
    # Fit the scaler
    feature_extractor.fit_scaler(all_features)
    
    # Transform all features
    for i, txn in enumerate(sample_data):
        features = feature_extractor.extract_features(txn)
        sample_data[i]['transformed_features'] = feature_extractor.transform_features(features)
    
    # Prepare sequences for LSTM
    X, y = prepare_sequences(sample_data, feature_extractor)
    
    print(f"Created {len(X)} sequences with shape {X.shape}")
    print(f"Fraud ratio: {np.mean(y):.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    model = FraudDetectionLSTM(input_size=input_size)
