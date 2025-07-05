#!/usr/bin/env python3
"""
FinShield Link - Data Ingestion Service
Simulates live transaction data streaming to Kafka
"""

import json
import time
import random
from datetime import datetime, timezone
from kafka import KafkaProducer
from kafka.errors import KafkaError
import uuid

class TransactionGenerator:
    """Generates realistic transaction data for fraud detection testing"""
    
    def __init__(self):
        self.merchants = [
            "Amazon", "Walmart", "Target", "Starbucks", "McDonald's",
            "Shell", "Exxon", "Best Buy", "Home Depot", "CVS",
            "Walgreens", "Uber", "Lyft", "Netflix", "Spotify"
        ]
        
        self.locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
            "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL"
        ]
        
        self.card_types = ["credit", "debit", "prepaid"]
        
    def generate_transaction(self):
        """Generate a single transaction with realistic patterns"""
        
        # 95% normal transactions, 5% potentially fraudulent
        is_fraud_like = random.random() < 0.05
        
        if is_fraud_like:
            # Fraudulent patterns: high amounts, unusual times, foreign locations
            amount = random.uniform(500, 5000)  # Higher amounts
            merchant = random.choice(["Unknown Merchant", "Cash Advance", "ATM Withdrawal"])
            location = random.choice(["Lagos, Nigeria", "Moscow, Russia", "Unknown Location"])
            # Fraud-like features: higher risk indicators
            features = [
                random.uniform(0.7, 1.0),  # High transaction frequency
                random.uniform(0.8, 1.0),  # Unusual location
                random.uniform(0.6, 1.0),  # High amount
                random.uniform(0.7, 1.0),  # Unusual time
                random.uniform(0.5, 1.0)   # Merchant risk
            ]
        else:
            # Normal transactions
            amount = random.uniform(5, 500)
            merchant = random.choice(self.merchants)
            location = random.choice(self.locations)
            # Normal features: lower risk indicators
            features = [
                random.uniform(0.0, 0.4),  # Normal transaction frequency
                random.uniform(0.0, 0.3),  # Known location
                random.uniform(0.0, 0.5),  # Normal amount
                random.uniform(0.0, 0.3),  # Normal time
                random.uniform(0.0, 0.2)   # Trusted merchant
            ]
        
        transaction = {
            "transaction_id": f"txn_{uuid.uuid4().hex[:8]}",
            "user_id": f"user_{random.randint(1000, 9999)}",
            "amount": round(amount, 2),
            "merchant": merchant,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": location,
            "card_type": random.choice(self.card_types),
            "features": [round(f, 3) for f in features],
            "is_fraud_simulation": is_fraud_like  # For testing purposes
        }
        
        return transaction

class KafkaTransactionProducer:
    """Kafka producer for streaming transaction data"""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='transactions'):
        self.topic = topic
        self.generator = TransactionGenerator()
        
        # Initialize Kafka producer with JSON serialization
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas to acknowledge
            retries=3,
            retry_backoff_ms=100
        )
        
    def start_streaming(self, transactions_per_second=2):
        """Start streaming transactions to Kafka"""
        print(f"Starting transaction stream to topic '{self.topic}'...")
        print(f"Rate: {transactions_per_second} transactions/second")
        print("Press Ctrl+C to stop\n")
        
        try:
            transaction_count = 0
            while True:
                transaction = self.generator.generate_transaction()
                
                # Use transaction_id as key for partitioning
                key = transaction['transaction_id']
                
                # Send to Kafka
                future = self.producer.send(
                    self.topic, 
                    key=key, 
                    value=transaction
                )
                
                # Add callback for success/error handling
                future.add_callback(self._on_send_success)
                future.add_errback(self._on_send_error)
                
                transaction_count += 1
                
                # Log every 10 transactions
                if transaction_count % 10 == 0:
                    fraud_indicator = "🚨 FRAUD-LIKE" if transaction.get('is_fraud_simulation') else "✅ NORMAL"
                    print(f"Sent transaction #{transaction_count}: {transaction['transaction_id']} "
                          f"(${transaction['amount']:.2f} at {transaction['merchant']}) {fraud_indicator}")
                
                # Wait before next transaction
                time.sleep(1.0 / transactions_per_second)
                
        except KeyboardInterrupt:
            print(f"\nStopping... Sent {transaction_count} transactions total")
        except Exception as e:
            print(f"Error in streaming: {e}")
        finally:
            self.producer.close()
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message send"""
        pass  # Silent success
    
    def _on_send_error(self, exception):
        """Callback for send errors"""
        print(f"Failed to send message: {exception}")

def main():
    """Main function to start the transaction ingestion service"""
    print("=== FinShield Link - Transaction Ingestion Service ===")
    
    # Wait for Kafka to be ready
    print("Waiting for Kafka to be ready...")
    time.sleep(20)
    
    # Create and start the producer
    producer = KafkaTransactionProducer()
    
    # Start streaming (2 transactions per second by default)
    producer.start_streaming(transactions_per_second=2)

if __name__ == "__main__":
    main()
