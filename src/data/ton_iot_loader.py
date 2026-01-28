import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TON_IoT_Loader")

class TonIoTLoader:
    def __init__(self, csv_path, test_size=0.2, seed=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.seed = seed
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
        # Define categorical columns based on TON_IoT standard features
        self.categorical_cols = [
            'proto', 'service', 'conn_state', 'dns_query', 'dns_AA', 'dns_RD', 
            'dns_RA', 'dns_rejected', 'ssl_version', 'ssl_cipher', 'http_method', 
            'http_uri', 'http_version', 'http_user_agent', 'http_orig_mime', 
            'http_resp_mime'
        ]
        # Common irrelevant columns to drop
        self.drop_cols = ['ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'type'] 
        # Note: 'type' is the parent category (normal, xss, etc), 'label' is binary usually, 
        # but TON_IoT has specific attack sub-versions. We need to be careful.
        # Generally we predict the 'type' or 'label'. 
        # For this research, we likely want multi-class prediction of 'type'.

    def load_and_process(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        logger.info(f"Loading dataset from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        # Basic cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Log distribution before processing
        logger.info("Class Distribution (Raw):")
        logger.info(df['type'].value_counts())

        # Encode Targets
        # specific TON_IoT 'type' usually contains: normal, scanning, ddos, ransomware, etc.
        y = self.label_encoder.fit_transform(df['type'])
        self.classes_ = self.label_encoder.classes_
        
        # Drop labels and high-cardinality/irrelevant identifiers
        X_raw = df.drop(columns=['type', 'label'] + [c for c in self.drop_cols if c in df.columns], errors='ignore')
        
        # Handle categorical features
        # We use simple Label Encoding for categorical to keep dimensions low for the Autoencoder 
        # or OneHot if we want strictly correct representation. 
        # Given "Representation Learning" requirement, OneHot -> Autoencoder is better.
        # BUT if cardinality is huge, LabelEncoding is practical. 
        # Let's stick to numerical mapping (LabelEncoding) for categoricals for now as per "Explicitly map features".
        
        for col in X_raw.columns:
            if X_raw[col].dtype == 'object' or col in self.categorical_cols:
                if col in X_raw.columns:
                    le = LabelEncoder()
                    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

        # Split first to avoid leakage
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        
        # Normalize
        logger.info("Normalizing features (Fit on Train)...")
        self.scaler.fit(X_train_raw)
        X_train = self.scaler.transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)
        
        logger.info(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test stub
    # loader = TonIoTLoader("src/data/Train_Test_Network.csv")
    # X_train, X_test, y_train, y_test = loader.load_and_process()
    pass
