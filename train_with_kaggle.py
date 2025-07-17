#!/usr/bin/env python3
"""
Script to train the fake news detector with local Kaggle datasets
"""

import os
import pandas as pd
from main import FakeNewsDetector
import sys

def check_local_datasets():
    """Check if local dataset files exist"""
    fake_file = "Fake.csv"
    true_file = "True.csv"
    
    if os.path.exists(fake_file) and os.path.exists(true_file):
        print("✅ Found local Kaggle datasets:")
        print(f"   📄 {fake_file}")
        print(f"   📄 {true_file}")
        return fake_file, true_file
    else:
        print("❌ Local dataset files not found!")
        print("   Expected: Fake.csv and True.csv")
        return None, None

def preview_dataset(fake_file, true_file):
    """Preview the dataset structure"""
    print("\n📊 Dataset Preview:")
    print("=" * 50)
    
    try:
        # Preview fake news dataset
        fake_df = pd.read_csv(fake_file)
        print(f"📰 Fake News Dataset:")
        print(f"   Shape: {fake_df.shape}")
        print(f"   Columns: {list(fake_df.columns)}")
        print(f"   Sample:")
        print(fake_df.head(2))
        
        print("\n" + "=" * 50)
        
        # Preview true news dataset
        true_df = pd.read_csv(true_file)
        print(f"📰 True News Dataset:")
        print(f"   Shape: {true_df.shape}")
        print(f"   Columns: {list(true_df.columns)}")
        print(f"   Sample:")
        print(true_df.head(2))
        
        return fake_df, true_df
        
    except Exception as e:
        print(f"❌ Error reading datasets: {e}")
        return None, None

def train_with_kaggle_data():
    """Train the model with local Kaggle datasets"""
    print("\n🚀 Training with Kaggle Datasets")
    print("=" * 50)
    
    # Check for local dataset files
    fake_file, true_file = check_local_datasets()
    
    if not fake_file or not true_file:
        return False
    
    # Preview datasets
    fake_df, true_df = preview_dataset(fake_file, true_file)
    
    if fake_df is None or true_df is None:
        return False
    
    # Train the model
    print("\n🤖 Training Model...")
    detector = FakeNewsDetector()
    
    try:
        detector.train(use_kaggle=True, 
                      fake_csv_path=fake_file, 
                      true_csv_path=true_file)
        
        print("✅ Model trained successfully with Kaggle data!")
        print(f"📊 Training completed with {len(fake_df)} fake + {len(true_df)} real articles")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_trained_model():
    """Test the newly trained model"""
    print("\n🧪 Testing Trained Model")
    print("=" * 50)
    
    detector = FakeNewsDetector()
    
    test_cases = [
        ("Breaking: Scientists Discover Cure for Cancer", 
         "Researchers at major university publish peer-reviewed study showing breakthrough in cancer treatment"),
        ("SHOCKING: Government Hiding Alien Evidence", 
         "Anonymous sources claim government has been hiding alien evidence for decades"),
        ("Stock Market Reaches All-Time High", 
         "Wall Street celebrated as major indices closed at record levels amid positive economic indicators"),
        ("Miracle Weight Loss Pill Doctors Don't Want You to Know", 
         "Amazing new pill helps you lose 50 pounds in 30 days without diet or exercise"),
        ("Climate Change Report Released", 
         "Scientists publish comprehensive study on global warming impacts and solutions"),
        ("Celebrity Secret Revealed", 
         "Insider sources reveal shocking truth about Hollywood star's hidden life")
    ]
    
    for title, text in test_cases:
        result = detector.predict(text, title)
        print(f"📰 '{title}'")
        print(f"   Result: {result['prediction']} ({result['confidence']:.1f}%)")
        print()

def main():
    """Main function to orchestrate the training process"""
    print("🔍 Fake News Detection - Training with Local Kaggle Data")
    print("=" * 60)
    
    # Train with local Kaggle data
    if train_with_kaggle_data():
        # Test the model
        test_trained_model()
        
        print("\n🎉 Training Complete!")
        print("✅ Your model is now trained with real Kaggle datasets")
        print("✅ You can now use main.py or app.py as usual")
        print("✅ The model will automatically load the new trained weights")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()