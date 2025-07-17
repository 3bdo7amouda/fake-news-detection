"""
Main script for the fake news detection project.
This script provides command-line interface for training and testing the models.
"""

import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import create_directories, logger
from src.model_training import ModelTrainer
from src.prediction import FakeNewsDetector, ModelComparator

def train_models(data_path: str = None):
    """
    Train all machine learning models.
    
    Args:
        data_path: Path to the dataset CSV file
    """
    print("🤖 Starting Fake News Detection Model Training")
    print("=" * 50)
    
    # Create necessary directories
    create_directories()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    results = trainer.train_full_pipeline(data_path)
    
    print("\n✅ Training completed successfully!")
    print(f"Best model: {results['best_model']}")
    print("\n💡 You can now use the trained models for predictions!")
    print("   - Run: python main.py --predict")
    print("   - Or use the web interface: streamlit run app.py")

def interactive_prediction():
    """
    Interactive prediction mode for testing individual articles.
    """
    print("🔍 Fake News Detection - Interactive Mode")
    print("=" * 50)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        detector.load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the models first by running: python main.py --train")
        return
    
    print("✅ Model loaded successfully!")
    print("Enter news articles to check (type 'quit' to exit, 'compare' for multi-model comparison)")
    print("-" * 50)
    
    while True:
        print("\n📰 Enter news article details:")
        title = input("Title (optional): ").strip()
        
        if title.lower() == 'quit':
            break
        elif title.lower() == 'compare':
            multi_model_comparison()
            continue
        
        text = input("Article text: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            print("⚠️ Please enter some text to analyze.")
            continue
        
        # Make prediction
        result = detector.predict(text, title)
        
        # Display result
        print("\n" + "="*50)
        print(detector.get_prediction_summary(result))
        print("="*50)

def multi_model_comparison():
    """
    Compare predictions from multiple models.
    """
    print("\n🔬 Multi-Model Comparison Mode")
    print("-" * 30)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    title = input("Title (optional): ").strip()
    text = input("Article text: ").strip()
    
    if not text:
        print("⚠️ Please enter some text to analyze.")
        return
    
    # Compare predictions
    comparison = comparator.compare_predictions(text, title)
    
    # Display results
    print("\n" + "="*60)
    print(comparator.get_comparison_summary(comparison))
    print("="*60)

def demo_examples():
    """
    Run demonstration with example articles.
    """
    print("📚 Fake News Detection - Demo Examples")
    print("=" * 50)
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        detector.load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please train the models first by running: python main.py --train")
        return
    
    # Example articles
    examples = [
        {
            'title': 'Scientists Discover New Treatment for Cancer',
            'text': 'Researchers at a major university have developed a promising new treatment for cancer that shows significant results in clinical trials. The treatment has been tested on hundreds of patients with positive outcomes.',
            'expected': 'REAL'
        },
        {
            'title': 'SHOCKING: Aliens Found in Government Facility',
            'text': 'Government officials deny but sources confirm that extraterrestrial beings are being held at a secret facility. This amazing discovery will change everything we know about life in the universe.',
            'expected': 'FAKE'
        },
        {
            'title': 'Miracle Cure Discovered by Local Mom',
            'text': 'Local mother discovers amazing cure that doctors hate using this one simple trick from her kitchen. This miracle cure has helped thousands of people lose weight instantly.',
            'expected': 'FAKE'
        }
    ]
    
    print("🧪 Testing example articles:\n")
    
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Title: {example['title']}")
        print(f"Expected: {example['expected']}")
        
        result = detector.predict(example['text'], example['title'])
        
        status = "REAL" if result['is_real'] else "FAKE"
        confidence = result['confidence']
        
        print(f"Predicted: {status} ({confidence}% confidence)")
        print(f"✅ Correct!" if status == example['expected'] else "❌ Incorrect!")
        print("-" * 50)

def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(description='Fake News Detection AI Project')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--predict', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--demo', action='store_true', help='Run demonstration examples')
    parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    
    args = parser.parse_args()
    
    if args.train:
        train_models(args.data)
    elif args.predict:
        interactive_prediction()
    elif args.demo:
        demo_examples()
    else:
        print("🚀 Welcome to Fake News Detection AI Project!")
        print("=" * 50)
        print("Available commands:")
        print("  --train     : Train the machine learning models")
        print("  --predict   : Interactive prediction mode")
        print("  --demo      : Run demonstration examples")
        print("  --data PATH : Specify dataset path for training")
        print("\nFor web interface, run: streamlit run app.py")
        print("\nExample usage:")
        print("  python main.py --train")
        print("  python main.py --predict")
        print("  python main.py --demo")

if __name__ == "__main__":
    main()