# predict.py
import joblib
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/bbc_news_model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run 'train.py' first to generate the model.")
        return None
    
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    return model

def main():
    model = load_model()
    if model is None:
        return

    print("\n" + "="*40)
    print(" BBC NEWS CLASSIFIER (INTERACTIVE)")
    print(" Type 'quit' to exit")
    print("="*40)

    while True:
        user_input = input("\nEnter a headline or article text:\n> ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue

        # Predict
        # We wrap input in a list [] because the model expects a list of texts
        prediction = model.predict([user_input])[0]
        
        # Get Probability (Confidence score)
        probs = model.predict_proba([user_input])
        confidence = probs.max() * 100
        
        print(f"\nCategory:  {prediction.upper()}")
        print(f"Confidence: {confidence:.1f}%")
        print("-" * 20)

if __name__ == "__main__":
    main()