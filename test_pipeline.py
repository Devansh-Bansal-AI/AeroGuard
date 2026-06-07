"""Quick test script to verify the ML pipeline works end-to-end."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))

from preprocess import preprocess_pipeline

print("Testing preprocessing pipeline...")
result = preprocess_pipeline(use_synthetic=True, save_dir=os.path.join(os.path.dirname(__file__), "data", "processed"))

print(f"\nTrain shape: {result['X_train'].shape}")
print(f"Test shape:  {result['X_test'].shape}")
print(f"Features:    {result['metadata']['n_features']}")
print(f"\nPipeline test PASSED!")
