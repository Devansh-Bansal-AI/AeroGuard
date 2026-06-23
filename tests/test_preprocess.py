"""Test the preprocessing pipeline works end-to-end."""
import sys
import os
import pytest

# Add project root and ml dir to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ml"))

from preprocess import preprocess_pipeline


def test_preprocess_pipeline():
    """Verify preprocessing produces correctly shaped data."""
    result = preprocess_pipeline(
        use_synthetic=True,
        save_dir=os.path.join(PROJECT_ROOT, "data", "processed")
    )

    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]

    # Shape checks
    assert X_train.ndim == 3, "X_train should be 3D (samples, seq_len, features)"
    assert X_test.ndim == 3, "X_test should be 3D"
    assert X_train.shape[1] == 50, "Sequence length should be 50"
    assert X_train.shape[2] == 112, "Should have 112 features"
    assert len(y_train) == len(X_train), "Labels must match samples"
    assert len(y_test) == len(X_test), "Labels must match samples"

    # Metadata check
    meta = result["metadata"]
    assert meta["n_features"] == 112
    assert meta["sequence_length"] == 50
