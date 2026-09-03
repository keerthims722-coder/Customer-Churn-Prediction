"""
Automated Test Verification Script for Customer Churn Model and App.
Tests:
1. Model loading
2. Prediction and Probability Generation
3. Sample Customer Pre-sets
4. Feature Pipeline integrity
"""

import os
import joblib
import pandas as pd


def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED MODEL & PREDICTION TESTS")
    print("=" * 60)

    model_path = os.path.join("model", "churn_model_pipeline.joblib")
    meta_path = os.path.join("model", "metadata.joblib")

    assert os.path.exists(model_path), f"Model file missing at {model_path}"
    assert os.path.exists(meta_path), f"Metadata file missing at {meta_path}"
    print("[PASS] Model and metadata artifact files exist.")

    pipeline = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    print("[PASS] Artifacts loaded successfully via joblib.")

    # Test Sample Customers
    sample_customers = metadata.get("sample_customers", {})
    assert len(sample_customers) > 0, "No sample customers in metadata"

    for idx, (name, sample_dict) in enumerate(sample_customers.items()):
        sample_clean = {k: v for k, v in sample_dict.items() if k != "description"}
        input_df = pd.DataFrame([sample_clean])

        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        assert pred in [0, 1], f"Prediction invalid: {pred}"
        assert 0.0 <= prob <= 1.0, f"Probability invalid: {prob}"

        print(f"[PASS] Sample {idx+1}: Pred={pred}, Churn Prob={prob*100:.1f}%")

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
