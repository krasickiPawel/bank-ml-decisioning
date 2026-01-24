# src/api/feature_schema.py

from __future__ import annotations
from typing import Any, Dict, List

# 20 kolumn z German Credit (credit-g)
EXPECTED_COLUMNS: List[str] = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
]

# Minimalny przykład — dla demo dajemy kilka sensownych wartości, reszta null
EXAMPLE_FEATURES: Dict[str, Any] = {
    "checking_status": "<0",
    "duration": 12,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 1000,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 2,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "real estate",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "none",
    "foreign_worker": "yes",
}
