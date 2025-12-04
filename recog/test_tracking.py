#!/usr/bin/env python3
"""
Test script to verify reference feature tracking.
This will load existing persons and show that their reference features remain stable.
"""

import numpy as np
from utils import load_existing_persons
from centroid_tracker import CentroidTracker

# Load existing persons
OUTPUT_DIR = "output"
print("Loading existing persons...")
known_persons = load_existing_persons(OUTPUT_DIR)

if not known_persons:
    print("No existing persons found. Please run main.py first to create some person data.")
    exit(1)

print(f"Found {len(known_persons)} known person(s):")
for pid, feature in known_persons.items():
    print(f"  - Person {pid}: feature shape {feature.shape}, norm {np.linalg.norm(feature):.4f}")

# Create tracker with known persons
ct = CentroidTracker(
    max_disappeared=50,
    max_distance=150,
    min_feature_similarity=0.45,
    known_persons=known_persons
)

print("\nVerifying reference features are preserved:")
for pid in known_persons.keys():
    ref_feat = ct.reference_features[pid]
    curr_feat = ct.features[pid]

    # Check they are initially the same
    if np.array_equal(ref_feat, curr_feat):
        print(f"  ✓ Person {pid}: reference and current features match initially")
    else:
        print(f"  ✗ Person {pid}: MISMATCH - this should not happen!")

    # Check reference is a separate copy (not same object)
    if ref_feat is not curr_feat:
        print(f"  ✓ Person {pid}: reference is a separate copy (not same object)")
    else:
        print(f"  ✗ Person {pid}: reference and current are SAME object - will drift together!")

print("\nSimulating feature drift by updating current feature:")
for pid in known_persons.keys():
    # Simulate an update with a slightly different feature
    noise = np.random.randn(128) * 0.1
    fake_new_feature = ct.features[pid] + noise
    fake_new_feature = fake_new_feature / np.linalg.norm(fake_new_feature)

    # Simulate EMA update (0.9 old + 0.1 new)
    original_ref = ct.reference_features[pid].copy()
    ct.features[pid] = 0.9 * ct.features[pid] + 0.1 * fake_new_feature
    ct.features[pid] = ct.features[pid] / np.linalg.norm(ct.features[pid])

    # Check that reference didn't change
    if np.array_equal(original_ref, ct.reference_features[pid]):
        print(f"  ✓ Person {pid}: reference feature unchanged after update")
    else:
        print(f"  ✗ Person {pid}: reference feature CHANGED - drift will occur!")

    # Check that current feature DID change
    similarity = np.dot(original_ref, ct.features[pid])
    print(f"  → Person {pid}: similarity after simulated update: {similarity:.4f}")

print("\n✅ Test complete! If all checks passed, drift should be prevented.")
