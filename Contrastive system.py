#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:45:28 2024

@author: hirtikamirghani
"""
import numpy as np
import random

# Helper functions for data loading and manipulation
def load_data(script_file, label_file):
    with open(script_file, 'r') as f:
        words = f.read().splitlines()[1:]  # Skip the title line
    with open(label_file, 'r') as f:
        labels = f.read().splitlines()[1:]  # Skip the title line
    return list(zip(words, labels))

def sort_data_by_word(data):
    from collections import defaultdict
    sorted_data = defaultdict(list)
    for word, label in data:
        sorted_data[word].append((word, label))
    return sorted_data

# Load and partition data
data = load_data('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnscr', '/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnlbls')
np.random.seed(42)
np.random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
held_out_data = data[split_idx:]
train_sorted = sort_data_by_word(train_data)
held_out_sorted = sort_data_by_word(held_out_data)

# HMM functionalities
def initialize_hmm(word):
    # Initialize HMM with random emission probabilities
    return {
        'transition_matrix': np.array([[0.8, 0.2, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 0.8]]),
        'emission_probabilities': np.random.rand(3, 256)
    }

def train_hmm(hmm, data):
    # Simulated training step (adjust the emission probabilities slightly)
    hmm['emission_probabilities'] += np.random.randn(3, 256) * 0.01
    np.clip(hmm['emission_probabilities'], 0, 1, out=hmm['emission_probabilities'])  # Keep probabilities valid
    return hmm

def test_hmm_system(hmms, data):
    # Simulated testing using random accuracy
    correct = 0
    total = 0
    for word, instances in data.items():
        for _, label in instances:
            predicted_word = max(hmms.keys(), key=lambda w: random.random())  # Randomly predict word
            if predicted_word == word:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# Initialize HMMs for each word
hmms = {word: initialize_hmm(word) for word in train_sorted.keys()}

# Training and Validation
max_iterations = 20
best_accuracy = 0
best_iteration = 0

for iteration in range(max_iterations):
    for word, instances in train_sorted.items():
        for _, label in instances:
            hmms[word] = train_hmm(hmms[word], label)
    
    accuracy = test_hmm_system(hmms, held_out_sorted)
    print(f"Iteration {iteration + 1}, Accuracy: {accuracy}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_iteration = iteration
    else:
        break  # Early stopping if no improvement

print(f"Best iteration: {best_iteration+1}, Best Accuracy: {best_accuracy}")

# Final training using all data
final_data_sorted = sort_data_by_word(data)
final_hmms = {word: initialize_hmm(word) for word in final_data_sorted.keys()}

for _ in range(best_iteration + 1):
    for word, instances in final_data_sorted.items():
        for _, label in instances:
            final_hmms[word] = train_hmm(final_hmms[word], label)
