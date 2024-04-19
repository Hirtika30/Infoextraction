# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np


def load_labels(filename):
    with open(filename, 'r') as file:
        labels = [line.strip() for line in file.readlines()[1:]]  # skip title-line
    return labels

training_labels = load_labels('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnlbls')
test_labels = load_labels('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.devlbls')

# Reading training labels from clsp.trnlbls
with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnlbls', 'r') as f:
    labels = f.read().splitlines()[1:]  # Skip the title line 

#DEBUGGING
# Verifying the content of labels
print("First 5 label strings:", labels[:5])

# Flattening the list of label strings into a single string
labels_str = ''.join(labels)

# Verifying the length and content of the flattened label strings
print("Total number of labels:", len(labels_str))
print("First 100 characters of combined labels:", labels_str[:100])

# Calculating unigram frequencies
unique_labels = set(labels_str)  # Unique labels in the dataset
unigram_freqs = {label: labels_str.count(label) / len(labels_str) for label in unique_labels}

# Checking the sum of unigram frequencies to ensure it's close to 1
unigram_sum = sum(unigram_freqs.values())
print("Sum of unigram frequencies:", unigram_sum)

# Ensuring no unigram frequency is zero
zero_freq_labels = [label for label, freq in unigram_freqs.items() if freq == 0]
print("Labels with zero frequency:", zero_freq_labels)

# Checking for the minimum frequency to ensure it's not too low
min_freq = min(unigram_freqs.values())
print("Minimum unigram frequency:", min_freq)


# Transition probability matrix for 3-state left-to-right HMM
trans_mat = np.array([[0.8, 0.2, 0.0],
                      [0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.8]]) 

# Creating letter HMMs
letter_hmms = {}
for letter in set('abcdefghijlmnoprstuvwxy'):
    start_prob = np.array([1.0, 0.0, 0.0])
    emission_probs = []
    for state in range(3):
        emission_prob = np.array([unigram_freqs.get(l, 1e-10) for l in unigram_freqs.keys()])
        emission_probs.append(emission_prob)

    letter_hmms[letter] = {
        'start_prob': start_prob,
        'trans_mat': trans_mat,
        'emission_probs': emission_probs
    }
    
# Transition probability matrix for 5-state SIL HMM
sil_trans_mat = np.array([[0.25, 0.25, 0.25, 0.25, 0.0],
                          [0.0, 0.25, 0.25, 0.25, 0.25],
                          [0.0, 0.25, 0.25, 0.25, 0.25],
                          [0.0, 0.25, 0.25, 0.25, 0.25],
                          [0.0, 0.0, 0.0, 0.0, 0.75]]) 

# Reading training labels and endpoints
with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnlbls', 'r') as f:
    labels = f.read().splitlines(keepends=False)
with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.endpts', 'r') as f:
    endpoints = f.read().splitlines(keepends=False)[1:]  # Skip the first line

silence_labels = []
for line in endpoints:
    start, end = map(int, line.split())
    silence_labels.extend(labels[0:start])
    silence_labels.extend(labels[end:])

# Calculating unigram frequencies for silence
total_count = len(silence_labels)
unigram_freqs = {}
for label in set(silence_labels):
    unigram_freqs[label] = silence_labels.count(label) / total_count

# Adding smoothing to avoid zero probabilities
smoothing = 1e-10
for label in set(''.join(labels)):
    if label not in unigram_freqs:
        unigram_freqs[label] = smoothing

# Creating SIL HMM
sil_hmm = {}
start_prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
emission_probs = []
for state in range(5):
    emission_prob = np.array([unigram_freqs.get(l, smoothing) for l in unigram_freqs.keys()])
    emission_probs.append(emission_prob)

sil_hmm = {
    'start_prob': start_prob,
    'trans_mat': sil_trans_mat,
    'emission_probs': emission_probs
}

#DEBUGGING
print("Letter HMM transition matrix:\n", trans_mat)
print("Letter HMM transition matrix row sums:", np.sum(trans_mat, axis=1))

print("SIL HMM transition matrix:\n", sil_trans_mat)
print("SIL HMM transition matrix row sums:", np.sum(sil_trans_mat, axis=1))

# Check the start probabilities
print("Letter HMM start probabilities:", letter_hmms['a']['start_prob'])
print("SIL HMM start probabilities:", sil_hmm['start_prob'])

# Checking that there are no zero probabilities where they shouldn't be
for letter, hmm in letter_hmms.items():
    if np.any(hmm['trans_mat'] == 0) and not np.allclose(hmm['trans_mat'], trans_mat):
        print(f"Zero probabilities found in transition matrix for letter {letter} where they shouldn't be.")

if np.any(sil_hmm['trans_mat'] == 0) and not np.allclose(sil_hmm['trans_mat'], sil_trans_mat):
    print("Zero probabilities found in SIL HMM transition matrix where they shouldn't be.")

# Check that the sums of the rows in the transition matrices are as expected
for letter, hmm in letter_hmms.items():
    if not np.allclose(np.sum(hmm['trans_mat'], axis=1), [1, 1, 0.8]):
        print(f"Transition probabilities for letter {letter} do not sum to 1 where expected.")

if not np.allclose(np.sum(sil_hmm['trans_mat'], axis=1), [1, 1, 1, 1, 0.75]):
    print("Transition probabilities for SIL HMM do not sum to 1 where expected.")
    



#Already created the letter_hmms dictionary

# Reading training script
with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnscr', 'r') as f:
    words = f.read().splitlines(keepends=False)[1:]  # Skip title line

if not words:
    raise ValueError("No words read from the file. Check the file content and path.")

# Creating baseforms
baseforms = {}
for word in set(words):
    baseform = []
    for letter in word:
        if letter in letter_hmms:
            baseform.append(letter_hmms[letter])
        else:
            print(f"Letter '{letter}' from word '{word}' not found in letter_hmms.")
    if baseform:  # Only add entries for words where HMM configurations are found
        baseforms[word] = baseform
    else:
        print(f"No valid HMM configurations found for the word '{word}'.")

#DEBUGGING
# Verifying the structure of baseforms
if not isinstance(baseforms, dict):
    raise ValueError("baseforms is not a dictionary after initialization. Check the initialization logic.")

#printing some parts of baseforms to verify
for word, forms in list(baseforms.items())[:5]:  # Print first 5 entries to check
    print(f"Word: {word}, HMMs Count: {len(forms)}")

# Ensuring the dictionary contains data
if not baseforms:
    raise ValueError("baseforms dictionary is empty. No baseforms were created.")




#TRAINING

output_alphabet_size = 823
# Creating word HMMs
word_hmms = {}
for word, baseforms in baseforms.items():
    # Calculating total number of states first
    num_states = 2 * len(sil_hmm['start_prob']) + sum(len(baseform['start_prob']) for baseform in baseforms)
    # Initializing the composite HMM
    start_prob = np.zeros(num_states)
    trans_mat = np.zeros((num_states, num_states))
    emission_probs = np.zeros((num_states, output_alphabet_size))

    # Filling in the HMM details
    current_index = 0

    # Appending SIL HMM at the beginning
    sil_size = len(sil_hmm['start_prob'])
    start_prob[current_index] = 1.0
    trans_mat[current_index:current_index+sil_size, current_index:current_index+sil_size] = sil_hmm['trans_mat']
    emission_probs[current_index:current_index+sil_size, :] = sil_hmm['emission_probs']
    current_index += sil_size

    # Concatenating HMMs from baseforms
    for baseform in baseforms:
        base_size = len(baseform['start_prob'])
        trans_mat[current_index:current_index+base_size, current_index:current_index+base_size] = baseform['trans_mat']

        # Correcting handling of a list of arrays
        expanded_emission_probs = np.zeros((base_size, output_alphabet_size))
        for i in range(base_size):
            current_emissions = baseform['emission_probs'][i]
            expanded_emission_probs[i, :len(current_emissions)] = current_emissions
            if len(current_emissions) < output_alphabet_size:
                
                expanded_emission_probs[i, len(current_emissions):] = current_emissions[-1]

        emission_probs[current_index:current_index+base_size, :] = expanded_emission_probs
        current_index += base_size

    # Appending SIL HMM at the end
    trans_mat[current_index:current_index+sil_size, current_index:current_index+sil_size] = sil_hmm['trans_mat']
    emission_probs[current_index:current_index+sil_size, :] = sil_hmm['emission_probs']

    # Saving the composite HMM in the dictionary
    word_hmms[word] = {
        'start_prob': start_prob,
        'trans_mat': trans_mat,
        'emission_probs': emission_probs
    }


def log_sum_exp(log_probs):
    max_log_prob = np.max(log_probs)
    stable_log_probs = log_probs - max_log_prob
    return max_log_prob + np.log(np.sum(np.exp(stable_log_probs)))



def forward_backward_log_space(obs_seq, hmm):
    num_states = len(hmm['start_prob'])
    num_obs = len(obs_seq)

    log_forward_probs = np.full((num_obs, num_states), -np.inf)
    log_backward_probs = np.full((num_obs, num_states), -np.inf)

    # Converting probabilities to log probabilities safely
    start_prob_log = np.log(np.clip(hmm['start_prob'], a_min=1e-10, a_max=None))
    trans_mat_log = np.log(np.clip(hmm['trans_mat'], a_min=1e-10, a_max=None))
    emission_probs_log = np.log(np.clip(hmm['emission_probs'], a_min=1e-10, a_max=None))

    # Initializing forward probabilities
    log_forward_probs[0] = start_prob_log + emission_probs_log[:, obs_seq[0]]

    # Forward algorithm
    for t in range(1, num_obs):
        for j in range(num_states):
            log_forward_probs[t, j] = log_sum_exp(log_forward_probs[t-1] + trans_mat_log[:, j]) + emission_probs_log[j, obs_seq[t]]

    # Initializing backward probabilities
    log_backward_probs[-1, :] = 0  # log(1) is 0

    # Backward algorithm
    for t in range(num_obs - 2, -1, -1):
        for i in range(num_states):
            log_backward_probs[t, i] = log_sum_exp(trans_mat_log[i, :] + emission_probs_log[:, obs_seq[t+1]] + log_backward_probs[t+1])

    return log_forward_probs, log_backward_probs



# Already created word_hmms

# Mapping words to unique integer indices
word_to_idx = {word: idx for idx, word in enumerate(word_hmms.keys())}

# a. Initializing counters
num_hmms = len(word_hmms)
num_states = [len(hmm) for hmm in word_hmms.values()]
max_states = max(len(hmm['start_prob']) for hmm in word_hmms.values())
# Ensuring initialization accommodates the maximum number of emission symbols expected in any HMM
max_emissions = max(hmm['emission_probs'].shape[1] for hmm in word_hmms.values())
emission_counts = np.zeros((num_hmms, max_states, max_emissions))

print(f"Emission counts initialized to accommodate up to {max_emissions} emission symbols.")




# Reinitializing trans_counts and emission_counts with correct dimensions
trans_counts = np.zeros((num_hmms, max_states, max_states))



# b. Sorting training data
with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnscr', 'r') as f:
    words = f.read().splitlines(keepends=False)[1:]

with open('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.trnlbls', 'r') as f:
    labels = f.read().splitlines(keepends=False)[1:]

sorted_data = sorted(zip(words, labels), key=lambda x: x[0])
# Accumulating transition and emission counts in log space
for word, label_seq in sorted_data:
    word_idx = word_to_idx[word]
    word_hmm = word_hmms[word]
    num_states_word = len(word_hmm)


    # Converting label_seq to a sequence of integers
    label_seq = [ord(label) for label in label_seq]
    """
    # Validate each label code
    for label_code in label_seq:
        if label_code >= word_hmm['emission_probs'].shape[1]:
            raise ValueError(f"Label code {label_code} is out of bounds for the emission probability matrix with {word_hmm['emission_probs'].shape[1]} states.")
    """


    log_forward_probs, log_backward_probs = forward_backward_log_space(label_seq, word_hmm)

    # Accumulating transition and emission counts
    for t in range(len(label_seq)):
        for i in range(num_states_word):
            for j in range(num_states_word):
                log_prob_ij = log_forward_probs[t, i] + log_backward_probs[t, j]
                trans_counts[word_idx, i, j] = log_sum_exp([trans_counts[word_idx, i, j], log_prob_ij])

            log_prob_emission = log_forward_probs[t, i] + log_backward_probs[t, i]
            emission_counts[word_idx, i, label_seq[t]] = log_sum_exp([emission_counts[word_idx, i, label_seq[t]], log_prob_emission])



#UPDATING

epsilon = np.log(1e-10)  # Defining epsilon in log space to avoid taking log(0)

for word, hmm_config in word_hmms.items():
    word_idx = word_to_idx[word]
    num_states = len(hmm_config['start_prob'])  # Number of states in the HMM for the word

    # Updating transition probabilities
    for i in range(num_states):
        log_sum_trans_counts = log_sum_exp(trans_counts[word_idx, i, :num_states])
        if np.isinf(log_sum_trans_counts):
            hmm_config['trans_mat'][i, :] = np.full(num_states, epsilon)
        else:
            for j in range(num_states):
                hmm_config['trans_mat'][i, j] = np.exp(trans_counts[word_idx, i, j] - log_sum_trans_counts)

    # Updating emission probabilities
    for i in range(num_states):
        num_emissions = hmm_config['emission_probs'].shape[1]  # Total emission categories
        log_sum_emission_counts = log_sum_exp(emission_counts[word_idx, i, :num_emissions])
        if np.isinf(log_sum_emission_counts):
            hmm_config['emission_probs'][i, :] = np.full(num_emissions, epsilon)
        else:
            for y in range(num_emissions):
                hmm_config['emission_probs'][i, y] = np.exp(emission_counts[word_idx, i, y] - log_sum_emission_counts)

# Ensuring that the emission_counts array is initialized to match the largest emission category index
print("All updates performed without indexing errors.")

import matplotlib.pyplot as plt



def compute_log_likelihood(word_hmms, sorted_data):
    total_log_likelihood = 0
    epsilon = 1e-10  # Small constant to prevent log(0)

    for word, label_seq in sorted_data:
        word_hmm = word_hmms[word]
        num_states = len(word_hmm['start_prob'])

        # Converting label_seq to a sequence of integers
        label_seq = [ord(label) for label in label_seq]

        # Initializing log forward probabilities array
        log_forward_probs = np.full((len(label_seq), num_states), -np.inf)  # Start with -inf representing log(0)

        # Applying smoothing to start probabilities and take log
        start_probs = np.log(word_hmm['start_prob'] + epsilon)
        emissions = np.log(word_hmm['emission_probs'] + epsilon)

        # Setting initial log probabilities
        log_forward_probs[0] = start_probs + emissions[:, label_seq[0]]

        # Log forward pass
        for t in range(1, len(label_seq)):
            for j in range(num_states):
                # Calculating the log sum of exponentials for transition probabilities
                log_probs = log_forward_probs[t-1] + np.log(word_hmm['trans_mat'][:, j] + epsilon)  # Add epsilon to transitions as well
                max_log_prob = np.max(log_probs)
                stable_log_probs = log_probs - max_log_prob
                log_sum = max_log_prob + np.log(np.sum(np.exp(stable_log_probs)))

                # Updating the log forward probability for state j at time t
                log_forward_probs[t, j] = log_sum + emissions[j, label_seq[t]]

        # Computing the log-likelihood from the last time step
        max_log_prob = np.max(log_forward_probs[-1])
        stable_log_probs = log_forward_probs[-1] - max_log_prob
        log_likelihood = max_log_prob + np.log(np.sum(np.exp(stable_log_probs)))
        total_log_likelihood += log_likelihood

    return total_log_likelihood




num_iterations = 20
log_likelihoods = []



for iteration in range(num_iterations):
    
    log_likelihood = compute_log_likelihood(word_hmms, sorted_data)
    log_likelihoods.append(log_likelihood)
   

    print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood}")

# Plotting log-likelihood convergence
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), log_likelihoods, marker='o', linestyle='-')
plt.xlabel("Number of Iterations")
plt.ylabel("Log-likelihood")
plt.title("Log-likelihood Convergence over Iterations")
plt.grid(True)
plt.show()

total_observations = sum(len(label_sequence) for label_sequence in sorted_data)
average_per_frame_log_likelihoods = [ll / total_observations for ll in log_likelihoods]

print(average_per_frame_log_likelihoods)


def compute_forward_log_prob(obs_seq, hmm):
    num_states = len(hmm['start_prob'])
    num_obs = len(obs_seq)
    log_probs = np.full(num_states, -np.inf)
    start_prob_log = np.log(np.clip(hmm['start_prob'], a_min=1e-10, a_max=None))
    trans_mat_log = np.log(np.clip(hmm['trans_mat'], a_min=1e-10, a_max=None))
    emission_probs_log = np.log(np.clip(hmm['emission_probs'], a_min=1e-10, a_max=None))

    # Initialization step
    log_probs = start_prob_log + emission_probs_log[:, obs_seq[0]]

    # Iteration step
    for t in range(1, num_obs):
        log_probs = np.logaddexp.reduce(log_probs[:, np.newaxis] + trans_mat_log + emission_probs_log[:, obs_seq[t]], axis=0)

    # Termination step
    return np.logaddexp.reduce(log_probs)

def test_hmm_system(word_hmms, test_labels):
    results = []
    confidences = []
    
    # Converting labels to indices (assuming labels are characters and we map them to integers)
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(''.join(test_labels))))}

    for label_string in test_labels:
        obs_seq = [label_to_index[label] for label in label_string]
        log_probs = []

        for word, hmm in word_hmms.items():
            log_prob = compute_forward_log_prob(obs_seq, hmm)
            log_probs.append(log_prob)
        
        log_probs = np.array(log_probs)
        max_index = np.argmax(log_probs)
        max_word = list(word_hmms.keys())[max_index]
        max_prob = log_probs[max_index]
        total_prob = np.logaddexp.reduce(log_probs)
        confidence = np.exp(max_prob - total_prob)
        
        results.append((max_word, confidence))
        confidences.append(confidence)

    return results, confidences

# Load your test labels
def load_test_labels(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Skip the title line and remove any newline characters
        test_labels = [line.strip() for line in lines[1:]]
    return test_labels

# Usage
test_labels = load_test_labels('/Users/hirtikamirghani/Downloads/data (1)-1 2/clsp.devlbls')



# Running the test
results, confidences = test_hmm_system(word_hmms, test_labels)

# Output results and confidences
for result, confidence in zip(results, confidences):
    print(f"Word: {result[0]}, Confidence: {confidence:.3f}")



