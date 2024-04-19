# Infoextraction

# Speech Recognition with HMM Training
This Python script implements training of Hidden Markov Models (HMMs) for speech recognition using the Baum-Welch algorithm. The script reads in training data consisting of word transcriptions and corresponding label sequences, and iteratively updates the HMM parameters to maximize the likelihood of the training data.

## System Features
•	Feature Extraction: Processes speech signals to extract spectral features.

•	HMM Definition: Constructs Hidden Markov Models for individual letters and silence.

•	Word Composition: Forms words by concatenating letter HMMs and integrates silence models.

•	Recognition: Uses trained HMMs to recognize words in speech data.

•	Accuracy Evaluation: Measures and reports the recognition accuracy of the system.

## Data Files
•	clsp.lblnames: Label names for quantized outputs.

•	clsp.trnscr: Script read by speakers, containing examples of each word.

•	clsp.trnlbls: Label strings for each spoken word.

•	clsp.endpts: Leading and trailing silence information for each word.

•	clsp.devlbls: Test set label strings.

## Prerequisites
•	Python 3.x

•	NumPy library

## Usage
1.	Ensure that the required data files (clsp.trnlbls, clsp.devlbls, clsp.endpts, clsp.trnscr) are present in the specified paths.
2.	Temp.py is the main code.
3.	Run the script, and it will load the data, initialize the HMMs, and perform iterative training. After that it tests on the test data and produces the output.
4.	Contrastive system.py is the code for the Contrastive system developed.

## Output
1.	The temp.py outputs the identity of the most likely word for each utterance and a confidence score. It also generates a log-likelihood plot for the training data as a function of the training iterations.
2.	The Contrastive System.py outputs the Best Accuracy. 

## Documentation
•	Extensive documentation is provided within the source code, detailing the required files and command-line usage for both training and testing modules.

•	A report is also provided displaying the output with the explanation of the source code.

## Notes
•	The script assumes a specific file structure and naming conventions for the training data files. Modify the file paths if necessary.

•	The code includes various debugging statements and checks to ensure the correctness of the implementation.

•	Discussed the project with Janvi Prasad and Darshil Shah.
