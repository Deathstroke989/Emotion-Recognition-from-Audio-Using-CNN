
# Project Description: Emotion Recognition from Audio Using CNN

## Overview
This project aims to develop a convolutional neural network (CNN) model to classify emotions from audio speech samples. By extracting Mel-frequency cepstral coefficients (MFCCs) from audio files, the model learns to identify emotional states such as happiness, sadness, anger, and others, based on vocal patterns. The system is designed to process audio inputs, transform them into a suitable format for deep learning, and predict the corresponding emotion with high accuracy.

## Objectives
- **Feature Extraction**: Utilize MFCCs to capture the spectral characteristics of audio signals, which are effective for emotion recognition.
- **Model Development**: Build and train a CNN to classify emotions from processed audio features.
- **Evaluation**: Assess the model's performance using accuracy metrics on a test dataset.
- **Application**: Enable potential applications in human-computer interaction, mental health monitoring, and automated customer service systems.

## Dataset
The dataset used is the **RAVDESS Emotional Speech Audio** dataset, available on Kaggle: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data). 

### Dataset Details
- **Source**: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- **Content**: Contains 1,440 audio files from 24 actors (12 male, 12 female) vocalizing two lexically matched statements in a neutral North American accent.
- **Emotions**: Includes 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, and surprised.
- **File Format**: WAV files with a sample rate of 48 kHz, 16-bit depth.
- **Structure**: Files are named with a 7-part numerical identifier (e.g., `03-01-01-01-01-01-01.wav`), where the third part indicates the emotion (01 = neutral, 02 = calm, etc.).
- **Usage**: The dataset is preprocessed to a uniform sample rate of 22,050 Hz and fixed length of 3 seconds for consistency in feature extraction.

### Preprocessing
- Audio files are resampled to 22,050 Hz and trimmed or padded to a fixed 3-second duration.
- MFCC features (40 coefficients) are extracted using the `librosa` library, resulting in a 2D feature matrix for each audio file.
- The emotion labels are extracted from filenames and encoded as categorical variables for classification.

## Methodology
1. **Data Loading**: Audio files are loaded, and MFCC features are extracted.
2. **Data Preparation**: Features are reshaped to include a channel dimension for CNN compatibility, and labels are one-hot encoded.
3. **Model Architecture**: A CNN with two convolutional layers (32 and 64 filters), max-pooling layers, a flatten layer, a dense layer with 128 units, dropout (0.5), and a softmax output layer.
4. **Training**: The model is trained for 20 epochs using the Adam optimizer and categorical cross-entropy loss, with a 80-20 train-test split.
5. **Evaluation**: Model performance is evaluated on the test set, reporting accuracy and predicted probabilities.
6. **Model Saving**: The trained model is saved for future use.

## Tools and Libraries
- **Python Libraries**: NumPy, Librosa (for audio processing), Scikit-learn (for data splitting and label encoding), Keras (for CNN implementation).
- **Hardware**: Standard CPU/GPU for training the CNN model.

