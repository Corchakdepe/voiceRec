import librosa
import numpy as np
import os
import pandas as pd

# Path to the folder containing your audio files
audio_folder = 'Audios'


# Function to extract audio features using librosa
def extract_audio_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # Extract features
    f0_mean = np.mean(librosa.piptrack(y=y, sr=sr)[0])  # Pitch (F0) mean
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)  # MFCCs mean

    return f0_mean, spectral_centroid, zero_crossing_rate, mfccs


# List to store the extracted features for each audio file
audio_parameters = []

# Iterate through audio files and extract features
for file_name in os.listdir(audio_folder):
    if file_name.endswith(('.wav', '.mp3')):
        file_path = os.path.join(audio_folder, file_name)
        try:
            f0_mean, spectral_centroid, zero_crossing_rate, mfccs = extract_audio_features(file_path)
            audio_parameters.append({
                'File Name': file_name,
                'Mean F0 (Hz)': f0_mean,
                'Spectral Centroid': spectral_centroid,
                'Zero Crossing Rate': zero_crossing_rate,
                'MFCCs': mfccs.tolist()  # Convert numpy array to list
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# Convert the list to a DataFrame
audio_df = pd.DataFrame(audio_parameters)

# Save the DataFrame to a CSV file
output_csv_path = os.path.join(audio_folder, 'audio_parameters_librosa.csv')
audio_df.to_csv(output_csv_path, index=False)

print(f"Audio parameters saved to {output_csv_path}")
