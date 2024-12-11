Here's a sample README file for your repository:

---

# Project Repository for Voice Command Pipeline

This repository contains all the necessary components for building a voice-based pipeline, including speech-to-text, text-to-speech, and various integrated services. The project uses models like Moonshine, Whisper, and AWS Polly to process audio commands and generate responses.

## Folder Structure

### 1. **`audios/`**
   - Contains the input audio files that will be processed in the pipeline. These audio files are sent to the speech-to-text models for transcription.

### 2. **`responses_audios/`**
   - This folder contains all the response audio files that were generated by the model (Moonshine and Whisper) and then converted into audio using AWS Polly. These responses are returned after processing the user's input.

### 3. **`moonshine/`**
   - This folder contains all the required files and dependencies to run the pipeline using the Moonshine speech-to-text model.
   - If you're using Moonshine, make sure to run the scripts inside this folder where the Moonshine model is loaded and saved.

### 4. **`Main Scripts/`**
   - This folder contains the main scripts for using various services:
     - **Moonshine Setup**: A Jupyter notebook for setting up and using the Moonshine model. Follow the instructions in this notebook to get started.
     - **Service Scripts**: Individual scripts for interacting with different services such as:
       - **Case Research**: A script for using the "Case Research" service.
       - **Co-Pilot and Case Research**: A script that integrates both "Co-Pilot" and "Case Research".
       - **Drafting**: A script for generating draft responses.
       - **AWS Polly and Whisper**: Scripts to use AWS Polly for text-to-speech and Whisper for speech-to-text functionalities.

## Prerequisites

Before you begin, make sure you have the following dependencies installed:

- **Python** (Version > 3.9 and < 3.11)
- **Librosa**: A Python package required for audio processing. Ensure that your Python version is within the supported range (between Python 3.9 and 3.11) for Librosa to work correctly.

## Installation Steps for moonshine

1. Clone the repository to your local machine:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install the necessary Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   This will install all required dependencies for the project, including libraries for speech-to-text and text-to-speech processing.

3. **For Moonshine**:
   - Ensure that you have the Moonshine model correctly set up as described in the **Moonshine/Setup** folder. Follow the instructions provided in the Jupyter notebook for the Main Scripts folder.

4. **For AWS Polly and Whisper**:
   - Make sure you have the credentials for AWS Polly and Whisper properly configured in your environment to use their respective APIs for text-to-speech and speech-to-text functionalities.

## Usage

1. **To process audio files**:
   - Place your input audio files in the `audios/` folder.
   - Use the relevant script in the **Main Scripts** folder for processing the audio through services like "Case Research", "Co-Pilot", "Drafting", etc. Or "production_pipeline" for testing full flow.

2. **To generate responses using AWS Polly**:
   - After processing the audio input, you can convert the response text into an audio file using AWS Polly, and it will be stored in the `responses_audios/` folder. (If specified).

3. **Moonshine Integration**:
   - If you wish to use the Moonshine model for speech-to-text, run the appropriate scripts inside the `moonshine/` folder. Be sure that the necessary model files are in place and follow the setup instructions for using Moonshine.

## Example Workflow

1. **Preprocessing Audio**:
   - Place the raw audio file in the `audios/` folder.
   - Ensure you give the correct path of the audio file.
   - Run the service script for processing the audio (e.g., "case research", "drafting").
   - If you want to run the whole pipeline, then production_pipeline.py is the script.

2. **Model Processing**:
   - The processed text or response will be generated by the relevant model (Moonshine, Whisper, etc.).

3. **Generating Audio Response**:
   - The model response will be converted to audio using AWS Polly and saved in the desired folder. For better organizing I saved them in `responses_audios/` folder.

## Troubleshooting

- **Librosa Compatibility**: If you face issues with Librosa, make sure you are using Python version 3.9 or higher, but lower than 3.11.
  
- **Moonshine Not Loading**: Ensure that you are running the Moonshine scripts inside the correct folder where the model is set up and saved.

This repository provides a complete pipeline for processing audio input, running it through various AI models, and generating spoken responses using AWS Polly. It integrates multiple services and models for tasks like case research, drafting, and speech-to-text processing, making it a robust solution for building voice-powered applications.

---

This README should provide a clear and concise guide to understand the structure and usage of the repository.