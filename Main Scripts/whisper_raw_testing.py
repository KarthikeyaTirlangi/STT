from transformers import WhisperProcessor, WhisperForConditionalGeneration #type: ignore
import soundfile as sf
import torch
import librosa

def transcribe_audio(audio_path):
    """
    Transcribe an audio file using the Whisper model.
    Automatically handles resampling to 16kHz if needed.
    
    Args:
        audio_path (str): Path to the audio file
    
    Returns:
        str: Transcribed text
    """
    try:
        # Load model and processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        model.config.forced_decoder_ids = None

        # Load audio file with automatic resampling to 16kHz
        print("Loading and processing audio file...")
        audio_input, orig_sr = librosa.load(audio_path, sr=None)  # Load with original sampling rate
        if orig_sr != 16000:
            print(f"Resampling audio from {orig_sr}Hz to 16000Hz...")
            audio_input = librosa.resample(y=audio_input, orig_sr=orig_sr, target_sr=16000)
        
        # Convert to mono if stereo
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)

        # Process audio
        input_features = processor(
            audio_input,
            sampling_rate=16000,  # Now we know it's always 16kHz
            return_tensors="pt"
        ).input_features

        # Generate transcription
        print("Generating transcription...")
        predicted_ids = model.generate(input_features)
        
        # Decode and return transcription
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription

    except Exception as e:
        return f"Error during transcription: {str(e)}"

if __name__ == "__main__":
    # If want to get the audio file path from the user.
    # audio_file_path = input("Please enter the path to your audio file: ")

    # Or from local
    audio_file_path = r"C:\Users\karth\OneDrive\Desktop\BlueKyte Work\VTT\audios\KB.wav"
    
    # Perform transcription
    print("\nTranscribing audio... Please wait...")
    result = transcribe_audio(audio_file_path)
    
    # Print results
    print("\nTranscription:")
    print("-" * 50)
    print(result)
    print("-" * 50)
