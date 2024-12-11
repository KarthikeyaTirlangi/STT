import boto3 #type: ignore

def synthesize_speech(text, voice_id="Raveena", output_format="wav", output_file="speech.wav"):
    polly_client = boto3.client("polly")

    try:
        response = polly_client.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            OutputFormat=output_format
        )

        with open(output_file, "wb") as file:
            file.write(response['AudioStream'].read())

        print(f"Audio content saved to {output_file}")

    except Exception as e:
        print("Error synthesizing speech:", e)

text_to_speak = "Hello, Karthik!! How's your day?."
synthesize_speech(text=text_to_speak)
