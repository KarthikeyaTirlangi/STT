# Documentation for Production Pipeline Code

## Overview
This document provides an overview of the production pipeline script designed to streamline and automate various tasks related to legal processing. The script integrates advanced functionalities like audio transcription, intent detection, and API-based interactions to handle legal queries efficiently. It supports a wide range of operations including transcribing audio inputs, identifying user intent, generating legal drafts, performing case research, and synthesizing text-to-speech outputs. This ensures a seamless and structured approach to processing complex legal tasks, reducing manual effort and improving accuracy.

---

## Key Features

1. **Configuration Validation:** Ensures all necessary environment variables are present and valid.
2. **Audio Transcription:** Converts audio input into text using OpenAI Whisper models.
3. **Intent Detection:** Determines the purpose of the transcribed text (e.g., case research, drafting, copilot query).
4. **Legal Draft System:** Handles drafting processes, including template retrieval and case information extraction.
5. **Case Research:** Searches for case information using external APIs.
6. **Speech Synthesis:** Converts text to audio using Amazon Polly.

---

## Prerequisites

### Environment Variables
Ensure the following environment variables are set:
- `CASE_RESEARCH_URL`
- `CASE_RESEARCH_API_KEY`
- `AWS_ACCESS_KEY`
- `AWS_SECRET_KEY`
- `MODEL_ID`
- `COPILOT_BASE_URL`
- `DRAFTING_BASE_URL`

### Required Python Libraries
Install the following dependencies:
- `boto3`
- `aioboto3`
- `librosa`
- `aiohttp`
- `sseclient`
- `requests`
- `transformers`
- `python-dotenv`

Use `pip install` to install the above packages.

---

## Code Modules and Functions

### 1. **Configuration Validation**

- **Function:** `validate_config`
- **Purpose:** Ensures that all required environment variables are set and valid.
- **Details:**
  - Checks for missing variables.
  - Validates API URLs and formats.
  - Issues warnings for incorrect API keys.

### 2. **Audio Transcription**

- **Function:** `transcribe_audio`
- **Purpose:** Converts audio files to text.
- **Details:**
  - Uses `librosa` to load and resample audio to 16,000 Hz.
  - Processes audio using OpenAI Whisper.
  - Generates transcriptions.

### 3. **Case Name Extraction**

- **Function:** `extract_case_name`
- **Purpose:** Extracts case names from the text using keyword patterns.
- **Details:**
  - Searches for patterns like "case of", "versus", "v."
  - Attempts fallback extraction if patterns fail.

### 4. **Key Term Extraction**

- **Function:** `extract_key_terms`
- **Purpose:** Extracts significant terms from a case name.
- **Details:**
  - Filters out stop words.
  - Processes terms for cleaner representation.

### 5. **Intent Detection**

- **Function:** `detect_intent`
- **Purpose:** Determines the purpose of the query.
- **Details:**
  - Identifies intents such as drafting, case research, or copilot queries.
  - Uses keywords and extracted case names.

### 6. **Text Correction**

- **Function:** `correct_indian_legal_text`
- **Purpose:** Corrects legal terms and proper nouns using Bedrock AI.
- **Details:**
  - Integrates with Bedrock runtime.
  - Formats text correction prompts.

### 7. **Copilot API Invocation**

- **Function:** `invoke_copilot`
- **Purpose:** Interacts with the Copilot API to process user queries.
- **Details:**
  - Streams JSON responses from the API.
  - Logs and processes the output.

### 8. **Case Research API**

- **Function:** `invoke_case_research`
- **Purpose:** Queries the case research API.
- **Details:**
  - Constructs search queries from case names and terms.
  - Processes MeiliSearch and Pinecone results.

### 9. **Legal Draft System**

- **Class:** `LegalDraftSystem`
- **Methods:**
  - `extract_case_info`: Extracts court type and case title.
  - `get_templates`: Retrieves legal draft templates.
  - `retrieve_template`: Fetches specific templates.
  - `process_final_draft`: Processes and saves drafts.

### 10. **Speech Synthesis**

- **Function:** `synthesize_speech`
- **Purpose:** Converts text to speech using Amazon Polly.
- **Details:**
  - Truncates text to meet Polly's character limits.
  - Outputs audio in MP3 format.

### 11. **Query Processing**

- **Function:** `process_query`
- **Purpose:** Handles intent-based query processing.
- **Details:**
  - Delegates tasks to drafting, case research, or copilot systems.

### 12. **Main Function**

- **Function:** `main`
- **Purpose:** Entry point for processing audio input.
- **Details:**
  - Transcribes audio.
  - Corrects transcription text.
  - Processes the query based on intent.

---

## Execution

To run the pipeline:
1. Set up environment variables using a `.env` file.
2. Provide the path to the audio file in the `audio_file_path` variable.
3. Execute the script.

---

## Logging

- Logs are configured using the `logging` module.
- Output includes timestamps, log levels, and messages.
- Log levels: INFO, ERROR, WARNING.

---

## Error Handling

- Validates configurations and inputs at multiple stages.
- Provides detailed logs for debugging.
- Handles API exceptions and network errors.

---