# Imports and Logging Configuration

import os
import json
import logging
import boto3  # type: ignore
import asyncio
import aiohttp  # type: ignore
import aioboto3  # type: ignore
import librosa # type: ignore
import nest_asyncio # type: ignore
import sseclient  # type: ignore
import requests # type: ignore
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv  # type: ignore
from botocore.config import Config  # type: ignore
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # type: ignore

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration validation
def validate_config():
    required_vars = [
        "CASE_RESEARCH_URL",
        "CASE_RESEARCH_API_KEY",
        "AWS_ACCESS_KEY",
        "AWS_SECRET_KEY",
        "MODEL_ID",
        "COPILOT_BASE_URL",
        "DRAFTING_BASE_URL"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    case_research_url = os.getenv("CASE_RESEARCH_URL")
    copilot_base_url = os.getenv("COPILOT_BASE_URL")

    for url in [case_research_url, copilot_base_url]:
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid API URL format: {url}")

    case_research_api_key = os.getenv("CASE_RESEARCH_API_KEY")
    if not case_research_api_key.startswith("Bearer "):
        logger.warning("API key should start with 'Bearer '")


try:
    validate_config()
except Exception as e:
    logger.error(f"Configuration error: {str(e)}")
    raise

# Environment variables
case_research_url = os.getenv("CASE_RESEARCH_URL")
case_research_api_key = os.getenv("CASE_RESEARCH_API_KEY")
aws_access_key = os.getenv("AWS_ACCESS_KEY")
aws_secret_key = os.getenv("AWS_SECRET_KEY")
model_id = os.getenv("MODEL_ID")
correction_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
copilot_base_url = os.getenv("COPILOT_BASE_URL")
drafting_base_url = os.getenv("DRAFTING_BASE_URL")

nest_asyncio.apply()

# Initialize Whisper model and processor
try:
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    whisper_model.config.forced_decoder_ids = None
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {str(e)}")
    raise


# Transcribe audio
def transcribe_audio(audio_path: str) -> str:
    audio_input, orig_sr = librosa.load(audio_path, sr=None)
    if orig_sr != 16000:
        audio_input = librosa.resample(y=audio_input, orig_sr=orig_sr, target_sr=16000)
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)
    input_features = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt", language="en").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def extract_case_name(text: str) -> Optional[str]:
    """
    Extracts case name from the text using pattern matching
    """
    text_lower = text.lower()

    # Add pattern for "summary of case" format
    patterns = [
        (text_lower.find("summary of case"), "summary of case"),
        (text_lower.find("case research for"), "case research for"),
        (text_lower.find("search for"), "search for"),
        (text_lower.find("case of"), "case of"),
        (text_lower.find("versus"), "versus"),
        (text_lower.find(" v. "), " v. "),
        (text_lower.find(" vs "), " vs "),
    ]

    for idx, pattern in patterns:
        if idx != -1:
            start_idx = idx + len(pattern)
            case_name = text[start_idx:].strip(" ,.?![]'\"")
            # Keep the original case formatting from the input text
            case_name = text[start_idx : start_idx + len(case_name)].strip()
            return case_name

    # If no pattern matches, try to find a case name format directly

    # Example: "State of Maharashtra v. ABC Corporation"
    # part[0] = SOM, part[1] = ABC Co
    if " v. " in text:
        parts = text.split(" v. ")
        if len(parts) >= 2:
            return f"{parts[0].strip()} v. {parts[1].strip()}"
    elif " vs " in text:
        parts = text.split(" vs ")
        if len(parts) >= 2:
            return f"{parts[0].strip()} vs {parts[1].strip()}"

    return None


def extract_key_terms(case_name: Optional[str]) -> List[str]:
    """
    Extracts key terms from case name
    """
    if not case_name:
        return []

    terms = case_name.replace(" v. ", " versus ").replace(" vs ", " versus ").split()
    stop_words = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to"}
    clean_terms = [
        term.strip(",.?![]'\"") for term in terms if term.lower() not in stop_words
    ]

    return clean_terms


# Intent detection to identify copilot, case research, summary, or drafting
def detect_intent(text: str) -> Dict[str, Any]:
    """
    Detects the intent of the query based on keywords and patterns,
    and uses extract_case_name and extract_key_terms to identify case-related intents.
    """
    text_lower = text.lower()

    # Keywords for different intents
    drafting_keywords = ["draft", "legal draft"]
    case_research_keywords = ["case research", "find case"]
    summary_keywords = [""]
    copilot_keywords = ["search", "what is", "tell me", "explain", "how to", "define", "summary"]

    # Extract potential case name
    extracted_case_name = extract_case_name(text)

    # Intent determination
    if any(keyword in text_lower for keyword in drafting_keywords):
        return {
            "intent_type": "drafting",
            "confidence": 0.9,
            "entities": {"case_name": extracted_case_name},
        }
    elif any(keyword in text_lower for keyword in summary_keywords):
        return {
            "intent_type": "summary",
            "confidence": 0.9,
            "entities": {"case_name": extracted_case_name},
        }
    elif any(keyword in text_lower for keyword in case_research_keywords):
        return {
            "intent_type": "case_research",
            "confidence": 0.9,
            "entities": {
                "case_name": extracted_case_name,
                "key_terms": (
                    extract_key_terms(extracted_case_name)
                    if extracted_case_name
                    else []
                ),
            },
        }
    elif any(keyword in text_lower for keyword in copilot_keywords):
        return {
            "intent_type": "copilot",
            "confidence": 0.8,
            "entities": {"query": text.strip()},
        }

    return {"intent_type": "unknown", "confidence": 0.0, "entities": {}}


async def correct_indian_legal_text(text: str) -> str:
    session = aioboto3.Session()
    async with session.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        config=Config(
            connect_timeout=300, read_timeout=300, retries={"max_attempts": 10}
        ),
    ) as bedrock:
        correction_prompt = f"""Please review and correct only the Indian proper nouns, names, and legal case names...
        Original text: {text}"""
        response = await bedrock.invoke_model(
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": correction_prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            ),
            modelId=model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(await response.get("body").read())
        return response_body.get("content", [{"text": text}])[0].get("text", text)



# SERVICE INVOKING ---

async def invoke_copilot(
    query: str,
    user_id: str = "1877",
    session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538",
) -> Optional[str]:
    """
    Invokes the Copilot API with the given query, accumulates content from each JSON chunk in the response,
    and returns the full response once complete.
    """
    # Construct the URL with parameters
    url = f"{os.getenv('COPILOT_BASE_URL')}/add-message-to-session-documentqa/{user_id}/{session_id}/{query}/total"
    headers = {"Content-Type": "application/json"}

    try:
        # Establish a streaming connection
        response = requests.get(url, headers=headers, stream=True)
        client = sseclient.SSEClient(response)

        full_content = ""

        # Iterate over incoming events from the SSE stream
        for event in client.events():
            if event.data:  # Check if there's data in the event
                # Parse the event data as JSON
                data = json.loads(event.data)

                # Extract content if it exists
                delta = data.get("delta", {})
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)
                    full_content += content

                # Break if the finish reason is 'stop'
                if data.get("finish_reason") == "stop":
                    break

        logger.info(f"Copilot API response: {full_content[:50]}")  # Log first 50 chars
        return full_content

    except Exception as e:
        logger.error(f"Error in Copilot API call: {str(e)}")
        return None


async def invoke_case_research(
    case_name: str, additional_terms: Optional[List[str]] = None
) -> Optional[str]:
    """
    Invokes the case research API with enhanced error handling and logging
    """
    try:
        if not case_name:
            logger.error("No case name provided")
            return None

        search_query = [case_name]
        if additional_terms:
            search_query.extend(
                [term for term in additional_terms if term and len(term) > 1]
            )
        search_query = [" ".join(search_query)]

        logger.info(f"Prepared search query: {search_query}")

        headers = {
            "authorization": case_research_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "semantic_search_query": search_query,
            "type": [],
            "case_type": [],
        }

        response = requests.post(
            case_research_url, json=payload, headers=headers, timeout=60
        )

        logger.info(f"API Response Status Code: {response.status_code}")

        if response.status_code != 200:
            logger.error(
                f"API Error: Status {response.status_code}, Response: {response.text}"
            )
            return None

        response_data = response.json().get("data", {})

        # Check MeiliSearch results first
        meili_results = response_data.get("meiliSearchResults", [])
        if meili_results:
            first_title = meili_results[0].get("title", "No title found")
            synthesize_speech(first_title)
            print(first_title)
            return first_title

        # Fall back to Pinecone results
        pinecone_results = response_data.get("pinconeResultsMapList", [])
        if pinecone_results:
            first_title = pinecone_results[0].get("title", "No title found")
            synthesize_speech(first_title)
            return first_title

        logger.warning("No results found in API response data.")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in case research API: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error in case research API: {str(e)}")
        return None


class LegalDraftSystem:
    def __init__(self):
        self.url = drafting_base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        logger.info("LegalDraftSystem initialized")
        print("Legal Draft System Ready")


    async def extract_case_info(self, text: str) -> Dict[str, str]:
        print("\nExtracting case information...")
        """Extracts case title and court type from text"""
        text_lower = text.lower()

        # Determine court
        court_type = "supreme court" if "supreme court" in text_lower else "high court"
        folder = "Special Leave Petition" if "supreme court" in text_lower else "Writ"

        logger.info(f"Detected court type: {court_type}")
        logger.info(f"Detected case folder: {folder}")

        # Extract title using existing patterns
        patterns = [
            (text_lower.find("case research for"), "case research for"),
            (text_lower.find("search for"), "search for"),
            (text_lower.find("case of"), "case of"),
            (text_lower.find("versus"), "versus"),
            (text_lower.find(" v. "), " v. "),
            (text_lower.find(" vs "), " vs "),
        ]

        title = None
        for idx, pattern in patterns:
            if idx != -1:
                start_idx = idx + len(pattern)
                title = text[start_idx:].strip(" ,.?![]'\"")
                title = title.split(" case")[0].strip()
                logger.info(f"Extracted case title using pattern: {pattern}")
                logger.info(f"Extracted title: {title}")
                break

        if not title:
            # Extract based on context if pattern matching fails
            logger.warning(
                "Failed to extract title using patterns, using fallback method"
            )

            title = "Legal Matter"  # Default title
            for phrase in ["regarding", "about", "concerning"]:
                if phrase in text_lower:
                    idx = text_lower.find(phrase) + len(phrase)
                    next_period = text[idx:].find(".")
                    if next_period != -1:
                        title = text[idx : idx + next_period].strip()
                        break

        return {"title": title, "court_type": court_type, "folder": folder}

    async def get_templates(self, folder: str) -> List[str]:
        print(f"\nRetrieving templates for folder: {folder}")
        if not folder:
            logger.error("Empty folder name provided")
            print("Error: No folder specified for templates")
            return []

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    f"{self.url}/list-folders",
                    headers=self.headers,
                    json={"bucket_name": "drafts-legal", "folder": folder},
                )
                if response.status == 200:
                    templates = (await response.json()).get("folders", [])
                    logger.info(f"Successfully retrieved {len(templates)} templates")
                    print(f"Found {len(templates)} templates")
                    return templates
                logger.error(f"Failed to get templates. Status: {response.status}")
                print(f"Error: Failed to retrieve templates (Status {response.status})")
                return []
        except Exception as e:
            logger.error(f"Error getting templates: {str(e)}")
            print(f"Error: Template retrieval failed - {str(e)}")
            return []

    async def retrieve_template(self, template_name: str, folder: str) -> Dict:
        print(f"\nRetrieving template: {template_name} from {folder}")

        template_name = template_name.rstrip("/")

        if not template_name or not folder:
            logger.error("Missing template name or folder")
            print("Error: Missing template details")
            return {}

        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    f"{self.url}/retrieve-files",
                    headers=self.headers,
                    json={"filename": template_name, "draft_folder": folder},
                )
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully retrieved template: {template_name}")
                    print(f"Template retrieved successfully")
                    return result
                logger.error(f"Failed to retrieve template. Status: {response.status}")
                print(f"Error: Failed to retrieve template (Status {response.status})")
                return {}
        except Exception as e:
            logger.error(f"Error retrieving template: {str(e)}")
            print(f"Error: Template retrieval failed - {str(e)}")
            return {}

    async def process_final_draft(self, template_name: str, complaint: str) -> str:
        print("\nProcessing final draft...")
        if not template_name or not complaint:
            logger.error("Missing template name or complaint text")
            print("Error: Missing required content for draft processing")
            return "Error: Missing template name or complaint text"

        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "markdown_file_content": f"**{complaint}\n",
                    "json_file_content": f'"{template_name}.json file not found."',
                    "user_prompt": complaint,
                }
                async with session.post(
                    f"{self.url}/process-content", headers=self.headers, json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        processed_content = result.get(
                            "processed_content", "Processing error"
                        )
                        logger.info("Successfully processed final draft")
                        print("Final draft processed successfully")
                        return f"Draft Processed: {processed_content}"
                    logger.error(f"Draft processing failed. Status: {response.status}")
                    print(f"Error: Draft processing failed (Status {response.status})")
                    return f"Error processing draft. Status: {response.status}"
        except Exception as e:
            logger.error(f"Error processing final draft: {str(e)}")
            print(f"Error: Draft processing failed - {str(e)}")
            return f"Error processing draft: {str(e)}"


def synthesize_speech(
    text: str,
    voice_id: str = "Raveena",
    output_format: str = "mp3",
    output_file: str = "speech3.mp3",
):
    """
    Synthesizes speech using Amazon Polly, with text length management for longer responses.
    """
    try:
        # Limit the text length to Amazon Polly’s limit of 1,500 characters
        truncated_text = text[:1500]

        polly_client = boto3.client(
            "polly",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name="us-east-1",
        )

        response = polly_client.synthesize_speech(
            Text=truncated_text, VoiceId=voice_id, OutputFormat=output_format
        )

        with open(output_file, "wb") as file:
            file.write(response["AudioStream"].read())
        logger.info(f"Audio content saved to {output_file}")

    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise


async def process_query(
    transcribed_text: str,
    user_id: str = "1877",
    session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538",
) -> Optional[str]:
    print("\n=== Starting Query Processing ===")
    logger.info(
        f"Processing query with text length: {len(transcribed_text) if transcribed_text else 0}"
    )

    if not transcribed_text:
        logger.error("Empty transcribed text provided")
        print("Error: No text to process")
        return None

    intent_result = detect_intent(transcribed_text)
    intent_type = intent_result.get("intent_type")
    print(f"Detected intent: {intent_type}")
    logger.info(f"Intent detection completed. Type: {intent_type}")

    if intent_type == "drafting":
        print("\n--- Starting Draft Processing ---")
        draft_system = LegalDraftSystem()

        # Extract case info
        case_info = await draft_system.extract_case_info(transcribed_text)
        if not case_info:
            logger.error("Failed to extract case information")
            print("Error: Could not extract case information")
            return None
        print(f"Case Info Extracted: {case_info['court_type']} - {case_info['folder']}")
        logger.info(f"Successfully extracted case info: {case_info}")

        # Get templates
        templates = await draft_system.get_templates(case_info["folder"])
        if not templates:
            logger.error(f"No templates found for folder: {case_info['folder']}")
            print(f"Error: No templates available for {case_info['folder']}")
            return None
        print(f"Found {len(templates)} templates")
        logger.info(f"Successfully retrieved {len(templates)} templates")

        # Process draft
        selected_template = templates[0]
        print(f"Selected template: {selected_template}")
        template_content = await draft_system.retrieve_template(
            selected_template, case_info["folder"]
        )
        result = await draft_system.process_final_draft(
            selected_template, transcribed_text
        )

        output_file_path = "sample_draft.md"
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(result)

        print("\n--- Draft Processing Completed ---")
        return result

    elif intent_type == "case_research":
        print("\n--- Starting Case Research ---")
        case_name = intent_result["entities"].get("case_name")
        key_terms = intent_result["entities"].get("key_terms", [])

        if case_name:
            print(f"Researching case: {case_name}")
            print(f"Additional terms: {', '.join(key_terms) if key_terms else 'None'}")
            result = await invoke_case_research(case_name, key_terms)
            if result:
                print("\n--- Case Research Completed Successfully ---")
                logger.info(f"Successfully completed case research for: {case_name}")
            return result
        else:
            logger.error("No case name found in the query")
            print("Error: No case name provided for research")
            return None

    elif intent_type == "copilot":
        print("\n--- Starting Copilot Query ---")
        query = intent_result["entities"].get("query")
        if query:
            print(f"Processing copilot query: {query[:100]}...")  # Show first 100 chars
            result = await invoke_copilot(query, user_id, session_id)
            if result:
                print("\n--- Copilot Query Completed Successfully ---")
                logger.info("Successfully completed copilot query")
            return result
        else:
            logger.error("No query found in copilot intent")
            print("Error: Empty query for copilot")
            return None

    print("\n=== Query Processing Completed with No Valid Intent ===")
    logger.warning(
        f"No valid intent detected or missing required entities. Intent: {intent_type}"
    )
    return None


async def main(
    audio_file_path: str,
    user_id: str = "1877",
    session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538",
) -> Optional[str]:
    
    print("\n=== Starting Main Processing ===")
    logger.info(f"Processing audio file: {audio_file_path}")

    print("Transcribing audio...")
    initial_transcription = transcribe_audio(audio_file_path)
    print("Audio transcription completed")
    logger.info("Audio transcription successful")

    print("Correcting transcribed text...")
    corrected_transcription = await correct_indian_legal_text(initial_transcription)
    print("Text correction completed")
    logger.info("Text correction successful")

    print("Processing query...")
    result = await process_query(corrected_transcription, user_id, session_id)

    if result:
        print("\n=== Processing Completed Successfully ===")
        logger.info("Main processing completed successfully")
    else:
        print("\n=== Processing Completed with Errors ===")
        logger.warning("Main processing completed with errors or no result")

    return result


if __name__ == "__main__":
    nest_asyncio.apply()
    audio_file_path = (r"/Users/karthikeyatirlangi/Desktop/Work/BlueKyte/VTT/audios/drafting.wav")
    asyncio.run(main(audio_file_path))