import os
import requests
import logging
import json
import asyncio
import aiohttp
import nest_asyncio
import boto3
import aioboto3
from botocore.config import Config
from dotenv import load_dotenv
import torch
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Union, Any

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
        "COPILOT_BASE_URL"
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

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes audio using Whisper model
    """
    try:
        logger.info("Starting audio transcription with Whisper...")
        
        audio_input, orig_sr = librosa.load(audio_path, sr=None)
        if orig_sr != 16000:
            audio_input = librosa.resample(y=audio_input, orig_sr=orig_sr, target_sr=16000)
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)

        input_features = whisper_processor(
            audio_input, 
            sampling_rate=16000, 
            return_tensors="pt", 
            language='en'
        ).input_features
        
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        logger.info("Audio transcription completed successfully")
        return transcription
        
    except Exception as e:
        logger.error(f"Error in audio transcription: {str(e)}")
        raise

async def query_bedrock_correction(transcription: str) -> str:
    """
    Sends transcription to Claude for correction of Indian proper nouns and legal case names.
    """
    session = aioboto3.Session()
    async with session.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        config=Config(
            connect_timeout=300,
            read_timeout=300,
            retries={"max_attempts": 10}
        ),
    ) as bedrock:
        correction_prompt = f"""Please review and correct only the Indian proper nouns, names, and legal case names in the following text. 
        Ensure that any Indian legal case names, personal names, or geographical names are spelled correctly and formatted according to standard Indian legal conventions.

        Original text: {transcription}

        Strict Correction Rules:
        - Only correct Indian case names, personal names, or geographical names, keeping all other text exactly the same.
        - Maintain accurate spelling and capitalization for names related to Indian law, including proper noun formats.
        - Do not rephrase, reword, or modify non-Indian names, any common words, or any parts of the text other than the specified proper nouns.
        - Ensure case titles adhere to Indian legal citation formats, such as "XYZ vs. ABC" or "In re: XYZ."
        - If unsure of a correction, leave the text exactly as it is.

        Output should contain only the strictly corrected text, with no added notes or explanations.
        """

        body = json.dumps({
            "anthropic_version": "bedrock-31",
            "messages": [
                {
                    "role": "user",
                    "content": correction_prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.5,
            "top_p": 0.9
        })

        try:
            response = await bedrock.invoke_model(
                body=body,
                modelId=correction_model_id,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(await response.get("body").read())
            corrected_text = response_body.get("content", [{"text": transcription}])[0].get("text", transcription)
            
            logger.info("Transcription correction completed successfully")
            return corrected_text

        except Exception as e:
            logger.error(f"Error in transcription correction: {str(e)}")
            return transcription

def detect_intent(text: str) -> Dict[str, Any]:
    """
    Detects the intent of the query based on keywords and patterns
    Returns a dictionary with intent type and extracted entities
    """
    text_lower = text.lower()
    
    # Case Research related keywords
    case_research_keywords = ["case research", "find case"]
    search_with_case = any(keyword in text_lower for keyword in case_research_keywords) and "search" in text_lower
    
    # Copilot related keywords
    copilot_keywords = ["search", "what is", "tell me", "explain", "how to", "define"]
    
    # Extract case name if present
    case_name = extract_case_name(text) if search_with_case or any(keyword in text_lower for keyword in case_research_keywords) else None
    
    # Determine intent
    if search_with_case or (any(keyword in text_lower for keyword in case_research_keywords) and case_name):
        return {
            "intent_type": "case_research",
            "confidence": 0.9,
            "entities": {
                "case_name": case_name,
                "key_terms": extract_key_terms(case_name) if case_name else []
            }
        }
    elif any(keyword in text_lower for keyword in copilot_keywords):
        return {
            "intent_type": "copilot",
            "confidence": 0.8,
            "entities": {
                "query": text.strip()
            }
        }
    
    return {
        "intent_type": "unknown",
        "confidence": 0.0,
        "entities": {}
    }

def extract_case_name(text: str) -> Optional[str]:
    """
    Extracts case name from the text using pattern matching
    """
    text_lower = text.lower()
    
    patterns = [
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
            case_name = case_name.split(" case")[0].strip()
            return case_name
            
    return None

def extract_key_terms(case_name: Optional[str]) -> List[str]:
    """
    Extracts key terms from case name
    """
    if not case_name:
        return []
        
    terms = case_name.replace(" v. ", " versus ").replace(" vs ", " versus ").split()
    stop_words = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to"}
    clean_terms = [term.strip(",.?![]'\"") for term in terms if term.lower() not in stop_words]
    
    return clean_terms

# def invoke_copilot(query: str, user_id: str = "1877", session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
#     """
#     Invokes the copilot API with the given query
#     Parameters:
#         query (str): The case name or query from user
#         user_id (str): User ID (default: "1877")
#         session_id (str): Session ID (default: "86068bd2-4311-11ee-a6b3-014c891e3538")
#     Returns:
#         Optional[str]: Response from copilot API or None if failed
#     """
#     try:
#         # Format URL with required parameters
#         url = f"{copilot_base_url}/add-message-to-session-documentqa/{user_id}/{session_id}/{query}/total"
        
#         # URL encode the query part to handle special characters
#         url = requests.utils.quote(url, safe=':/?=&')
        
#         headers = {
#             "Content-Type": "application/json"
#         }
        
#         logger.info(f"Calling Copilot API with URL: {url}")
        
#         response = requests.post(url, headers=headers, timeout=60)
        
#         logger.info(f"Copilot API Response Status: {response.status_code}")
        
#         if response.status_code != 200:
#             logger.error(f"Copilot API Error: {response.status_code} - {response.text}")
#             return None
            
#         response_data = response.json()
#         # Process response according to actual API response structure
#         # Adjust the response data extraction based on actual API response format
#         result = response_data.get("answer", response_data.get("response", "No answer found"))
        
#         # Synthesize speech for the response
#         synthesize_speech(result)
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Error in copilot API call: {str(e)}")
#         return None

async def invoke_copilot(query: str, user_id: str = "1877", session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
    """
    Invokes the copilot API with the given query using a streaming request
    Parameters:
        query (str): The case name or query from user
        user_id (str): User ID (default: "1877")
        session_id (str): Session ID (default: "86068bd2-4311-11ee-a6b3-014c891e3538")
    Returns:
        Optional[str]: Response from copilot API or None if failed
    """
    try:
        # Format URL with required parameters
        url = f"{copilot_base_url}/add-message-to-session-documentqa/{user_id}/{session_id}/{query}/total"
        
        # URL encode the query part to handle special characters
        url = requests.utils.quote(url, safe=':/?=&')
        
        headers = {
            "Content-Type": "application/json"
        }
        
        logger.info(f"Calling Copilot API with URL: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, timeout=60) as response:
                if response.status != 200:
                    logger.error(f"Copilot API Error: {response.status} - {await response.text()}")
                    return None
                
                result = ""
                async for chunk in response.content.iter_any():
                    result += chunk.decode()
                
                # Synthesize speech for the response
                synthesize_speech(result)
                
                return result
    except Exception as e:
        logger.error(f"Error in copilot API call: {str(e)}")
        return None

def invoke_case_research(case_name: str, additional_terms: Optional[List[str]] = None) -> Optional[str]:
    """
    Invokes the case research API with enhanced error handling and logging
    """
    try:
        if not case_name:
            logger.error("No case name provided")
            return None

        search_query = [case_name]
        if additional_terms:
            search_query.extend([term for term in additional_terms if term and len(term) > 1])
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
            case_research_url, 
            json=payload, 
            headers=headers,
            timeout=60
        )
        
        logger.info(f"API Response Status Code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API Error: Status {response.status_code}, Response: {response.text}")
            return None

        response_data = response.json().get("data", {})
        
        # Check MeiliSearch results first
        meili_results = response_data.get("meiliSearchResults", [])
        if meili_results:
            first_title = meili_results[0].get("title", "No title found")
            synthesize_speech(first_title)
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

def synthesize_speech(text: str, voice_id: str = "Raveena", output_format: str = "mp3", output_file: str = "speech2.mp3"):
    """
    Synthesizes speech using Amazon Polly
    """
    try:
        polly_client = boto3.client(
            "polly",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name="us-east-1"
        )
        
        response = polly_client.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            OutputFormat=output_format
        )
        
        with open(output_file, "wb") as file:
            file.write(response['AudioStream'].read())
        logger.info(f"Audio content saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise

# async def process_query(transcribed_text: str, 
#                        user_id: str = "1877", 
#                        session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
#     """
#     Processes the query and routes to appropriate service based on intent
#     """
#     try:
#         # Detect intent
#         intent_result = detect_intent(transcribed_text)
        
#         logger.info(f"Detected intent: {intent_result['intent_type']} with confidence: {intent_result['confidence']}")
        
#         if intent_result["intent_type"] == "case_research":
#             case_name = intent_result["entities"].get("case_name")
#             key_terms = intent_result["entities"].get("key_terms", [])
#             if case_name:
#                 return invoke_case_research(case_name, key_terms)
        
#         elif intent_result["intent_type"] == "copilot":
#             query = intent_result["entities"].get("query")
#             if query:
#                 return invoke_copilot(query, user_id, session_id)
        
#         logger.warning(f"No valid intent detected or missing required entities. Intent: {intent_result['intent_type']}")
#         return None
            
#     except Exception as e:
#         logger.error(f"Error in query processing: {str(e)}")
#         return None

async def process_query(transcribed_text: str, 
                       user_id: str = "1877", 
                       session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
    """
    Processes the query and routes to appropriate service based on intent
    """
    try:
        # Detect intent
        intent_result = detect_intent(transcribed_text)
        
        logger.info(f"Detected intent: {intent_result['intent_type']} with confidence: {intent_result['confidence']}")
        
        if intent_result["intent_type"] == "case_research":
            case_name = intent_result["entities"].get("case_name")
            key_terms = intent_result["entities"].get("key_terms", [])
            if case_name:
                return await invoke_case_research(case_name, key_terms)
        
        elif intent_result["intent_type"] == "copilot":
            query = intent_result["entities"].get("query")
            if query:
                return await invoke_copilot(query, user_id, session_id)
        
        logger.warning(f"No valid intent detected or missing required entities. Intent: {intent_result['intent_type']}")
        return None
            
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        return None

# async def main(audio_file_path: str,
#                user_id: str = "1877",
#                session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
#     """
#     Main pipeline to process audio input and return appropriate response
#     """
#     try:
#         if not os.path.exists(audio_file_path):
#             raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
#         initial_transcription = transcribe_audio(audio_file_path)
#         logger.info(f"Initial Transcription: {initial_transcription}")
        
#         corrected_transcription = await query_bedrock_correction(initial_transcription)
#         logger.info(f"Corrected Transcription: {corrected_transcription}")
        
#         result = await process_query(corrected_transcription, user_id, session_id)
        
#         if result:
#             logger.info(f"Process result: {result}")
#             return result
#         else:
#             logger.warning("No result available from processing.")
#             return None
            
#     except Exception as e:
#         logger.error(f"Error in main pipeline: {str(e)}")
#         raise


# if __name__ == "__main__":
#     try:
#         audio_file_path = r"C:\Users\karth\OneDrive\Desktop\BlueKyte Work\VTT\audios\copilot.wav"
#         user_id = "1877"
#         session_id = "86068bd2-4311-11ee-a6b3-014c891e3538"
        
#         result = asyncio.run(main(audio_file_path, user_id, session_id))
#         if result:
#             print(f"Process completed successfully. Result: {result}")
#         else:
#             print("Process completed but no results were found.")
#     except Exception as e:
#         logger.error(f"Application error: {str(e)}")
#         print(f"An error occurred: {str(e)}")

async def main(audio_file_path: str,
               user_id: str = "1877",
               session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]:
    """
    Main pipeline to process audio input and return appropriate response
    """
    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
        initial_transcription = transcribe_audio(audio_file_path)
        logger.info(f"Initial Transcription: {initial_transcription}")
        
        corrected_transcription = await query_bedrock_correction(initial_transcription)
        logger.info(f"Corrected Transcription: {corrected_transcription}")
        
        result = await process_query(corrected_transcription, user_id, session_id)
        
        if result:
            logger.info(f"Process result: {result}")
            return result
        else:
            logger.warning("No result available from processing.")
            return None
            
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        audio_file_path = r"C:\Users\karth\OneDrive\Desktop\BlueKyte Work\VTT\audios\copilot.wav"
        user_id = "1877"
        session_id = "86068bd2-4311-11ee-a6b3-014c891e3538"
        
        result = asyncio.run(main(audio_file_path, user_id, session_id))
        if result:
            print(f"Process completed successfully. Result: {result}")
        else:
            print("Process completed but no results were found.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"An error occurred: {str(e)}")
