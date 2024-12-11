import os
import json
import asyncio
import aiohttp #type: ignore
import aioboto3 #type: ignore
import logging
import librosa
import nest_asyncio
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from botocore.config import Config #type: ignore
from transformers import WhisperProcessor, WhisperForConditionalGeneration #type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalDraftSystem:
    def __init__(self):
        # Initialize environment variables
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY")
        self.aws_secret_key = os.getenv("AWS_SECRET_KEY")
        # self.model_id = os.getenv("MODEL_ID")
        self.correction_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        self.template_selection_model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        # Initialize Whisper
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
            self.whisper_model.config.forced_decoder_ids = None
            logger.info("Whisper model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Whisper: {str(e)}")
            raise

        # API configuration
        self.base_url = "https://aiengine.counsello.ai/draft"
        self.headers = {
            'accept': 'application/json, text/plain, */*',
            'content-type': 'application/json',
            'origin': 'https://suite.counsello.ai',
            'referer': 'https://suite.counsello.ai',
            'sec-ch-ua': '"Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribes audio using Whisper model"""
        try:
            logger.info("Starting audio transcription...")
            
            # Load and preprocess audio
            audio_input, orig_sr = librosa.load(audio_path, sr=None)
            if orig_sr != 16000:
                audio_input = librosa.resample(y=audio_input, orig_sr=orig_sr, target_sr=16000)
            if len(audio_input.shape) > 1:
                audio_input = audio_input.mean(axis=1)

            # Process audio through Whisper
            input_features = self.whisper_processor(
                audio_input, sampling_rate=16000, return_tensors="pt", language="en"
            ).input_features

            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            logger.info("Audio transcription completed")
            return transcription

        except Exception as e:
            logger.error(f"Error in audio transcription: {str(e)}")
            raise

    async def correct_indian_legal_text(self, text: str) -> str:
        """Corrects Indian legal terms and names using Claude"""
        session = aioboto3.Session()
        async with session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            config=Config(connect_timeout=300, read_timeout=300, retries={"max_attempts": 10})
        ) as bedrock:
            correction_prompt = f"""Please review and correct only the Indian proper nouns, names, and legal case names in the following text. 
            Ensure that any Indian legal case names, personal names, or geographical names are spelled correctly and formatted according to standard Indian legal conventions.

            Original text: {text}

            Strict Correction Rules:
            - Only correct Indian case names, personal names, or geographical names
            - Maintain accurate spelling and capitalization for names related to Indian law
            - Do not modify any other parts of the text
            - Ensure case titles follow Indian legal citation formats
            - If unsure of a correction, leave the text as is

            Output the corrected text only."""

            try:
                response = await bedrock.invoke_model(
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": correction_prompt}],
                        "max_tokens": 2000,
                        "temperature": 0.5,
                        "top_p": 0.9,
                    }),
                    modelId=self.correction_model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(await response.get("body").read())
                return response_body.get("content", [{"text": text}])[0].get("text", text)

            except Exception as e:
                logger.error(f"Error in text correction: {str(e)}")
                return text

    async def extract_case_info(self, text: str) -> Dict[str, str]:
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
            logger.warning("Failed to extract title using patterns, using fallback method")

            title = "Legal Matter"  # Default title
            for phrase in ["regarding", "about", "concerning"]:
                if phrase in text_lower:
                    idx = text_lower.find(phrase) + len(phrase)
                    next_period = text[idx:].find(".")
                    if next_period != -1:
                        title = text[idx:idx + next_period].strip()
                        break

        return {
            "title": title,
            "court_type": court_type,
            "folder": folder
        }

    async def select_template(self, case_info: Dict[str, str], templates: List[str]) -> str:
        """Selects appropriate template using Claude"""
        session = aioboto3.Session()

        logger.info(f"Available templates for selection: {templates}")
        logger.info(f"Case info for template selection: {case_info}")

        async with session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            config=Config(connect_timeout=300, read_timeout=300, retries={"max_attempts": 10})
        ) as bedrock:
            template_prompt = f"""Based on the following case information, select the most appropriate legal template from the available options.

            Case Information:
            Title: {case_info['title']}
            Court: {case_info['court_type']}

            Available Templates:
            {json.dumps(templates, indent=2)}

            Selection Rules:
            - Consider the nature of the case and appropriate legal remedy
            - Match case requirements with template purpose
            - Select only one template that best fits the case
            - Provide only the exact template name from the list

            Output the selected template name only."""

            try:
                response = await bedrock.invoke_model(
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "messages": [{"role": "user", "content": template_prompt}],
                        "max_tokens": 2000,
                        "temperature": 0.2,
                        "top_p": 0.9,
                    }),
                    modelId=self.template_selection_model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                
                response_body = json.loads(await response.get("body").read())
                selected_template = response_body.get("content", [{"text": templates[0]}])[0].get("text", templates[0])
                
                logger.info(f"Selected template: {selected_template}")

                # Verify template exists in list
                if selected_template in templates:
                    return selected_template
                return templates[0]  # Default to first template (if selection fails)

            except Exception as e:
                logger.error(f"Error in template selection: {str(e)}")
                return templates[0]

    async def get_templates(self, folder: str) -> List[str]:
        """Fetches available templates from the API"""
        async with aiohttp.ClientSession() as session:
            try:
                data = {
                    "bucket_name": "drafts-legal",
                    "folder": folder
                }
                
                async with session.post(
                    f"{self.base_url}/list-folders",
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("folders", [])
                    else:
                        logger.error(f"Failed to get templates: {response.status}")
                        return []
                        
            except Exception as e:
                logger.error(f"Error getting templates: {str(e)}")
                return []

    async def retrieve_template(self, template_name: str, folder: str) -> Dict:
        """Retrieves specific template content"""

        logger.info(f"Attempting to retrieve template: {template_name} from folder: {folder}")

        async with aiohttp.ClientSession() as session:
            try:
                data = {
                    "filename": template_name,
                    "draft_folder": folder
                }
                
                async with session.post(
                    f"{self.base_url}/retrieve-files",
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to retrieve template: {response.status}")
                        return {}
                        
            except Exception as e:
                logger.error(f"Error retrieving template: {str(e)}")
                return {}

    async def process_final_draft(self, template_name: str, complaint: str) -> str:
        """Processes the final draft with the complaint"""

        logger.info(f"Processing draft with template: {template_name}")
        logger.info(f"Complaint preview: {complaint[:200]}...")

        async with aiohttp.ClientSession() as session:
            try:
                data = {
                    "markdown_file_content": "**" + complaint + "\\n",
                    "json_file_content": f'"{template_name}.json file not found."',
                    "user_prompt": complaint
                }
                
                async with session.post(
                    f"{self.base_url}/process-content",
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        processed_content = result.get("processed_content", "")
                        if processed_content:
                            logger.info("Draft processed successfully")
                            return f"Yes worked: {processed_content[:50]}"
                        return "Failed to process content"
                    else:
                        return f"Error: {response.status}"
                        
            except Exception as e:
                logger.error(f"Error processing draft: {str(e)}")
                return f"Error: {str(e)}"

async def main():
    # Initialize the system
    system = LegalDraftSystem()
    
    try:
        # 1. Transcribe audio
        audio_path = r"C:\Users\karth\OneDrive\Desktop\BlueKyte Work\VTT\audios\drafting.wav"  # Replace with audio path
        transcript = await system.transcribe_audio(audio_path)
        logger.info(f"Transcription result: {transcript[:200]}...")  # First 200 chars for preview
        logger.info("Transcription completed")

        # 2. Correct Indian legal terms
        corrected_transcript = await system.correct_indian_legal_text(transcript)
        logger.info(f"Original text: {transcript[:200]}...")
        logger.info(f"Corrected text: {corrected_transcript[:200]}...")
        logger.info("Text correction completed")

        # 3. Extract case information
        case_info = await system.extract_case_info(corrected_transcript)
        logger.info(f"Extracted case info: {case_info}")

        # 4. Get available templates
        templates = await system.get_templates(case_info['folder'])
        if not templates:
            raise Exception("No templates found")
        logger.info(f"Retrieved {len(templates)} templates")

        # 5. Select appropriate template
        selected_template = await system.select_template(case_info, templates)
        logger.info(f"Selected template: {selected_template}")

        # 6. Retrieve template content
        template_content = await system.retrieve_template(selected_template, case_info['folder'])
        if not template_content:
            raise Exception("Failed to retrieve template")

        # 7. Process final draft
        result = await system.process_final_draft(selected_template, corrected_transcript)
        print(result)

    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
