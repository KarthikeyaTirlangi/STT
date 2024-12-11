--- STEP 1 ---

If case is to Supreem Court

curl 'https://aiengine.counsello.ai/draft/list-folders' \
 -H 'accept: application/json, text/plain, _/_' \
 -H 'accept-language: en-US,en;q=0.9' \
 -H 'content-type: application/json' \
 -H 'origin: https://suite.counsello.ai' \
 -H 'priority: u=1, i' \
 -H 'referer: https://suite.counsello.ai/' \
 -H 'sec-ch-ua: "Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"' \
 -H 'sec-ch-ua-mobile: ?0' \
 -H 'sec-ch-ua-platform: "Windows"' \
 -H 'sec-fetch-dest: empty' \
 -H 'sec-fetch-mode: cors' \
 -H 'sec-fetch-site: same-site' \
 -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0' \
 --data-raw '{"bucket_name":"drafts-legal","folder":"Special Leave Petition"}'

If case is to High Court

curl 'https://aiengine.counsello.ai/draft/list-folders' \
 -H 'accept: application/json, text/plain, _/_' \
 -H 'accept-language: en-US,en;q=0.9' \
 -H 'content-type: application/json' \
 -H 'origin: https://suite.counsello.ai' \
 -H 'priority: u=1, i' \
 -H 'referer: https://suite.counsello.ai/' \
 -H 'sec-ch-ua: "Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"' \
 -H 'sec-ch-ua-mobile: ?0' \
 -H 'sec-ch-ua-platform: "Windows"' \
 -H 'sec-fetch-dest: empty' \
 -H 'sec-fetch-mode: cors' \
 -H 'sec-fetch-site: same-site' \
 -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0' \
 --data-raw '{"bucket_name":"drafts-legal","folder":"Writ"}'

--- Response Sample

Supreem Court

{
"folders": [
"Application for exemption from filing officially translated documents with Supreme Court of India/",
"Format of AOR Certificate to be filed with SLP. Advocate on Record certificate for Special Leave Petition in Supreme Court/",
"Format of Counter Affidavit against Special Leave Petition (SLP) under Article 136 of the Constitution of India. Counter affidavit from Respondents/",
"Format of Rejoinder Affidavit against Counter Affidavit in Supreme Court of India/",
"Format of affidavit to be filed with SLP (Special Leave Petition) in Supreme Court/",
"SLP Application for Withdrawal of case from Supreme Court on settlement/",
"SYNOPSIS AND LIST OF DATES/",
"Special Leave Petition (Criminal) format under Article 136 of the Constitution to be filed in Supreme Court of India against Judgment of High Court/"
]
}

High Court.

{
"folders": [
"Affidavit for Writ of Certiorari Writ Petition under Article 226 and 227 filed in High Court/",
"Affidavit for Writ of Habeas Corpus under Article 226 of the Constitution to be filed in High Court/",
"Affidavit for Writ of Mandamus under Article 226 of the Constitution to file in High Court/",
"Affidavit for Writ of Prohibition under Article 226 of the Constitution to be filed in High Court as Public Interest Litigation/",
"Affidavit for Writ of Quo Warranto Article 226 of the Constitution to file in High Court/",
"PIL format for Supreme Court under Article 32 of the Constitution of India. Writ Petition Public Interest Litigation/",
"Writ Petition format to file Writ under Article 226 and 227 to challenge order passed by Central Administrative Tribunal against OA of Petitioner/",
"Writ Petition of Habeas Corpus to High Court under Article 226 of Constitution to release a person, Quash order and pay compensation/",
"Writ Petition of Mandamus to High Court under Article 226 of Constitution to quash termination order, Reinstate Petitioner and pay back wages/",
"Writ Petition seeking Writ of Certiorari in High Court under Article 226 of Constitution to quash order of Respondent/",
"Writ Petition seeking Writ of Prohibition from High Court under Article 226 of Constitution to prohibit Respondents from proceeding with disciplinary action/",
"Writ Petition seeking Writ of Quo Warranto in High Court under Article 226 of Constitution to cancel illegal appointment order and remove person illegally appointed/",
"Writ Petition to High Court under Article 226 of Constitution of India seeking appropriate Writ for Fundamental Right, Format Download/",
"rit Petition format to file Writ under Article 226 and 227 to challenge order passed by Central Administrative Tribunal against OA of Petitioner/"
]
}

--- SAMPLE TEMPLATE FOR A CASE ---

curl 'https://aiengine.counsello.ai/draft/retrieve-files' \
 -H 'accept: application/json, text/plain, _/_' \
 -H 'accept-language: en-US,en;q=0.9' \
 -H 'content-type: application/json' \
 -H 'origin: https://suite.counsello.ai' \
 -H 'priority: u=1, i' \
 -H 'referer: https://suite.counsello.ai/' \
 -H 'sec-ch-ua: "Chromium";v="130", "Microsoft Edge";v="130", "Not?A_Brand";v="99"' \
 -H 'sec-ch-ua-mobile: ?0' \
 -H 'sec-ch-ua-platform: "Windows"' \
 -H 'sec-fetch-dest: empty' \
 -H 'sec-fetch-mode: cors' \
 -H 'sec-fetch-site: same-site' \
 -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0' \
 --data-raw '{"filename":"Affidavit for Writ of Habeas Corpus under Article 226 of the Constitution to be filed in High Court","draft_folder":"Writ"}'

!!! Here "filename" and "draft_folder" will base changed. "filename" will change by which template we choose and "draft_folder" by which court we are filing case. (Supreem Court - [Sample Leave Petition], High Court - [Writ Petition])

--- WHEN USER SUBMITS THE RESPONSE (FINAL API CALL)

curl 'https://aiengine.counsello.ai/draft/process-content' \
 -H 'accept: application/json, text/plain, _/_' \
 -H 'accept-language: en-US,en;q=0.9' \
 -H 'content-type: application/json' \
 -H 'origin: https://suite.counsello.ai' \
 -H 'priority: u=1, i' \
 -H 'referer: https://suite.counsello.ai/' \
 -H 'sec-ch-ua: "Chromium";v="130", "Microsoft Edge";v="130", "Not?A*Brand";v="99"' \
 -H 'sec-ch-ua-mobile: ?0' \
 -H 'sec-ch-ua-platform: "Windows"' \
 -H 'sec-fetch-dest: empty' \
 -H 'sec-fetch-mode: cors' \
 -H 'sec-fetch-site: same-site' \
 -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0' \
 --data-raw $'{"markdown_file_content":"**IN THE HIGH COURT OF DELHI AT NEW DELHI**\\n\\n**CRIMINAL ORIGINAL JURISDICTION**\\n\\n\*\*WRIT PETITION (CRIMINAL) NO. OF 20\\\\*\\\\_**\\n\\n**IN THE MATTER OF:\*\*\\n\\n\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_                                     **        \\nPETITIONER**\\n\\n**VERSUS**\\n\\nGOVERNMENT OF NCT OF DELHI & ORS                        **RESPONDENTS**\\n\\n**AFFIDAVIT**\\n\\nI, \\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_, S/O\\n\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_, aged \\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_, Occupation\\n\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_ Resident of \\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\ndo hereby solemnly affirm and state as follows :-\\n\\n1\\\\. I am the father of Petitioner and filing this Writ Petition on his\\nbehalf and I am well conversant with the facts of the present writ\\npetition and hence, am competent to swear this affidavit.\\n\\n2\\\\. That the Petitioner is detained in Tihar Jail and he is unable to\\nmake the affidavit himself.\\n\\n3\\\\. That on \\\\_\\\\_\\\\\_day of\\\\_\\\\_\\\\_\\\\_, the Petitioner was arrested and\\ndetained for a period of 2 months in the Tihar Jail, New Delhi, wherein\\nthe Respondent No. 3 is the Superintendent, with an order passed by the\\nRespondent No.1 dated \\\\_\\\\_\\\\\_under the National Security Act, 1980.\\n\\n4\\\\. That, on the date of getting detained and arrested in the Tihar\\nJail. The Petitioner was not informed about the grounds of his detention\\nby Respondent No. 3.\\\\\\n\\\\\\n5. That after Ten days of getting arrested and detained, the Petitioner\\nwas informed of his ground of arrest and detention.\\n\\n6\\\\. The report of the ground of detention was furnished to the\\nPetitioner in English, which is not understood by the Petitioner.\\n\\n7\\\\. I have done whatsoever inquiry which was in my power to do, to\\ncollect all data which was available and which was relevant for this\\nHon\\\\\'ble Court to entertain the present petition. I confirm that I have\\nnot concealed in the present petition any data/material/information\\nwhich may have enabled this Hon\\\\\'ble Court to form an opinion whether to\\nentertain his petition or not and/or whether to grant any relief or not.\\n\\n8\\\\. That the accompanying Petition has been drafted under my\\ninstructions and the contents thereof except the legal averments\\ncontained therein are true and correct based on my knowledge and belief.\\nThe legal averments contained therein are true and correct on the basis\\nof legal advice received by me and believed by me to be true and\\ncorrect. The contents of the Petition are not being repeated here for\\nthe sake of brevity and to avoid prolixity. The contents of the same may\\nbe read as a part of this Affidavit.\\n\\n9\\\\. That no part of this Affidavit is false and no material facts have\\nbeen concealed therefrom.\\n\\n10\\\\. That the Petitioners have no other efficacious remedy except to\\napproach this Hon\\\\\'ble Court by way of this Petition under Article 226\\nof the Constitution of India.\\n\\n11\\\\. That the Petitioners have not filed any other petition or preceding\\nin any court or tribunal throughout the territory of India regarding the\\nmatter.\\n\\n12\\\\. That I have read and understood the content of Writ Petition. I\\nhave read and understood the contents of the accompanying synopsis &\\nList of Dates at Pages \\\\_\\\\_\\\\_\\\\_ to \\\\_\\\\_\\\\_\\\\_\\\\_\\\\_, Writ Petition at Pages\\n\\\\_\\\\_\\\\_ to \\\\_\\\\_\\\\_\\\\_, Para \\\\_\\\\_\\\\_\\\\_ to \\\\_\\\\_\\\\_\\\\_, Grounds \\\\_\\\\_\\\\_\\\\_ to\\n\\\\_\\\\_\\\\_\\\\_ and all accompanying Applications. I state that the facts\\ntherein are true and correct to the best of my knowledge and belief. I\\nfurther state that the Annexures annexed to the Writ Petition are true\\ncopies of their respective originals.\\n\\n**DEPONENT**\\n\\n**VERIFICATION:**\\\\\\nVerified at New Delhi on this \\\\_\\\\_\\\\_ day of \\\\_\\\\_\\\\_\\\\_\\\\_\\\\_\\\\_ 20\\\\_\\\\_ that\\nthe contents of my aforesaid affidavit are true and correct to my\\nknowledge and belief. No part of it is false nor anything material has\\nbeen concealed therefrom.\\n\\n**DEPONENT**\\\\\\n\\\\\\n\\\\\\n \\n\\n**Writ of Habeas Corpus**\\n\\nA writ of habeas corpus is issued to an authority or person to produce\\nin court a person who is either missing or kept in illegal custody.\\nWhere the detention is found to be without authority of law, the Court\\nmay order compensation to the person illegally detained.\\n","json_file_content":"\\"Affidavit for Writ of Habeas Corpus under Article 226 of the Constitution to be filed in High Court.json file not found.\\"","user_prompt":"I am driving a car, a guy came beside me and dashed me, i fell into a porthole, petitioner name is karthik and accused name is tarun. "}'

--- ACTION ---

We take audio input from the user, from that input we decide/extract the title of the draft, and court ther want to file.
There are two api's, one for supreem court and another for high court. Say user want to file at high court, then that particular api will be called.
When that api called there will be list of templates that user can select for that case, what template will be choosen will taken care by LLM.

After extracting title and court name, then we will extract the actual complaint of the user. We give the actual whole complaint and will call the api.

--- MANUAL PIPELINE ---

1. User gives a title (in my case, we extract this from user audio)
   User selects the court (in my case, we extract this from user audio)
   User selects the template (in my case, LLM will decide which template to choose based on the extracted title)

2. User gives the whole complaint details

3. After giving whole input, user submits it, then we will direct him to the ui of the draft.

--- FURTHER STEPS ---

-> Should try to save the content in a .md file.
-> Should combine both the code files. [Drafting and Version-1]

case_info = await system.extract_case_info(corrected_transcript)
logger.info(f"Extracted case info: {case_info}")


--- FINAL PLAN ---

# Imports and Logging Configuration
import os
import json
import logging
import asyncio
import aiohttp  # type: ignore
import aioboto3  # type: ignore
import librosa
import nest_asyncio
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv  # type: ignore
from botocore.config import Config  # type: ignore
from transformers import WhisperProcessor, WhisperForConditionalGeneration  # type: ignore
import sseclient  # type: ignore
import requests

# 1. Load environment variables and initialize logging
# 2. Initialize Whisper model and processor
# 3. Define helper functions for handling transcription and intent detection

# Function Definitions

## Configuration and Initialization
load_dotenv()

def transcribe_audio(audio_path: str) -> str
def detect_intent(text: str) -> Dict[str, Any]
async def correct_indian_legal_text(text: str) -> str

# Case research, copilot, and summary functions (from Code 1):
def invoke_case_research(case_name: str, additional_terms: Optional[List[str]] = None) -> Optional[Dict[str, str]]
def invoke_copilot(query: str, user_id: str = "1877", session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]
def invoke_summary_service(user_id: str, doc_id: str) -> Optional[str]

# Drafting functionality class
class LegalDraftSystem:
    def __init__(self)
    
    async def extract_case_info(self, text: str) -> Dict[str, str]
    async def get_templates(self, folder: str) -> List[str]
    async def retrieve_template(self, template_name: str, folder: str) -> Dict
    async def process_final_draft(self, template_name: str, complaint: str) -> str

# Intent handling and processing function
async def process_query(transcribed_text: str, user_id: str = "1877", session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]

# Main entry point
async def main(audio_file_path: str, user_id: str = "1877", session_id: str = "86068bd2-4311-11ee-a6b3-014c891e3538") -> Optional[str]

if __name__ == "__main__":
    nest_asyncio.apply()
    audio_file_path = "path_to_audio.wav"
    asyncio.run(main(audio_file_path))

    def __init__(self):
        # self.base_url = "https://aiengine.counsello.ai/draft"
        self.url = drafting_base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        logger.info("LegalDraftSystem initialized")
        print("Legal Draft System Ready")