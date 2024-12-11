import os
import requests #type: ignore
import logging
import json
from dotenv import load_dotenv #type: ignore

# Load environment variables
load_dotenv()

class CaseResearchService:
    def __init__(self):
        # Configuration for the Case Research API
        self.case_research_url = os.getenv("CASE_RESEARCH_URL")
        self.case_research_api_key = os.getenv("CASE_RESEARCH_API_KEY")

        

        # Setting up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def invoke_case_research(self, query, entities=None):
        try:
            payload = {
                "semantic_search_query": [query],
                "type": entities.get("case_type", []) if entities else [],
                "case_type": entities.get("key_terms", []) if entities else [],
            }
            headers = {
                "authorization": f"{self.case_research_api_key}",
                "content-type": "application/json",
            }
            response = requests.post(
                self.case_research_url, json=payload, headers=headers, timeout=10
            )
            response.raise_for_status()
            self.logger.info("Case research request successful.")
            return response.json()
        except Exception as e:
            self.logger.error(f"Error in case research API: {e}")
            raise

# Running the Case Research Service
if __name__ == "__main__":
    case_research_service = CaseResearchService()
    query = "chitra sharma vs union of india"
    try:
        result = case_research_service.invoke_case_research(query)
        print(result)
    except Exception as e:
        print(f"Error during execution: {e}")
