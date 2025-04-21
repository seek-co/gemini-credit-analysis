import os
import re
import pathlib
import json
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode
from google import genai
from google.genai import types
from google.cloud import storage

from src.util import gcloud_connect_credentials, get_company_list


# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()


def get_company_keywords():
    """
    Get company keywords from storage or use default mapping.
    Returns a dictionary with company names as keys and list of keywords as values.
    """
    try:
        # Try to load keywords from storage
        storage_client = storage.Client(credentials=gcloud_connect_credentials())
        bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))
        blob = bucket.blob('config/company_keywords.json')
        
        if blob.exists():
            contents = blob.download_as_bytes()
            company_keywords = json.loads(contents.decode("utf-8"))
            storage_client.close()
            return company_keywords
        
        # Fall back to default keywords if file doesn't exist
        storage_client.close()
    except Exception as e:
        print(f"Error loading company keywords: {e}")
    
    # Default keywords mapping
    default_keywords = {
        "Sasol Limited": ["sasol", "sasol limited", "south africa", "south african energy", "south african chemicals"],
        "Arabian Centres": ["arabian centres", "cenomi", "saudi arabia", "saudi retail", "saudi mall"],
        "Tullow Oil": ["tullow", "tullow oil", "ghana", "kenya", "oil exploration", "african oil"]
    }
    return default_keywords


def detect_news_company(news_title, news_body):
    """
    Detect which company the news is related to based on the content.
    Returns the company name and a confidence score.
    """
    # Get dynamic company list
    company_list = get_company_list()
    company_keywords = get_company_keywords()
    
    # Filter the keywords to only include companies in the current company list
    # and add any new companies with empty keyword list
    filtered_keywords = {}
    for company in company_list:
        if company in company_keywords:
            filtered_keywords[company] = company_keywords[company]
        else:
            # New company with default empty keyword list
            # At minimum, use the company name itself as a keyword
            filtered_keywords[company] = [company.lower()]
    
    # Combine title and body for analysis
    combined_text = (news_title + " " + news_body).lower()
    
    # Check for company mentions and calculate a simple score
    results = {}
    for company, keywords in filtered_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in combined_text:
                # Add more weight if found in the title
                if keyword.lower() in news_title.lower():
                    score += 3
                else:
                    score += 1
        results[company] = score
    
    # If no companies in our database, return empty results
    if not results:
        return "", 0
        
    # Get the company with highest score
    detected_company = max(results.items(), key=lambda x: x[1])
    
    # If no clear match (score 0), try advanced detection using Gemini
    if detected_company[1] == 0:
        try:
            return detect_company_with_gemini(news_title, news_body, company_list)
        except Exception as e:
            # Fall back to best guess if Gemini fails
            return max(results.items(), key=lambda x: x[1])[0], 0
    
    return detected_company[0], detected_company[1]


def detect_company_with_gemini(news_title, news_body, company_list=None):
    """Use Gemini to detect which company the news is related to"""
    client_gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash-preview-04-17"
    
    # If company_list is not provided, fetch it
    if company_list is None:
        company_list = get_company_list()
    
    # Create company options string for prompt
    company_options = "\n".join([f"- {company}" for company in company_list])
    
    prompt = f"""
    Determine which company the following news is most related to. The options are:
    {company_options}
    
    News title: {news_title}
    News body: {news_body}
    
    Respond with just the company name exactly as one of the options above.
    """
    
    response = client_gem.models.generate_text(
        model=model,
        prompt=prompt,
        config=types.GenerateTextConfig(temperature=0)
    )
    
    result = response.text.strip()
    
    # Check for exact company name match
    detected_company = None
    for company in company_list:
        if company.lower() in result.lower():
            detected_company = company
            break
    
    # If still no match, use the first company as default if available
    if not detected_company and company_list:
        detected_company = company_list[0]
    elif not detected_company:
        detected_company = ""
    
    return detected_company, 5  # High confidence score for AI detection


def evaluate_credit_impact_dummy():
    with open('./data/news_alert_sample.json', "r") as f:
        file_ctn = f.read()
        response_sample = json.loads(file_ctn)
    return response_sample


def evaluate_credit_impact(company_name, news_title, news_body, sasol_true_rpt=False):
    # load files classifying from gcs blob
    storage_client = storage.Client(credentials=gcloud_connect_credentials())
    bucket = storage_client.bucket(os.getenv('GCLOUD_BUCKET_NAME'))

    if sasol_true_rpt:
        blob = bucket.blob(f'generated/{company_name}/credit_report.pdf')
    else:
        blob = bucket.blob(f'generated/{company_name}/credit_report.md')

    client_gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    # model name
    model = "gemini-2.5-flash-preview-04-17"
    # system prompt
    prompts_path = pathlib.Path().parent.joinpath("./prompts").resolve()
    system_prompt_file_name = "alert_system_prompt.txt"
    with open(prompts_path.joinpath(system_prompt_file_name), "r") as f:
        system_prompt = f.read()
    system_prompt = system_prompt.replace("_COMPANY_NAME_", company_name)

    # files involved
    file_ctn_parts = [
        types.Part.from_bytes(
            data=blob.download_as_bytes(),
            mime_type="text/plain" if not sasol_true_rpt else "application/pdf"
        )
    ]
    user_prompt_parts = [
        types.Part.from_text(text=(
            f"The credit report is provided, and the incoming news is as follows: \n"
            f"Incoming news title: {news_title} \n"
            f"Incoming news body: {news_body}"
        ))
    ]
    contents = [types.Content(role="user", parts=file_ctn_parts + user_prompt_parts)]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["credit_impact", "news_title", "news_summary", "basis", "reason"],
            properties={
                "credit_impact": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description=(
                        "Exactly one of 'bullish', 'bearish', 'neutral' according to the trading opportunity you "
                        "identified."
                    ),
                    enum=["bullish", "bearish", "neutral"],
                ),
                "news_title": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Title of the incoming news you receive",
                ),
                "news_summary": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Brief summary of the incoming news you receive in 2 sentences only",
                ),
                "basis": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description="Evaluation based on credit report generated on <date of credit report generation>",
                ),
                "reason": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    description=(
                        "Reasoning or thinking process to justify your 'bullish' or 'bearish' or 'neutral' "
                        "trading alert decision"
                    ),
                ),
            },
        ),
        system_instruction=[types.Part.from_text(text=system_prompt)],
    )
    response = client_gem.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config
    )
    response_json = json.loads(re.sub(".+\n```.+\n|.+\n```\n|```", "", response.text))
    return response_json

def build_message(from_email, destination, subject, body):
    message = MIMEText(body)
    message['to'] = destination
    message['from'] = from_email
    message['subject'] = subject
    return {'raw': urlsafe_b64encode(message.as_bytes()).decode()}


def analyze_raw_news(raw_text):
    """
    Uses Gemini to analyze raw news text and extract the title, body, and company.
    
    Args:
        raw_text (str): The raw news text
        
    Returns:
        dict: A dictionary containing the news title, body, company, and confidence
    """
    client_gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemini-2.5-flash-preview-04-17"
    
    # Create schema for structured response
    response_schema = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["title", "body", "company", "confidence"],
        properties={
            "title": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="A concise news title that captures the main point"
            ),
            "body": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="The extracted news body (main content)"
            ),
            "company": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="The company name this news is most related to",
                enum=["Sasol Limited", "Arabian Centres", "Tullow Oil"]
            ),
            "confidence": genai.types.Schema(
                type=genai.types.Type.INTEGER,
                description="Confidence score for company identification (1-10)",
                minimum=1,
                maximum=10
            )
        }
    )
    
    # System instruction
    system_instruction = """
    Analyze the provided news text and extract key information including title, body content, 
    and which company it relates to from the given options. Respond with structured data only.
    """
    
    # User prompt
    user_prompt = f"""
    Raw news text:
    {raw_text}
    
    Extract the title, body, related company, and your confidence level in the company identification.
    Company options:
    - Sasol Limited (South African energy and chemicals company)
    - Arabian Centres/Cenomi (Saudi Arabian retail and mall company)
    - Tullow Oil (Oil exploration company focused on Africa)
    """
    
    # Configure the request
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        system_instruction=[types.Part.from_text(text=system_instruction)],
        temperature=0
    )
    
    # Make the request
    response = client_gem.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
        config=generate_content_config
    )
    
    try:
        # Parse the response directly as JSON
        result = json.loads(response.text)
        
        # Normalize company name if needed
        company_mapping = {
            "sasol": "Sasol Limited",
            "arabian": "Arabian Centres",
            "cenomi": "Arabian Centres",
            "tullow": "Tullow Oil"
        }
        
        for key, value in company_mapping.items():
            if key.lower() in result["company"].lower():
                result["company"] = value
                break
                
        return result
    except Exception as e:
        # Fallback to manual extraction if JSON parsing fails
        return {
            "title": raw_text.split("\n")[0][:100],  # First line as title, max 100 chars
            "body": raw_text,
            "company": "Sasol Limited",  # Default company
            "confidence": 1
        }
