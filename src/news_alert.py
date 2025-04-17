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

from src.util import gcloud_connect_credentials


# load .env if not in production
if not bool(os.getenv("PRODUCTION_MODE")):
    from dotenv import load_dotenv
    load_dotenv()


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
    model = "gemini-2.5-pro-preview-03-25"  # "gemini-2.5-pro-preview-03-25"  gemini-2.0-flash
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


def gmail_authenticate():
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(
                client_config={
                    "installed":{
                        "client_id":os.getenv("GMAIL_CLIENT_ID"),
                        "project_id":os.getenv("GCLOUD_PROJECT_ID"),
                        "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                        "token_uri":"https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                        "client_secret":os.getenv("GMAIL_CLIENT_SECRET"),
                        "redirect_uris":["http://localhost"]
                    }},
                scopes=['https://mail.google.com/']
            )
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)


def build_message(from_email, destination, subject, body):
    message = MIMEText(body)
    message['to'] = destination
    message['from'] = from_email
    message['subject'] = subject
    return {'raw': urlsafe_b64encode(message.as_bytes()).decode()}
