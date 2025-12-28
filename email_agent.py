import os
import smtplib
import imaplib
import email
import ssl
import json
import re
import requests
import time
from email.message import EmailMessage

# --- Configuration & Secrets (Loaded from GitHub Environment Variables) ---
from google import genai
from google.genai import types

# Load the API key from the environment variable (set via GitHub Secrets)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
IMAP_SERVER = "imap.gmail.com"

# --- LangSmith Configuration for Tracing ---
langsmith_key = os.environ.get("LANGCHAIN_API_KEY")

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "Email_automation_schedule"
    print("STATUS: LangSmith tracing configured.")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("STATUS: LANGCHAIN_API_KEY not found. LangSmith tracing is disabled.")

# Initialize Gemini Client
gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("STATUS: Gemini client initialized successfully using API key from environment.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini client. Error: {e}")
else:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not found. Gemini client not initialized.")


# --- Knowledge Base & Persona Configuration ---
DATA_SCIENCE_KNOWLEDGE = """
# Data Science Project & Service Knowledge Base

## Core Services
- Time Series (ARIMA, LSTM, Prophet)
- ML pipelines
- NLP
- Computer Vision
- MLOps
- Dashboards & Data Engineering

## Model Guidance
- ARIMA = short / interpretable
- LSTM = long-term / complex patterns

## Meeting Availability
Mon / Wed / Fri — 2PM to 5PM IST
"""

AUTOMATION_CONDITION = (
    "Does the incoming email contain a technical question or project inquiry "
    "related to Data Science, Machine Learning, Deep Learning, Data Engineering, "
    "Statistics, or AI services?"
)

AGENTIC_SYSTEM_INSTRUCTIONS = (
    """You are EMAYAN R M, a Senior Data Scientist and AI/ML Engineering Specialist.

RESPONSIBILITIES:
- ONLY reply as EMAYAN R M
- Provide helpful, professional replies
- Focus on clarity and value

FORMAT RULES:
- Plain text only
- Always sign exactly:
Best regards,
EMAYAN R M
"""
)

RESPONSE_SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "is_technical": {"type": "boolean"},
        "simple_reply_draft": {"type": "string"},
        "non_technical_reply_draft": {"type": "string"},
        "request_meeting": {"type": "boolean"},
        "meeting_suggestion_draft": {"type": "string"},
    },
    "required": [
        "is_technical",
        "simple_reply_draft",
        "non_technical_reply_draft",
        "request_meeting",
        "meeting_suggestion_draft",
    ],
}

response_schema = RESPONSE_SCHEMA_JSON


# --- EMAIL HELPERS ---

def _send_smtp_email(to_email, subject, content):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials missing.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg.set_content(content)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print("STATUS: Email sent.")
        return True
    except Exception as e:
        print(f"SMTP ERROR: {e}")
        return False


def _fetch_latest_unread_email():
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials missing.")
        return None, None, None

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")

        status, data = mail.search(None, 'UNSEEN')
        
        ids = data[0].split()
        if not ids:
            print("STATUS: No unread emails.")
            return None, None, None

        latest_id = ids[-1]
        mail.store(latest_id, '+FLAGS', '\\Seen')

        status, msg_data = mail.fetch(latest_id, "(RFC822)")
        email_message = email.message_from_bytes(msg_data[0][1])

        from_header = email_message.get("From", "")
        subject = email_message.get("Subject", "No Subject")

        match = re.search(r"<([^>]+)>", from_header)
        from_email = match.group(1) if match else from_header

        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()

        return from_email, subject, body

    except Exception as e:
        print(f"IMAP ERROR: {e}")
        return None, None, None


def _run_ai_agent(email_data):
    global gemini_client
    if not gemini_client:
        return None

    user_prompt = (
        f"{DATA_SCIENCE_KNOWLEDGE}\n\n"
        f"Ensure replies always end with:\nBest regards,\\nEMAYAN R M\n\n"
        f"FROM: {email_data['from_email']}\n"
        f"SUBJECT: {email_data['subject']}\n"
        f"BODY:\n{email_data['body']}"
    )

    config = types.GenerateContentConfig(
        system_instruction=AGENTIC_SYSTEM_INSTRUCTIONS,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.3,
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=config,
        )

        raw = response.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            return json.loads(match.group(0))

    except Exception as e:
        print(f"AI ERROR: {e}")

    return None


def main_agent_workflow():
    print("=== START WORKFLOW ===")

    from_email, subject, body = _fetch_latest_unread_email()
    if not from_email:
        return

    ai_output = _run_ai_agent(
        {"from_email": from_email, "subject": subject, "body": body}
    )

    SAFE_DEFAULT_REPLY = (
        "Thank you for reaching out. I will review your message and respond shortly.\n\n"
        "Best regards,\nEMAYAN R M"
    )

    if not ai_output:
        reply = SAFE_DEFAULT_REPLY
    else:
        is_technical = ai_output.get("is_technical", False)
        request_meeting = ai_output.get("request_meeting", False)

        if is_technical and request_meeting:
            reply = ai_output.get("meeting_suggestion_draft", SAFE_DEFAULT_REPLY)
        elif is_technical:
            reply = ai_output.get("simple_reply_draft", SAFE_DEFAULT_REPLY)
        else:
            reply = ai_output.get("non_technical_reply_draft", SAFE_DEFAULT_REPLY)

    reply = re.sub(r"<[^>]+>", "", reply).strip()

    if not reply.lower().startswith(("hello", "hi", "dear", "thank you")):
        reply = f"Hello,\n\n{reply}"

    _send_smtp_email(from_email, f"Re: {subject}", reply)

    print("=== DONE ===")


if __name__ == "__main__":
    main_agent_workflow()


if __name__ == "__main__":
    main_agent_workflow()

