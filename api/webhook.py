from http.server import BaseHTTPRequestHandler
import json
import os
import logging
import re
import requests
import google.generativeai as genai

# --- Configuration ---
COMBINED_TEXT_FILENAME = "all_pdf_text_combined.txt"
MAX_CONTEXT_PARAGRAPHS = 3
AI_MODEL_NAME = 'gemini-2.0-flash'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Load & cache PDF text once ---
CACHED_FULL_TEXT = None

def load_text_from_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            logger.info(f"Loaded text from {filename} (length: {len(text)})")
            return text
    except Exception as e:
        logger.error(f"Error loading text file: {e}")
        return None

def get_cached_full_text():
    global CACHED_FULL_TEXT
    if CACHED_FULL_TEXT is None:
        CACHED_FULL_TEXT = load_text_from_file(COMBINED_TEXT_FILENAME)
    return CACHED_FULL_TEXT

# --- Text processing ---
def find_relevant_paragraphs(full_text, question, max_paragraphs):
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    keywords = set(q.lower() for q in question.split() if len(q) > 2)
    relevant = []
    for para in paragraphs:
        if any(k in para.lower() for k in keywords):
            relevant.append(para)
            if len(relevant) >= max_paragraphs:
                break
    if not relevant:
        relevant = paragraphs[:max_paragraphs]
    return relevant

def find_youtube_links(text):
    """Find all YouTube URLs in the given text."""
    youtube_regex = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)(?:\S+)?"
    links = re.findall(youtube_regex, text)
    full_links = [f"https://www.youtube.com/watch?v={link_id}" for link_id in links]
    unique_links = list(set(full_links))
    if unique_links:
        logger.info(f"Found YouTube links in context: {unique_links}")
    return unique_links

def construct_prompt(context_paragraphs, question, youtube_links):
    context = "\n\n".join(context_paragraphs)
    links_text = ""
    if youtube_links:
        links_text = "\n\nRelevant YouTube links found in the documents:\n" + "\n".join(youtube_links)
    prompt = f"""You are Jenny, Sal's Personal Assistant. You have expert knowledge of the following documents. Please answer the user's question *only* using the information from these excerpts.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

{links_text}

User's question: {question}

Please respond clearly, helpfully, and in a warm conversational tone as Jenny. Do NOT speculate or provide information outside these documents. If a relevant YouTube link supports your answer, include only the most relevant link at the end.

Jenny's answer:"""
    return prompt

def get_ai_response(api_key, prompt):
    if not api_key:
        return "Error: Google API key is missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(AI_MODEL_NAME)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        logger.error(f"Google AI error: {e}")
        return "Sorry, an error occurred contacting the AI."

def send_message(chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = requests.post(url, json=payload)
        return resp.json()
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return None

# --- Webhook Handler ---
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("Jenny is online! This is the webhook endpoint for the Telegram bot.".encode())

    def do_POST(self):
        try:
            length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(length)
            update = json.loads(post_data.decode('utf-8'))

            message = update.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text')

            if chat_id and text:
                if text.startswith('/start'):
                    reply = "Hey! I'm Jenny, Sal's Personal Assistant. How can I help you today?"
                else:
                    full_text = get_cached_full_text()
                    if not full_text:
                        reply = "Sorry, I couldn't load the documents right now."
                    else:
                        relevant = find_relevant_paragraphs(full_text, text, MAX_CONTEXT_PARAGRAPHS)
                        youtube_links = find_youtube_links("\n\n".join(relevant))
                        prompt = construct_prompt(relevant, text, youtube_links)
                        reply = get_ai_response(GOOGLE_API_KEY, prompt)

                send_message(chat_id, reply)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())

        except Exception as e:
            logger.error(f"Webhook error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
