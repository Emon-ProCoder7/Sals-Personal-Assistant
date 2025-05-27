from http.server import BaseHTTPRequestHandler
import json
import os
import logging
import re
import requests
import google.generativeai as genai
from collections import defaultdict  # Add this for session management
import time  # Add this for session timeout

# --- Configuration ---
COMBINED_TEXT_FILENAME = "all_pdf_text_combined.txt"
MAX_CONTEXT_PARAGRAPHS = 5  # Match Zara's setting
AI_MODEL_NAME = 'gemini-2.0-flash'

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Session Management ---
# --- Session Management for Personalization ---
USER_SESSIONS = defaultdict(lambda: {"name": None, "last_interaction": 0})
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

def update_user_session(chat_id, first_name, last_name):
    """Store user's name and update interaction time."""
    name = first_name or last_name or "friend"  # Fallback to "friend" if no name
    USER_SESSIONS[chat_id]["name"] = name
    USER_SESSIONS[chat_id]["last_interaction"] = time.time()

def get_user_name(chat_id):
    """Retrieve user's name, checking session timeout."""
    session = USER_SESSIONS.get(chat_id, {})
    if not session or (time.time() - session["last_interaction"]) > SESSION_TIMEOUT:
        del USER_SESSIONS[chat_id]  # Clear expired session
        return None
    return session["name"]

# --- Cache the PDF text once ---
CACHED_FULL_TEXT = None

# --- AI Logic ---
def load_text_from_file(filename):
    """Reads the entire content of a text file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            logger.info(f"Successfully loaded text from {filename}")
            return f.read()
    except FileNotFoundError:
        logger.error(f"Error: Text file {filename} not found.")
        return None
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return None

def find_relevant_paragraphs(full_text, question, max_paragraphs):
    """Finds paragraphs relevant to the question."""
    logger.info(f"Searching paragraphs for: '{question}'")
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    keywords = set(q.lower() for q in question.split() if len(q) > 2)
    relevant_paragraphs = []
    for para in paragraphs:
        if any(keyword in para.lower() for keyword in keywords):
            relevant_paragraphs.append(para)
            if len(relevant_paragraphs) >= max_paragraphs:
                break
    if not relevant_paragraphs:
        logger.warning("Found no specific paragraphs. Using start of text.")
        return paragraphs[:max_paragraphs]
    else:
        logger.info(f"Found {len(relevant_paragraphs)} relevant paragraph(s).")
        return relevant_paragraphs

def find_youtube_links(text):
    """Finds all YouTube URLs in a given text."""
    youtube_regex = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)(?:\S+)?"
    links = re.findall(youtube_regex, text)
    full_links = [f"https://www.youtube.com/watch?v={link_id}" for link_id in links]
    unique_links = list(set(full_links))
    if unique_links:
        logger.info(f"Found potential YouTube links in context: {unique_links}")
    return unique_links

def construct_prompt(context_paragraphs, question, youtube_links, user_name):
    """Creates the prompt for the AI model, incorporating Jenny's persona and link selection."""
    context = "\n\n".join(context_paragraphs) if context_paragraphs else "No relevant context found."
    links_string = ""
    if youtube_links:
        links_string = "\n\nPotential relevant YouTube links found in the source text:\n" + "\n".join(youtube_links)

    prompt = f"""You are Jenny, Sal's Personal Assistant, an expert on the provided documents with a world-class engineering background. Answer the user's question based *only* on the following text excerpts from Sal's documents, addressing the user by their name ({user_name}) in a natural, conversational way:

--- TEXT EXCERPTS ---
{context}
--- END TEXT EXCERPTS ---

{links_string}

User's question: {question}

Instructions for Jenny:
1. Answer the question naturally and conversationally, as Sal's helpful assistant. Do NOT mention "based on the text" or "according to the document".
2. If one of the 'Potential relevant YouTube links' listed above directly supports or illustrates your answer, include ONLY that single most relevant link at the end of your response. Do not include links otherwise.
3. Use a warm, friendly tone, addressing the user by their name ({user_name}) throughout. For the first message, start with "Hey {user_name}!" and for follow-ups, weave the name naturally into the response.
4. If the question is outside the scope of the documents, politely explain that you can only assist with the provided knowledge base.

Jenny's Answer:"""
    logger.info("Prompt constructed for AI with Jenny's persona and link selection instructions.")
    return prompt

def get_ai_response(api_key, prompt):
    """Connects to Google AI and gets the response."""
    if not api_key:
        logger.error("GOOGLE_API_KEY not found.")
        return f"Oops, something went wrong with my configuration. Please try again later!"
    try:
        logger.info(f"Configuring Google AI with model {AI_MODEL_NAME}...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(AI_MODEL_NAME)
        logger.info("Sending prompt to Google AI...")
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        logger.info("Received response from AI.")
        try:
            return response.text
        except ValueError:
            logger.warning(f"AI response was blocked or empty. Block reason: {response.prompt_feedback.block_reason}")
            return f"Sorry, I couldn't generate a response for that request based on the provided information. It might be outside the scope of my documents."
    except Exception as e:
        logger.error(f"Error contacting Google AI: {e}")
        return f"Sorry, I ran into an issue while processing your request: {str(e)}. Could you try again?"

def send_message(chat_id, text):
    """Send a message via Telegram API."""
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

        # Get user info
        first_name = message.get('from', {}).get('first_name', '')
        last_name = message.get('from', {}).get('last_name', '')

        # Update session with user name
        update_user_session(chat_id, first_name, last_name)
        user_name = get_user_name(chat_id)

        # Handle /start command
        if text.startswith('/start'):
            reply = f"Hey {user_name}, I'm Jenny, Sal's Personal Assistant! I'm here to help with any questions about our company documents or YouTube transcripts. What's on your mind?"
        else:
            full_text = get_cached_full_text()
            if not full_text:
                reply = f"Sorry, {user_name}, I couldn't load the document text right now."
            else:
                relevant = find_relevant_paragraphs(full_text, text, MAX_CONTEXT_PARAGRAPHS)
                youtube_links = find_youtube_links("\n\n".join(relevant))
                prompt = construct_prompt(relevant, text, youtube_links, user_name)
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
