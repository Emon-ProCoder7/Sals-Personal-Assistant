from http.server import BaseHTTPRequestHandler
import json
import os
import logging
import re
import requests
import google.generativeai as genai
from collections import defaultdict
import time

# --- Configuration ---
COMBINED_TEXT_FILENAME = "all_pdf_text_combined.txt"
MAX_CONTEXT_PARAGRAPHS = 3
AI_MODEL_NAME = 'gemini-2.0-flash'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Session Management ---
# Store user info (name) and last interaction time per chat_id
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

# --- Text & YouTube processing ---
def find_relevant_paragraphs(full_text, question, max_paragraphs):
    """Rank paragraphs by keyword overlap for better relevance."""
    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Extract keywords (words > 2 chars, lowercase)
    keywords = set(word.lower() for word in question.split() if len(word) > 2)
    if not keywords:
        return paragraphs[:max_paragraphs]  # Return first few if no keywords

    # Score paragraphs based on keyword matches
    scored_paragraphs = []
    for para in paragraphs:
        score = sum(1 for keyword in keywords if keyword in para.lower())
        if score > 0:  # Only include paragraphs with at least one match
            scored_paragraphs.append((score, para))

    # Sort by score (descending) and select top paragraphs
    scored_paragraphs.sort(reverse=True)
    relevant = [para for _, para in scored_paragraphs[:max_paragraphs]]
    
    # If no matches, return first few paragraphs
    return relevant if relevant else paragraphs[:max_paragraphs]

def find_youtube_links(text):
    """Extract unique YouTube links from text."""
    youtube_regex = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)(?:\S+)?"
    links = re.findall(youtube_regex, text)
    full_links = [f"https://www.youtube.com/watch?v={link_id}" for link_id in links]
    unique_links = list(set(full_links))
    if unique_links:
        logger.info(f"Found YouTube links: {unique_links}")
    return unique_links

def construct_prompt(context_paragraphs, question, youtube_links, user_name):
    """Build a prompt for the AI with a warm, personalized tone."""
    context = "\n\n".join(context_paragraphs) if context_paragraphs else "No relevant context found."
    links_text = ""
    if youtube_links:
        links_text = "\n\nRelevant YouTube links:\n" + "\n".join(youtube_links)

    prompt = f"""You are Jenny, Sal's Personal Assistant, a friendly and highly intelligent assistant with a world-class engineering background. Your role is to provide clear, accurate, and helpful answers based *only* on the provided company documents and YouTube transcripts. Use a warm, conversational tone, addressing the user by their name ({user_name}) throughout the conversation. Avoid generic greetings like "Hi there" and maintain a natural, human-like flow. If the question is outside the scope of the documents, politely explain that you can only assist with the provided knowledge base. If a YouTube link is highly relevant, mention *only* the most relevant one in your response.

--- DOCUMENT EXCERPTS ---
{context}
--- END OF EXCERPTS ---

{links_text}

User's question: {question}

Jenny's answer (start with a friendly greeting like "Hey {user_name}" for the first message, then use the name naturally in follow-ups):"""
    return prompt

def get_ai_response(api_key, prompt):
    """Get response from Gemini AI model."""
    if not api_key:
        return "Oops, something went wrong with my configuration. Please try again later!"
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
        return "Sorry, I ran into an issue while processing your request. Could you try again?"

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
            user_name = get_user_name(chat_id)

            # Update session with user name
            if not user_name:
                update_user_session(chat_id, first_name, last_name)
                user_name = get_user_name(chat_id)

            # Handle /start command
            if text.startswith('/start'):
                reply = f"Hey {user_name}, I'm Jenny, Sal's Personal Assistant! I'm here to help with any questions about our company documents or YouTube transcripts. What's on your mind?"
            else:
                full_text = get_cached_full_text()
                if not full_text:
                    reply = f"Sorry {user_name}, I couldn't load the documents right now. Could you try again in a moment?"
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
