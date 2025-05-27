import os
import json
import logging
import re
from http.server import BaseHTTPRequestHandler
import google.generativeai as genai
from telegram import Bot, Update

# --- Configuration ---
COMBINED_TEXT_FILENAME = "all_pdf_text_combined.txt"
MAX_CONTEXT_PARAGRAPHS = 5
AI_MODEL_NAME = 'gemini-2.0-flash'
# -------------------

# Configure logging
logging.basicConfig(
    format="%(asctime )s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Get Secrets from Environment Variables ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# -------------------

# --- PDF & AI Logic ---
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
    youtube_regex = r"(?:https?:// )?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)(?:\S+)?"
    links = re.findall(youtube_regex, text)
    full_links = [f"https://www.youtube.com/watch?v={link_id}" for link_id in links]
    unique_links = list(set(full_links ))
    if unique_links:
        logger.info(f"Found potential YouTube links in context: {unique_links}")
    return unique_links

def construct_prompt(context_paragraphs, question, youtube_links_in_context):
    """Creates the prompt for the AI model, incorporating persona and link selection."""
    context = "\n\n".join(context_paragraphs)
    
    # Prepare the list of links found in the context for the prompt
    links_string = ""
    if youtube_links_in_context:
        links_string = "\n\nPotential relevant YouTube links found in the source text:\n" + "\n".join(youtube_links_in_context)

    # The core instruction for the AI
    prompt = f"""You are Jenny, Sal's Personal Assistant, an expert on the provided documents. Answer the user's question based *only* on the following text excerpts from Sal's documents:

--- TEXT EXCERPTS ---
{context}
--- END TEXT EXCERPTS ---

{links_string}

User's question: {question}

Instructions for Jenny:
1. Answer the question naturally and conversationally, as Sal's helpful assistant. Do NOT mention "based on the text" or "according to the document".
2. If one of the 'Potential relevant YouTube links' listed above directly supports or illustrates your answer, include ONLY that single most relevant link at the end of your response. Do not include links otherwise.

Jenny's Answer:"""
    logger.info("Prompt constructed for AI with persona and link selection instructions.")
    return prompt

def get_ai_response(api_key, prompt):
    """Connects to Google AI and gets the response."""
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return "Error: Google API Key not configured."
    try:
        logger.info(f"Configuring Google AI with model {AI_MODEL_NAME}...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(AI_MODEL_NAME)
        logger.info("Sending prompt to Google AI...")
        # Add safety settings to potentially reduce overly verbose or off-topic responses
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        logger.info("Received response from AI.")
        # Check if response has text, handle potential blocks
        try:
            return response.text
        except ValueError:
             # If the response was blocked, response.text might raise an error.
             logger.warning(f"AI response was blocked or empty.")
             return "I'm sorry, I couldn't generate a response for that request based on the provided information. It might be outside the scope of the documents or triggered a safety filter."

    except Exception as e:
        logger.error(f"Error contacting Google AI: {e}")
        error_message = str(e)
        return f"Sorry, an error occurred while contacting the AI: {error_message}"

# --- Webhook Handler ---
class handler(BaseHTTPRequestHandler):
    async def do_POST(self):
        """Handle POST requests (Telegram webhook)"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Parse the update from Telegram
        update_dict = json.loads(post_data.decode('utf-8'))
        update = Update.de_json(update_dict, None)
        
        # Process the message
        await self.process_update(update)
        
        # Send response to Telegram
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))
    
    async def process_update(self, update):
        """Process the Telegram update"""
        if not update.message or not update.message.text:
            return
        
        # Initialize bot
        bot = Bot(token=TELEGRAM_TOKEN)
        
        # Get user's message
        user_question = update.message.text
        chat_id = update.message.chat_id
        user_name = update.message.from_user.first_name
        
        # Handle /start command
        if user_question.startswith('/start'):
            welcome_message = f"Hey {user_name}! I am Jenny, Sal's Personal Assistant. How can I help you today?"
            await bot.send_message(chat_id=chat_id, text=welcome_message)
            return
        
        # Process regular messages
        logger.info(f"Received question from {user_name}: {user_question}")
        
        # Send typing indicator
        await bot.send_chat_action(chat_id=chat_id, action='typing')
        
        # Load text from file
        full_text = load_text_from_file(COMBINED_TEXT_FILENAME)
        if not full_text:
            await bot.send_message(chat_id=chat_id, text="Sorry, I couldn't load the document text.")
            return
        
        # Find relevant paragraphs
        relevant_context_paragraphs = find_relevant_paragraphs(full_text, user_question, MAX_CONTEXT_PARAGRAPHS)
        relevant_context_text = "\n\n".join(relevant_context_paragraphs)
        youtube_links = find_youtube_links(relevant_context_text)
        
        # Construct prompt and get AI response
        ai_prompt = construct_prompt(relevant_context_paragraphs, user_question, youtube_links)
        ai_answer = get_ai_response(GOOGLE_API_KEY, ai_prompt)
        
        # Send response back to user
        await bot.send_message(chat_id=chat_id, text=ai_answer)
