from http.server import BaseHTTPRequestHandler
import json
import os
from telegram import Bot

# Get Telegram token from environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN" )

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests (for testing)"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("Jenny is online! This is the webhook endpoint for the Telegram bot.".encode('utf-8'))
    
    def do_POST(self):
        """Handle POST requests (Telegram webhook)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Log the incoming data
            print(f"Received webhook data: {post_data.decode('utf-8')}")
            
            # Parse the update from Telegram
            update_dict = json.loads(post_data.decode('utf-8'))
            
            # Get basic message info
            message = update_dict.get('message', {})
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '')
            
            # Only process if we have a chat_id and text
            if chat_id and text:
                # Initialize bot
                bot = Bot(token=TELEGRAM_TOKEN)
                
                # Send a simple response
                if text.startswith('/start'):
                    response = "Hello! I am Jenny, Sal's Personal Assistant. This is a test response."
                else:
                    response = f"You said: {text}\nThis is a test response from the simplified webhook."
                
                bot.send_message(chat_id=chat_id, text=response)
                print(f"Sent response to chat {chat_id}")
            
            # Send response to Telegram
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))
        except Exception as e:
            print(f"Error in webhook handler: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
