"""
Gradio user interface for the Manzil Chatbot.
"""

import time
import gradio as gr
from datetime import datetime

from config import settings
from utils.logging_config import get_logger

logger = get_logger("ui.gradio_app")


class ChatHistory:
    """Manages chat history for the UI."""

    def __init__(self, max_history=10):
        """Initialize chat history manager."""
        self.history = []
        self.max_history = max_history

    def add(self, user_message, assistant_message):
        """Add a conversation turn to history."""
        self.history.append(
            {
                "user": user_message,
                "assistant": assistant_message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get(self):
        """Get the conversation history."""
        return self.history

    def clear(self):
        """Clear the conversation history."""
        self.history = []

    def save_to_file(self, filename=None):
        """Save chat history to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.txt"

        with open(filename, "w") as f:
            for turn in self.history:
                f.write(f"User: {turn['user']}\n")
                f.write(f"Assistant: {turn['assistant']}\n")
                f.write(f"Time: {turn['timestamp']}\n")
                f.write("-" * 50 + "\n")

        return filename


def create_ui(chatbot):
    """
    Create the Gradio user interface for the chatbot.

    Args:
        chatbot: The initialized chatbot instance

    Returns:
        gradio.Interface: The Gradio interface
    """
    # Initialize chat history manager
    chat_history = ChatHistory()

    # CSS for custom styling
    css = """
    .chat-message-user {
        background-color: #e3f6fc;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-message-assistant {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .company-logo {
        text-align: center;
        margin-bottom: 20px;
    }
    .welcome-message {
        text-align: center;
        margin-bottom: 30px;
    }
    """

    # Helper function to format the conversation for display
    def format_history(history):
        formatted = []
        for turn in history:
            formatted.append((turn["user"], turn["assistant"]))
        return formatted

    # Function to process user message
    def process_message(message, chatbot_ui):
        if not message.strip():
            return chatbot_ui, ""

        # Add user message to UI
        chatbot_ui.append((message, None))

        try:
            # Process the query
            start_time = time.time()
            response = chatbot.process_query(message, chat_history.get())
            processing_time = time.time() - start_time

            # Log processing time
            logger.info(
                f"Query processed in {processing_time:.2f} seconds: {message[:50]}..."
            )

            # Update chat history
            chat_history.add(message, response)

            # Update UI with response
            chatbot_ui[-1] = (message, response)

            return chatbot_ui, ""
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_message = f"I'm sorry, I encountered an error while processing your question. Please try again or contact {settings.COMPANY_NAME} directly for assistance."

            # Update UI with error message
            chatbot_ui[-1] = (message, error_message)

            return chatbot_ui, ""

    # Function to clear chat history
    def clear_history():
        chat_history.clear()
        return [], ""

    # Function to save chat history
    def save_history():
        try:
            filename = chat_history.save_to_file()
            return f"Chat history saved to {filename}"
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            return "Failed to save chat history. Please try again."

    # Function to initialize knowledge base
    def init_knowledge_base(progress=gr.Progress()):
        try:
            progress(0, desc="Starting initialization...")

            # Clear existing data
            progress(0.2, desc="Clearing existing data...")
            chatbot.vector_store.clear_collection()

            # Crawl website
            progress(0.3, desc="Crawling website...")
            pages_content = chatbot.crawler.crawl()

            if not pages_content:
                return "No content extracted from website. Initialization failed."

            progress(0.6, desc=f"Processing {len(pages_content)} pages...")

            # Process and chunk content
            all_chunks = []
            for i, page in enumerate(pages_content):
                progress(
                    0.6 + (0.2 * (i / len(pages_content))),
                    desc=f"Processing page {i+1}/{len(pages_content)}",
                )

                # Use header-based chunking for better semantic division
                chunks = chatbot.chunker.chunk_by_headers(page)
                all_chunks.extend(chunks)

            # Store chunks in vector database
            progress(0.8, desc=f"Storing {len(all_chunks)} chunks...")
            chatbot.vector_store.store_documents(all_chunks)

            progress(1.0, desc="Completed")
            return f"Knowledge base initialized successfully with {len(all_chunks)} chunks from {len(pages_content)} pages."

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {str(e)}")
            return f"Failed to initialize knowledge base: {str(e)}"

    # Create the Gradio interface
    with gr.Blocks(css=css) as demo:
        gr.HTML(
            """
        <div class="company-logo">
            <h1>Manzil Help Center Chatbot</h1>
        </div>
        <div class="welcome-message">
            <p>Welcome to the Manzil Help Center Chatbot. I'm here to answer your questions about Manzil's Shariah-compliant financial products and services.</p>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=4):
                # Chat interface
                chatbot_ui = gr.Chatbot(height=500, show_label=False)

                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Ask about Manzil's services, Islamic finance principles, or how to get started...",
                        show_label=False,
                        scale=9,
                    )
                    submit_button = gr.Button("Send", scale=1)

                with gr.Row():
                    clear_button = gr.Button("Clear Chat")
                    save_button = gr.Button("Save Chat History")

            with gr.Column(scale=1):
                # Sidebar with information and options
                gr.Markdown("## Manzil Services")

                gr.Markdown(
                    """
                - Halal Home Financing
                - Halal Investing & Wealth Management
                - Islamic Wills (Wasiyyah)
                - Halal Prepaid MasterCard
                - Manzil Communities
                - Zakat Calculation
                - Financial Education
                """
                )

                gr.Markdown("## Admin Functions")
                initialize_button = gr.Button("Initialize Knowledge Base")
                status_output = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("## Example Questions")
                example_questions = gr.Examples(
                    examples=[
                        "What is Murabaha financing?",
                        "How do I qualify for halal home financing?",
                        "What makes Manzil's investments Shariah-compliant?",
                        "How do I create an Islamic will?",
                        "What is the difference between conventional and Islamic finance?",
                        "Can you explain how zakat works?",
                    ],
                    inputs=message_input,
                )

        # Set up event handlers
        submit_button.click(
            process_message,
            inputs=[message_input, chatbot_ui],
            outputs=[chatbot_ui, message_input],
        )

        message_input.submit(
            process_message,
            inputs=[message_input, chatbot_ui],
            outputs=[chatbot_ui, message_input],
        )

        clear_button.click(clear_history, outputs=[chatbot_ui, message_input])

        save_button.click(save_history, outputs=[status_output])

        initialize_button.click(init_knowledge_base, outputs=[status_output])

    return demo
