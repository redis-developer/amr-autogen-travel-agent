import asyncio
import gradio as gr
import os
from typing import List, Tuple, Optional, Dict
import uuid

from agent import TravelAgent


class TravelAgentUI:
    """
    Gradio UI wrapper for the TravelAgent.
    """
    
    def __init__(self):
        """Initialize the UI with a travel agent instance."""
        self.agent = None
        self.session_id = str(uuid.uuid4())
        
    async def initialize_agent(self):
        """Initialize the travel agent asynchronously."""
        if self.agent is None:
            self.agent = TravelAgent(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                tavily_api_key=os.environ["TAVILY_API_KEY"]
            )
    
    async def chat_with_agent(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Handle chat interaction with the travel agent.
        
        Args:
            message: User's input message
            history: Chat history as list of message dicts with 'role' and 'content'
            
        Returns:
            Tuple of (empty string for clearing input, updated history)
        """
        if not self.agent:
            await self.initialize_agent()
        
        try:
            # Get response from the agent
            response = await self.agent.chat(message, user_id=self.session_id)
            
            # Update history with new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    async def get_user_preferences(self) -> str:
        """
        Retrieve and format user preferences for display.
        
        Returns:
            Formatted HTML string of user preferences
        """
        if not self.agent:
            await self.initialize_agent()
        
        try:
            preferences = await self.agent.get_user_preferences(user_id=self.session_id)
            
            if not preferences:
                return """
                <div style="text-align: center; padding: 40px; color: var(--dusk-50p);">
                    <h3>No Preferences Found</h3>
                    <p>Start chatting to learn and save your travel preferences!</p>
                </div>
                """
            
            html_content = """
            <div style="font-family: 'Space Grotesk', sans-serif; color: var(--white);">
                <h3 style="color: var(--yellow); margin-bottom: 20px;">🧠 Your Travel Preferences</h3>
            """
            
            for i, pref in enumerate(preferences):
                # Format timestamp
                timestamp = pref.get('timestamp', 'Unknown time')
                if timestamp and timestamp != 'Unknown time':
                    try:
                        # Try to parse and format the timestamp
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%B %d, %Y at %I:%M %p')
                    except:
                        formatted_time = timestamp
                else:
                    formatted_time = 'Recently'
                
                html_content += f"""
                <div style="
                    background: linear-gradient(135deg, var(--dusk) 0%, var(--dusk-90p) 100%);
                    border: 1px solid var(--dusk-70p);
                    border-left: 4px solid var(--violet);
                    border-radius: 8px;
                    padding: 16px;
                    margin-bottom: 12px;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                        margin-bottom: 8px;
                    ">
                        <span style="
                            background-color: var(--violet);
                            color: var(--midnight);
                            padding: 4px 8px;
                            border-radius: 4px;
                            font-size: 12px;
                            font-weight: 500;
                            font-family: 'Space Mono', monospace;
                        ">#{i+1}</span>
                        <span style="
                            color: var(--dusk-30p);
                            font-size: 12px;
                        ">{formatted_time}</span>
                    </div>
                    <p style="
                        margin: 0;
                        line-height: 1.5;
                        color: var(--white);
                    ">{pref.get('content', 'No content')}</p>
                </div>
                """
            
            html_content += "</div>"
            return html_content
            
        except Exception as e:
            return f"""
            <div style="text-align: center; padding: 40px; color: var(--redis-red);">
                <h3>Error Loading Preferences</h3>
                <p>Could not retrieve preferences: {str(e)}</p>
            </div>
            """
    
    def create_interface(self) -> gr.Interface:
        """Create and return the Gradio interface."""
        
        # Redis-themed CSS styling
        css = """
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono&display=swap');

        /* Redis Color Variables */
        :root {
            --white: #FFFFFF;
            --redis-red: #FF4438;
            --black: #000000;
            --black-90p: #191919;
            --black-70p: #4c4c4c;
            --black-50p: #808080;
            --black-30p: #B2B2B2;
            --black-10p: #E5E5E5;
            --light-gray: #F0F0F0;
            --midnight: #091A23;
            --dusk: #163341;
            --dusk-90p: #2D4754;
            --dusk-70p: #5C707A;
            --dusk-50p: #8A99A0;
            --dusk-30p: #B9C2C6;
            --dusk-10p: #E8EBEC;
            --violet: #C795E3;
            --violet-50p: #E3CAF1;
            --violet-10p: #F9F4FC;
            --sky-blue: #80DBFF;
            --sky-blue-50p: #BFEDFF;
            --sky-blue-10p: #F2FBFF;
            --yellow: #DCFF1E;
            --yellow-40p: #F1FFA5;
            --yellow-10p: #FBFFE8;
        }

        /* Global styling */
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
            background-color: var(--midnight) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--white) !important;
        }
        
        body {
            background-color: var(--midnight) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            color: var(--white) !important;
        }

        /* Main layout containers */
        .main-container {
            display: flex !important;
            height: 80vh !important;
            min-height: 600px !important;
            gap: 20px !important;
            background-color: var(--midnight) !important;
        }
        
        .chat-panel {
            flex: 1 !important;
            min-width: 400px !important;
            background-color: var(--midnight) !important;
            border: 1px solid var(--dusk-70p) !important;
            border-radius: 12px !important;
            display: flex !important;
            flex-direction: column !important;
            overflow: hidden !important;
        }
        
        .memory-panel {
            flex: 0 0 400px !important;
            min-width: 300px !important;
            max-width: 500px !important;
            background-color: var(--dusk) !important;
            border: 1px solid var(--dusk-70p) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
            resize: horizontal !important;
        }
        
        /* Chat container */
        .chat-container {
            height: 100% !important;
            background-color: var(--midnight) !important;
        }
        
        .chatbot-scrollable {
            flex: 1 !important;
            min-height: 400px !important;
            max-height: calc(80vh - 300px) !important;
            background-color: var(--midnight) !important;
            overflow-y: auto !important;
            scroll-behavior: smooth !important;
        }
        
        #component-0 {
            height: 100% !important;
            background-color: var(--midnight) !important;
        }
        
        /* Resizable splitter */
        .resizer {
            width: 6px !important;
            background: var(--dusk-70p) !important;
            cursor: col-resize !important;
            transition: background-color 0.3s ease !important;
        }
        
        .resizer:hover {
            background: var(--violet) !important;
        }

        /* Chatbot styling */
        .chatbot {
            background-color: var(--midnight) !important;
            border: 1px solid var(--dusk) !important;
            height: 100% !important;
            overflow-y: auto !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Ensure chatbot messages container is scrollable */
        .chatbot .wrap {
            height: 100% !important;
            overflow-y: auto !important;
            padding: 12px !important;
        }
        
        /* Auto-scroll to bottom styling */
        .chatbot .messages {
            display: flex !important;
            flex-direction: column !important;
            gap: 8px !important;
            padding-bottom: 20px !important;
        }

        /* Message styling */
        .message.user {
            background-color: var(--dusk) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }
        
        .message.bot {
            background-color: var(--dusk-90p) !important;
            color: var(--white) !important;
            border: 1px solid var(--violet) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        /* Input field styling */
        .gr-textbox textarea, .gr-textbox input {
            background-color: var(--midnight) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .gr-textbox textarea:focus, .gr-textbox input:focus {
            border-color: var(--yellow) !important;
            box-shadow: 0 0 0 2px var(--yellow-40p) !important;
        }

        /* Button styling */
        .gr-button {
            font-family: 'Space Mono', monospace !important;
            transition: all 0.3s ease !important;
            border-radius: 6px !important;
        }

        .gr-button.primary {
            background-color: var(--yellow) !important;
            border: 1px solid var(--yellow) !important;
            color: var(--midnight) !important;
            font-weight: 500 !important;
        }

        .gr-button.primary:hover {
            background-color: var(--midnight) !important;
            color: var(--white) !important;
            border-color: var(--yellow) !important;
        }

        .gr-button.secondary {
            background-color: var(--dusk-70p) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
        }

        .gr-button.secondary:hover {
            background-color: var(--dusk-90p) !important;
            border-color: var(--dusk-90p) !important;
        }

        /* Accordion styling */
        .gr-accordion {
            background-color: var(--dusk) !important;
            border: 1px solid var(--dusk-70p) !important;
        }

        .gr-accordion .label-wrap {
            background-color: var(--dusk) !important;
            color: var(--white) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .gr-accordion .label-wrap:hover {
            color: var(--violet) !important;
        }

        /* Examples styling */
        .gr-examples .examples {
            background-color: var(--dusk) !important;
        }

        .gr-examples .example {
            background-color: var(--dusk-90p) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .gr-examples .example:hover {
            background-color: var(--violet) !important;
            border-color: var(--violet) !important;
        }

        /* Header text styling */
        h1, h2, h3, h4, h5, h6 {
            color: var(--white) !important;
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 500 !important;
        }

        /* Panel and container backgrounds */
        .gr-panel, .gr-box, .gr-form {
            background-color: var(--dusk) !important;
            border: 1px solid var(--dusk-70p) !important;
        }

        /* Markdown content */
        .gr-markdown {
            color: var(--white) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .gr-markdown p, .gr-markdown li {
            color: var(--white) !important;
        }

        .gr-markdown strong {
            color: var(--yellow) !important;
        }

        .gr-markdown code {
            background-color: var(--dusk-90p) !important;
            color: var(--sky-blue) !important;
            font-family: 'Space Mono', monospace !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: var(--dusk);
        }

        ::-webkit-scrollbar-thumb {
            background: var(--dusk-70p);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--violet);
        }
        
        /* Specific scrollbar styling for chatbot */
        .chatbot::-webkit-scrollbar,
        .chatbot .wrap::-webkit-scrollbar {
            width: 6px;
        }

        .chatbot::-webkit-scrollbar-track,
        .chatbot .wrap::-webkit-scrollbar-track {
            background: var(--midnight);
        }

        .chatbot::-webkit-scrollbar-thumb,
        .chatbot .wrap::-webkit-scrollbar-thumb {
            background: var(--dusk-70p);
            border-radius: 3px;
        }

        .chatbot::-webkit-scrollbar-thumb:hover,
        .chatbot .wrap::-webkit-scrollbar-thumb:hover {
            background: var(--yellow);
        }

        /* Custom Redis element classes */
        .redis-input {
            background-color: var(--midnight) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
            font-family: 'Space Grotesk', sans-serif !important;
        }

        .redis-input:focus {
            border-color: var(--yellow) !important;
            box-shadow: 0 0 0 2px var(--yellow-40p) !important;
        }

        .redis-button-primary {
            background-color: var(--yellow) !important;
            border: 1px solid var(--yellow) !important;
            color: var(--midnight) !important;
            font-family: 'Space Mono', monospace !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
        }

        .redis-button-primary:hover {
            background-color: var(--midnight) !important;
            color: var(--white) !important;
            border-color: var(--yellow) !important;
        }

        .redis-button-secondary {
            background-color: var(--dusk-70p) !important;
            color: var(--white) !important;
            border: 1px solid var(--dusk-70p) !important;
            font-family: 'Space Mono', monospace !important;
        }

        .redis-button-secondary:hover {
            background-color: var(--dusk-90p) !important;
            border-color: var(--dusk-90p) !important;
        }

        /* Chat message bubble improvements */
        .chatbot .message {
            border-radius: 12px !important;
            margin: 8px 0 !important;
            padding: 12px 16px !important;
        }

        .chatbot .message.user {
            background: linear-gradient(135deg, var(--dusk) 0%, var(--dusk-90p) 100%) !important;
            border-left: 4px solid var(--yellow) !important;
        }

        .chatbot .message.bot {
            background: linear-gradient(135deg, var(--dusk-90p) 0%, var(--violet) 100%) !important;
            border-left: 4px solid var(--sky-blue) !important;
        }

        /* Add subtle animations */
        .redis-button-primary, .redis-button-secondary {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        .redis-input {
            transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        }

        /* Memory panel specific styling */
        .memory-content {
            height: calc(100vh - 300px) !important;
            overflow-y: auto !important;
            padding: 16px !important;
            background-color: var(--dusk) !important;
        }

        .memory-content::-webkit-scrollbar {
            width: 6px;
        }

        .memory-content::-webkit-scrollbar-track {
            background: var(--dusk-90p);
        }

        .memory-content::-webkit-scrollbar-thumb {
            background: var(--violet);
            border-radius: 3px;
        }

        /* Responsive design for smaller screens */
        @media (max-width: 1200px) {
            .main-container {
                flex-direction: column !important;
                height: auto !important;
            }
            
            .memory-panel {
                flex: none !important;
                min-width: 100% !important;
                max-width: none !important;
            }
            
            .memory-content {
                height: 300px !important;
            }
        }

        /* Enhanced chat panel styling */
        .chat-panel {
            padding: 16px !important;
        }
        
        /* Input section styling to stay at bottom */
        .chat-panel > .gr-row:last-of-type {
            margin-top: auto !important;
            flex-shrink: 0 !important;
        }

        .memory-panel {
            padding: 0 !important;
        }

        /* Improve spacing and layout */
        .gradio-container .app {
            max-width: 100% !important;
            padding: 0 20px !important;
        }
        """
        
        # Create custom Redis theme
        redis_theme = gr.themes.Soft(
            font=[
                gr.themes.GoogleFont("Space Grotesk"),
                "ui-sans-serif",
                "system-ui", 
                "sans-serif"
            ],
            font_mono=[
                gr.themes.GoogleFont("Space Mono"),
                "ui-monospace",
                "Consolas",
                "monospace"
            ],
            primary_hue=gr.themes.colors.yellow,
            secondary_hue=gr.themes.colors.slate,
            neutral_hue=gr.themes.colors.slate
        ).set(
            # Background colors
            background_fill_primary="#091A23",
            background_fill_secondary="#163341",
            background_fill_primary_dark="#091A23",
            background_fill_secondary_dark="#163341",
            
            # Button colors
            button_primary_background_fill="#DCFF1E",
            button_primary_background_fill_dark="#DCFF1E",
            button_primary_text_color="#091A23",
            button_primary_text_color_dark="#091A23",
            button_primary_background_fill_hover="#091A23",
            button_primary_background_fill_hover_dark="#091A23",
            button_primary_text_color_hover="#FFFFFF",
            button_primary_text_color_hover_dark="#FFFFFF",
            button_primary_border_color_hover="#091A23",
            
            button_secondary_background_fill="#5C707A",
            button_secondary_background_fill_hover="#2D4754",
            button_secondary_border_color="#5C707A",
            button_secondary_border_color_hover="#2D4754",
            button_secondary_text_color="#FFFFFF",
            button_secondary_text_color_hover="#FFFFFF",
            
            # Input and text colors
            input_background_fill="#091A23",
            body_text_color="#FFFFFF",
            checkbox_label_background_fill="#091A23",
            checkbox_label_text_color="#FFFFFF",
            checkbox_label_text_color_selected="#FFFFFF",
            
            # Accent colors
            color_accent_soft="#163341",
            border_color_accent_subdued="#FFFFFF",
            
            # Panel colors
            panel_border_color_dark="#8A99A0",
            panel_border_width_dark="1px",
            panel_background_fill_dark="#091A23",
            
            # Stat colors
            stat_background_fill="#C795E3",
            stat_background_fill_dark="#C795E3"
        )
        
        # Create the chat interface
        with gr.Blocks(
            title="AI Travel Concierge",
            theme=redis_theme,
            css=css
        ) as interface:
            
            # Header
            gr.HTML("""
                <div style="text-align: center; margin: 20px 0; font-family: 'Space Grotesk', sans-serif;">
                    <h1 style="color: #FFFFFF; margin-bottom: 10px; font-weight: 500;">🌍 AI Travel Concierge</h1>
                    <p style="color: #DCFF1E; font-size: 18px; font-weight: 500;">
                        Your intelligent travel planning assistant powered by AutoGen & Redis
                    </p>
                    <p style="color: #8A99A0; font-size: 14px;">
                        I can help you plan trips, find flights, recommend activities, and handle dietary restrictions!
                    </p>
                </div>
            """)
            
            # Main split-view container
            with gr.Row(elem_classes="main-container"):
                
                # Left Panel - Chat Interface
                with gr.Column(elem_classes="chat-panel", scale=2):
                    
                    # Chat interface
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        show_label=False,
                        container=True,
                        type="messages",
                        avatar_images=("🧳", "🤖"),
                        show_copy_button=True,
                        elem_classes="chatbot-scrollable",
                        autoscroll=True
                    )
                    
                    # Input components
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about planning your next trip! Example: 'I want to plan a week in Japan this fall for 2 people with a $7000 budget'",
                            show_label=False,
                            container=False,
                            scale=4,
                            lines=2,
                            elem_classes="redis-input"
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, elem_classes="redis-button-primary")
                    
                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", elem_classes="redis-button-secondary")
                        refresh_memory_btn = gr.Button("🔄 Refresh Memory", variant="secondary", elem_classes="redis-button-secondary")
                
                # Right Panel - Memory Explorer
                with gr.Column(elem_classes="memory-panel", scale=1):
                    gr.HTML("""
                        <div style="padding: 16px; background: linear-gradient(135deg, var(--dusk) 0%, var(--dusk-90p) 100%); border-bottom: 1px solid var(--dusk-70p);">
                            <h3 style="color: var(--yellow); margin: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 500;">
                                🧠 Memory Explorer
                            </h3>
                            <p style="color: var(--dusk-30p); margin: 8px 0 0 0; font-size: 14px;">
                                Your learned travel preferences
                            </p>
                        </div>
                    """)
                    
                    # Memory display area
                    memory_display = gr.HTML(
                        value="""
                        <div style="text-align: center; padding: 40px; color: var(--dusk-50p);">
                            <h3>No Preferences Found</h3>
                            <p>Start chatting to learn and save your travel preferences!</p>
                        </div>
                        """,
                        elem_classes="memory-content"
                    )    
            
            
            # Set up the chat functionality
            def handle_chat(message, history):
                """Wrapper to handle async chat in Gradio."""
                result = asyncio.run(self.chat_with_agent(message, history))
                # Also refresh memory after chat
                memory_html = asyncio.run(self.get_user_preferences())
                return result[0], result[1], memory_html
            
            def handle_refresh_memory():
                """Refresh the memory display."""
                return asyncio.run(self.get_user_preferences())
            
            # Event handlers
            msg.submit(
                handle_chat,
                [msg, chatbot],
                [msg, chatbot, memory_display],
                queue=True
            )
            
            send_btn.click(
                handle_chat,
                [msg, chatbot],
                [msg, chatbot, memory_display],
                queue=True
            )
            
            # Clear chat functionality
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg],
                queue=False
            )
            
            # Refresh memory functionality
            refresh_memory_btn.click(
                handle_refresh_memory,
                outputs=[memory_display],
                queue=True
            )
        
        return interface


def create_app() -> gr.Interface:
    """Create and return the Gradio app."""
    ui = TravelAgentUI()
    return ui.create_interface()


def main():
    """Main function to launch the Gradio app."""
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not found in environment variables")
        print("Please set your Tavily API key:")
        print("export TAVILY_API_KEY='your-api-key-here'")
    
    # Create and launch the app
    app = create_app()
    
    print("\n🚀 Launching AI Travel Concierge...")
    print("📱 The app will open in your browser shortly")
    
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
