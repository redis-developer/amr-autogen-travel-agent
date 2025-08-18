import asyncio
import gradio as gr
import os
from typing import List, Tuple, Optional, Dict
import uuid
from pathlib import Path

from agent import TravelAgent
from config import get_config, validate_dependencies



def load_css() -> str:
    """Load CSS from external file."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Warning: CSS file not found at {css_path}")
        return ""



class TravelAgentUI:
    """
    Gradio UI wrapper for the TravelAgent.
    """
    
    def __init__(self, config=None):
        """Initialize the UI with a travel agent instance."""
        self.config = config or get_config()
        self.session_id = str(uuid.uuid4())
        self.agent = TravelAgent(config=self.config)
    
    async def chat_with_agent(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """
        Handle chat interaction with the travel agent.
        
        Args:
            message: User's input message
            history: Chat history as list of message dicts with 'role' and 'content'
            
        Returns:
            Tuple of (empty string for clearing input, updated history)
        """
        # Add user message immediately
        history.append({"role": "user", "content": message})
        print("CHATTING WITH AGENT", message, flush=True)
        
        try:
            # Get response from the agent
            response = await self.agent.chat(message, user_id=self.session_id)
            print("RESPONSE", response, flush=True)
            
            # Add assistant response
            history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    async def get_user_preferences(self) -> str:
        """
        Retrieve and format user preferences for display.
        
        Returns:
            Formatted HTML string of user preferences
        """
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
        
        # Load CSS from external file
        css = load_css()
        
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
                with gr.Column(elem_classes="chat-panel"):
                    
                    # Chat interface
                    chatbot = gr.Chatbot(
                        [],
                        show_label=False,
                        container=False,
                        type="messages",
                        avatar_images=None,
                        show_copy_button=False,
                        height=400,
                        autoscroll=True
                    )
                    
                    # Input components
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about planning your next trip...",
                            show_label=False,
                            container=False,
                            scale=4,
                            lines=1
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary", size="sm")
                        refresh_memory_btn = gr.Button("Refresh Memory", variant="secondary", size="sm")
                
                # Right Panel - Memory Explorer
                with gr.Column(elem_classes="memory-panel"):
                    gr.HTML("""
                        <div style="padding: 20px; border-bottom: 1px solid var(--dusk-50p);">
                            <h3 style="color: var(--white); margin: 0; font-size: 18px; font-weight: 500;">
                                Travel Preferences
                            </h3>
                            <p style="color: var(--dusk-30p); margin: 8px 0 0 0; font-size: 14px;">
                                Learned from our conversations
                            </p>
                        </div>
                    """)
                    
                    # Memory display area - start with loading state
                    memory_display = gr.HTML(
                        value="""
                        <div style="text-align: center; padding: 40px; color: var(--dusk-50p);">
                            <h3>Loading preferences...</h3>
                        </div>
                        """,
                        elem_classes="memory-content"
                    )    
            
            
            # Set up the chat functionality
            async def handle_chat_start(message, history):
                """Show typing indicator when chat starts."""
                print("CHAT START", message, flush=True)
                if not message.strip():
                    return "", history, await self.get_user_preferences()
                
                # Add user message and typing indicator immediately
                updated_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Thinking..."}
                ]
                return "", updated_history, await self.get_user_preferences()
            
            async def handle_chat_complete(message, history):
                """Complete the chat interaction."""
                print("CHAT COMPLETE", repr(message), flush=True)
                print("HISTORY BEFORE:", [h.get("content", "")[:50] for h in history], flush=True)
                
                # Extract the actual message from history since the message parameter gets cleared
                if not history or len(history) < 2:
                    return history, await self.get_user_preferences()
                
                # Get the user message from history (should be second to last, before "Thinking...")
                user_message = None
                if history[-1]["content"] == "Thinking..." and len(history) >= 2:
                    user_message = history[-2]["content"]
                    # Remove the typing indicator
                    history = history[:-1]
                    print("REMOVED THINKING, HISTORY NOW:", [h.get("content", "")[:50] for h in history], flush=True)
                
                if not user_message:
                    return history, await self.get_user_preferences()
                
                # The history now has the user message, so we need to pass the history without the user message
                # since chat_with_agent will add it again
                clean_history = history[:-1] if history and history[-1]["role"] == "user" else history
                print("CLEAN HISTORY:", [h.get("content", "")[:50] for h in clean_history], flush=True)
                print("USER MESSAGE:", repr(user_message), flush=True)
                result = await self.chat_with_agent(user_message, clean_history)
                # Also refresh memory after chat
                memory_html = await self.get_user_preferences()
                return result[1], memory_html
            
            async def handle_refresh_memory():
                """Refresh the memory display with loading indicator."""
                # First show loading state
                loading_html = """
                <div style="text-align: center; padding: 40px; color: var(--dusk-50p);">
                    <h3>Refreshing...</h3>
                </div>
                """
                return loading_html
            
            async def handle_refresh_memory_complete():
                """Complete the memory refresh."""
                return await self.get_user_preferences()
            
            # Event handlers with two-step process for better UX
            msg.submit(
                handle_chat_start,
                [msg, chatbot],
                [msg, chatbot, memory_display],
                queue=True
            ).then(
                handle_chat_complete,
                [msg, chatbot],
                [chatbot, memory_display],
                queue=True
            )
            
            send_btn.click(
                handle_chat_start,
                [msg, chatbot],
                [msg, chatbot, memory_display],
                queue=True
            ).then(
                handle_chat_complete,
                [msg, chatbot],
                [chatbot, memory_display],
                queue=True
            )
            
            # Clear chat functionality
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, msg],
                queue=False
            )
            
            # Refresh memory functionality with loading state
            refresh_memory_btn.click(
                handle_refresh_memory,
                outputs=[memory_display],
                queue=True
            ).then(
                handle_refresh_memory_complete,
                outputs=[memory_display],
                queue=True
            )
            
            # Auto-load memory on interface startup
            interface.load(
                handle_refresh_memory_complete,
                outputs=[memory_display],
                queue=True
            )
        
        return interface


def create_app(config=None) -> gr.Interface:
    """Create and return the Gradio app."""
    if config is None:
        config = get_config()
    ui = TravelAgentUI(config=config)
    print("Travel agent built")
    return ui.create_interface()


def main():
    """Main function to launch the Gradio app."""
    print("🌍 AI Travel Concierge - Starting up...")
    
    # Load and validate configuration
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
    except SystemExit:
        return
    
    # Validate dependencies
    if not validate_dependencies():
        print("\n❌ Dependency validation failed. Please check your configuration.")
        return
    
    # Create and launch the app
    try:
        app = create_app(config)
        
        print(f"\n🚀 Launching AI Travel Concierge on {config.server_name}:{config.server_port}")
        
        app.queue().launch(
            server_name=config.server_name,
            server_port=config.server_port,
            share=config.share,
            show_error=True,
            inbrowser=True
        )

    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        raise


if __name__ == "__main__":
    main()
