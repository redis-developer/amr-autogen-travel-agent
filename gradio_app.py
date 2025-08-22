import asyncio
import gradio as gr
import os
from typing import List, Dict
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
        self.agent = TravelAgent(config=self.config)
        self.current_user_id = "Tyler"  # Hardcoded user ID
        # Initialize the user context
        self.agent._get_or_create_user_ctx(self.current_user_id)
    
    async def chat_with_agent(self, message: str) -> List[dict]:
        """
        Handle chat interaction with the travel agent.
        
        Args:
            message: User's input message
            
        Returns:
            Complete chat history from Redis for the current user
        """
        # Using hardcoded user Tyler
        # No need to check for user selection since it's hardcoded
        
        try:
            # Send message to agent (it handles adding to Redis automatically)
            await self.agent.chat(message, user_id=self.current_user_id)
            
            # Retrieve updated chat history from Redis
            history = await self.agent.get_chat_history(self.current_user_id, n=-1)
            return history
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            # Get current history and add error message
            try:
                history = await self.agent.get_chat_history(self.current_user_id, n=-1)
                history.append({"role": "assistant", "content": error_msg})
                return history
            except:
                return [{"role": "assistant", "content": error_msg}]
    
    # User management methods removed - using hardcoded Tyler user
    

    

    
    async def clear_chat_history(self) -> List[dict]:
        """Clear chat history for the current user."""
        if not self.current_user_id:
            return []
        
        try:
            # Get the user context and clear its Redis history
            ctx = self.agent._get_or_create_user_ctx(self.current_user_id)
            await ctx.supervisor.model_context.clear()
            return []
        except Exception as e:
            print(f"Error clearing chat history for user {self.current_user_id}: {e}")
            return []
    
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
                        Your intelligent travel planning assistant powered by AutoGen & TCM
                    </p>
                    <p style="color: #8A99A0; font-size: 14px;">
                        Your personal travel planning assistant ready to help!
                    </p>
                </div>
            """)
            
            # Main single-column layout for chat interface
            with gr.Column():
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">💬 Chat Interface</h3>
                        </div>
                    """)
                    
                    chatbot = gr.Chatbot(
                        [],
                        show_label=False,
                        container=True,
                        type="messages",
                        avatar_images=None,
                        show_copy_button=False,
                        height=500,
                        autoscroll=True
                    )
                    
                    # Input components
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me about planning your next trip...",
                            show_label=False,
                            container=False,
                            scale=5,
                            lines=1,
                            interactive=True
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1, interactive=True)
                    
                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
            
            
            # Event handler functions
            async def handle_streaming_chat(message, history):
                """Handle streaming chat with console logging for insights."""
                if not message.strip():
                    yield history
                    return
                
                # Add user message to history and show it immediately
                history = history + [{"role": "user", "content": message}]
                yield history
                
                # Initialize assistant message with animated thinking dots
                history = history + [{"role": "assistant", "content": '<span class="thinking-animation">●●●</span>'}]
                yield history
                
                try:
                    # Stream the response
                    async for partial_response in self.agent.stream_chat_turn(self.current_user_id, message):
                        # Update the last assistant message with the streaming content
                        history[-1] = {"role": "assistant", "content": partial_response}
                        yield history
                    
                    # After streaming is complete, log insights to console
                    insights = await self.agent.insights_for_task(self.current_user_id, message, limit=4)
                    
                    if insights:
                        print(f"\n🧠 INSIGHTS APPLIED (user: {self.current_user_id}) 🧠", flush=True)
                        for insight in insights:
                            print(f"• {insight}", flush=True)
                        print("--- END INSIGHTS ---\n", flush=True)
                    else:
                        print(f"--- NO INSIGHTS APPLIED (user: {self.current_user_id}) ---\n", flush=True)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    # Replace thinking indicator with error message
                    history[-1] = {"role": "assistant", "content": error_msg}
                    print(f"❌ Error during streaming chat: {e}", flush=True)
                    yield history
            
            # User management handlers removed - using hardcoded Tyler user
            

            

            
            # Event handlers for chat - streaming version
            async def handle_submit_and_clear(message, history):
                """Handle message submission and immediately clear input."""
                async for result in handle_streaming_chat(message, history):
                    yield result, ""
            
            msg.submit(
                handle_submit_and_clear,
                [msg, chatbot],
                [chatbot, msg],
                queue=True
            )
            
            send_btn.click(
                handle_submit_and_clear,
                [msg, chatbot],
                [chatbot, msg],
                queue=True
            )
            
            # User management handlers removed - using hardcoded Tyler user
            
            # Clear chat functionality
            async def handle_clear_chat():
                """Handle clearing chat history."""
                cleared_history = await self.clear_chat_history()
                return cleared_history, ""
            
            clear_btn.click(
                handle_clear_chat,
                outputs=[chatbot, msg],
                queue=True
            )
        
        return interface


def create_app(config=None) -> gr.Interface:
    """Create and return the Gradio app."""
    if config is None:
        config = get_config()
    ui = TravelAgentUI(config=config)
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
