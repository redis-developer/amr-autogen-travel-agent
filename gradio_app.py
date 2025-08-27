import asyncio
import gradio as gr
import os
from typing import List, Dict
from pathlib import Path

from agent import TravelAgent
from config import get_config, validate_dependencies
from auth.entra import get_redis_client

_redis_client = None
_config = get_config()
_redis_host = _config.redis_host
_redis_port = _config.redis_port

def get_redis():
    """Lazy singleton Redis client using entra.py"""
    global _redis_client
    if _redis_client is None:
        print("[app] creating Redis client…")
        
        _redis_client = get_redis_client(_redis_host, _redis_port)
    return _redis_client

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
        self.agent = TravelAgent(config=self.config, redis_client_factory=get_redis)
        self.current_user_id = "Tyler"  # Default user ID
        self.initial_history = []  # Will be populated async
    
    async def initialize_chat_history(self):
        """Load initial chat history for the default user."""
        try:
            self.initial_history = await self.agent.get_chat_history(self.current_user_id, n=-1)
            print(f"✅ Loaded {len(self.initial_history)} initial messages for user: {self.current_user_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load initial chat history: {e}")
            self.initial_history = []

    
    async def switch_user(self, new_user_id: str) -> List[dict]:
        """Switch to a different user profile and load their chat history.
        
        Args:
            new_user_id: The user ID to switch to
            
        Returns:
            Chat history for the new user
        """
        if new_user_id == self.current_user_id:
            # No change needed, return current history
            return await self.agent.get_chat_history(self.current_user_id, n=-1)
        
        # Switch to the new user
        self.current_user_id = new_user_id
        
        # Initialize the new user context (this will preseed if needed)
        self.agent._get_or_create_user_ctx(new_user_id)
        
        # Return the new user's chat history
        return await self.agent.get_chat_history(new_user_id, n=-1)
    
    async def clear_chat_history(self) -> List[dict]:
        """Clear chat history for the current user from Redis and UI."""
        if not self.current_user_id:
            return []
        
        try:
            # Get the user context and clear its Redis history
            ctx = self.agent._get_or_create_user_ctx(self.current_user_id)
            await ctx.agent.model_context.clear()
            print(f"✅ Cleared chat history for user: {self.current_user_id}")
            return []
        except Exception as e:
            print(f"❌ Error clearing chat history for user {self.current_user_id}: {e}")
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
                <div style="text-align: center; margin: 8px 0; font-family: 'Space Grotesk', sans-serif;">
                    <h1 style="color: #FFFFFF; margin-bottom: 10px; font-weight: 500;">🌍 AI Travel Concierge</h1>
                    <p style="color: #DCFF1E; font-size: 18px; font-weight: 500;">
                        Your intelligent travel planning assistant powered by AI & Long Term Memory
                    </p>
                    <p style="color: #8A99A0; font-size: 14px;">
                        Built on Redis, Autogen, Tavily, and OpenAI
                    </p>
                </div>
            """)
            
            # User Profile Switcher (top)
            with gr.Row(elem_classes=["user-selector-container"], equal_height=True):
                with gr.Tabs(selected=0) as user_tabs:
                    with gr.Tab("Tyler") as tyler_tab:
                        pass
                    with gr.Tab("Amanda") as amanda_tab:
                        pass
            
            # Two-column layout: Chat (left 70%) and Agent Logs (right 30%)
            with gr.Row(equal_height=False, variant="panel"):
                # Chat Column (70% width)
                with gr.Column(scale=7, min_width=0):
                    
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">💬 Chat</h3>
                        </div>
                    """)
                    
                    chatbot = gr.Chatbot(
                        value=self.initial_history,
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

                # Agent Logs Panel (30% width)
                with gr.Column(scale=3, min_width=0):
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">🛈 Agent Logs</h3>
                        </div>
                    """)
                    events_state = gr.State([])
                    events_panel = gr.HTML(
                        value="""
                        <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                            <div style="text-align: center; margin-top: 200px;">
                                <p>🤖 Agent events will appear here during chat</p>
                            </div>
                        </div>
                        """, 
                        elem_id="events-panel"
                    )
            
            
            # Event handler functions
            async def handle_streaming_chat(message, history, events):
                """Handle streaming chat and side-panel event rendering."""
                if not message.strip():
                    events_html = ""
                    yield history, events, events_html
                    return
                
                # Add user message to history and show it immediately
                history = history + [{"role": "user", "content": message}]
                # Clear events at the start of each new submit
                events = []
                yield history, events, ""
                
                # Initialize assistant message with animated thinking dots
                history = history + [{"role": "assistant", "content": '<span class="thinking-animation">●●●</span>'}]
                events_html = "".join([e.get("html", "") for e in events[-200:]])
                yield history, events, events_html
                
                try:
                    # Stream the response with events
                    async for partial_response, evt in self.agent.stream_chat_turn_with_events(self.current_user_id, message):
                        # Keep the thinking dots visible while streaming
                        history[-1] = {"role": "assistant", "content": partial_response}
                        # Append new event if any
                        if evt is not None:
                            events = events + [evt]
                        # Render compact HTML for events
                        events_html = "".join([e.get("html", "") for e in events[-200:]])
                        yield history, events, events_html
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    # Replace thinking indicator with error message
                    history[-1] = {"role": "assistant", "content": error_msg}
                    print(f"❌ Error during streaming chat: {e}", flush=True)
                    events_html = "".join([e.get("html", "") for e in events[-200:]])
                    yield history, events, events_html
            
            
            # Event handlers for chat - streaming version
            async def handle_submit_and_clear(message, history, events):
                """Handle message submission and immediately clear input."""
                async for h, ev, html in handle_streaming_chat(message, history, events):
                    yield h, ev, html, ""
            
            msg.submit(
                handle_submit_and_clear,
                [msg, chatbot, events_state],
                [chatbot, events_state, events_panel, msg],
                queue=True
            )
            
            send_btn.click(
                handle_submit_and_clear,
                [msg, chatbot, events_state],
                [chatbot, events_state, events_panel, msg],
                queue=True
            )
            
            # User switcher handlers via Tabs
            async def handle_tab_tyler():
                try:
                    history = await self.switch_user("Tyler")
                    # Reset events panel to initial state
                    initial_events_html = """
                    <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                        <div style="text-align: center; margin-top: 200px;">
                            <p>🤖 Agent events will appear here during chat</p>
                        </div>
                    </div>
                    """
                    return history, [], initial_events_html, ""
                except Exception as e:
                    print(f"Error switching to user Tyler: {e}")
                    return [], [], "", ""

            async def handle_tab_amanda():
                try:
                    history = await self.switch_user("Amanda")
                    # Reset events panel to initial state
                    initial_events_html = """
                    <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                        <div style="text-align: center; margin-top: 200px;">
                            <p>🤖 Agent events will appear here during chat</p>
                        </div>
                    </div>
                    """
                    return history, [], initial_events_html, ""
                except Exception as e:
                    print(f"Error switching to user Amanda: {e}")
                    return [], [], "", ""

            tyler_tab.select(
                handle_tab_tyler,
                outputs=[chatbot, events_state, events_panel, msg],
                queue=True
            )

            amanda_tab.select(
                handle_tab_amanda,
                outputs=[chatbot, events_state, events_panel, msg],
                queue=True
            )
            
            # Clear chat functionality
            async def handle_clear_chat():
                """Handle clearing chat history from Redis and UI."""
                cleared_history = await self.clear_chat_history()
                # Reset events panel to initial state
                initial_events_html = """
                <div style="height: 500px; padding: 15px; background: #163341; border-radius: 8px; color: #8A99A0; overflow-y: auto;">
                    <div style="text-align: center; margin-top: 200px;">
                        <p>🤖 Agent events will appear here during chat</p>
                    </div>
                </div>
                """
                return cleared_history, [], initial_events_html, ""
            
            clear_btn.click(
                handle_clear_chat,
                outputs=[chatbot, events_state, events_panel, msg],
                queue=True
            )
        
        return interface


async def create_app(config=None) -> gr.Interface:
    """Create and return the Gradio app."""
    if config is None:
        config = get_config()
    ui = TravelAgentUI(config=config)
    # Initialize chat history before creating interface
    await ui.initialize_chat_history()
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
        # Use asyncio to handle async app creation
        import asyncio
        app = asyncio.run(create_app(config))
        
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
