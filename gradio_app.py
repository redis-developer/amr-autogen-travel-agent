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
        self.agent = TravelAgent(config=self.config)
        self.current_user_id = None  # Currently selected user
    
    async def chat_with_agent(self, message: str) -> List[dict]:
        """
        Handle chat interaction with the travel agent.
        
        Args:
            message: User's input message
            
        Returns:
            Complete chat history from Redis for the current user
        """
        # Ensure we have a selected user
        if not self.current_user_id:
            return [{"role": "assistant", "content": "Please select a user first."}]
        
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
    
    def create_new_user(self, user_id_input: str) -> Tuple[gr.update, List[dict], str]:
        """
        Create a new user and switch to them.
        
        Args:
            user_id_input: User ID entered by user (can be empty)
            
        Returns:
            Tuple of (updated_user_list, empty_chat_history, status_message)
        """
        # Use provided user ID or generate a new one
        if user_id_input.strip():
            new_user_id = user_id_input.strip()
        else:
            # Generate a UUID for anonymous users
            new_user_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        
        # Check if user already exists in the agent
        if self.agent.user_exists(new_user_id):
            return gr.update(), [], f"❌ User '{new_user_id}' already exists. Please choose a different ID."
        
        # Create new user by making the agent initialize context for them
        self.current_user_id = new_user_id
        # Actually create the user context in the agent
        self.agent._get_or_create_user_ctx(new_user_id)
        
        # Update user list from agent (should now include the new user)
        user_choices = self.agent.get_user_list()
        
        return gr.update(choices=user_choices, value=new_user_id), [], f"✅ Created and switched to user: {new_user_id}"
    
    async def switch_user(self, selected_user_id: str) -> Tuple[List[dict], str]:
        """
        Switch to a different user.
        
        Args:
            selected_user_id: The user ID to switch to
            
        Returns:
            Tuple of (chat_history_for_user, status_message)
        """
        if not selected_user_id or not self.agent.user_exists(selected_user_id):
            return [], "❌ Invalid user selection."
        
        self.current_user_id = selected_user_id
        
        # Load chat history from Redis
        try:
            chat_history = await self.agent.get_chat_history(selected_user_id, n=-1)
        except Exception as e:
            print(f"Error loading chat history for user {selected_user_id}: {e}")
            chat_history = []
        
        return chat_history, f"✅ Switched to user: {selected_user_id}"
    
    def delete_user(self, user_to_delete: str) -> Tuple[gr.update, List[dict], str]:
        """
        Delete a user and their chat history.
        
        Args:
            user_to_delete: The user ID to delete
            
        Returns:
            Tuple of (updated_user_list, empty_chat_history, status_message)
        """
        if not user_to_delete or not self.agent.user_exists(user_to_delete):
            return gr.update(), [], "❌ Invalid user to delete."
        
        # Don't allow deleting the currently selected user
        if user_to_delete == self.current_user_id:
            return gr.update(), [], "❌ Cannot delete currently selected user. Switch to another user first."
        
        # Delete user from agent (this clears their memory and context)
        self.agent.reset_user_memory(user_to_delete)
        
        # Update user list
        user_choices = self.agent.get_user_list()
        new_value = self.current_user_id if self.current_user_id in user_choices else (user_choices[0] if user_choices else None)
        
        return gr.update(choices=user_choices, value=new_value), [], f"✅ Deleted user: {user_to_delete}"
    
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
                        Create users, switch between them, and maintain separate travel conversations!
                    </p>
                </div>
            """)
            
            # Main two-column layout
            with gr.Row():
                # Left Column - User Management
                with gr.Column(scale=1, min_width=300):
                    gr.HTML("""
                        <div style="text-align: center; margin-bottom: 15px; padding: 15px; background: linear-gradient(135deg, #163341 0%, #091A23 100%); border-radius: 8px;">
                            <h3 style="color: #DCFF1E; margin: 0; font-size: 16px;">👥 User Management</h3>
                        </div>
                    """)
                    
                    # Create new user section
                    with gr.Group():
                        gr.HTML("<h4 style='color: #FFFFFF; margin: 10px 0;'>Create New User</h4>")
                        with gr.Row():
                            new_user_input = gr.Textbox(
                                placeholder="Enter User ID (optional)",
                                show_label=False,
                                scale=3,
                                container=False
                            )
                            create_user_btn = gr.Button("Create", variant="primary", scale=1)
                    
                    # User list section
                    with gr.Group():
                        gr.HTML("<h4 style='color: #FFFFFF; margin: 10px 0;'>Select User</h4>")
                        user_list = gr.Dropdown(
                            choices=[],
                            value=None,
                            label="Active Users",
                            interactive=True,
                            allow_custom_value=False
                        )
                        with gr.Row():
                            delete_user_btn = gr.Button("Delete User", variant="secondary", size="sm")
                    
                    # Status display
                    status_display = gr.HTML(
                        value="""
                        <div style="text-align: center; color: #8A99A0; font-size: 14px; margin-top: 15px; padding: 10px; background: #091A23; border-radius: 6px;">
                            👆 Create a new user to get started
                        </div>
                        """,
                        visible=True
                    )
                
                # Right Column - Chat Interface
                with gr.Column(scale=2):
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
                            placeholder="Select a user first, then ask me about planning your next trip...",
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
                        current_user_display = gr.HTML(
                            value="<div style='color: #8A99A0; font-size: 12px; text-align: center; margin-top: 5px;'>No user selected</div>"
                        )    
            
            
            # Event handler functions
            async def handle_chat_start(message, history):
                """Show typing indicator when chat starts."""
                if not message.strip():
                    return "", history
                
                # Add user message and typing indicator immediately
                updated_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "Thinking..."}
                ]
                return "", updated_history
            
            async def handle_chat_complete(message, history):
                """Complete the chat interaction."""
                
                # Extract the actual message from history since the message parameter gets cleared
                if not history or len(history) < 2:
                    return history
                
                # Get the user message from history (should be second to last, before "Thinking...")
                user_message = None
                if history[-1]["content"] == "Thinking..." and len(history) >= 2:
                    user_message = history[-2]["content"]
                
                if not user_message:
                    return history
                
                # Send message to agent and get updated history from Redis
                updated_history = await self.chat_with_agent(user_message)
                return updated_history
            
            def handle_create_user(user_id_input):
                """Handle creating a new user."""
                updated_user_list, empty_chat, status_msg = self.create_new_user(user_id_input)
                
                # Update placeholder based on success
                chat_enabled = "✅" in status_msg
                current_user_msg = f"<div style='color: #DCFF1E; font-size: 12px; text-align: center; margin-top: 5px;'>Current: {self.current_user_id}</div>" if self.current_user_id else "<div style='color: #8A99A0; font-size: 12px; text-align: center; margin-top: 5px;'>No user selected</div>"
                
                return (
                    updated_user_list,  # user_list
                    empty_chat,  # chatbot
                    status_msg,  # status_display
                    "",  # clear new_user_input
                    current_user_msg  # current_user_display
                )
            
            async def handle_switch_user(selected_user_id):
                """Handle switching to a different user."""
                chat_history, status_msg = await self.switch_user(selected_user_id)
                
                # Update display based on success
                current_user_msg = f"<div style='color: #DCFF1E; font-size: 12px; text-align: center; margin-top: 5px;'>Current: {self.current_user_id}</div>" if self.current_user_id else "<div style='color: #8A99A0; font-size: 12px; text-align: center; margin-top: 5px;'>No user selected</div>"
                
                return (
                    chat_history,  # chatbot
                    status_msg,  # status_display
                    current_user_msg  # current_user_display
                )
            
            def handle_delete_user(selected_user_id):
                """Handle deleting a user."""
                updated_user_list, empty_chat, status_msg = self.delete_user(selected_user_id)
                
                # Update display based on current user
                current_user_msg = f"<div style='color: #DCFF1E; font-size: 12px; text-align: center; margin-top: 5px;'>Current: {self.current_user_id}</div>" if self.current_user_id else "<div style='color: #8A99A0; font-size: 12px; text-align: center; margin-top: 5px;'>No user selected</div>"
                
                return (
                    updated_user_list,  # user_list
                    status_msg,  # status_display
                    current_user_msg  # current_user_display
                )
            
            # Event handlers for chat
            msg.submit(
                handle_chat_start,
                [msg, chatbot],
                [msg, chatbot],
                queue=True
            ).then(
                handle_chat_complete,
                [msg, chatbot],
                [chatbot],
                queue=True
            )
            
            send_btn.click(
                handle_chat_start,
                [msg, chatbot],
                [msg, chatbot],
                queue=True
            ).then(
                handle_chat_complete,
                [msg, chatbot],
                [chatbot],
                queue=True
            )
            
            # User management handlers
            create_user_btn.click(
                handle_create_user,
                inputs=[new_user_input],
                outputs=[
                    user_list,
                    chatbot,
                    status_display,
                    new_user_input,
                    current_user_display
                ],
                queue=False
            )
            
            # Allow Enter key in new user input to create user
            new_user_input.submit(
                handle_create_user,
                inputs=[new_user_input],
                outputs=[
                    user_list,
                    chatbot,
                    status_display,
                    new_user_input,
                    current_user_display
                ],
                queue=False
            )
            
            # User selection handler
            user_list.change(
                handle_switch_user,
                inputs=[user_list],
                outputs=[
                    chatbot,
                    status_display,
                    current_user_display
                ],
                queue=False
            )
            
            # Delete user handler
            delete_user_btn.click(
                handle_delete_user,
                inputs=[user_list],
                outputs=[
                    user_list,
                    status_display,
                    current_user_display
                ],
                queue=False
            )
            
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
