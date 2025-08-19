# 🌍 AI Travel Concierge - User-Partitioned Memory Demo

A sophisticated AI travel planning assistant built with AutoGen and Task-Centric Memory (TCM), featuring user-specific memory persistence and intelligent chat interface.

## 🚀 What This Demo Does

- **Smart Travel Chat**: Ask about destinations, flights, hotels, and activities
- **User-Partitioned Memory**: Each user gets their own persistent memory context using TCM
- **Cross-Session Learning**: Your preferences persist across different chat sessions
- **Real-time Search**: Gets current travel information from the web via Tavily
- **Split Interface**: Chat on the left, your personalized insights on the right

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your API Keys
You'll need two free API keys:
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [tavily.com](https://tavily.com) (free tier available)

### Step 2: No Additional Setup Required
The new TCM-based memory system stores user data locally in `./bank/users/{user_id}/` directories. No Redis installation needed!

### Step 3: Setup the Project
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repository-url>
cd amr-autogen-travel-agent

uv sync
```

### Step 4: Configure Your Environment
Create a `.env` file in the project directory (copy from `env.example`):
```bash
cp env.example .env
```

Edit the `.env` file with your actual API keys:
```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=your-tavily-api-key-here

# Optional Model Configuration (defaults shown)
TRAVEL_AGENT_MODEL_NAME=gpt-4o
MEMORY_MODEL_NAME=gpt-4o-mini
```

### Step 5: Run the Demo
```bash
uv run python gradio_app.py
```

The app will open at `http://localhost:7860` 🎉

The application will automatically:
- ✅ Validate your configuration and API keys
- ✅ Load with responsive UI and user-partitioned memory management
- ✅ Generate a unique User ID for your session (or you can set your own)

## 💬 Try These Examples

Once the app is running, try this user-partitioned memory demo:

**Session 1 (User ID: "tyler"):**
1. "Plan a 3-day San Francisco trip in October, under $250/night, near transit."
2. "Avoid red-eyes; prefer arrival by 8pm. I like boutique hotels near BART."

**Session 2 (User ID: "jane"):**
1. "I want to plan a luxury weekend in Paris. I prefer 5-star hotels and fine dining."

**Session 3 (User ID: "tyler" again):**
1. "Now plan a weekend in Seattle in November."

Notice how Tyler's preferences are remembered from Session 1, but Jane has a completely separate memory context!

## 🛠️ How It Works

**User-Partitioned TCM Architecture:**
- **Chat Interface**: Powered by Gradio with user session management
- **AI Agent**: Uses AutoGen framework with OpenAI GPT models
- **Memory**: Task-Centric Memory (TCM) with per-user isolation
- **Search**: Tavily provides real-time travel information
- **Persistence**: User insights stored in `./bank/users/{user_id}/`

**Key Files:**
- `gradio_app.py` - The main UI application with user session support
- `agent.py` - The travel AI agent with user-partitioned TCM
- `config.py` - Configuration management (no Redis needed!)
- `assets/styles.css` - UI styling and themes
- `example_usage.py` - Demo script showing user memory isolation
- `test_memory.py` - Test script for TCM functionality
- `env.example` - Example environment configuration
- `pyproject.toml` - Dependencies

**Memory System Features:**
- ✨ **User Isolation**: Each user_id gets separate memory context
- 🧠 **Cross-Session Persistence**: Insights survive app restarts
- 📚 **Automatic Learning**: Preferences extracted from conversations
- 🔍 **Relevant Retrieval**: Context-aware insight application
- 🗂️ **Clean Architecture**: UserCtx dataclass manages per-user state


---

**Start chatting and watch your travel preferences get smarter! 🧳✈️**
