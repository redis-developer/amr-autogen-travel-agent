# 🌍 AI Travel Concierge - Basic Demo

A simple AI travel planning assistant built with AutoGen and Redis, featuring an intelligent chat interface with memory capabilities.

## 🚀 What This Demo Does

- **Smart Travel Chat**: Ask about destinations, flights, hotels, and activities
- **Memory System**: Remembers your preferences across conversations using Redis
- **Real-time Search**: Gets current travel information from the web
- **Split Interface**: Chat on the left, your learned preferences on the right

## 🚀 Quick Setup (5 minutes)

### Step 1: Get Your API Keys
You'll need two free API keys:
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [tavily.com](https://tavily.com) (free tier available)

### Step 2: Install Redis
```bash
docker run -d -p 6379:6379 --name redis redis:8.0.3
```

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

# Optional Configuration (defaults shown)
REDIS_URL=redis://localhost:6379
```

### Step 5: Run the Demo
```bash
uv run python gradio_app.py
```

The app will open at `http://localhost:7860` 🎉

The application will automatically:
- ✅ Validate your configuration and API keys
- ✅ Check Redis connectivity
- ✅ Load with responsive UI and memory management

## 💬 Try These Examples

Once the app is running, try asking:

TBD

## 🛠️ How It Works

**Simple Architecture:**
- **Chat Interface**: Powered by Gradio for the UI
- **AI Agent**: Uses AutoGen framework with OpenAI GPT
- **Memory**: Redis stores your learned preferences
- **Search**: Tavily provides real-time travel information

**Key Files:**
- `gradio_app.py` - The main UI application
- `agent.py` - The travel AI agent with memory and tools
- `config.py` - Configuration management with validation
- `ui_utils.py` - UI utility functions
- `assets/styles.css` - UI styling and themes
- `env.example` - Example environment configuration
- `pyproject.toml` - Dependencies


---

**Start chatting and watch your travel preferences get smarter! 🧳✈️**
