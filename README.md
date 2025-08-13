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

### Step 4: Set Your API Keys
```bash
# Set your API keys (replace with your actual keys)
export OPENAI_API_KEY="your-openai-api-key-here"
export TAVILY_API_KEY="your-tavily-api-key-here"
```

### Step 5: Run the Demo
```bash
python gradio_app.py
```

The app will open at `http://localhost:7860` 🎉

## 💬 Try These Examples

Once the app is running, try asking:

**Basic Travel Questions:**
- "I want to plan a week-long trip to Japan this fall"
- "Help me find flights from New York to Paris for 2 people"
- "What are the best restaurants in Tokyo for someone with food allergies?"

**Watch the Memory Panel:**
- As you chat, your preferences will appear on the right side
- The AI learns your budget, travel style, dietary needs, etc.
- This memory persists across conversations

## 🛠️ How It Works

**Simple Architecture:**
- **Chat Interface**: Powered by Gradio for the UI
- **AI Agent**: Uses AutoGen framework with OpenAI GPT
- **Memory**: Redis stores your learned preferences
- **Search**: Tavily provides real-time travel information

**Key Files:**
- `gradio_app.py` - The main UI application
- `agent.py` - The travel AI agent with memory
- `pyproject.toml` - Dependencies

## 🔍 Troubleshooting

**App won't start?**
- Check Redis is running: `docker ps` or `redis-cli ping`
- Verify API keys are set: `echo $OPENAI_API_KEY`

**No preferences showing?**
- Make sure Redis is connected
- Try asking travel questions with specific preferences

**Import errors?**
- Run `uv sync` to reinstall dependencies
- Make sure you're in the project directory


---

**Start chatting and watch your travel preferences get smarter! 🧳✈️**
