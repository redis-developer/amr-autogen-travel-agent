# 🌍 Autogen Travel Concierge 

A sophisticated AI travel planning assistant showcasing AutoGen's advanced memory capabilities with dual-layer memory architecture: Redis-backed chat history and Task-Centric Memory (TCM) for intelligent learning.

## 🧠 What's included?

- **🎯 Dual-Layer Memory**: Short-term chat history (Redis) + Long-term learning (TCM)
- **👥 User Isolation**: Each user gets completely separate memory contexts
- **🔄 Session Persistence**: Your conversations and preferences survive app restarts
- **📚 Intelligent Learning**: The agent learns your travel preferences automatically
- **🌐 Real-time Search**: Live travel information via Tavily search API
- **💬 Clean Chat UI**: Gradio interface with user management

## 🚀 Quick Setup (<5 minutes)

### Step 1: Get Your API Keys
You'll need three API keys:
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [tavily.com](https://tavily.com) (free tier available)  
- **Redis URL**: For chat history storage (see Step 2)

### Step 2: Set Up Redis
You have 3 options for Redis:

#### Option A: Local Redis with Docker**
```bash
docker run --name redis -p 6379:6379 -d redis:8.0.3
```

#### Option B: Redis Cloud
Get a free db [here](https://redis.io/cloud).

#### Option C: Azure Managed Redis
Here's a quickstart guide to create Azure Managed Redis for as low as $12 monthly: https://learn.microsoft.com/en-us/azure/redis/quickstart-create-managed-redis

### Step 3: Setup the Project
```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repository-url>
cd amr-autogen-travel-agent

# Get uv env setup
uv sync
```

### Step 4: Configure Your Environment
Create a `.env` file with your API keys:
```bash
cp env.example .env
```

Edit the `.env` file as needed.

### Step 5: Launch the Application
```bash
uv run python gradio_app.py
```

🎉 **Open [http://localhost:7860](http://localhost:7860)** to start chatting!

The application will:
- ✅ Validate your configuration and API connections
- ✅ Initialize the dual-layer memory system
- ✅ Load the user management interface

## 💬 Demo: Multi-User Memory Isolation

Test the dual-layer memory system with multiple users:

### 👤 User: "Tyler"
1. **Create user** "Tyler" in the UI
2. **Chat**: "Plan a 3-day San Francisco trip in October, under $250/night, near transit."
3. **Chat**: "I prefer boutique hotels and hate red-eye flights."
4. **Switch away** to another user, then **switch back** to Tyler
5. **Chat**: "Now plan a weekend in Seattle." 
   
   ✨ *Notice: Tyler's preferences are remembered!*

### 👤 User: "Jane"  
1. **Create user** "Jane" in the UI
2. **Chat**: "I want a luxury Paris weekend with 5-star hotels and fine dining."

   ✨ *Notice: Jane has completely separate preferences from Tyler!*

### 🔄 Test Persistence
- **Restart the app** completely
- **Switch back to Tyler** - his conversation history and preferences persist
- **Switch to Jane** - her separate context is also preserved

## 🏗️ Architecture: Dual-Layer Memory System

This demo showcases AutoGen's advanced memory capabilities with a sophisticated dual-layer architecture:

### 🧠 Memory Layers

**Layer 1: Short-Term Memory (Redis)**
- **Purpose**: Complete conversation history storage
- **Technology**: Redis-backed `ChatCompletionContext`
- **Scope**: Per-user chat sessions with full message history
- **Persistence**: Survives app restarts, immediate retrieval

**Layer 2: Long-Term Memory (TCM)**  -- TO BE PORTED TO REDIS BACKEND
- **Purpose**: Intelligent preference learning and insights
- **Technology**: AutoGen's Task-Centric Memory with ChromaDB
- **Scope**: Cross-conversation learning and preference extraction
- **Persistence**: Permanent insight storage in `./memory/users/{user_id}/`

### 👥 User Context Management

Each user gets completely isolated contexts:

```python
# Per-User Context Structure
UserCtx:
  ├── Redis Chat History (short-term)
  ├── TCM Memory Controller (long-term learning)  
  ├── Teachability Adapter (auto-learning)
  └── Supervisor Agent (bound to user's memory)
```

**Flow**: `User Message → Agent → Redis Storage → TCM Learning → Personalized Response`

### 🔧 Key Components

- **`agent.py`**: Core travel agent with dual-layer memory management
- **`gradio_app.py`**: UI with user session management and chat interface
- **`memory/redis_chat_completion_context.py`**: Custom Redis-backed chat context
- **`config.py`**: Configuration management for Redis and OpenAI
- **`memory/users/{user_id}/`**: Per-user TCM storage directories

---

**🚀 Ready to see AI memory in action? Start chatting and watch your travel preferences get smarter!**
