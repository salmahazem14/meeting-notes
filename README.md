# 🧠 Transcription to Notion & Jira Automation System

## 📌 Overview

This project takes a meeting or call transcription as input, then automatically:
- Generates a structured summary stored in a Notion database
- Extracts action items from the transcription
- Creates and assigns tasks in Jira with priorities and due dates

It helps teams turn conversations into **organized, trackable work** without manual effort.

---

## 🚀 Features

- 🎙️ **Transcription Processing**: Accepts raw meeting or call transcripts
- 📝 **Smart Summarization**: Generates a clean structured summary
- 📚 **Notion Integration**: Stores summaries in a Notion database
- 📊 **Task Extraction**: Identifies action items from conversations
- 🎯 **Jira Integration**: Automatically creates Jira tickets
- 👥 **Task Assignment**: Assigns tasks to team members
- ⏳ **Priority & Deadlines**: Adds priority levels and due dates
- 🔄 **Workflow Automation**: Converts unstructured meetings into structured workflows

---

## 🏗️ System Flow

1. Input: Meeting / call transcription  
2. AI Processing:
   - Summarization
   - Action item extraction  
3. Output:
   - Notion page (meeting summary)
   - Jira tickets (action items)

---

## 🧩 Tech Stack

- Node.js / Python (backend)
- Notion API
- Jira REST API
- LLM (for summarization & task extraction)
- Express / FastAPI (optional API layer)

---

## 📦 Example Output

### Notion Summary
- Meeting overview
- Key decisions
- Important discussion points
- Action items overview

### Jira Tasks
- Review authentication flow (High priority, due 2026-07-01)
- Update API documentation (Medium priority, due 2026-07-05)
- Fix notification duplication bug (High priority, due 2026-06-30)

---

## ⚙️ Environment Variables

```env
NOTION:
NOTION_API_KEY=your_notion_token
NOTION_DATABASE_ID=your_database_id
NOTION_CLIENT_ID
NOTION_CLIENT_SECRET
NOTION_API_KEY
NOTION_DATABASE_ID
NOTION_TASKS_DATABASE_ID
AUTH_URL

JIRA:
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your_email
JIRA_API_TOKEN=your_jira_token
JIRA_PROJECT_KEY=MP

LLM:
GROQ_API_KEY
