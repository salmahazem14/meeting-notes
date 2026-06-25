# 🧠 Transcription to Notion & Jira Automation System

## 📌 Overview

This project processes meeting or call transcriptions and automatically transforms them into structured knowledge and actionable tasks.

It:
- Generates a high-quality summary stored in a Notion database
- Extracts actionable tasks from conversations
- Creates and assigns Jira tickets with priorities and due dates

The system is designed to eliminate manual note-taking and turn unstructured conversations into structured, trackable workflows.

---

## 🚀 Features

- 🎙️ **Transcription Processing**: Handles raw meeting or call transcripts
- 🧠 **High-Accuracy Summarization Pipeline**
- 📚 **Notion Integration**: Stores structured summaries in a database
- 📊 **LLM-Based Action Item Extraction**
- 🎯 **Jira Integration**: Automatically creates issues in Jira
- 👥 **Task Assignment**: Assigns tasks to relevant owners
- ⏳ **Priority & Due Dates**: Automatically enriches tasks with metadata
- 🔄 **End-to-End Automation**

---

## 🧠 Summarization Technique

The summarization pipeline uses a **chunking strategy with overlapping segments**, followed by a **Map-Reduce summarization approach**:

### 1. Chunking with Overlap
- The transcription is split into smaller chunks
- Overlapping context is preserved between chunks to avoid information loss
- This ensures continuity across long conversations

### 2. Map Phase
- Each chunk is independently summarized using an LLM
- Focus is on extracting key ideas, decisions, and important context

### 3. Reduce Phase
- All chunk summaries are merged
- A final LLM pass produces a coherent global summary

### 4. Evaluation
- Tested against multiple summarization strategies
- Achieved the highest accuracy and coherence score
- Evaluated using an **LLM-as-a-Judge framework**, assessing:
  - Factual consistency
  - Coverage of key points
  - Structural clarity

---

## 🧠 Task Extraction & Jira Generation

After the summarization phase, the system performs **LLM-based action item extraction**:

### 🔹 Action Item Extraction
- The LLM analyzes the summarized content (and relevant transcript context)
- It identifies actionable tasks mentioned or implied in the conversation
- Each task is structured with:
  - Clear description
  - Assigned owner
  - Priority level
  - Due date (if available or inferred)

### 🔹 Jira Task Creation
Once extracted, action items are automatically converted into Jira issues:

- Each task is created under the configured Jira project (`JIRA_PROJECT_KEY`)
- Metadata is attached:
  - Summary
  - Description
  - Assignee
  - Priority
  - Due date

---

## 🏗️ System Flow

1. Input: Raw transcription  
2. Summarization:
   - Chunking with overlap  
   - Map-Reduce summarization  
3. Action Item Extraction:
   - LLM extracts tasks from summarized content  
4. Jira Integration:
   - Tasks are converted into Jira issues  
5. Notion Integration:
   - Final summary stored in Notion database  

---

## 🧩 Tech Stack

- Node.js / Python (backend)
- Notion API
- Jira REST API
- LLM (summarization + extraction)
- Map-Reduce prompting architecture

---

## 📦 Example Output

### 📚 Notion Summary
- Meeting overview
- Key decisions
- Discussion highlights
- Action items overview

### 🎯 Jira Tasks
- Review authentication flow (High priority, due 2026-07-01)
- Update API documentation (Medium priority, due 2026-07-05)
- Fix notification duplication bug (High priority, due 2026-06-30)

---

## ⚙️ Environment Variables

```env
NOTION_API_KEY=your_notion_token
NOTION_DATABASE_ID=your_database_id

JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your_email
JIRA_API_TOKEN=your_jira_token
JIRA_PROJECT_KEY=MP
