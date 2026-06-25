
"""action_tasks.py — Takes action_items from notetaker output,
parses each one with an LLM to extract task, assignee, and due date,
then creates Jira tasks via the local Jira MCP server."""
 
from __future__ import annotations
 
import json
import logging
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
 
from dotenv import load_dotenv
load_dotenv()
 
logger = logging.getLogger(__name__)
 
 
# ------------------------------------------------------------------ #
#  Resolve assignee name → email from recipient_emails list
# ------------------------------------------------------------------ #
 
def resolve_email_from_recipients(name: str | None, recipient_emails: list[str]) -> str | None:
    """Match an assignee name to an email from the recipient list.
    Matches on first name, last name, or full name (case-insensitive)."""
    if not name or name.lower() == "unassigned":
        return None
 
    name_lower = name.lower().strip()
 
    for email in recipient_emails:
        local = email.split("@")[0].lower()
        parts = local.replace(".", " ").replace("_", " ").replace("-", " ").split()
        name_parts = name_lower.split()
        for np in name_parts:
            if np in parts or any(np in p for p in parts):
                return email
 
    return None
 
 
# ------------------------------------------------------------------ #
#  1. Parse action items with LLM
# ------------------------------------------------------------------ #
 
def parse_action_items(llm: ChatGroq, action_items: list[str]) -> list[dict]:
    """Send all action items to LLM and extract structured task data."""
 
    prompt = f"""You are a project management assistant.
Below is a list of action items extracted from a meeting transcript.
For each action item, extract:
- "task": a clean, concise task title
- "assignee_name": the full name of the person responsible if mentioned, otherwise "Unassigned"
- "due_date": due date in YYYY-MM-DD format if mentioned, otherwise null
- "priority": one of "Highest", "High", "Medium", "Low", "Lowest" based on urgency, default "Medium"
- "notes": any additional context worth keeping, otherwise null
 
Return ONLY a valid JSON array, no markdown, no backticks.
[
  {{
    "task": "...",
    "assignee_name": "...",
    "due_date": "...",
    "priority": "...",
    "notes": "..."
  }}
]
 
Action items:
{json.dumps(action_items, indent=2)}
"""
 
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    logger.info("Parsed %d action items from LLM", len(parsed))
    return parsed
 
 
# ------------------------------------------------------------------ #
#  2. Main node
# ------------------------------------------------------------------ #
 
async def action_tasks_node(state: dict) -> dict:
    """LangGraph node: parses action items and creates Jira tasks via MCP."""
 
    notes            = state.get("notes", {})
    action_items     = notes.get("action_items", [])
    meeting_id       = state.get("meeting_id", "unknown-meeting")
    recipient_emails: list[str] = state.get("recipient_emails", [])
 
    if not action_items:
        logger.info("No action items found.")
        return {"tasks": []}
 
    logger.info("Processing %d action items for meeting: %s", len(action_items), meeting_id)
    logger.info("Recipient emails available: %s", recipient_emails)
 
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
 
    try:
        parsed_tasks = parse_action_items(llm, action_items)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Failed to parse action items from LLM: %s", exc)
        return {"tasks": [], "error_log": [str(exc)]}
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {"tasks": [], "error_log": [str(exc)]}
 
    bulk_tasks = []
    for task in parsed_tasks:
        assignee_name  = task.get("assignee_name", "Unassigned")
        assignee_email = resolve_email_from_recipients(assignee_name, recipient_emails)
 
        if assignee_email:
            logger.info("Resolved '%s' -> %s", assignee_name, assignee_email)
        else:
            logger.warning("No email match for '%s', task will be unassigned", assignee_name)
 
        notes_text = task.get("notes") or ""
        bulk_tasks.append({
            "summary":        task.get("task", "Untitled Task"),
            "description":    f"Meeting: {meeting_id}\nAssignee: {assignee_name}\n\n{notes_text}".strip(),
            "assignee_email": assignee_email,
            "due_date":       task.get("due_date"),
            "priority":       task.get("priority", "Medium"),
            "issue_type":     "Task",
        })
 
    # ------------------------------------------------------------------ #
    #  Send to Jira via MCP
    # ------------------------------------------------------------------ #
    created_tasks = []
 
    try:
        jira_mcp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jira_mcp.py")
 
        server_params = StdioServerParameters(
            command="python",
            args=[jira_mcp_path],
            env={
                **os.environ,
                "JIRA_BASE_URL":    os.environ["JIRA_BASE_URL"],
                "JIRA_EMAIL":       os.environ["JIRA_EMAIL"],
                "JIRA_API_TOKEN":   os.environ["JIRA_API_TOKEN"],
                "JIRA_PROJECT_KEY": os.environ["JIRA_PROJECT_KEY"],
            }
        )
 
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Jira MCP connected successfully")
 
                result = await session.call_tool(
                    "create_jira_tasks_bulk",
                    arguments={"tasks": bulk_tasks}
                )
 
                # debug: print full result
                logger.info("MCP result type: %s", type(result))
                logger.info("MCP result content: %s", result.content)
 
                # collect all text blocks
                raw_result = "".join(
                    block.text for block in (result.content or [])
                    if hasattr(block, "text") and block.text
                ).strip()
 
                logger.info("Raw result text: '%s'", raw_result)
 
                if not raw_result:
                    logger.error("MCP returned empty response — check jira_mcp.py for errors")
                    return {"tasks": [], "error_log": ["MCP returned empty response"]}
 
                raw_result = raw_result.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                results = json.loads(raw_result)
 
                for r, task in zip(results, parsed_tasks):
                    if r.get("success"):
                        logger.info("Created Jira task: %s -> %s (%s)",
                                    r["summary"], r["issue_key"], r["url"])
                        created_tasks.append({
                            "task":          r["summary"],
                            "issue_key":     r["issue_key"],
                            "url":           r["url"],
                            "assignee_name": task.get("assignee_name", "Unassigned"),
                            "due_date":      task.get("due_date"),
                            "priority":      task.get("priority", "Medium"),
                            "notes":         task.get("notes"),
                        })
                    else:
                        logger.warning("Failed to create task '%s': %s",
                                       r.get("summary"), r.get("error"))
 
    except Exception as exc:
        import traceback
        logger.warning("Jira MCP call failed (non-fatal): %s", exc)
        traceback.print_exc()
 
    logger.info("Done. %d/%d tasks created in Jira.", len(created_tasks), len(parsed_tasks))
    return {"tasks": created_tasks}

# ------------------------------------------------------------------ #
#  Run standalone
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import asyncio

    sample_state = {
        "meeting_id": "quarterly-product-engineering-review-2024-08-05",
        "recipient_emails": [
            "layla.hassan@company.com",
            "omar.sharaf@company.com",
            "salma.ahmed@company.com",
            "mona.youssef@company.com",
            "karim.bassem@company.com",
            "ahmed.mostafa@company.com",
        ],
        "notes": {
            "action_items": [
                "layla will improve the authentication flow by reviewing security gaps and implementing stronger authorization checks. This task is High priority, due date on 2026-07-01, and is currently In Progress."
            ]
        }
    }

    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(action_tasks_node(sample_state))
    print(json.dumps(result, indent=2))