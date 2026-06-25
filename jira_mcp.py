"""jira_mcp.py — MCP server exposing a tool to create Jira issues
with assignee, due date, priority, and description."""

from __future__ import annotations

import os
import json
import logging
from dotenv import load_dotenv
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JIRA_BASE_URL  = os.environ["JIRA_BASE_URL"]       # e.g. https://yourcompany.atlassian.net
JIRA_EMAIL     = os.environ["JIRA_EMAIL"]           # your Atlassian account email
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]       # API token from id.atlassian.com
JIRA_PROJECT   = os.environ["JIRA_PROJECT_KEY"]     # e.g. MEET


# ------------------------------------------------------------------ #
#  Jira REST helper
# ------------------------------------------------------------------ #

async def create_jira_issue(
    summary: str,
    description: str | None,
    assignee_email: str | None,
    due_date: str | None,
    priority: str = "Medium",
    issue_type: str = "Task",
) -> dict:
    """POST /rest/api/3/issue to create a Jira issue."""

    url = f"{JIRA_BASE_URL}/rest/api/3/issue"
    auth = (JIRA_EMAIL, JIRA_API_TOKEN)

    fields: dict = {
        "project":   {"key": JIRA_PROJECT},
        "summary":   summary,
        "issuetype": {"name": issue_type},
        "priority":  {"name": priority},
    }

    if description:
        fields["description"] = {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}]
                }
            ]
        }

    if assignee_email:
        # Jira Cloud requires accountId — first resolve it from email
        account_id = await resolve_account_id(assignee_email, auth)
        if account_id:
            fields["assignee"] = {"accountId": account_id}
        else:
            logger.warning("Could not resolve accountId for %s", assignee_email)

    if due_date:
        fields["duedate"] = due_date   # YYYY-MM-DD

    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"fields": fields},
            auth=auth,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()


async def resolve_account_id(email: str, auth: tuple) -> str | None:
    """Look up a Jira Cloud accountId by email."""
    url = f"{JIRA_BASE_URL}/rest/api/3/user/search"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            params={"query": email},
            auth=auth,
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if response.status_code == 200:
            users = response.json()
            if users:
                return users[0].get("accountId")
    return None


# ------------------------------------------------------------------ #
#  MCP Server
# ------------------------------------------------------------------ #

app = Server("jira-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="create_jira_task",
            description="Create a Jira task with summary, description, assignee email, due date, and priority.",
            inputSchema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Task title / summary"
                    },
                    "description": {
                        "type": ["string", "null"],
                        "description": "Additional context or notes for the task"
                    },
                    "assignee_email": {
                        "type": ["string", "null"],
                        "description": "Atlassian account email of the assignee"
                    },
                    "due_date": {
                        "type": ["string", "null"],
                        "description": "Due date in YYYY-MM-DD format"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Highest", "High", "Medium", "Low", "Lowest"],
                        "description": "Task priority (default: Medium)"
                    },
                    "issue_type": {
                        "type": "string",
                        "enum": ["Task", "Story", "Bug", "Subtask"],
                        "description": "Jira issue type (default: Task)"
                    },
                },
                "required": ["summary"],
            },
        ),
        Tool(
            name="create_jira_tasks_bulk",
            description="Create multiple Jira tasks at once from a list of action items.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "List of tasks to create",
                        "items": {
                            "type": "object",
                            "properties": {
                                "summary":        {"type": "string"},
                                "description":    {"type": ["string", "null"]},
                                "assignee_email": {"type": ["string", "null"]},
                                "due_date":       {"type": ["string", "null"]},
                                "priority":       {"type": ["string", "null"]},
                            },
                            "required": ["summary"],
                        }
                    }
                },
                "required": ["tasks"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    if name == "create_jira_task":
        try:
            result = await create_jira_issue(
                summary        = arguments["summary"],
                description    = arguments.get("description"),
                assignee_email = arguments.get("assignee_email"),
                due_date       = arguments.get("due_date"),
                priority       = arguments.get("priority", "Medium"),
                issue_type     = arguments.get("issue_type", "Task"),
            )
            issue_key = result.get("key", "unknown")
            issue_url = f"{JIRA_BASE_URL}/browse/{issue_key}"
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "issue_key": issue_key,
                    "url": issue_url,
                })
            )]
        except httpx.HTTPStatusError as exc:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": str(exc),
                "detail": exc.response.text,
            }))]
        except Exception as exc:
            return [TextContent(type="text", text=json.dumps({
                "success": False,
                "error": str(exc),
            }))]

    elif name == "create_jira_tasks_bulk":
        results = []
        for task in arguments.get("tasks", []):
            try:
                result = await create_jira_issue(
                    summary        = task["summary"],
                    description    = task.get("description"),
                    assignee_email = task.get("assignee_email"),
                    due_date       = task.get("due_date"),
                    priority       = task.get("priority", "Medium"),
                    issue_type     = task.get("issue_type", "Task"),
                )
                issue_key = result.get("key", "unknown")
                results.append({
                    "success":   True,
                    "summary":   task["summary"],
                    "issue_key": issue_key,
                    "url":       f"{JIRA_BASE_URL}/browse/{issue_key}",
                })
            except Exception as exc:
                results.append({
                    "success": False,
                    "summary": task.get("summary"),
                    "error":   str(exc),
                })

        return [TextContent(type="text", text=json.dumps(results, indent=2))]

    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())