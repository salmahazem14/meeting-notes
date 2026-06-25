"""graph.py — LangGraph simulation connecting notetaker_node and action_tasks_node."""

import asyncio
import json
import logging
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any

from notetaker_node_mapreduce import notetaker_node
from task_organizer import action_tasks_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  State schema
# ------------------------------------------------------------------ #

class MeetingState(TypedDict, total=False):
    meeting_id: str
    transcript: Any
    recipient_emails: list
    notes: dict
    tasks: list
    error_log: list


# ------------------------------------------------------------------ #
#  Build graph
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    graph = StateGraph(MeetingState)

    graph.add_node("notetaker_node", notetaker_node)
    graph.add_node("action_tasks_node", action_tasks_node)

    graph.set_entry_point("notetaker_node")
    graph.add_edge("notetaker_node", "action_tasks_node")
    graph.add_edge("action_tasks_node", END)

    return graph.compile()


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #

async def main():
    app = build_graph()

    initial_state: MeetingState = {
        "meeting_id": "frontend-sprint-planning-2024-09-02",
        "recipient_emails": [
            "ali.hassan@devteam.com",
            "maya.saleh@devteam.com",
            "tamer.nour@devteam.com",
            "rania.khaled@devteam.com",
        ],
        "transcript": [
            {"speaker": {"name": "Ali Hassan"}, "text": "Good morning everyone. Today we are planning the frontend sprint for September. We need to finalize what goes into this sprint and who owns what.", "started_at": {"absolute_timestamp": "2024-09-02T09:00:00Z"}},
            {"speaker": {"name": "Maya Saleh"}, "text": "I reviewed the backlog. The highest priority items are the login page redesign and fixing the mobile navbar bug.", "started_at": {"absolute_timestamp": "2024-09-02T09:01:10Z"}},
            {"speaker": {"name": "Tamer Nour"}, "text": "I can take the login page redesign. I already have the Figma mockups from Rania so I can start immediately.", "started_at": {"absolute_timestamp": "2024-09-02T09:02:20Z"}},
            {"speaker": {"name": "Rania Khaled"}, "text": "Yes the mockups are ready. Tamer please make sure the new design matches the updated color tokens we agreed on last week.", "started_at": {"absolute_timestamp": "2024-09-02T09:03:30Z"}},
            {"speaker": {"name": "Ali Hassan"}, "text": "Good. Tamer can you finish the login redesign by September 10th?", "started_at": {"absolute_timestamp": "2024-09-02T09:04:40Z"}},
            {"speaker": {"name": "Tamer Nour"}, "text": "Yes that should be fine.", "started_at": {"absolute_timestamp": "2024-09-02T09:05:00Z"}},
            {"speaker": {"name": "Maya Saleh"}, "text": "I will handle the mobile navbar bug. It is affecting users on iOS mostly. I will investigate and push a fix by September 8th.", "started_at": {"absolute_timestamp": "2024-09-02T09:06:10Z"}},
            {"speaker": {"name": "Ali Hassan"}, "text": "Perfect. Rania can you update the design system documentation to reflect the new color tokens?", "started_at": {"absolute_timestamp": "2024-09-02T09:07:20Z"}},
            {"speaker": {"name": "Rania Khaled"}, "text": "Sure I will update the documentation and share it with the team by September 6th.", "started_at": {"absolute_timestamp": "2024-09-02T09:08:30Z"}},
            {"speaker": {"name": "Maya Saleh"}, "text": "We also need to set up the new component library structure we discussed. Should that go into this sprint?", "started_at": {"absolute_timestamp": "2024-09-02T09:09:40Z"}},
            {"speaker": {"name": "Ali Hassan"}, "text": "Yes let's include it. Tamer after the login redesign can you start on the component library setup?", "started_at": {"absolute_timestamp": "2024-09-02T09:10:50Z"}},
            {"speaker": {"name": "Tamer Nour"}, "text": "I can start on it in parallel actually. I will create the initial folder structure and conventions document.", "started_at": {"absolute_timestamp": "2024-09-02T09:12:00Z"}},
            {"speaker": {"name": "Ali Hassan"}, "text": "Great. Let's wrap up. To summarize: Tamer handles login redesign by September 10th and component library setup, Maya fixes the mobile navbar by September 8th, and Rania updates the design system docs by September 6th. Any questions?", "started_at": {"absolute_timestamp": "2024-09-02T09:13:10Z"}},
            {"speaker": {"name": "Maya Saleh"}, "text": "No that is clear. Let's get started.", "started_at": {"absolute_timestamp": "2024-09-02T09:14:00Z"}},
        ]
    }

    logger.info("Starting graph run for meeting: %s", initial_state["meeting_id"])
    final_state = await app.ainvoke(initial_state)

    print("\n========== FINAL STATE ==========")
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    asyncio.run(main())