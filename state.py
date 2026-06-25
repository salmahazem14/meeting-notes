"""LangGraph state definition for the meeting pipeline."""

from typing import TypedDict


class MeetingState(TypedDict):
    meeting_id: str
    transcript: dict          
    notes: dict               # structured output: key_decisions, action_items, discussion_points
    error_log: list[str]
    recipient_emails : list[str]
