"""Notetaker agent node — LLM summarizes transcript in chunks,
then merges results and writes to Notion via MCP stdio transport."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp.client.session import ClientSession

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent / "notetaker.txt"
CHUNK_TOKEN_LIMIT = 3000
CHARS_PER_TOKEN = 4  # rough estimate: 1 token ≈ 4 chars


def chunk_transcript(transcript: str, max_tokens: int = CHUNK_TOKEN_LIMIT, overlap_lines: int = 5) -> list[str]:
    """Split transcript into chunks of ~max_tokens each with overlapping lines at boundaries."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    lines = transcript.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0
 
    for line in lines:
        line_len = len(line)
        if current_len + line_len > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            # carry over last `overlap_lines` lines into the next chunk
            overlap = current_chunk[-overlap_lines:]
            current_chunk = overlap + [line]
            current_len = sum(len(l) for l in current_chunk)
        else:
            current_chunk.append(line)
            current_len += line_len
 
    if current_chunk:
        chunks.append("\n".join(current_chunk))
 
    return chunks
 


def merge_summaries(summaries: list[dict]) -> dict:
    """Merge multiple chunk summaries into one final summary."""
    merged_decisions = []
    merged_actions = []
    merged_discussion = []
    seen_decisions = set()
    seen_actions = set()
    seen_headers = {}

    for summary in summaries:
        for d in summary.get("key_decisions", []):
            if d.lower() not in seen_decisions:
                seen_decisions.add(d.lower())
                merged_decisions.append(d)

        for a in summary.get("action_items", []):
            if a.lower() not in seen_actions:
                seen_actions.add(a.lower())
                merged_actions.append(a)

        for section in summary.get("discussion_points", []):
            header = section.get("header", "")
            notes = section.get("notes", [])
            if header in seen_headers:
                existing_notes = seen_headers[header]
                for note in notes:
                    if note not in existing_notes:
                        existing_notes.append(note)
            else:
                seen_headers[header] = list(notes)
                merged_discussion.append({"header": header, "notes": seen_headers[header]})

    return {
        "key_decisions": merged_decisions,
        "action_items": merged_actions,
        "discussion_points": merged_discussion,
    }


async def summarize_chunk(llm: ChatGroq, prompt_template: str, chunk: str, chunk_idx: int) -> dict:
    """Send a single chunk to the LLM and return parsed JSON."""
    response = llm.invoke([
        HumanMessage(content=prompt_template.replace("{transcript}", chunk))
    ])
    raw = response.content
    clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(clean)
    logger.info("Chunk %d summarized: %d decisions, %d actions, %d topics",
                chunk_idx,
                len(data.get("key_decisions", [])),
                len(data.get("action_items", [])),
                len(data.get("discussion_points", [])))
    return data


async def notetaker_node(state: dict) -> dict:
    """LangGraph node: LLM summarizes transcript in chunks, then writes to Notion via MCP."""

    transcript = state.get("transcript", "")
    if not transcript:
        logger.info("Notetaker: empty transcript.")
        return {"notes": {}}

    if isinstance(transcript, list):
        transcript = "\n".join(
            f"{entry['speaker']['name']}: {entry['text']}"
            for entry in transcript
        )
    transcript = transcript.strip()

    total_tokens = len(transcript) // CHARS_PER_TOKEN
    logger.info("Transcript length: %d chars (~%d tokens)", len(transcript), total_tokens)

    # ------------------------------------------------------------------ #
    #  1. Load prompt
    # ------------------------------------------------------------------ #
    try:
        prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        msg = f"Prompt file not found: {PROMPT_PATH}"
        logger.error(msg)
        return {"error_log": [msg]}

    # ------------------------------------------------------------------ #
    #  2. Chunk + summarize each chunk
    # ------------------------------------------------------------------ #
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    chunks = chunk_transcript(transcript)
    logger.info("Split into %d chunks", len(chunks))

    summaries = []
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(chunk) // CHARS_PER_TOKEN
        logger.info("Processing chunk %d/%d (~%d tokens)", i + 1, len(chunks), chunk_tokens)
        try:
            summary = await summarize_chunk(llm, prompt_template, chunk, i + 1)
            summaries.append(summary)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("Failed to parse chunk %d: %s", i + 1, exc)
            continue
        except Exception as exc:
            logger.error("LLM call failed for chunk %d: %s", i + 1, exc)
            continue

    if not summaries:
        return {"error_log": ["All chunks failed to summarize"]}

    # ------------------------------------------------------------------ #
    #  3. Merge all chunk summaries
    # ------------------------------------------------------------------ #
    merged = merge_summaries(summaries)

    logger.info("Merged: %d decisions, %d actions, %d topics",
                len(merged["key_decisions"]),
                len(merged["action_items"]),
                len(merged["discussion_points"]))

    # ------------------------------------------------------------------ #
    #  3.5. LLM finalization — clean up, deduplicate, and coherence pass
    # ------------------------------------------------------------------ #
    final_prompt = f"""You are a meeting notes assistant.
Below are combined meeting notes extracted from multiple transcript chunks.
Clean them up: remove any duplicates, fix inconsistencies, and ensure the final output is coherent.
Return ONLY valid JSON, no markdown, no backticks.
{{
  "key_decisions": [...],
  "action_items": [...],
  "discussion_points": [...]
}}
Combined notes:
{json.dumps(merged, indent=2)}
"""

    try:
        response = llm.invoke([HumanMessage(content=final_prompt)])
        raw = response.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        final = json.loads(raw)
        logger.info("Finalized: %d decisions, %d actions, %d topics",
                    len(final.get("key_decisions", [])),
                    len(final.get("action_items", [])),
                    len(final.get("discussion_points", [])))
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("LLM finalization parse failed, falling back to merged: %s", exc)
        final = merged
    except Exception as exc:
        logger.warning("LLM finalization call failed, falling back to merged: %s", exc)
        final = merged

    key_decisions = final.get("key_decisions", merged["key_decisions"])
    action_items = final.get("action_items", merged["action_items"])
    discussion_points = final.get("discussion_points", merged["discussion_points"])

    # ------------------------------------------------------------------ #
    #  4. Write to Notion via MCP stdio
    # ------------------------------------------------------------------ #
    try:
        meeting_id = state["meeting_id"]
        database_id = os.environ["NOTION_DATABASE_ID"]
        if "-" not in database_id:
            database_id = f"{database_id[0:8]}-{database_id[8:12]}-{database_id[12:16]}-{database_id[16:20]}-{database_id[20:]}"

        with open("notion_token.txt") as f:
            notion_token = f.read().strip()

        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={"OPENAPI_MCP_HEADERS": f'{{"Authorization": "Bearer {notion_token}", "Notion-Version": "2022-06-28"}}'}
        )

        children = []

        children.append({"type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "📋 Key Decisions Made"}}]}})
        for d in key_decisions:
            children.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": d}}]}})

        children.append({"type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "✅ Action Items"}}]}})
        for a in action_items:
            children.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": a}}]}})

        children.append({"type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": "📌 Main Discussion Points"}}]}})
        for section in discussion_points:
            children.append({"type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": section["header"]}}]}})
            for note in section["notes"]:
                children.append({"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": note}}]}})

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info("Notion MCP connected successfully")

                result = await session.call_tool(
                    "API-post-page",
                    arguments={
                        "parent": {"database_id": database_id},
                        "properties": {
                            "title": {"title": [{"text": {"content": f"Meeting Notes – {meeting_id}"}}]}
                        },
                        "children": children
                    }
                )
                logger.info("Notion page created: %s", result)

    except Exception as exc:
        import traceback
        logger.warning("Notion MCP push failed (non-fatal): %s", exc)
        traceback.print_exc()

    return {
        "notes": {
            "key_decisions": key_decisions,
            "action_items": action_items,
            "discussion_points": discussion_points,
        }
    }

if __name__ == "__main__":
    import asyncio

    test_state = {
        "meeting_id": "quarterly-product-engineering-review-2024-08-05 finalll",
        "transcript": [
            {"speaker": {"name": "Layla Hassan"}, "text": "Good morning everyone. Let's get started. This is our quarterly product and engineering review. We have a lot to cover today including product metrics, customer feedback, engineering progress, upcoming releases, and technical challenges.", "started_at": {"absolute_timestamp": "2024-08-05T10:00:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "The objective of this meeting is not only to review what happened last quarter but also to align everyone on priorities for the next three months.", "started_at": {"absolute_timestamp": "2024-08-05T10:01:15Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "I'll start with product analytics. Overall usage increased significantly, but we noticed some important areas where users are still struggling.", "started_at": {"absolute_timestamp": "2024-08-05T10:02:30Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Our active users increased by around twenty percent. However, activation after signup remains lower than our target.", "started_at": {"absolute_timestamp": "2024-08-05T10:03:50Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "Customer support has seen the same pattern. New customers are interested in the product, but some need additional guidance during their first week.", "started_at": {"absolute_timestamp": "2024-08-05T10:05:10Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "From the design side, we reviewed recordings from new users. The biggest confusion happens when users first open the dashboard.", "started_at": {"absolute_timestamp": "2024-08-05T10:06:25Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Engineering can support onboarding improvements. Most of the required changes should not require major backend modifications.", "started_at": {"absolute_timestamp": "2024-08-05T10:07:40Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Before discussing solutions, let's understand the biggest problems. Omar, can you explain where users are dropping off?", "started_at": {"absolute_timestamp": "2024-08-05T10:09:00Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "The first issue is feature discovery. Users complete registration but they do not understand what actions they should take next.", "started_at": {"absolute_timestamp": "2024-08-05T10:10:20Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "The second issue is that advanced features are hidden. Users who would benefit from them never discover them.", "started_at": {"absolute_timestamp": "2024-08-05T10:11:40Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "I think the dashboard currently tries to serve experienced users and new users with the same interface.", "started_at": {"absolute_timestamp": "2024-08-05T10:13:00Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "A new user probably needs guidance, while an experienced user needs speed and customization.", "started_at": {"absolute_timestamp": "2024-08-05T10:14:15Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Personalized dashboards would require some additional logic, but we already collect enough activity data to support basic recommendations.", "started_at": {"absolute_timestamp": "2024-08-05T10:15:40Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Would personalization delay the release timeline?", "started_at": {"absolute_timestamp": "2024-08-05T10:17:00Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "A complete recommendation system would, but a simpler version based on user role and activity would be achievable.", "started_at": {"absolute_timestamp": "2024-08-05T10:18:20Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "We should validate whether personalization actually improves engagement before investing heavily.", "started_at": {"absolute_timestamp": "2024-08-05T10:19:40Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Agreed. Let's keep it as an experiment rather than a full product commitment.", "started_at": {"absolute_timestamp": "2024-08-05T10:21:00Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I want to bring up another customer issue. Several enterprise customers requested better reporting capabilities.", "started_at": {"absolute_timestamp": "2024-08-05T10:22:30Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Analytics is actually one of the most requested improvements. Customers want more visibility into their performance.", "started_at": {"absolute_timestamp": "2024-08-05T10:23:50Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The current reporting system was designed for internal usage. Supporting customers will require permission handling and scalability improvements.", "started_at": {"absolute_timestamp": "2024-08-05T10:25:15Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's add customer reporting as a strategic initiative. It may not be the immediate sprint priority, but it should be planned.", "started_at": {"absolute_timestamp": "2024-08-05T10:26:40Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "Moving to design updates, we completed research for the new navigation structure.", "started_at": {"absolute_timestamp": "2024-08-05T10:28:00Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "The main recommendation is reducing the number of primary navigation items and grouping related features.", "started_at": {"absolute_timestamp": "2024-08-05T10:29:20Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "That matches the usability results. Users spend too much time scanning menus instead of completing tasks.", "started_at": {"absolute_timestamp": "2024-08-05T10:30:40Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "What are the possible risks of changing navigation?", "started_at": {"absolute_timestamp": "2024-08-05T10:32:00Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The main risk is breaking existing workflows. We need redirects and compatibility checks before deployment.", "started_at": {"absolute_timestamp": "2024-08-05T10:33:20Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Okay. Let's make backward compatibility a mandatory requirement.", "started_at": {"absolute_timestamp": "2024-08-05T10:34:40Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "I'll prepare the updated navigation prototype and include the new information hierarchy.", "started_at": {"absolute_timestamp": "2024-08-05T10:35:50Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Once the prototype is ready, I'll organize usability sessions with both existing and new users.", "started_at": {"absolute_timestamp": "2024-08-05T10:37:10Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Great. Let's move to engineering updates. Karim, can you walk us through the current technical status?", "started_at": {"absolute_timestamp": "2024-08-05T10:38:30Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Overall, the platform is stable. We completed the database optimization work and reduced average response time for the main APIs.", "started_at": {"absolute_timestamp": "2024-08-05T10:39:45Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The main technical challenge currently is managing growth. Some services were designed when we had fewer users.", "started_at": {"absolute_timestamp": "2024-08-05T10:41:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Which areas are becoming a concern?", "started_at": {"absolute_timestamp": "2024-08-05T10:42:10Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The notification service and reporting system are the biggest concerns. Both generate a large amount of background processing.", "started_at": {"absolute_timestamp": "2024-08-05T10:43:25Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "From the infrastructure side, we noticed higher resource usage during peak hours, especially when many users generate reports at the same time.", "started_at": {"absolute_timestamp": "2024-08-05T10:44:40Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "One solution is moving heavy report generation into asynchronous jobs instead of processing everything during the user request.", "started_at": {"absolute_timestamp": "2024-08-05T10:46:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Would that affect the user experience?", "started_at": {"absolute_timestamp": "2024-08-05T10:47:10Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Users would receive a notification when reports are ready instead of waiting on a loading screen.", "started_at": {"absolute_timestamp": "2024-08-05T10:48:20Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "That actually fits well with the notification improvements we already discussed.", "started_at": {"absolute_timestamp": "2024-08-05T10:49:30Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "We should test whether users prefer instant results or background generation. Some reports may need immediate access.", "started_at": {"absolute_timestamp": "2024-08-05T10:50:45Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "We can categorize reports based on complexity. Simple reports can remain synchronous while larger ones run in the background.", "started_at": {"absolute_timestamp": "2024-08-05T10:52:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "That sounds like a balanced approach. Please include it in the technical roadmap.", "started_at": {"absolute_timestamp": "2024-08-05T10:53:15Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "I also want to discuss infrastructure monitoring. We currently have basic alerts, but they do not always identify problems early.", "started_at": {"absolute_timestamp": "2024-08-05T10:54:30Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Improving observability would help the team identify slow services before customers report issues.", "started_at": {"absolute_timestamp": "2024-08-05T10:55:45Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Is this something we can include in the upcoming sprint?", "started_at": {"absolute_timestamp": "2024-08-05T10:57:00Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "A basic improvement is possible. We can add additional monitoring dashboards and improve alert rules.", "started_at": {"absolute_timestamp": "2024-08-05T10:58:15Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Okay, let's include that as an engineering improvement task.", "started_at": {"absolute_timestamp": "2024-08-05T10:59:20Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I want to discuss customer feedback around integrations. Enterprise users are requesting more connections with external tools.", "started_at": {"absolute_timestamp": "2024-08-05T11:00:40Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "The most requested integrations are currently project management tools and communication platforms.", "started_at": {"absolute_timestamp": "2024-08-05T11:01:55Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Building many direct integrations can become difficult to maintain. We should consider creating a more flexible integration layer.", "started_at": {"absolute_timestamp": "2024-08-05T11:03:10Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Would an API-first approach solve this problem?", "started_at": {"absolute_timestamp": "2024-08-05T11:04:25Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Yes, exposing stable APIs would allow partners to build integrations without us maintaining every connection.", "started_at": {"absolute_timestamp": "2024-08-05T11:05:40Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "Customers would appreciate that because some companies already have internal systems they need to connect.", "started_at": {"absolute_timestamp": "2024-08-05T11:06:55Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's add API expansion to the long-term roadmap. It is important but not more urgent than the current usability issues.", "started_at": {"absolute_timestamp": "2024-08-05T11:08:10Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Before we move on, I want to mention that user research also showed customers want more customization options.", "started_at": {"absolute_timestamp": "2024-08-05T11:09:20Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "Customization came up several times. Users want more control over their workspace, especially teams that use the product daily.", "started_at": {"absolute_timestamp": "2024-08-05T11:10:35Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "What kind of customization are users asking for specifically?", "started_at": {"absolute_timestamp": "2024-08-05T11:11:45Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Mostly dashboard layouts, saved filters, and the ability to hide features they do not use frequently.", "started_at": {"absolute_timestamp": "2024-08-05T11:13:00Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "From a design perspective, customizable dashboards are possible, but we need to avoid making the interface complicated.", "started_at": {"absolute_timestamp": "2024-08-05T11:14:15Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The backend already stores user preferences in some areas. Expanding that system would be manageable.", "started_at": {"absolute_timestamp": "2024-08-05T11:15:30Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's prioritize customization for enterprise users first since they have the strongest need.", "started_at": {"absolute_timestamp": "2024-08-05T11:16:45Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I'll collect more detailed feedback from enterprise customers and identify the highest-value customization requests.", "started_at": {"absolute_timestamp": "2024-08-05T11:18:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Good. Now let's discuss the mobile application roadmap.", "started_at": {"absolute_timestamp": "2024-08-05T11:19:15Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "The mobile experience is functional, but the user experience is inconsistent compared to the web version.", "started_at": {"absolute_timestamp": "2024-08-05T11:20:30Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Mobile users have lower engagement with some features, especially reporting and account management.", "started_at": {"absolute_timestamp": "2024-08-05T11:21:45Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The mobile application shares most backend services with the web platform, so improvements can be reused.", "started_at": {"absolute_timestamp": "2024-08-05T11:23:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "What are the biggest technical limitations on mobile right now?", "started_at": {"absolute_timestamp": "2024-08-05T11:24:10Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The main issue is performance on older devices. Some screens load too much information at once.", "started_at": {"absolute_timestamp": "2024-08-05T11:25:25Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "We can improve this by simplifying screens and loading additional information only when users request it.", "started_at": {"absolute_timestamp": "2024-08-05T11:26:40Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "That approach should also improve usability because users currently see too much information immediately.", "started_at": {"absolute_timestamp": "2024-08-05T11:27:55Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's include mobile performance improvements in the roadmap but focus first on the highest-impact screens.", "started_at": {"absolute_timestamp": "2024-08-05T11:29:10Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "I want to discuss security improvements. We completed the latest security review and identified some areas for improvement.", "started_at": {"absolute_timestamp": "2024-08-05T11:30:30Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Were there any critical issues?", "started_at": {"absolute_timestamp": "2024-08-05T11:31:40Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "No critical vulnerabilities were found, but we should improve access auditing and permission management.", "started_at": {"absolute_timestamp": "2024-08-05T11:32:55Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "The permission system has grown over time. Refactoring it would make future changes safer.", "started_at": {"absolute_timestamp": "2024-08-05T11:34:10Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Security improvements should not be delayed. Let's create a dedicated task for reviewing permissions.", "started_at": {"absolute_timestamp": "2024-08-05T11:35:25Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "I'll prepare a security improvement plan including auditing, permissions, and authentication enhancements.", "started_at": {"absolute_timestamp": "2024-08-05T11:36:40Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I also want to discuss the upcoming customer training sessions. Several new customers requested onboarding workshops.", "started_at": {"absolute_timestamp": "2024-08-05T11:37:55Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "That could help with activation. Can we measure whether trained customers retain better?", "started_at": {"absolute_timestamp": "2024-08-05T11:39:10Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "Yes, we can compare activation and retention metrics between customers who attend training and those who do not.", "started_at": {"absolute_timestamp": "2024-08-05T11:40:25Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I'll coordinate with customer success and prepare a training schedule.", "started_at": {"absolute_timestamp": "2024-08-05T11:41:40Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Excellent. Let's take a short break and continue after reviewing the remaining roadmap items.", "started_at": {"absolute_timestamp": "2024-08-05T11:43:00Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Welcome back everyone. Let's continue with the roadmap discussion.", "started_at": {"absolute_timestamp": "2024-08-05T11:45:00Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "I'll start with the release pipeline updates. We improved our deployment process, but there are still manual steps that slow down releases.", "started_at": {"absolute_timestamp": "2024-08-05T11:46:15Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Which steps are still manual?", "started_at": {"absolute_timestamp": "2024-08-05T11:47:20Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Mainly database migrations, environment configuration checks, and final verification before production deployment.", "started_at": {"absolute_timestamp": "2024-08-05T11:48:35Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "We can automate some of those checks through the deployment pipeline. It would reduce human errors during releases.", "started_at": {"absolute_timestamp": "2024-08-05T11:49:50Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "Exactly. The goal is not fully automatic deployment immediately, but improving confidence in every release.", "started_at": {"absolute_timestamp": "2024-08-05T11:51:05Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's include it as an engineering efficiency initiative.", "started_at": {"absolute_timestamp": "2024-08-05T11:54:40Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "I want to discuss experimentation. Currently, product decisions rely heavily on feedback but we need more controlled experiments.", "started_at": {"absolute_timestamp": "2024-08-05T11:55:55Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "For example, onboarding changes. Instead of releasing one version, we can compare different flows and measure activation rates.", "started_at": {"absolute_timestamp": "2024-08-05T11:58:20Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "We would need feature flag support and reliable analytics tracking before running experiments.", "started_at": {"absolute_timestamp": "2024-08-05T12:00:50Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's consider experimentation infrastructure as part of the analytics roadmap.", "started_at": {"absolute_timestamp": "2024-08-05T12:02:00Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "Some customers asked for better collaboration features between team members.", "started_at": {"absolute_timestamp": "2024-08-05T12:03:20Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "We could start with comments and sharing before implementing real-time collaboration.", "started_at": {"absolute_timestamp": "2024-08-05T12:08:15Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Let's add collaboration improvements to the future roadmap, but keep the initial scope small.", "started_at": {"absolute_timestamp": "2024-08-05T12:10:45Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "We need clearer data retention policies, improved audit logs, and easier ways for customers to manage their data.", "started_at": {"absolute_timestamp": "2024-08-05T12:14:25Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Privacy should be considered in every upcoming feature rather than handled separately.", "started_at": {"absolute_timestamp": "2024-08-05T12:18:10Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "I'll prepare the updated onboarding flows, including the first-time user experience and guided product discovery.", "started_at": {"absolute_timestamp": "2024-08-05T12:28:15Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "I'll define the testing scenarios and measure activation rate, completion rate, and feature discovery improvements.", "started_at": {"absolute_timestamp": "2024-08-05T12:29:30Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "I'll review the technical impact of the navigation changes and identify any migration requirements.", "started_at": {"absolute_timestamp": "2024-08-05T12:32:00Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "I'll prepare the API design, database changes, and estimated implementation timeline for the notification system.", "started_at": {"absolute_timestamp": "2024-08-05T12:34:30Z"}},
            {"speaker": {"name": "Nadia Youssef"}, "text": "I'll design the notification categories, priority indicators, and user preference settings.", "started_at": {"absolute_timestamp": "2024-08-05T12:35:45Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "I'll analyze the current checkout state machine and prepare the required backend modifications.", "started_at": {"absolute_timestamp": "2024-08-05T12:38:15Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "I'll create a usability study comparing the current checkout process with the proposed two-step flow.", "started_at": {"absolute_timestamp": "2024-08-05T12:39:30Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I'll continue collecting customer feedback and coordinate with customer success for enterprise requirements.", "started_at": {"absolute_timestamp": "2024-08-05T12:40:45Z"}},
            {"speaker": {"name": "Ahmed Mostafa"}, "text": "I'll prepare a security improvement plan including auditing, permissions, and authentication enhancements.", "started_at": {"absolute_timestamp": "2024-08-05T12:52:00Z"}},
            {"speaker": {"name": "Sara Ahmed"}, "text": "I'll send the customer insights document after incorporating the latest support feedback.", "started_at": {"absolute_timestamp": "2024-08-05T12:55:45Z"}},
            {"speaker": {"name": "Karim Bassem"}, "text": "I'll share the technical breakdown and estimates before the next planning session.", "started_at": {"absolute_timestamp": "2024-08-05T12:57:00Z"}},
            {"speaker": {"name": "Omar Sharaf"}, "text": "I'll prepare the research plan and make sure we have measurable success criteria for each initiative.", "started_at": {"absolute_timestamp": "2024-08-05T12:58:15Z"}},
            {"speaker": {"name": "Layla Hassan"}, "text": "Thank you everyone. This was a productive discussion. I'll share the meeting notes and action items today.", "started_at": {"absolute_timestamp": "2024-08-05T12:59:30Z"}},
        ]
    }


    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(notetaker_node(test_state))
    print(json.dumps(result, indent=2))



#MAP REDUCE WITH OVERLAPPING CHUNKING 
