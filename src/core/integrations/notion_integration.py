import logging
import re
from notion_client import Client
from config import (
    NOTION_KEY,
    NOTION_DATABASE_ID,
    NOTION_FULL_SUMMARY_PARENT_ID
)

logger = logging.getLogger(__name__)

notion = None
if NOTION_KEY and NOTION_KEY.strip():
    try:
        notion = Client(auth=NOTION_KEY)
        if not NOTION_DATABASE_ID:
            logger.error(
                "NOTION_DATABASE_ID is not set"
            )
    except Exception:
        logger.error(
            "Error initializing Notion client",
            exc_info=True
        )
else:
    logger.error(
        "NOTION_KEY is not set"
    )


def split_text_into_chunks(text: str, max_length: int = 2000, suffix: str = ""):
    """
    Splits the given text into a list of chunks, each with a maximum length of max_length.
    It attempts to split on whitespace to avoid cutting words in half.
    If a suffix is provided, it is appended to each chunk (except the final one).
    """
    chunks = []
    while len(text) > max_length:
        cutoff = max_length - len(suffix) if suffix else max_length
        split_idx = text.rfind(" ", 0, cutoff)
        if split_idx == -1:
            split_idx = cutoff
        # Append the suffix if there is more text after this chunk and suffix is not empty.
        chunk = text[:split_idx] + (suffix if len(text) > split_idx and suffix else "")
        chunks.append(chunk)
        text = text[split_idx:].strip()
    if text:
        chunks.append(text)
    return chunks


def parse_inline_formatting(line: str) -> list:
    """
    Converts inline Markdown (bold, italic, links) into a list of rich_text segments.
    Supported syntax:
      - **bold** text
      - *italic* text
      - [label](url)
    This is a naive approach, but handles basic usage for inline formatting.
    
    Returns a list of Notion 'rich_text' dictionaries, each containing:
       {
         "type": "text",
         "text": { "content": "...", "link": optional },
         "annotations": { "bold": True/False, "italic": True/False, etc. }
       }
    """
    import re

    # We will build a list of rich_text items:
    rich_text_list = []
    pos = 0

    # Regex to detect:
    #   1. Bold: **(.+?)**
    #   2. Italic: \*(.+?)\*
    #   3. Link: \[(.+?)\]\((.+?)\)
    pattern = re.compile(r"(\*\*(.+?)\*\*)|(\*(.+?)\*)|(\[(.+?)\]\((.+?)\))")
    for match in pattern.finditer(line):
        start, end = match.span()
        # Text before the match (plain)
        if start > pos:
            segment = line[pos:start]
            if segment:
                rich_text_list.append({
                    "type": "text",
                    "text": {"content": segment},
                    "annotations": {}
                })

        bold_group = match.group(2)   # inside ** **
        italic_group = match.group(4)  # inside * *
        link_label = match.group(6)    # label in [label]
        link_url = match.group(7)      # url in (url)

        if bold_group:
            # Matched **bold**
            rich_text_list.append({
                "type": "text",
                "text": {"content": bold_group},
                "annotations": {"bold": True}
            })
        elif italic_group:
            # Matched *italic*
            rich_text_list.append({
                "type": "text",
                "text": {"content": italic_group},
                "annotations": {"italic": True}
            })
        elif link_label and link_url:
            # Matched [label](url)
            rich_text_list.append({
                "type": "text",
                "text": {"content": link_label, "link": {"url": link_url}},
                "annotations": {}
            })

        pos = end

    # Any remaining text after last match is plain
    if pos < len(line):
        tail = line[pos:]
        if tail:
            rich_text_list.append({
                "type": "text",
                "text": {"content": tail},
                "annotations": {}
            })

    return rich_text_list


def parse_markdown_to_blocks(markdown_text: str) -> list:
    """
    Splits the markdown text by lines. Each line is checked for:
      - # Heading 1
      - ## Heading 2
      - ### Heading 3
      - - (Bullet point)
      - Otherwise it's a paragraph
    Then we parse inline formatting (bold, italic, links) within each line.
    Returns a list of Notion blocks (JSON) that can be passed as children to the Notion API.
    """
    blocks = []
    lines = markdown_text.split("\n")

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            # skip empty lines, or create an empty paragraph if desired
            continue

        if line_stripped.startswith("### "):  # Heading 3
            content = line_stripped[4:]
            rich_text_segments = parse_inline_formatting(content)
            block = {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": rich_text_segments
                }
            }
        elif line_stripped.startswith("## "):  # Heading 2
            content = line_stripped[3:]
            rich_text_segments = parse_inline_formatting(content)
            block = {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": rich_text_segments
                }
            }
        elif line_stripped.startswith("# "):   # Heading 1
            content = line_stripped[2:]
            rich_text_segments = parse_inline_formatting(content)
            block = {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": rich_text_segments
                }
            }
        elif line_stripped.startswith("- "):   # Bullet
            content = line_stripped[2:]
            rich_text_segments = parse_inline_formatting(content)
            block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": rich_text_segments
                }
            }
        else:
            # Paragraph
            rich_text_segments = parse_inline_formatting(line_stripped)
            block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": rich_text_segments
                }
            }

        blocks.append(block)

    return blocks


def create_full_summary_page(summary_text: str, media_type: str = "Full Summary"):
    """
    Creates a single Notion sub-page (child of NOTION_FULL_SUMMARY_PARENT_ID)
    and appends *all* blocks in increments of up to 100, so they appear on
    the same page (not multiple sub-pages).
    """
    if not NOTION_FULL_SUMMARY_PARENT_ID:
        logger.error("NOTION_FULL_SUMMARY_PARENT_ID is not set in config.py.")
        return None

    if not notion:
        logger.error("Notion client is not initialized (missing NOTION_KEY).")
        return None

    # 1) Convert the raw text into Notion blocks (may be > 100 blocks).
    try:
        all_blocks = parse_markdown_to_blocks(summary_text)
    except Exception:
        logger.error(
            "Error parsing Markdown for Notion blocks",
            exc_info=True
        )
        # fallback: just chunk text into big paragraphs
        chunked = split_text_into_chunks(summary_text, max_length=2000)
        all_blocks = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": c}, "annotations": {}}]
                }
            }
            for c in chunked
        ]

    full_title = f"{media_type} Full Summary"

    # 2) First, create an *empty* page with just a title (and no children).
    try:
        page_data = {
            "parent": {"page_id": NOTION_FULL_SUMMARY_PARENT_ID},
            "properties": {
                "title": [
                    {
                        "type": "text",
                        "text": {"content": full_title}
                    }
                ]
            }
            # No "children" here -> we create the page empty
        }
        response = notion.pages.create(**page_data)
        page_id = response["id"]
        page_url_for_return = response.get("url")
        logger.info(
            "Created Notion page for full summary",
            extra={"page_id": page_id, "url": page_url_for_return}
        )
    except Exception:
        logger.error(
            "Error creating base full summary page",
            exc_info=True
        )
        return None

    # 3) Now append blocks in batches of up to 100
    PAGE_BLOCK_LIMIT = 100
    for i in range(0, len(all_blocks), PAGE_BLOCK_LIMIT):
        chunk_of_blocks = all_blocks[i : i + PAGE_BLOCK_LIMIT]
        try:
            notion.blocks.children.append(
                block_id=page_id,
                children=chunk_of_blocks
            )
            logger.info(
                "Appended blocks to Notion page",
                extra={"page_id": page_id, "block_count": len(chunk_of_blocks)}
            )
        except Exception:
            logger.error(
                "Error appending blocks to page",
                extra={"page_id": page_id},
                exc_info=True
            )
    
    return page_url_for_return


def split_rich_text_segment(seg, limit=2000):
    """
    Splits a single rich_text segment (preserving its annotations)
    into multiple segments so that each segment's text length is ≤ limit.
    """
    content = seg["text"]["content"]
    annotations = seg.get("annotations", {})
    # If content already within limit, return as is
    if len(content) <= limit:
        return [seg]
    segments = []
    for i in range(0, len(content), limit):
        chunk = content[i:i+limit]
        segments.append({
            "type": "text",
            "text": {"content": chunk},
            "annotations": annotations
        })
    return segments


def ensure_all_segments_within_limit(segments, limit=2000):
    """
    Processes a list of rich_text segments so that each individual segment's text
    does not exceed the limit. If a segment is too long, it is split into multiple segments.
    Returns a flattened list of segments.
    """
    new_segments = []
    for seg in segments:
        new_segments.extend(split_rich_text_segment(seg, limit))
    return new_segments


def send_tracked_content(
    media_type: str = "",
    url: str = "",
    executive_summary: str = "",
    summary: str = "",
    tags: list = None
):
    """
    Creates a new page in the Notion database with the specified properties:
      - URL (url property): user-provided URL (if any)
      - Executive Summary (rich_text)
      - Summary (rich_text or a link to a separate page if it's large)
      - Tags (rich_text, comma-separated) or multi_select
      - Media Type (rich_text), using media_type parameter
      - Project/Org (multi_select), empty for now

    Note: You must have the NOTION_KEY and NOTION_DATABASE_ID set in config.py.
    In your Notion DB, the "Tags" property must allow new tag values to be created.
    """
    if not notion:
        logger.error("Notion client is not initialized (missing or invalid NOTION_KEY).")
        return

    if not NOTION_DATABASE_ID:
        logger.error("NOTION_DATABASE_ID is not set in config.py")
        return

    if tags is None:
        tags = []

    # If the media type is a Twitter digest, remove the legend block
    if media_type in ("Twitter Digest", "Twitter Account Digest"):
        executive_summary = clean_exec_summary_for_notion(executive_summary)

    # 1) Parse the user's executive_summary into inline segments
    es_segments = parse_inline_formatting(executive_summary or "")

    # 2)  Ensure each rich_text segment is within the 2000-character limit.
    es_segments = ensure_all_segments_within_limit(es_segments, limit=2000)

    # 3) Merge if > 95 to avoid Notion's 100-object limit
    es_segments = merge_surplus_segments(es_segments, keep_count=95)

    # We'll store the entire summary in a "Full Summary" property as well.
    if summary:
        full_summary_chunks = split_text_into_chunks(summary, max_length=1000, suffix="")
        full_summary_rich_text = [{"text": {"content": chunk}} for chunk in full_summary_chunks]
    else:
        full_summary_rich_text = []

    # If summary is huge, create a separate page (legacy behavior)
    if summary and len(summary) > 1500:
        full_page_url = create_full_summary_page(summary, media_type=f"{media_type} Full Summary")
        summary_property = {
            "rich_text": [{
                "text": {
                    "content": "Full summary available here.",
                    "link": {"url": full_page_url} if full_page_url else None
                }
            }]
        }
    else:
        summary_property = {
            "rich_text": [{"text": {"content": summary}}] if summary else []
        }

    properties = {
        "URL": {
            "url": url if url else None
        },
        "Executive Summary": {
            "rich_text": es_segments
        },
        "Summary": summary_property,
        "Full Summary": {
            "rich_text": full_summary_rich_text
        },
        "Tags": {
            "multi_select": [{"name": t} for t in tags]
        },
        "Media Type": {
            "title": [{"type": "text", "text": {"content": media_type}}] if media_type else []
        },
        "Project/Org": {
            "multi_select": []
        }
    }

    try:
        response = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=properties
        )
        logger.info(
            "Successfully created Notion page",
            extra={"page_id": response.get("id")}
        )
    except Exception:
        logger.error(
            "Error creating Notion page",
            exc_info=True
        )


def merge_surplus_segments(segments, keep_count=95):
    """
    If len(segments) <= keep_count, return them as is.
    Otherwise:
      - keep the first keep_count segments unchanged,
      - gather the TEXT from all subsequent segments and combine them into ONE final plain segment
        (no bold/italic).
      - If the final combined leftover exceeds 2000 chars, trim it to 2000 minus some placeholder text
        so that we avoid 'rich_text[...] must be <= 2000 characters'.
    """
    if len(segments) <= keep_count:
        return segments

    # The first keep_count segments remain unchanged
    kept = segments[:keep_count]

    # Combine leftover text (ignoring inline formatting, i.e. only grabbing seg["text"]["content"])
    leftover_text = ""
    for seg in segments[keep_count:]:
        leftover_text += seg["text"]["content"]

    # If leftover_text > 2000, we forcibly trim
    #  - Save ~20 chars for a '... (truncated)' notice
    max_allowed = 1980
    if len(leftover_text) > max_allowed:
        leftover_text = leftover_text[:max_allowed] + "... (truncated)"

    # Create one final plain-text segment
    plain_seg = {
        "type": "text",
        "text": {"content": leftover_text},
        "annotations": {}
    }

    return kept + [plain_seg]


def clean_exec_summary_for_notion(exec_text: str) -> str:
    """
    Remove any legend block that starts with :information_source:, **:information_source:**
    ℹ️, or **ℹ️**, and all subsequent lines, until we reach a line whose length is < 3 after stripping.
    """
    lines = exec_text.splitlines()
    cleaned_lines = []
    skip_block = False

    for line in lines:
        stripped = line.strip()

        # If we are currently skipping lines...
        if skip_block:
            # ... check if we found a short line (< 3 chars)
            # If so, stop skipping and keep this line.
            if len(stripped) < 3:
                skip_block = False
                cleaned_lines.append(line)
            # Otherwise, keep skipping.
            continue

        # Not skipping yet => check if this line starts with a legend marker
        if (stripped.startswith(":information_source:") or
            stripped.startswith("**:information_source:**") or
            stripped.startswith("ℹ️") or
            stripped.startswith("**ℹ️**")):
            # Turn on skipping and do *not* append this line
            skip_block = True
            continue

        # If no skip condition triggered, keep the line
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
