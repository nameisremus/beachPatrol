import discord
import json
import io
import redis
import logging
from config import REDIS_HOST, REDIS_PORT
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from bs4 import BeautifulSoup
from core.integrations.notion_integration import send_tracked_content
from core.media_configs import get_media_type_from_command

import asyncio


logger = logging.getLogger(__name__)

def safe_filename(name: str) -> str:
    """
    Returns a sanitized filename from 'name'
    """
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_')).rstrip()

def chunk_text(text: str, limit: int = 3846, suffix: str = "... (cont. on next page)"):
    """
    Character-based chunking approach preserving newlines and formatting.
    Slices off up to (limit - len(suffix)) for each chunk if text is longer,
    appending 'suffix' to indicate there's more to read.
    """
    chunks = []
    suffix_len = len(suffix)
    while len(text) > limit:
        cutoff = limit - suffix_len
        if cutoff < 0:
            cutoff = limit
        slice_ = text[:cutoff]
        last_space_idx = slice_.rfind(" ")
        if last_space_idx == -1:
            chunk = text[:cutoff] + suffix
            chunks.append(chunk)
            text = text[cutoff:]
        else:
            chunk = text[:last_space_idx] + suffix
            chunks.append(chunk)
            text = text[last_space_idx:].lstrip()
    if text:
        chunks.append(text)
    return chunks

def sanitize_html(text: str) -> str:
    """
    Replaces unsupported HTML tags with supported ones for Telegram.
    - Converts <h1> and <h2> to <b>.
    - Removes <ul> and <ol> tags.
    - Replaces <li> with a bullet "• " and a newline.
    - Removes any other unsupported tags if needed.
    """
    # Replace headings with bold
    replacements = {
        "<h1>": "<b>",
        "</h1>": "</b>",
        "<h2>": "<b>",
        "</h2>": "</b>",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Get rid of lists since they are not available within TG.
    text = text.replace("<ul>", "")
    text = text.replace("</ul>", "")
    text = text.replace("<ol>", "")
    text = text.replace("</ol>", "")
    # Replace <li> with a new line and a bullet; remove </li>
    text = text.replace("<li>", "\n• ")
    text = text.replace("</li>", "")
    
    # Optionally, remove any other tags that might not be supported.
    text = text.replace("<p>", "")
    text = text.replace("</p>", "\n\n")
    
    return text

def safe_telegram_html(text: str) -> str:
    """
    Uses BeautifulSoup to sanitize HTML so that only allowed tags (<b>, <i>, <a>) remain.
    Any <a> tag without a proper href is removed. This prevents parse errors in Telegram.
    """
    soup = BeautifulSoup(text, "html.parser")
    allowed_tags = {"b", "i", "a"}
    for tag in soup.find_all(True):
        if tag.name not in allowed_tags:
            tag.unwrap()
        elif tag.name == "a":
            href = tag.get("href")
            if not href:
                tag.unwrap()
    return soup.decode_contents()


class PersistentPaginatedEmbedView(discord.ui.View):
    """
    A persistent pagination View that:
      - Sets timeout=None so it never auto-expires
      - Defines custom_id for each button so that it can be re-registered on bot restarts
      - Saves state (pages, index) in Redis so it can be restored if the bot restarts
      - Also stores the message_id and channel_id to re-fetch the message after bot restarts.
    If len(self.pages) == 1, there's only one page and no navigation.
    """
    def __init__(
        self,
        report_id: str,
        pages: list[str],
        base_title: str = "Paginated Embed",
        current_index: int = 0,
        message_id: int = None,
        channel_id: int = None
    ):
        super().__init__(timeout=None)
        self.report_id = report_id
        self.pages = pages
        self.index = current_index
        self.base_title = base_title

        self.message: discord.Message | None = None
        self.message_id = message_id
        self.channel_id = channel_id

        # Notion-related fields:
        self.notion_status = "default"
        self.notion_executive = ""
        self.notion_summary = ""
        self.notion_tags = []
        self.notion_url = ""
        self.notion_media_type = ""

        # This flag tracks if the content has been sent to Notion. (Already existed)
        self.notion_sent = False

        # The @discord.ui.button decorators *must* have a static custom_id,
        # so we override them here to ensure uniqueness:
        self.first_page_button.custom_id = f"persistent_first_button_{self.report_id}"
        self.previous_button.custom_id = f"persistent_previous_button_{self.report_id}"
        self.next_button.custom_id = f"persistent_next_button_{self.report_id}"
        self.last_page_button.custom_id = f"persistent_last_button_{self.report_id}"
        self.download_txt_button.custom_id = f"persistent_download_txt_button_{self.report_id}"
        self.send_to_notion_button.custom_id = f"persistent_send_notion_button_{self.report_id}"

    def _get_embed(self) -> discord.Embed:
        """
        Builds an embed for the current page.
        If there's only 1 page total, we skip the "Use the buttons..." footer to avoid confusion.
        """
        page_count = len(self.pages)
        current_page_num = self.index + 1
        embed_title = f"{self.base_title} (page {current_page_num}/{page_count})"
        embed = discord.Embed(
            title=embed_title,
            description=self.pages[self.index],
            color=0x2F3136
        )
        if page_count > 1:
            embed.set_footer(text="Use the buttons below to navigate multiple pages.")
        return embed

    @discord.ui.button(
        label="|<",
        style=discord.ButtonStyle.gray,
        custom_id="placeholder_first_button"
    )
    async def first_page_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.index = 0
        await self.save_state_and_update(interaction)

    @discord.ui.button(
        label="<",
        style=discord.ButtonStyle.green,
        custom_id="placeholder_previous_button",
        disabled=True
    )
    async def previous_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if self.index > 0:
            self.index -= 1
        await self.save_state_and_update(interaction)

    @discord.ui.button(
        label=">",
        style=discord.ButtonStyle.green,
        custom_id="placeholder_next_button"
    )
    async def next_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if self.index < len(self.pages) - 1:
            self.index += 1
        await self.save_state_and_update(interaction)

    @discord.ui.button(
        label=">|",
        style=discord.ButtonStyle.gray,
        custom_id="placeholder_last_button"
    )
    async def last_page_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.index = len(self.pages) - 1
        await self.save_state_and_update(interaction)

    @discord.ui.button(
        label="Download .txt",
        style=discord.ButtonStyle.blurple,
        custom_id="placeholder_download_txt_button"
    )
    async def download_txt_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        """
        Collects all pages, combines them, and sends them as a .txt file attachment.
        This button is persistent, so it's always available for users to download the text.
        """
        combined_text = "\n\n----- PAGE BREAK -----\n\n".join(self.pages)
        buffer = io.StringIO(combined_text)
        filename = f"{safe_filename(self.base_title)}.txt"
        discord_file = discord.File(fp=buffer, filename=filename)
        await interaction.response.send_message(
            content="Here is a .txt download of the entire summary:",
            file=discord_file,
            ephemeral=True
        )

    @discord.ui.button(
    label="➤ Send to Notion",
    style=discord.ButtonStyle.secondary,
    custom_id="placeholder_send_notion"
    )
    async def send_to_notion_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        # Immediately defer the interaction so we don't hit a timeout
        await interaction.response.defer()

        # Step 1: Update status to "sending" and push the updated button (⏳ Sending..)
        self.notion_status = "sending"
        await self._save_state()
        await self._update_buttons(None)  # this rebuilds the button with the new label
        if self.message:
            await self.message.edit(view=self)

        # Step 2: Run the Notion post in a separate thread so we don't block the event loop
        notion_exec = self.notion_executive
        #if get_media_type_from_command(self.notion_media_type) in ("Twitter Digest", "Twitter Account Digest"):
        #    if "**:information_source:**" in notion_exec:
        #        notion_exec = notion_exec.split("**:information_source:**", 1)[0].strip()

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                send_tracked_content,
                get_media_type_from_command(self.notion_media_type),
                self.notion_url,
                notion_exec or "No executive summary",
                self.notion_summary or "No full summary",
                self.notion_tags
            )
            # If successful, update status to "sent"
            self.notion_status = "sent"
        except Exception as e:
            # Use a followup message if there's an error
            await interaction.followup.send(f"Failed to send to Notion: {e}", ephemeral=True)
            return

        # Step 3: Save state and update the button again so it shows "✅ Sent to Notion"
        await self._save_state()
        await self._update_buttons(None)
        if self.message:
            await self.message.edit(view=self)


    async def save_state_and_update(self, interaction: discord.Interaction):
        """
        Saves updated index/pages in Redis, then updates the embed message.
        Also re-edits the message so the new page is displayed.
        """
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        data = {
            "pages": self.pages,
            "current_index": self.index,
            "base_title": self.base_title,
            "message_id": self.message_id,
            "chat_id": self.channel_id,
            "notion_sent": self.notion_sent,
            "notion_status": self.notion_status,
            "notion_executive": self.notion_executive,
            "notion_summary": self.notion_summary,
            "notion_tags": self.notion_tags,
            "notion_url": self.notion_url,
            "notion_media_type": self.notion_media_type
        }
        r.set(f"report:{self.report_id}", json.dumps(data))
        await interaction.response.defer()
        await self._update_embed()

    async def _update_embed(self):
        """Updates button states and re-edits the message with the current page."""
        self.first_page_button.disabled = (self.index == 0)
        self.previous_button.disabled = (self.index == 0)
        last_idx = len(self.pages) - 1
        self.next_button.disabled = (self.index == last_idx)
        self.last_page_button.disabled = (self.index == last_idx)

        await self._update_buttons(None)

        if self.message:
            await self.message.edit(embed=self._get_embed(), view=self)

    async def _update_buttons(self, interaction: discord.Interaction | None):
        send_btn: discord.ui.Button = self.send_to_notion_button
        if self.notion_status == "default":
            send_btn.label = "➤ Send to Notion"
        elif self.notion_status == "sending":
            send_btn.label = "⏳ Sending.."
        elif self.notion_status == "sent":
            send_btn.label = "✅ Sent to Notion"

        if interaction:
            await interaction.message.edit(view=self)


    async def _save_state(self):
        """
        Similar to save_state_and_update, but can be used if we only want to store
        the notion-related fields (e.g. status) without changing pages.
        """
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        data = {
            "pages": self.pages,
            "current_index": self.index,
            "base_title": self.base_title,
            "message_id": self.message_id,
            "chat_id": self.channel_id,
            "notion_sent": self.notion_sent,
            "notion_status": self.notion_status,
            "notion_executive": self.notion_executive,
            "notion_summary": self.notion_summary,
            "notion_tags": self.notion_tags,
            "notion_url": self.notion_url,
            "notion_media_type": self.notion_media_type
        }
        r.set(f"report:{self.report_id}", json.dumps(data))


class PersistentPaginatedMessage:
    """
    A Telegram persistent paginated message.

    - Stores pages, current index, base title, message and chat IDs.
    - Can generate the current message text (with header).
    - Builds an inline keyboard with navigation buttons.
    - Saves state to Redis so that it can be rehydrated after bot restarts.
    """
    def __init__(self, report_id: str, pages: list[str], base_title: str = "Paginated Message",
                 current_index: int = 0, message_id: int = None, chat_id: int = None):
        self.report_id = report_id
        self.pages = pages
        self.base_title = base_title
        self.current_index = current_index
        self.message_id = message_id
        self.chat_id = chat_id
        # notion_status can be "default", "sending", or "sent"
        self.notion_status = "default"
        self.notion_executive = ""
        self.notion_summary = ""
        self.notion_tags = []
        self.notion_url = ""
        self.notion_media_type = ""

    def get_current_text(self) -> str:
        """
        Returns the message text for the current page.
        Includes a header with the title and page numbers.
        For pages other than the first (executive summary), adds a footer disclaimer in italic.
        """
        total_pages = len(self.pages)
        current_page_num = self.current_index + 1
        header = f"{self.base_title} page {current_page_num}/{total_pages}\n\n"
        body = self.pages[self.current_index]
        footer = "\n\n<i>Use the buttons below to navigate multiple pages.</i>" if current_page_num > 1 else ""
        return header + body + footer

    def build_keyboard(self):
        """
        Builds an inline keyboard with:
          - four navigation buttons,
          - a "Download .txt" button,
          - and a "Send to Notion" button whose label depends on self.notion_status.
        """
        if self.notion_status == "default":
            notion_label = "➤ Send to Notion"
        elif self.notion_status == "sending":
            notion_label = "⏳ Sending.."
        elif self.notion_status == "sent":
            notion_label = "✅ Sent to Notion"
        keyboard = [
            [
                InlineKeyboardButton("|<", callback_data=f"persistent|first|{self.report_id}"),
                InlineKeyboardButton("<", callback_data=f"persistent|previous|{self.report_id}"),
                InlineKeyboardButton(">", callback_data=f"persistent|next|{self.report_id}"),
                InlineKeyboardButton(">|", callback_data=f"persistent|last|{self.report_id}")
            ],
            [
                InlineKeyboardButton("Download .txt", callback_data=f"persistent|download|{self.report_id}"),
                InlineKeyboardButton(notion_label, callback_data=f"persistent|notion|{self.report_id}")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation for saving in Redis.
        """
        return {
            "pages": self.pages,
            "current_index": self.current_index,
            "base_title": self.base_title,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "notion_status": self.notion_status,
            "notion_executive": self.notion_executive,
            "notion_summary": self.notion_summary,
            "notion_tags": self.notion_tags,
            "notion_url": self.notion_url,
            "notion_media_type": self.notion_media_type,
        }

    @classmethod
    def from_dict(cls, report_id: str, data: dict):
        """
        Creates an instance from a dictionary (e.g., loaded from Redis).
        """
        instance = cls(
            report_id=report_id,
            pages=data.get("pages", []),
            base_title=data.get("base_title", "Paginated Message"),
            current_index=data.get("current_index", 0),
            message_id=data.get("message_id"),
            chat_id=data.get("chat_id")
        )
        instance.notion_status = data.get("notion_status", "default")
        instance.notion_executive = data.get("notion_executive", "")
        instance.notion_summary = data.get("notion_summary", "")
        instance.notion_tags = data.get("notion_tags", [])
        instance.notion_url = data.get("notion_url", "")
        instance.notion_media_type = data.get("notion_media_type", "")
        return instance

    def save_state(self):
        """
        Saves the current state to Redis using key format "report:{report_id}".
        """
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.set(f"report:{self.report_id}", json.dumps(self.to_dict()))
