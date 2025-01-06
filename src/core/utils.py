import discord
import json
import io

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
    while len(text) > limit - len(suffix):
        cutoff = limit - len(suffix)
        chunk = text[:cutoff] + suffix
        chunks.append(chunk)
        text = text[cutoff:]
    if text:
        chunks.append(text)
    return chunks

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
        super().__init__(timeout=None)  # no auto-timeout
        self.report_id = report_id
        self.pages = pages
        self.index = current_index
        self.base_title = base_title
        
        # We'll store a reference to the message and channel
        self.message: discord.Message | None = None
        self.message_id = message_id
        self.channel_id = channel_id

        # The @discord.ui.button decorators *must* have a static custom_id, but we override
        # them in the constructor to ensure uniqueness for each instance:
        self.first_page_button.custom_id = f"persistent_first_button_{self.report_id}"
        self.previous_button.custom_id = f"persistent_previous_button_{self.report_id}"
        self.next_button.custom_id = f"persistent_next_button_{self.report_id}"
        self.last_page_button.custom_id = f"persistent_last_button_{self.report_id}"
        self.download_txt_button.custom_id = f"persistent_download_txt_button_{self.report_id}"

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
        # Only add a footer if multiple pages
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
        # Combine pages with a separator to keep clarity
        combined_text = "\n\n----- PAGE BREAK -----\n\n".join(self.pages)

        # Create an in-memory text buffer
        buffer = io.StringIO(combined_text)

        # Generate a safe filename based on the embed's title
        filename = f"{safe_filename(self.base_title)}.txt"

        # Create a Discord file
        discord_file = discord.File(fp=buffer, filename=filename)

        # Respond with the file. ephemeral=True so only the user sees the file.
        await interaction.response.send_message(
            content="Here is a .txt download of the entire summary:",
            file=discord_file,
            ephemeral=True
        )

    async def save_state_and_update(self, interaction: discord.Interaction):
        """
        Saves updated index/pages in Redis, then updates the embed message.
        Also re-edits the message so the new page is displayed.
        """
        from config import REDIS_HOST, REDIS_PORT
        import redis

        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        data = {
            "pages": self.pages,
            "current_index": self.index,
            "base_title": self.base_title,
            # Store message_id/channel_id to re-fetch if the bot restarts
            "message_id": self.message_id,
            "channel_id": self.channel_id
        }
        r.set(f"report:{self.report_id}", json.dumps(data))

        # Defer the interaction to avoid 'interaction failed' errors
        await interaction.response.defer()
        await self._update_embed()

    async def _update_embed(self):
        """Updates button states and re-edits the message with the current page."""
        # If single-page, the arrows remain disabled:
        self.first_page_button.disabled = (self.index == 0)
        self.previous_button.disabled = (self.index == 0)
        last_idx = len(self.pages) - 1
        self.next_button.disabled = (self.index == last_idx)
        self.last_page_button.disabled = (self.index == last_idx)

        if self.message:
            await self.message.edit(embed=self._get_embed(), view=self)
        else:
            # In case there's no message (e.g. re-hydration issue), we can't update.
            pass