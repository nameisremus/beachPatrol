import asyncio
import logging_config
import logging
from tweety import Twitter
from tweety.types import Proxy, PROXY_TYPE_HTTP
from tweety.exceptions import TwitterError
from config import TWITTER_USERNAMES, TWITTER_PASSWORD, PROXIES_LIST

logger = logging.getLogger(__name__)


def is_rate_limit_error(exc: Exception) -> bool:
    """Tweety wraps HTTP-429/code 88 in plain Exception."""
    return "rate limit exceeded" in str(exc).lower() or getattr(exc, "code", None) == 88


def is_auth_404(exc: Exception) -> bool:
    """
    Detect Tweety/Twitter GraphQL 404 responses that really mean
    “viewer lacks permission” (protected or deleted tweet).
    We treat these as soft-failures so the pool can try the next account.
    """
    if not isinstance(exc, TwitterError):
        return False

    # TwitterError.response is an httpx.Response
    resp = getattr(exc, "response", None)
    if resp is None or resp.status_code != 404:
        return False

    msg = str(exc).lower()
    return (
        "page not found" in msg or 
        "authorization" in msg or 
        "not found" in msg
    )


class _TwitterPool:
    BACKOFF = 3 * 60  # seconds to wait after all accounts exhausted

    def __init__(self) -> None:
        self._usernames = TWITTER_USERNAMES
        self._proxies   = PROXIES_LIST
        self._pwd       = TWITTER_PASSWORD
        self._clients: list[Twitter | None] = [None] * len(self._usernames)
        self._idx = 0

        logger.info(
            "TwitterPool initialised",
            extra={"accounts": len(self._usernames), "proxies": len(self._proxies)},
        )

    async def _build_client(self, idx: int) -> Twitter:
        proxy_cfg = self._proxies[idx] if idx < len(self._proxies) else {}
        host = proxy_cfg.get("host")
        port = proxy_cfg.get("port")

        proxy_obj = None
        if host and port is not None:
            proxy_obj = Proxy(
                host=host,
                port=port,
                proxy_type=PROXY_TYPE_HTTP,
                username=proxy_cfg.get("username"),
                password=proxy_cfg.get("password"),
            )

        app = Twitter("session", proxy=proxy_obj) if proxy_obj else Twitter("session")

        uname = self._usernames[idx]
        if uname and self._pwd:
            try:
                logger.info("Signing-in twitter account", extra={"user": uname})
                await app.sign_in(uname, self._pwd)
                logger.info("Authenticated", extra={"user": uname})
            except Exception:
                logger.error("Auth failed", extra={"user": uname}, exc_info=True)
        return app

    async def _client(self) -> Twitter:
        if self._clients[self._idx] is None:
            self._clients[self._idx] = await self._build_client(self._idx)
        return self._clients[self._idx]

    async def _rotate(self) -> None:
        prev = self._idx
        self._idx = (self._idx + 1) % len(self._usernames)
        logger.warning(
            "Rotated twitter account",
            extra={"from": self._usernames[prev], "to": self._usernames[self._idx]},
        )

    async def rate_limit_hit(self) -> None:
        await self._rotate()

    async def safe_call(self, method_name: str, *args, **kwargs):
        """
        Execute Tweety.<method_name>(*args, **kwargs) with auto-rotation on:
          • rate-limit (code 88 / 429)
          • protected/deleted tweet 404 (“Page not Found / authorization”)
        """
        attempts = 0
        while True:
            app = await self._client()
            try:
                return await getattr(app, method_name)(*args, **kwargs)
            except Exception as exc:
                if is_rate_limit_error(exc) or is_auth_404(exc):
                    attempts += 1
                    await self._rotate()
                    if attempts >= len(self._usernames):
                        logger.warning("All accounts exhausted – backing off...")
                        await asyncio.sleep(self.BACKOFF)
                        attempts = 0
                    continue
                raise


twitter_pool = _TwitterPool()
