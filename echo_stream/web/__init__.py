"""Web Frontend Adapter。

Phase 0-4 的 host 介面。Discord（Phase 5）只是換一個 Frontend Adapter，
管線本身不動。
"""

from .server import StreamService, serve

__all__ = ["serve", "StreamService"]
