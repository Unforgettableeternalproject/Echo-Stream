"""沒有 Core 的宿主（Web／CLI）用的簽發器與靜態快照替身。

Core 宿主的 ticket 與快照由 Core 提供；這裡只是讓單機宿主能用同一套契約。
"""

from __future__ import annotations

from ..contracts.output import PolicySnapshot, TurnTicket
from ..contracts.types import Utterance


class StandaloneTicketIssuer:
    """本地簽發 ticket：``turn_id`` 直接沿用 ``Utterance.turn_id``
    （Web／CLI 的輸入端 id 就是唯一的 id 空間）。"""

    def __init__(self, snapshot_version: int = 0) -> None:
        self._snapshot_version = snapshot_version

    def issue(self, utterance: Utterance, *, reply_mode: str | None = None) -> TurnTicket:
        return TurnTicket(
            turn_id=utterance.turn_id,
            source_turn_id=utterance.turn_id,
            owner_id=utterance.speaker_id,
            reply_mode=reply_mode,
            snapshot_version=self._snapshot_version,
        )


class StaticPolicySource:
    """固定快照的 :class:`~echo_stream.contracts.output.PolicySource` 替身。"""

    def __init__(self, snapshot: PolicySnapshot | None = None) -> None:
        self._snapshot = snapshot or PolicySnapshot()

    def current(self) -> PolicySnapshot:
        return self._snapshot

    def replace(self, snapshot: PolicySnapshot) -> None:
        """換成新快照（測試用，模擬 Core 推送新版本）。"""
        self._snapshot = snapshot


__all__ = ["StandaloneTicketIssuer", "StaticPolicySource"]
