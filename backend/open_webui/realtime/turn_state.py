"""Voice-turn state machine for realtime sessions."""

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class VoiceTurn:
    turn_id: str
    user_message_id: str
    assistant_message_id: str
    parent_message_id: str

    input_item_ids: list[str] = field(default_factory=list)
    response_id: str | None = None

    user_transcript: str = ''
    finalized_user_segments: list[str] = field(default_factory=list)
    finalized_item_ids: set[str] = field(default_factory=set)
    assistant_transcript: str = ''
    assistant_output: list[dict] = field(default_factory=list)

    is_user_done: bool = False
    is_assistant_done: bool = False
    is_empty: bool = False

    def has_any_output(self) -> bool:
        return bool(self.assistant_transcript)


@dataclass
class TurnStateManager:
    turns: dict[str, VoiceTurn] = field(default_factory=dict)
    _input_item_to_turn: dict[str, str] = field(default_factory=dict)
    _response_to_turn: dict[str, str] = field(default_factory=dict)
    active_turn_id: str | None = None

    def create_turn(
        self,
        turn_id: str,
        user_message_id: str,
        assistant_message_id: str,
        parent_message_id: str,
    ) -> VoiceTurn:
        turn = VoiceTurn(
            turn_id=turn_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            parent_message_id=parent_message_id,
        )
        self.turns[turn_id] = turn
        self.active_turn_id = turn_id
        return turn

    def bind_input_item(self, item_id: str, turn_id: str) -> None:
        prior_turn_id = self._input_item_to_turn.get(item_id)
        if prior_turn_id and prior_turn_id != turn_id:
            prior_turn = self.turns.get(prior_turn_id)
            if prior_turn and item_id in prior_turn.input_item_ids:
                prior_turn.input_item_ids.remove(item_id)
        self._input_item_to_turn[item_id] = turn_id
        turn = self.turns.get(turn_id)
        if turn and item_id not in turn.input_item_ids:
            turn.input_item_ids.append(item_id)

    def bind_response(self, response_id: str, turn_id: str) -> None:
        turn = self.turns.get(turn_id)
        if not turn:
            return
        if turn.response_id:
            self._response_to_turn.pop(turn.response_id, None)
        self._response_to_turn[response_id] = turn_id
        turn.response_id = response_id

    def unbind_response(self, response_id: str) -> None:
        turn_id = self._response_to_turn.pop(response_id, None)
        if turn_id:
            turn = self.turns.get(turn_id)
            if turn and turn.response_id == response_id:
                turn.response_id = None

    def get_turn_by_input_item(self, item_id: str) -> VoiceTurn | None:
        turn_id = self._input_item_to_turn.get(item_id)
        return self.turns.get(turn_id) if turn_id else None

    def get_turn_by_response(self, response_id: str) -> VoiceTurn | None:
        turn_id = self._response_to_turn.get(response_id)
        return self.turns.get(turn_id) if turn_id else None

    def get_active_turn(self) -> VoiceTurn | None:
        if self.active_turn_id:
            return self.turns.get(self.active_turn_id)
        return None

    def get_pending_assistant_turn(self) -> VoiceTurn | None:
        for turn in self.turns.values():
            if not turn.response_id and not turn.is_assistant_done:
                return turn
        return self.get_active_turn()

    def append_user_transcript(self, turn_id: str, delta: str) -> None:
        turn = self.turns.get(turn_id)
        if turn:
            turn.user_transcript += delta

    def finalize_user_transcript(self, turn_id: str, transcript: str, item_id: str = '') -> None:
        turn = self.turns.get(turn_id)
        if not turn:
            return
        if item_id:
            if item_id in turn.finalized_item_ids:
                return
            turn.finalized_item_ids.add(item_id)
        elif turn.is_user_done:
            return
        turn.finalized_user_segments.append(transcript)
        merged = '\n'.join(s.strip() for s in turn.finalized_user_segments if s.strip())
        turn.user_transcript = merged
        turn.is_user_done = True
        turn.is_empty = not merged.strip()

    def append_assistant_transcript(self, turn_id: str, delta: str) -> None:
        turn = self.turns.get(turn_id)
        if turn:
            turn.assistant_transcript += delta

    def finalize_assistant_transcript(self, turn_id: str, transcript: str) -> None:
        turn = self.turns.get(turn_id)
        if turn:
            final_transcript = (transcript or '').strip()
            existing_transcript = (turn.assistant_transcript or '').strip()

            if final_transcript and len(final_transcript) >= len(existing_transcript):
                turn.assistant_transcript = transcript
            turn.is_assistant_done = True

    def gc_turn(self, turn_id: str) -> VoiceTurn | None:
        turn = self.turns.pop(turn_id, None)
        if turn:
            for item_id in turn.input_item_ids:
                self._input_item_to_turn.pop(item_id, None)
            if turn.response_id:
                self._response_to_turn.pop(turn.response_id, None)
            if self.active_turn_id == turn_id:
                self.active_turn_id = None
        return turn

    def clear(self) -> None:
        self.turns.clear()
        self._input_item_to_turn.clear()
        self._response_to_turn.clear()
        self.active_turn_id = None
