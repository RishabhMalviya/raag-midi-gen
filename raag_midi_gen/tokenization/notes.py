from enum import Enum
from typing import NamedTuple


class EventType(Enum):
    NO_NOTE    = 0
    NOTE_ON    = 1
    NOTE_OFF   = 2
    NOTE_HOLD  = 3


class NoteEvent(NamedTuple):
    position: int
    midi_pitch: int
    velocity: int
    event_type: EventType
