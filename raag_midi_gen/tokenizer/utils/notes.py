import math
import operator
from collections import deque
from typing import NamedTuple

from raag_midi_gen.tokenizer.vocab import EventTypeToken


class NoteEvent(NamedTuple):
    position: int
    midi_pitch: int
    velocity: int
    event_type: EventTypeToken


def separate_on_and_off_events(note):
    """
    Converts a muspy Note object into two NoteEvent objects, one for Note On one for Note Off

    Args:
        note: one muspy Note object, containing information on `time`, `pitch`, `duration`, and `velocity`

    Returns:
        A list of two NoteEvent objects, one for the Note On event and one for the Note Off event        
    """
    return [
        NoteEvent(note.time,                 note.pitch, note.velocity, EventTypeToken.NOTE_ON ),
        NoteEvent(note.time + note.duration, note.pitch, note.velocity, EventTypeToken.NOTE_OFF)
    ]


def insert_note_holds(note_events, length_in_ticks):
    """
    Converts a list of NOTE_ON and NOTE_OFF NoteEvents to a list of NOTE_ON, NOTE_OFF, and NOTE_HOLD NoteEvents.
    Adds NOTE_HOLDs corresponding to the latest NOTE_ON event. 

    Args:
        note_events: An input sequence of NoteEvents of type NOTE_ON or NOTE_OFF only
        length_in_ticks: The length of the entire clip in ticks (lowest timestep resolution on MIDI grid)
    
    Returns:
        A sequence of NoteEvents of length `length_in_ticks`.
        Each timestep will have either a NOTE_ON, NOTE_OFF, NOTE_HOLD, or NO_NOTE NoteEvent.
    """
    # STEP 1: Separate simultaneous events (except at the very end of the sequence)
    for i in range(0,len(note_events)-1):
        if note_events[i].position == note_events[i+1].position:
            if not note_events[i+1].position == length_in_ticks-1:
                note_events[i+1] = NoteEvent(
                    position=note_events[i+1].position + 1,
                    midi_pitch=note_events[i+1].midi_pitch,
                    velocity=note_events[i+1].velocity,
                    event_type=note_events[i+1].event_type
                )

    # STEP 2: Add NOTE_HOLD NoteEvents
    # Pre-fill the final output with NO_NOTE NoteEvents by default
    output_note_events = [NoteEvent(position=i, midi_pitch=-1, velocity=-1, event_type=EventTypeToken.NO_NOTE) for i in range(length_in_ticks)]

    # Convert input note_events to deque for easier processing
    note_events = deque(note_events)

    # Keep track of NOTE_ON and NOTE_OFF NoteEvents
    switched_on_notes = deque([])
    switched_off_notes = deque([])
    def _cancel_out_on_and_off_events():
        while switched_off_notes and (switched_off_notes[-1].midi_pitch == switched_on_notes[-1].midi_pitch):
            switched_on_notes.pop()
            switched_off_notes.pop()

    # Fill in final output with NoteEvents
    for i in range(length_in_ticks):
        if note_events and i == note_events[0].position:
            current_note_event = note_events.popleft()
            output_note_events[i] = current_note_event

            if current_note_event.event_type == EventTypeToken.NOTE_ON:
                switched_on_notes.append(current_note_event)        
            elif current_note_event.event_type == EventTypeToken.NOTE_OFF:
                switched_off_notes.append(current_note_event)
                _cancel_out_on_and_off_events()

        elif switched_on_notes:
            output_note_events[i] = NoteEvent(
                position=i,
                midi_pitch=switched_on_notes[-1].midi_pitch,
                velocity=switched_on_notes[-1].velocity,
                event_type=EventTypeToken.NOTE_HOLD
            )
    
    non_null_note_event_positions = [note_event.position for note_event in output_note_events if not note_event.event_type == EventTypeToken.NO_NOTE]
    return output_note_events, non_null_note_event_positions


def get_musical_lengths(muspy_midi):
    ticks_per_qn         =   muspy_midi.resolution
    beats_per_measure    =   muspy_midi.time_signatures[0].numerator
    qn_per_beat          =   4/muspy_midi.time_signatures[0].denominator
    qn_per_measure       =   beats_per_measure*qn_per_beat

    length_in_qn         =   math.ceil(muspy_midi.get_end_time()/muspy_midi.resolution)
    length_in_ticks      =   length_in_qn*ticks_per_qn
    length_in_beats      =   length_in_qn/qn_per_beat
    length_in_measures   =   length_in_qn/qn_per_measure
    return length_in_ticks,length_in_beats,length_in_measures


def get_note_event_rep(muspy_midi, length_in_ticks):
    note_events_without_holds = [note_event for note in muspy_midi.tracks[0].notes for note_event in separate_on_and_off_events(note)]
    note_events_without_holds = sorted(note_events_without_holds, key=operator.attrgetter('position'))
    note_events, non_null_note_event_positions = insert_note_holds(note_events_without_holds, length_in_ticks)
    return note_events,non_null_note_event_positions
