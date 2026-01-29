import math
import operator
from collections import deque

import torch
import numpy as np

from raag_midi_gen.architecture.embedding import NOTE_VOCAB_SIZE, OCTAVE_VOCAB_SIZE, EVENT_TYPE_VOCAB_SIZE
from raag_midi_gen.tokenization.notes import EventType, NoteEvent


def separate_on_and_off_events(note):
    """
    Converts a muspy Note object into two NoteEvent objects, one for Note On one for Note Off

    Args:
        note: one muspy Note object, containing information on `time`, `pitch`, `duration`, and `velocity`

    Returns:
        A list of two NoteEvent objects, one for the Note On event and one for the Note Off event        
    """
    return [
        NoteEvent(note.time,                 note.pitch, note.velocity, EventType.NOTE_ON ),
        NoteEvent(note.time + note.duration, note.pitch, note.velocity, EventType.NOTE_OFF)
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
    output_note_events = [NoteEvent(position=i, midi_pitch=-1, velocity=-1, event_type=EventType.NO_NOTE) for i in range(length_in_ticks)]

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

            if current_note_event.event_type == EventType.NOTE_ON:
                switched_on_notes.append(current_note_event)        
            elif current_note_event.event_type == EventType.NOTE_OFF:
                switched_off_notes.append(current_note_event)
                _cancel_out_on_and_off_events()

        elif switched_on_notes:
            output_note_events[i] = NoteEvent(
                position=i,
                midi_pitch=switched_on_notes[-1].midi_pitch,
                velocity=switched_on_notes[-1].velocity,
                event_type=EventType.NOTE_HOLD
            )
    
    non_null_note_event_positions = [note_event.position for note_event in output_note_events if not note_event.event_type == EventType.NO_NOTE]
    return output_note_events, non_null_note_event_positions


def compute_input_numpy_arrays(note_events, non_null_note_event_positions, length_in_ticks, length_in_beats, length_in_measures):
    # POSITION
    position = np.empty((length_in_ticks,6), dtype=float)
    position[...,0]  =  np.sin(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))       # Preiodicity of Beat
    position[...,1]  =  np.cos(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))
    position[...,2]  =  np.sin(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))    # Periodicity of Measure
    position[...,3]  =  np.cos(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))
    position[...,4]  =  np.sin(np.linspace(0, 2*np.pi, length_in_ticks))                       # Periodicity of Full MIDI Melody
    position[...,5]  =  np.cos(np.linspace(0, 2*np.pi, length_in_ticks))

    # PITCH
    pitch = np.zeros((length_in_ticks,1), dtype=int)
    pitch[non_null_note_event_positions,0] = np.mod(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12) + 1

    # OCTAVES
    octave = np.zeros((length_in_ticks,1), dtype=int)
    octave[non_null_note_event_positions,0] = np.floor_divide(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12)

    # VELOCITIES
    velocity = np.zeros((length_in_ticks,1), dtype=float)
    velocity[non_null_note_event_positions,0] = np.interp(np.array([note_event.velocity for note_event in note_events])[non_null_note_event_positions], [0,127], [0,1])

    # EVENT TYPES
    note_event_type = np.zeros((length_in_ticks,1), dtype=int)
    note_event_type[non_null_note_event_positions,0] = np.array([note_event.event_type.value for note_event in note_events])[non_null_note_event_positions]

    return position, pitch, octave, velocity, note_event_type


def encode_input(muspy_midi):
    ticks_per_qn         =   muspy_midi.resolution
    beats_per_measure    =   muspy_midi.time_signatures[0].numerator
    qn_per_beat          =   4/muspy_midi.time_signatures[0].denominator
    qn_per_measure       =   beats_per_measure*qn_per_beat

    length_in_qn         =   math.ceil(muspy_midi.get_end_time()/muspy_midi.resolution)
    length_in_ticks      =   length_in_qn*ticks_per_qn
    length_in_beats      =   length_in_qn/qn_per_beat
    length_in_measures   =   length_in_qn/qn_per_measure

    note_events_without_holds = [note_event for note in muspy_midi.tracks[0].notes for note_event in separate_on_and_off_events(note)]
    note_events_without_holds = sorted(note_events_without_holds, key=operator.attrgetter('position'))
    note_events, non_null_note_event_positions = insert_note_holds(note_events_without_holds, length_in_ticks)

    position, pitch, octave, velocity, note_event_type = compute_input_numpy_arrays(note_events, non_null_note_event_positions, length_in_ticks, length_in_beats, length_in_measures)

    return {
        'position': position,
        'pitch': pitch,
        'octave': octave,
        'velocity': velocity,
        'note_event_type': note_event_type
    }


def compute_output_torch_tensors(note_events, non_null_note_event_positions, length_in_ticks):
    # PITCH (one-hot)
    pitch_vals = torch.zeros(length_in_ticks, dtype=torch.int64)
    pitch_vals[non_null_note_event_positions] = torch.from_numpy(np.mod(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12) + 1).to(torch.int64)
    pitch = torch.nn.functional.one_hot(pitch_vals, num_classes=NOTE_VOCAB_SIZE).float()

    # OCTAVES (one-hot)
    octave_vals = torch.zeros(length_in_ticks, dtype=torch.int64)
    octave_vals[non_null_note_event_positions] = torch.from_numpy(np.floor_divide(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12)).to(torch.int64)
    octave = torch.nn.functional.one_hot(octave_vals, num_classes=OCTAVE_VOCAB_SIZE).float()

    # VELOCITIES (regression, 0-1)
    velocity = torch.zeros((length_in_ticks,1), dtype=torch.float32)
    velocity[non_null_note_event_positions,0] = torch.from_numpy(np.interp(np.array([note_event.velocity for note_event in note_events])[non_null_note_event_positions], [0,127], [0,1])).to(torch.float32)

    # EVENT TYPES (one-hot)
    note_event_type_vals = torch.zeros(length_in_ticks, dtype=torch.int64)
    note_event_type_vals[non_null_note_event_positions] = torch.from_numpy(np.array([note_event.event_type.value for note_event in note_events])[non_null_note_event_positions]).to(torch.int64)
    note_event_type = torch.nn.functional.one_hot(note_event_type_vals, num_classes=EVENT_TYPE_VOCAB_SIZE).float()

    return pitch, octave, velocity, note_event_type


def encode_target(muspy_midi):
    ticks_per_qn         =   muspy_midi.resolution
    length_in_qn         =   math.ceil(muspy_midi.get_end_time()/muspy_midi.resolution)
    length_in_ticks      =   length_in_qn*ticks_per_qn

    note_events_without_holds = [note_event for note in muspy_midi.tracks[0].notes for note_event in separate_on_and_off_events(note)]
    note_events_without_holds = sorted(note_events_without_holds, key=operator.attrgetter('position'))
    note_events, non_null_note_event_positions = insert_note_holds(note_events_without_holds, length_in_ticks)

    pitch, octave, velocity, note_event_type = compute_output_torch_tensors(note_events, non_null_note_event_positions, length_in_ticks)

    return {
        'pitch': pitch,
        'octave': octave,
        'velocity': velocity,
        'note_event_type': note_event_type
    }


if __name__ == "__main__":
    from raag_midi_gen.datasets.dataset import midi_files_dataset
    midi_files_dataset = midi_files_dataset()
    test_muspy_midi = midi_files_dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    encode_input(test_muspy_midi)
