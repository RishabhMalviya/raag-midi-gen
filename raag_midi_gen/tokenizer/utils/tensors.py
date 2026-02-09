import torch
import numpy as np
from numba import float64, int64, vectorize

from raag_midi_gen.tokenizer.vocab import NOTE_VOCAB_SIZE, OCTAVE_VOCAB_SIZE, EVENT_TYPE_VOCAB_SIZE


# PITCH

@vectorize([int64(int64)])
def midi_pitch_to_note_token_value(midi_pitch: int):
    """
    Converts a MIDI pitch number to its corresponding note and octave tokens.

    Args:
        midi_pitch: An integer MIDI pitch number (0-127)
    """
    return np.mod(midi_pitch, 12)


def get_pitch_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=False):
    midi_pitches = np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions]

    pitch = torch.zeros(length_in_ticks, dtype=torch.int64)
    pitch[non_null_note_event_positions] = torch.from_numpy(midi_pitch_to_note_token_value(midi_pitches)).to(torch.int64)

    if one_hot:
        pitch = torch.nn.functional.one_hot(pitch, num_classes=NOTE_VOCAB_SIZE).float()

    return pitch


# OCTAVE

@vectorize([int64(int64)])
def midi_pitch_to_octave_token_value(midi_pitch: int):
    """
    Converts a MIDI pitch number to its corresponding note and octave tokens.

    Args:
        midi_pitch: An integer MIDI pitch number (0-127)
    """
    return np.floor_divide(midi_pitch, 12)


def get_octave_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=False):
    midi_pitches = np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions]

    octave = torch.zeros(length_in_ticks, dtype=torch.int64)
    octave[non_null_note_event_positions] = torch.from_numpy(midi_pitch_to_octave_token_value(midi_pitches)).to(torch.int64)

    if one_hot:
        octave =torch.nn.functional.one_hot(octave, num_classes=OCTAVE_VOCAB_SIZE).float()

    return octave


# VELOCITY

@vectorize([float64(int64)])
def midi_velocity_to_normalized_value(midi_velocity: int):
    """
    Converts a MIDI pitch (between 0-127) to a value between 0-1

    Args:
         midi_velocity: An integer MIDI velocity value (0-127)
    """
    return np.interp(midi_velocity, [0,127], [0,1])


def get_velocity_tensor(note_events, non_null_note_event_positions, length_in_ticks):
    midi_velocities = np.array([note_event.velocity for note_event in note_events])[non_null_note_event_positions]

    velocity = np.zeros((length_in_ticks,1), dtype=float)
    velocity[non_null_note_event_positions,0] = np.interp(midi_velocities, [0,127], [0,1])
    velocity = torch.from_numpy(velocity)

    return velocity


# EVENT TYPE

def get_event_type_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=False):
    note_event_types = np.array([note_event.event_type.value for note_event in note_events])[non_null_note_event_positions]

    event_type = torch.zeros(length_in_ticks, dtype=torch.int64)
    event_type[non_null_note_event_positions] = torch.from_numpy(note_event_types).to(torch.int64)

    if one_hot:
        event_type = torch.nn.functional.one_hot(event_type, num_classes=EVENT_TYPE_VOCAB_SIZE).float()

    return event_type


# POSITION

def get_position_tensor(length_in_ticks, length_in_beats, length_in_measures):
    position = np.empty((length_in_ticks,6), dtype=float)

    # Periodicity of Beat
    position[...,0]  =  np.sin(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))
    position[...,1]  =  np.cos(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))
    
    # Periodicity of Measure
    position[...,2]  =  np.sin(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))
    position[...,3]  =  np.cos(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))

    # Periodicity of Full MIDI Melody    
    position[...,4]  =  np.sin(np.linspace(0, 2*np.pi, length_in_ticks))
    position[...,5]  =  np.cos(np.linspace(0, 2*np.pi, length_in_ticks))
    
    position = torch.from_numpy(position)

    return position
