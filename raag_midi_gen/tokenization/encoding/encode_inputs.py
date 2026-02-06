import math
import operator

import torch
import numpy as np

from raag_midi_gen.tokenization.encoding.utils import separate_on_and_off_events, insert_note_holds


def compute_input_numpy_arrays(note_events, non_null_note_event_positions, length_in_ticks, length_in_beats, length_in_measures):
    # POSITION
    position = np.empty((length_in_ticks,6), dtype=float)
    position[...,0]  =  np.sin(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))       # Periodicity of Beat
    position[...,1]  =  np.cos(np.linspace(0, 2*np.pi*length_in_beats, length_in_ticks))
    position[...,2]  =  np.sin(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))    # Periodicity of Measure
    position[...,3]  =  np.cos(np.linspace(0, 2*np.pi*length_in_measures, length_in_ticks))
    position[...,4]  =  np.sin(np.linspace(0, 2*np.pi, length_in_ticks))                       # Periodicity of Full MIDI Melody
    position[...,5]  =  np.cos(np.linspace(0, 2*np.pi, length_in_ticks))

    # PITCH
    pitch = np.zeros((length_in_ticks,1), dtype=int)
    pitch[non_null_note_event_positions,0] = np.mod(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12) + 1
    pitch = torch.from_numpy(pitch)

    # OCTAVES
    octave = np.zeros((length_in_ticks,1), dtype=int)
    octave[non_null_note_event_positions,0] = np.floor_divide(np.array([note_event.midi_pitch for note_event in note_events])[non_null_note_event_positions], 12)
    octave = torch.from_numpy(octave)

    # VELOCITIES
    velocity = np.zeros((length_in_ticks,1), dtype=float)
    velocity[non_null_note_event_positions,0] = np.interp(np.array([note_event.velocity for note_event in note_events])[non_null_note_event_positions], [0,127], [0,1])
    velocity = torch.from_numpy(velocity)

    # EVENT TYPES
    event_type = np.zeros((length_in_ticks,1), dtype=int)
    event_type[non_null_note_event_positions,0] = np.array([note_event.event_type.value for note_event in note_events])[non_null_note_event_positions]
    event_type = torch.from_numpy(event_type)

    return position, pitch, octave, velocity, event_type


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

    position, pitch, octave, velocity, event_type = compute_input_numpy_arrays(note_events, non_null_note_event_positions, length_in_ticks, length_in_beats, length_in_measures)

    return {
        'position': position,
        'pitch': pitch,
        'octave': octave,
        'velocity': velocity,
        'event_type': event_type
    }


if __name__ == "__main__":
    from raag_midi_gen.datasets import midi_files_dataset

    midi_files_dataset = midi_files_dataset.get_dataset()
    test_muspy_midi = midi_files_dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    encode_input(test_muspy_midi)
