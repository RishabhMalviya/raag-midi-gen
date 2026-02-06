import math
import operator

import torch
import numpy as np

from raag_midi_gen.tokenization.vocab import NOTE_VOCAB_SIZE, OCTAVE_VOCAB_SIZE, EVENT_TYPE_VOCAB_SIZE
from raag_midi_gen.tokenization.encoding.utils import separate_on_and_off_events, insert_note_holds


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
    from raag_midi_gen.datasets import midi_files_dataset

    midi_files_dataset = midi_files_dataset.get_dataset()
    test_muspy_midi = midi_files_dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    encode_target(test_muspy_midi)
