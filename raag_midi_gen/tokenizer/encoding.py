from raag_midi_gen.tokenizer.utils.notes import get_note_event_rep, get_musical_lengths
from raag_midi_gen.tokenizer.utils.tensors import (
    get_pitch_tensor, get_octave_tensor, get_velocity_tensor, get_event_type_tensor, get_position_tensor
)


def encode_input(muspy_midi):
    length_in_ticks, length_in_beats, length_in_measures = get_musical_lengths(muspy_midi)
    note_events, non_null_note_event_positions = get_note_event_rep(muspy_midi, length_in_ticks)

    return {
        'pitch'      :      get_pitch_tensor(note_events, non_null_note_event_positions, length_in_ticks),
        'octave'     :     get_octave_tensor(note_events, non_null_note_event_positions, length_in_ticks),
        'velocity'   :   get_velocity_tensor(note_events, non_null_note_event_positions, length_in_ticks),
        'event_type' : get_event_type_tensor(note_events, non_null_note_event_positions, length_in_ticks),
        'position'   :   get_position_tensor(length_in_ticks, length_in_beats, length_in_measures)
    }


def encode_target(muspy_midi):
    length_in_ticks, _, _ = get_musical_lengths(muspy_midi)
    note_events, non_null_note_event_positions = get_note_event_rep(muspy_midi, length_in_ticks)

    return {
        'pitch'      :      get_pitch_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=True),
        'octave'     :     get_octave_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=True),
        'velocity'   :   get_velocity_tensor(note_events, non_null_note_event_positions, length_in_ticks),
        'event_type' : get_event_type_tensor(note_events, non_null_note_event_positions, length_in_ticks, one_hot=True),
    }


if __name__ == "__main__":
    from raag_midi_gen.datasets import midi_files_dataset

    midi_files_dataset = midi_files_dataset.get_dataset()
    test_muspy_midi = midi_files_dataset['Taraana - Antra 2.0.mid'][-1]

    encode_input(test_muspy_midi)

    # from raag_midi_gen.datasets import midi_files_dataset

    # midi_files_dataset = midi_files_dataset.get_dataset()
    # test_muspy_midi = midi_files_dataset['Taraana - Antra 2.0.mid'][-1]

    # encode_target(test_muspy_midi)
