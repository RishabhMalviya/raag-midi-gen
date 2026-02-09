import muspy
import torch

from raag_midi_gen.tokenization.vocab import (
    EventTypeToken,
    NOTE_VOCAB_DECODABLE_TOKENS_START, NOTE_VOCAB_DECODABLE_TOKENS_END,
    OCTAVE_VOCAB_DECODABLE_TOKENS_START, OCTAVE_VOCAB_DECODABLE_TOKENS_END,
    EVENT_TYPE_VOCAB_DECODABLE_TOKENS_START, EVENT_TYPE_VOCAB_DECODABLE_TOKENS_END
)


DEFAULT_TEMPO = 120
DEFAULT_TPQ = 96


def _initialize_muspy_midi(tempo, tpq) -> muspy.Music:
    my_midi = muspy.Music()
    
    my_midi.resolution = tpq
    my_midi.tempos = [muspy.Tempo(0, tempo)]
    my_midi.time_signatures = [muspy.TimeSignature(0, 4, 4)]
    my_midi.tracks = [muspy.Track(program=0, is_drum=False, notes = [])]

    return my_midi


def _get_midi_pitches(pitches: torch.Tensor, octaves: torch.Tensor):
    with torch.no_grad():
        # Slice out special tokens before argmax, so that values are 0-indexed and correspond to actual note pitches/octaves
        pitches, octaves = \
            torch.argmax(pitches[..., NOTE_VOCAB_DECODABLE_TOKENS_START:NOTE_VOCAB_DECODABLE_TOKENS_END], dim=-1), \
                torch.argmax(octaves[..., OCTAVE_VOCAB_DECODABLE_TOKENS_START:OCTAVE_VOCAB_DECODABLE_TOKENS_END], dim=-1)

        # Because the pitches/octaves are 0-indexed, they can directly be used to compute MIDI pitches
        midi_pitches = octaves*12 + pitches

    return midi_pitches


def decode_output(
    velocities: torch.Tensor,
    pitches: torch.Tensor,
    octaves: torch.Tensor,
    event_types: torch.Tensor,
    tempo: int = DEFAULT_TEMPO,
    tpq: int = DEFAULT_TPQ
):    
    notes_tracker = {}
    final_midi = _initialize_muspy_midi(tempo, tpq)    

    midi_velocities = (velocities.detach() * 127).clamp(0,127).round().long()
    midi_pitches = _get_midi_pitches(pitches, octaves)

    len_in_ticks = velocities.shape[0]
    for curr_on_timestep in range(len_in_ticks):
        with torch.no_grad():
            event_type = EventTypeToken(torch.argmax(event_types[curr_on_timestep][EVENT_TYPE_VOCAB_DECODABLE_TOKENS_START:EVENT_TYPE_VOCAB_DECODABLE_TOKENS_END]).item())
    
        if event_type in [EventTypeToken.NOTE_ON, EventTypeToken.NOTE_OFF]:
            midi_pitch = midi_pitches[curr_on_timestep].item()
            current_velocity = midi_velocities[curr_on_timestep].item()

            # Close out any existing note for this pitch
            if midi_pitch in notes_tracker:
                prev_on_timestep, prev_velocity = notes_tracker[midi_pitch]

                final_midi.tracks[0].notes.append(muspy.Note(
                    time      = prev_on_timestep,
                    duration  = curr_on_timestep - prev_on_timestep,
                    pitch     = midi_pitch,
                    velocity  = prev_velocity
                ))

                del notes_tracker[midi_pitch]

            # Track new note for this pitch if event_type is NOTE_ON
            if event_type == EventTypeToken.NOTE_ON:
                notes_tracker[midi_pitch] = (curr_on_timestep, current_velocity)
    
    # Finally, close out any unclosed notes. Default to a duration of a quarter note
    for midi_pitch, info in notes_tracker.items():
        on_timestep, velocity = info

        final_midi.tracks[0].notes.append(muspy.Note(
            time     = on_timestep,
            duration = tpq if on_timestep + tpq < len_in_ticks else len_in_ticks - on_timestep,
            pitch    = midi_pitch,
            velocity = velocity
        ))
    
    return final_midi


if __name__ == "__main__":
    import random

    from raag_midi_gen.tokenization.encoding import encode_target
    from raag_midi_gen.datasets import midi_files_dataset
    from raag_midi_gen.utils.midi_utils import play_muspy_music


    dataset = midi_files_dataset.get_dataset()
    midi_filename, test_muspy_midi = dataset['Taraana - Antra 2.0.mid']

    target_encoding = encode_target(test_muspy_midi)
    pitches = target_encoding['pitch']
    octaves = target_encoding['octave']
    velocities = target_encoding['velocity']
    event_types = target_encoding['event_type']

    decoded_midi = decode_output(velocities, pitches, octaves, event_types)

    print(midi_filename)
    print('------------------')
    play_muspy_music(decoded_midi)
