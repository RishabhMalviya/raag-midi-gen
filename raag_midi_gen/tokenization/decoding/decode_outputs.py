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

# Run this in a torch.no_grad() context!
# That will save the memory overhead of tracking the computational graph nodes for the argmax functions below.
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

    len_in_ticks = velocities.shape[0]
    midi_velocities = (velocities.detach() * 127).clamp(0,127).round().long()
    
    for t in range(len_in_ticks):
        event_type = EventTypeToken(torch.argmax(event_types[t][EVENT_TYPE_VOCAB_DECODABLE_TOKENS_START:EVENT_TYPE_VOCAB_DECODABLE_TOKENS_END]).item())  # ignore special tokens
    
        if not event_type in [EventTypeToken.NO_NOTE, EventTypeToken.NOTE_HOLD]:
            pitch  = torch.argmax(pitches[t][NOTE_VOCAB_DECODABLE_TOKENS_START:NOTE_VOCAB_DECODABLE_TOKENS_END]).item()      # ignore special tokens
            octave = torch.argmax(octaves[t][OCTAVE_VOCAB_DECODABLE_TOKENS_START:OCTAVE_VOCAB_DECODABLE_TOKENS_END]).item()  # ignore special tokens
            midi_pitch = (octave-1)*12 + pitch
    
            current_velocity = midi_velocities[t].item()
    
            if event_type == EventTypeToken.NOTE_ON:
                if midi_pitch in notes_tracker:  # Close if there is an existing note
                    on_timestep, velocity = notes_tracker[midi_pitch]
                    duration = t - on_timestep
                    final_midi.tracks[0].notes.append(muspy.Note(time=on_timestep, duration=duration, pitch=midi_pitch, velocity=velocity))
                notes_tracker[midi_pitch] = (t, current_velocity)
            if event_type == EventTypeToken.NOTE_OFF:
                if midi_pitch in notes_tracker:  # Add note to final_midi if there is a corresponding NOTE_ON event
                    on_timestep, velocity = notes_tracker[midi_pitch]
                    duration = t - on_timestep
                    final_midi.tracks[0].notes.append(muspy.Note(time=on_timestep, duration=duration, pitch=midi_pitch, velocity=velocity))
    
    # Finally, close out any unclosed notes. Default to a duration of a quarter note
    for midi_pitch, info in notes_tracker.items():
        on_timestep, velocity = info
        duration = tpq if on_timestep + tpq < len_in_ticks else len_in_ticks - on_timestep
        final_midi.tracks[0].notes.append(muspy.Note(time=on_timestep, duration=duration, pitch=midi_pitch, velocity=velocity))
    
    return final_midi


if __name__ == "__main__":
    import random

    from raag_midi_gen.tokenization.encoding.encode_targets import encode_target
    from raag_midi_gen.datasets import midi_files_dataset
    from raag_midi_gen.utils.midi_utils import play_muspy_music


    dataset = midi_files_dataset.get_dataset()
    midi_filename, test_muspy_midi = dataset[random.randint(1,100)]

    target_encoding = encode_target(test_muspy_midi)
    pitches = target_encoding['pitch']
    octaves = target_encoding['octave']
    velocities = target_encoding['velocity']
    event_types = target_encoding['note_event_type']

    decoded_midi = decode_output(velocities, pitches, octaves, event_types)

    print(midi_filename)
    print('------------------')
    play_muspy_music(decoded_midi)
