from collections import defaultdict

import torch
import torch.nn.functional as F
from midiutil import MIDIFile

from raag_midi_gen.tokenization.encoding import EventType
from raag_midi_gen.utils.midi_utils import play_midiutil_output


DEFAULT_TEMPO = 120
DEFAULT_TPQ = 96


def decode_output(
    velocities: torch.Tensor,
    pitches: torch.Tensor,
    octaves: torch.Tensor,
    event_type: torch.Tensor
):
    notes_tracker = {}
    final_midi = MIDIFile(numTracks=1)
    final_midi.addTempo(0,0,DEFAULT_TEMPO)
    
    len_in_ticks = velocities.shape[0]
    midi_velocities = (velocities.detach() * 127).clamp(0,127).round().long()
    
    for t in range(len_in_ticks):
        event_type = EventType(torch.argmax(event_types[t]).item())
    
        if not event_type in [EventType.NO_NOTE, EventType.NOTE_HOLD]:
            pitch  = torch.argmax(pitches[t][1:]).item()  # ignore the first index of the output pitch tensor, which corresponds to NO_NOTE
            octave = torch.argmax(octaves[t][1:]).item()  # ignore the first index of the output pitch tensor, which corresponds to NO_NOTE
            midi_pitch = octave*12 + pitch
    
            current_velocity = midi_velocities[t].item()
    
            if event_type == EventType.NOTE_ON:
                if midi_pitch in notes_tracker:  # Close if there is an existing note
                    on_timestep, velocity = notes_tracker[midi_pitch]
                    duration = t - on_timestep
                    final_midi.addNote(0, 0, midi_pitch, on_timestep/DEFAULT_TPQ, duration/DEFAULT_TPQ, velocity)
                notes_tracker[midi_pitch] = (t, current_velocity)
            if event_type == EventType.NOTE_OFF:
                if midi_pitch in notes_tracker:  # Add note to final_midi if there is a corresponding NOTE_ON event
                    on_timestep, velocity = notes_tracker[midi_pitch]
                    duration = t - on_timestep
                    final_midi.addNote(0, 0, midi_pitch, on_timestep/DEFAULT_TPQ, duration/DEFAULT_TPQ, velocity)
    
    # Finally, close out any unclosed notes. Default to a duration of a quarter note
    for midi_pitch, info in notes_tracker.items():
        on_timestep, velocity = notes_tracker[midi_pitch]
        duration = DEFAULT_TPQ if on_timestep + DEFAULT_TPQ < len_in_ticks else len_in_ticks - on_timestep
        final_midi.addNote(0, 0, midi_pitch, on_timestep/DEFAULT_TPQ, duration/DEFAULT_TPQ, velocity)

    print(final_midi.tracks[0].MIDIdata)
    
    return final_midi


if __name__ == "__main__":
    seq_len = 786

    velocities = torch.sigmoid(torch.randn(seq_len, 1))
    pitches = F.softmax(torch.randn(seq_len, 13), dim=-1)
    octaves = F.softmax(torch.randn(seq_len, 12), dim=-1)
    event_types = F.softmax(torch.randn(seq_len, 4), dim=-1)

    decoded_midi = decode_output(velocities, pitches, octaves, event_types)

    play_midiutil_output(decoded_midi)
