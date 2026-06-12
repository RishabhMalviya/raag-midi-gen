import torch
import torch.nn as nn

from raag_midi_gen.model.components.embedding_layer import d_embeddings
from raag_midi_gen.tokenizer.vocab import (
    NOTE_VOCAB_SIZE, OCTAVE_VOCAB_SIZE, EVENT_TYPE_VOCAB_SIZE
)


class OutputLayer(nn.Module):
    def __init__(self, d_emdebbings: int=d_embeddings):
        super().__init__()
        
        # Velocity head: scalar output
        self.velocity_head = nn.Linear(d_emdebbings, 1)
        
        # Categorical heads: logits for each vocab
        self.pitch_head = nn.Linear(d_emdebbings, NOTE_VOCAB_SIZE)
        self.octave_head = nn.Linear(d_emdebbings, OCTAVE_VOCAB_SIZE)
        self.event_type_head = nn.Linear(d_emdebbings, EVENT_TYPE_VOCAB_SIZE)

    def forward(self, x: torch.Tensor):
        # --- Velocity ---
        velocities = torch.sigmoid(self.velocity_head(x))
        if not self.training:
            velocities = velocities.squeeze(-1)

        # --- Categorical ---
        # Output logits for training (feed it into nn.CrossEntropyLoss (which is nn.LogSoftmax + nn.NLLLoss))
        pitches = self.pitch_head(x)
        octaves = self.octave_head(x)
        event_types = self.event_type_head(x)

        # Actually convert to probabilities for inference
        if not self.training:
            pitches = torch.softmax(pitches, dim=-1)
            octaves = torch.softmax(octaves, dim=-1)
            event_types = torch.softmax(event_types, dim=-1)

        return velocities, pitches, octaves, event_types


if __name__ == "__main__":
    from raag_midi_gen.datasets import midi_files_dataset
    from raag_midi_gen.utils.midi_utils import play_muspy_music
    from raag_midi_gen.tokenizer import encode_input, decode_output

    from raag_midi_gen.model.components.embedding_layer import NoteEventEmbedding

    dataset = midi_files_dataset.get_dataset()
    muspy_midi = dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    embedder = NoteEventEmbedding()

    midi_info_arrays = encode_input(muspy_midi)
    embeddings = embedder(midi_info_arrays)
 
    output_layer = OutputLayer()
    velocities, pitches, octaves, event_types = output_layer(embeddings)

    play_muspy_music(decode_output(velocities, pitches, octaves, event_types))
