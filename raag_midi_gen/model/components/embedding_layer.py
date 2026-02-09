from typing import Dict

import torch
import numpy as np
from torch import nn

from raag_midi_gen.tokenizer.vocab import EVENT_TYPE_VOCAB_SIZE, NOTE_VOCAB_SIZE, OCTAVE_VOCAB_SIZE


class NoteEventEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Learnable Layer 1: Pitch
        NOTE_EMBED_DIM = 6
        self.pitch_embedder = nn.Embedding(NOTE_VOCAB_SIZE, NOTE_EMBED_DIM)
        
        # Learnable Layer 2: Octave
        OCTAVE_EMBED_DIM = 2
        self.octave_embedder = nn.Embedding(OCTAVE_VOCAB_SIZE, OCTAVE_EMBED_DIM)

        # Learnable Layer 3: Event Type
        EVENT_TYPE_EMBED_DIM = 1
        self.event_type_embedder = nn.Embedding(EVENT_TYPE_VOCAB_SIZE, EVENT_TYPE_EMBED_DIM)

    def forward(self, midi_info_arrays: Dict[str, np.ndarray]):
        """
        Forward pass with concatenated embeddings
        
        Args:
            midi_info_arrays: Dictionary containing the following keys:
                pitch: numpy int array of note pitches (values in [0,12], where 0 means no note)
                octave: numpy int array of octaves (values in [0,12], where 0 means no note)
                velocity: numpy float array of note velocities in [0,1] range
                event_type: numpy int array denoting event types (follows the convention in EventType enum)
                position: numpy float array of positions encoded with sin/cos trick at three resolutions - beat, measure, and full clip
        
        Returns:
            Concatenated embeddings
        """
        pitch       =  midi_info_arrays['pitch']
        octave      =  midi_info_arrays['octave']
        velocity    =  midi_info_arrays['velocity']
        event_type  =  midi_info_arrays['event_type']
        position    =  midi_info_arrays['position']

        # Pitch Embedding
        pitch_emb = self.pitch_embedder(pitch)

        # Octave Embedding
        oct_emb = self.octave_embedder(octave)

        # Event Type Embedding
        event_type_emb = self.event_type_embedder(event_type)

        return torch.cat([pitch_emb, oct_emb, velocity, event_type_emb, position], dim=-1)


if __name__ == "__main__":
    from raag_midi_gen.datasets import midi_files_dataset
    from raag_midi_gen.tokenizer import encode_input

    # Simple test
    embedder = NoteEventEmbedding()

    dataset = midi_files_dataset.get_dataset()
    muspy_midi = dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    midi_info_arrays = encode_input(muspy_midi)

    output_embeddings = embedder(midi_info_arrays)
 
    print("Output Embeddings Shape:", output_embeddings.shape)
