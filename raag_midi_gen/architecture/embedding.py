from typing import Dict

import torch
import numpy as np
from torch import nn

from raag_midi_gen.tokenization.encoding import EventType


NOTE_VOCAB_SIZE = 13  # 0-12, where 0 means no note
OCTAVE_VOCAB_SIZE = 12  # 0-11, where 0 means no note
EVENT_TYPE_VOCAB_SIZE = len(EventType)  # Number of event types


class NoteAttributeEmbedding(nn.Module):
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
                position: numpy float array of positions encoded with sin/cos trick at three resolutions - beat, measure, and full clip
                pitch: numpy int array of note pitches (values in [0,12], where 0 means no note)
                octave: numpy int array of octaves (values in [0,12], where 0 means no note)
                velocity: numpy float array of note velocities in [0,1] range
                note_event_type: numpy int array denoting event types (follows the convention in EventType enum)
        
        Returns:
            Concatenated embeddings
        """
        position=midi_info_arrays['position']
        pitch=midi_info_arrays['pitch']
        octave=midi_info_arrays['octave']
        velocity=midi_info_arrays['velocity']
        note_event_type=midi_info_arrays['note_event_type']

        # Pitch Embedding
        pitch_emb = self.pitch_embedder(torch.from_numpy(pitch[...,0]))

        # Octave Embedding
        oct_emb = self.octave_embedder(torch.from_numpy(octave[...,0]))

        # Event Type Embedding
        event_type_emb = self.event_type_embedder(torch.from_numpy(note_event_type[...,0]))

        return torch.cat([pitch_emb, oct_emb, torch.from_numpy(velocity), event_type_emb, torch.from_numpy(position)], dim=-1)


if __name__ == "__main__":
    from raag_midi_gen.datasets.dataset import midi_files_dataset
    from raag_midi_gen.tokenization.encoding import encode_input

    # Simple test
    embedder = NoteAttributeEmbedding()

    midi_files_dataset = midi_files_dataset()
    muspy_midi = midi_files_dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    midi_info_arrays = encode_input(muspy_midi)

    output_embeddings = embedder(midi_info_arrays)
 
    print("Output Embeddings Shape:", output_embeddings.shape)
