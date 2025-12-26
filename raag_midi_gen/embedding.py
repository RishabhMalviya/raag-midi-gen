import math

import torch
import numpy as np
from torch import nn

from raag_midi_gen.tokenization import EventType


class NoteAttributeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Learnable Layer 1: Pitch
        NOTE_VOCAB_SIZE = 13
        NOTE_EMBED_DIM = 6
        self.pitch_embedder = nn.Embedding(NOTE_VOCAB_SIZE, NOTE_EMBED_DIM)
        
        # Learnable Layer 2: Octave
        OCTAVE_VOCAB_SIZE = 12
        OCTAVE_EMBED_DIM = 2
        self.octave_embedder = nn.Embedding(OCTAVE_VOCAB_SIZE, OCTAVE_EMBED_DIM)

        # Learnable Layer 3: Event Type
        EVENT_TYPE_SIZE = len(EventType)
        EVENT_TYPE_EMBED_DIM = 1
        self.event_type_embedder = nn.Embedding(EVENT_TYPE_SIZE, EVENT_TYPE_EMBED_DIM)

    def forward(self, position: np.ndarray, pitch: np.ndarray, octave: np.ndarray, velocity: np.ndarray, note_event_type: np.ndarray):
        """
        Forward pass with concatenated embeddings
        
        Args:
            position: numpy float array of positions encoded with sin/cos trick at three resolutions - beat, measure, and full clip
            note: numpy int array of note pitches (values in [0,12], where 0 means no note)
            octave: numpy int array of octaves (values in [0,12], where 0 means no note)
            velocity: numpy float array of note velocities in [0,1] range
            note_event_type: numpy int array denoting event types (follows the convention in EventType enum)
        
        Returns:
            Concatenated embeddings
        """
        # Pitch Embedding
        pitch_emb = self.pitch_embedder(torch.from_numpy(pitch[...,0]))

        # Octave Embedding
        oct_emb = self.octave_embedder(torch.from_numpy(octave[...,0]))

        # Event Type Embedding
        event_type_emb = self.event_type_embedder(torch.from_numpy(note_event_type[...,0]))

        return torch.cat([pitch_emb, oct_emb, torch.from_numpy(velocity), event_type_emb, torch.from_numpy(position)], dim=-1)