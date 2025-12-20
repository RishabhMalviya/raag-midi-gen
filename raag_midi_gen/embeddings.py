import math

import muspy
import torch
import numpy as np

from torch.nn import Embedding


class MultiPartEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Learnable Layer 1: Notes
        NOTE_VOCAB_SIZE = 12
        NOTE_EMBED_DIM = 6
        self.note_embedder = nn.Embedding(NOTE_VOCAB_SIZE, NOTE_EMBED_DIM)
        
        # learnable Layer 2: Octaves
        OCTAVE_VOCAB_SIZE = 11
        OCTAVE_EMBED_DIM = 2
        self.octave_embedder = nn.Embedding(OCTAVE_VOCAB_SIZE, OCTAVE_EMBED_DIM)
    
    def forward(self, note_rep: np.ndarray, length, resolution, time_sig_num, time_sig_den):
        """
        Forward pass with concatenated embeddings
        
        Args:
            note_rep: muspy note representation of a MIDI clip (a numpy ndarray with dtype np.int64).
                        It has one `[position, pitch, length, velocity]` entry for each note.
                        `time` and `duration` are in ticks
            length: total length of the clip in beats (can be determined from muspy quite simply)
            resolution: PPQN (pulses per quarter note, a.k.a., ticks per quarter note)
            time_sig_num: numerator of time signature for MIDI clip
            time_sig_den: denominator of time signature for MIDI clip
        
        Returns:
            Concatenated embeddings
        """
        # Note Embedding
        pitches = note_rep[..., 1]%12
        note_emb = self.note_embedder(torch.from_numpy(pitches))

        # Octave Embedding
        octaves = note_rep[..., 1]//12 - 1
        oct_emb = self.octave_embedder(torch.from_numpy(octaves))

        # Velocity Embedding
        velocities = note_rep[...,3]
        vel_emb = torch.from_numpy(note_rep[..., 3])[...,None]/127
        
        # Length Embedding
        len_emb = torch.from_numpy(note_rep[..., 2])[...,None]/length
        
        # Position Embeddings (encoded at 3 different resolutions with sin and cos)
        positions = note_rep[..., 0]
        pos_emb = torch.from_numpy(positions)[...,None].repeat((1,)*positions.ndim + (6,)).float()  # create 6 copies of the start positions of the notes

        TICKS_PER_BEAT    = resolution
        W_BEAT            = 2*math.pi/TICKS_PER_BEAT
        pos_emb[...,0] = torch.sin(pos_emb[...,0]*W_BEAT)
        pos_emb[...,1] = torch.cos(pos_emb[...,1]*W_BEAT)
        
        TICKS_PER_MEASURE = resolution*((time_sig_num)/(time_sig_den/4))
        W_MEASURE         = 2*math.pi/TICKS_PER_MEASURE
        pos_emb[...,0] = torch.sin(pos_emb[...,0]*W_MEASURE)
        pos_emb[...,1] = torch.cos(pos_emb[...,1]*W_MEASURE)
        
        TICKS_PER_MELODY  = resolution*length
        W_MELODY          = 2*math.pi/TICKS_PER_MELODY
        pos_emb[...,0] = torch.sin(pos_emb[...,0]*W_MELODY)
        pos_emb[...,1] = torch.cos(pos_emb[...,1]*W_MELODY)
        
        
        combined = torch.cat([note_emb, oct_emb, vel_emb, len_emb, pos_emb], dim=-1)
        
        return combined


# FOR LOCAL TESTING
if __name__ == "__main__":
    from raag_midi_gen.dataset_muspy import get_dataset


    midi_files_dataset = get_dataset()
    test_muspy_midi = midi_files_dataset['Aeri Aali - Sthaayi 1.1_2.mid'][-1]

    embedding_module = MultiPartEmbedding()
    embedding = embedding_module(
        muspy.to_note_representation(test_muspy_midi),
        math.ceil(test_muspy_midi.get_end_time()/test_muspy_midi.resolution),
        test_muspy_midi.resolution,
        test_muspy_midi.time_signatures[0].numerator,
        test_muspy_midi.time_signatures[0].denominator
    )

