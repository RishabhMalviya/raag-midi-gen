from torch import nn

from raag_midi_gen.model.components.embedding_layer import NoteEventEmbedding, d_embeddings
from raag_midi_gen.model.components.output_layer import OutputLayer


class Model(nn.Module):
    def __init__(self):
        self.embedding_layer = NoteEventEmbedding()

        num_heads = 5
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_embeddings,
            dim_feedforward=d_embeddings,
            nhead=num_heads,
            batch_first=True  # Important! Expects [batch, seq, feature]
        )

        self.output_layer = OutputLayer()


    def forward(self, midi_info_arrays):
        embedding = self.embedding_layer(midi_info_arrays)
        encoded_embeddings = self.encoder_layer(embedding)
        velocities, pitches, octaves, event_types = self.output_layer(encoded_embeddings)

        return velocities, pitches, octaves, event_types
