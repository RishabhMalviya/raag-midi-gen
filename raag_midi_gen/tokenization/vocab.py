from enum import IntEnum


class NoteToken(IntEnum):
    """Note vocabulary tokens"""
    # No Note / Rest
    NO_NOTE = 0

    # Notes
    C        = 1
    C_SHARP  = 2
    D        = 3
    D_SHARP  = 4
    E        = 5
    F        = 6
    F_SHARP  = 7
    G        = 8
    G_SHARP  = 9
    A        = 10
    A_SHARP  = 11
    B        = 12

    # Special Tokens
    # MASK = 13
    # BOS = 14
    # EOS = 15


class OctaveToken(IntEnum):
    """Octave vocabulary tokens"""
    # No Octave / Rest
    NO_NOTE = 0

    # Octaves
    OCTAVE_0  = 1
    OCTAVE_1  = 2
    OCTAVE_2  = 3
    OCTAVE_3  = 4
    OCTAVE_4  = 5
    OCTAVE_5  = 6
    OCTAVE_6  = 7
    OCTAVE_7  = 8
    OCTAVE_8  = 9
    OCTAVE_9  = 10
    OCTAVE_10 = 11

    # Special tokens
    # MASK = 12
    # BOS = 13
    # EOS = 14


class EventTypeToken(IntEnum):
    """Event type vocabulary tokens"""
    # No Event / Rest
    NO_NOTE = 0

    # Event Types
    NOTE_ON    = 1
    NOTE_OFF   = 2
    NOTE_HOLD  = 3

    # Special tokens
    # MASK = 4
    # BOS = 5
    # EOS = 6


class Vocabulary:
    """
    Flexible vocabulary supporting both core tokens and special tokens.
    Provides bidirectional mapping between token names and IDs.
    """
    
    def __init__(self, token_type: type):
        """
        Args:
            token_enum: IntEnum class containing all tokens
            name: Optional name for the vocabulary (e.g., 'note', 'octave', 'event_type')
        """
        self.token_enum = token_type
        self.size = len(token_type)
        
        # Build mappings
        self.token_to_id = {token.name: token.value for token in token_type}
        self.id_to_token = {token.value: token.name for token in token_type}
    
    def encode(self, token_name: str) -> int:
        """
        Convert token name to ID.
        
        Args:
            token_name: Name of the token (e.g., 'C', 'MASK', 'NOTE_ON')
        
        Returns:
            Integer ID of the token
        
        Raises:
            KeyError: If token_name is not in vocabulary
        """
        return self.token_to_id[token_name]
    
    def decode(self, token_id: int) -> str:
        """
        Convert token ID to name.
        
        Args:
            token_id: Integer ID of the token
        
        Returns:
            Name of the token
        
        Raises:
            KeyError: If token_id is not in vocabulary
        """
        return self.id_to_token[token_id]
    
    def is_special_token(self, token_id: int) -> bool:
        """
        Check if a token ID corresponds to a special token.
        
        Args:
            token_id: Integer ID of the token
        
        Returns:
            True if the token is a special token (MASK, BOS, EOS)
        """
        token_name = self.id_to_token.get(token_id)
        return token_name in ['NO_NOTE', 'MASK', 'BOS', 'EOS'] if token_name else False
    
    def decodable_tokens_range(self):
        """
        Get a list of all non-special token IDs.
        
        Returns:
            List of integer IDs corresponding to non-special tokens
        """
        if self.token_enum == EventTypeToken:
            special_tokens = ['MASK', 'BOS', 'EOS']
        else:
            special_tokens = ['NO_NOTE', 'MASK', 'BOS', 'EOS']

        decodable_token_indices = [token.value for token in self.token_enum if token.name not in special_tokens]

        return min(decodable_token_indices), max(decodable_token_indices) + 1   


# Global vocabulary instances
NOTE_VOCAB = Vocabulary(token_type=NoteToken)
OCTAVE_VOCAB = Vocabulary(token_type=OctaveToken)
EVENT_TYPE_VOCAB = Vocabulary(token_type=EventTypeToken)

# Convenience constants
NOTE_VOCAB_SIZE = NOTE_VOCAB.size
NOTE_VOCAB_DECODABLE_TOKENS_START = NOTE_VOCAB.decodable_tokens_range()[0]
NOTE_VOCAB_DECODABLE_TOKENS_END = NOTE_VOCAB.decodable_tokens_range()[1]

OCTAVE_VOCAB_SIZE = OCTAVE_VOCAB.size
OCTAVE_VOCAB_DECODABLE_TOKENS_START = OCTAVE_VOCAB.decodable_tokens_range()[0]
OCTAVE_VOCAB_DECODABLE_TOKENS_END = OCTAVE_VOCAB.decodable_tokens_range()[1]

EVENT_TYPE_VOCAB_SIZE = EVENT_TYPE_VOCAB.size
EVENT_TYPE_VOCAB_DECODABLE_TOKENS_START = EVENT_TYPE_VOCAB.decodable_tokens_range()[0]
EVENT_TYPE_VOCAB_DECODABLE_TOKENS_END = EVENT_TYPE_VOCAB.decodable_tokens_range()[1]
