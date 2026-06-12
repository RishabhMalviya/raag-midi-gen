from enum import IntEnum


class NoteToken(IntEnum):
    """
    Note vocabulary tokens
    
    !!CAUTION!!: Only append to the Special Tokens section at the end of the enum.
    Everything else is chosen so that MIDI pitches can be efficiently encoded and decoded using numpy operations.
    """
    # Decodable Tokens
    C        = 0
    C_SHARP  = 1
    D        = 2
    D_SHARP  = 3
    E        = 4
    F        = 5
    F_SHARP  = 6
    G        = 7
    G_SHARP  = 8
    A        = 9
    A_SHARP  = 10
    B        = 11

    # Special Tokens
    NO_NOTE = 12
    PAD     = 13
    # MASK = 14
    # BOS = 15
    # EOS = 16


class OctaveToken(IntEnum):
    """
    Octave vocabulary tokens
    
    !!CAUTION!!: Only append to the Special Tokens section at the end of the enum.
    Everything else is chosen so that MIDI pitches can be efficiently encoded and decoded using numpy operations.
    """
    # Decodable Tokens
    OCTAVE_0  = 0
    OCTAVE_1  = 1
    OCTAVE_2  = 2
    OCTAVE_3  = 3
    OCTAVE_4  = 4
    OCTAVE_5  = 5
    OCTAVE_6  = 6
    OCTAVE_7  = 7
    OCTAVE_8  = 8
    OCTAVE_9  = 9
    OCTAVE_10 = 10

    # Special tokens
    NO_NOTE = 11
    PAD     = 12
    # MASK = 13
    # BOS = 14
    # EOS = 15


class EventTypeToken(IntEnum):
    """Event type vocabulary tokens"""
    # Decodable Tokens
    NO_NOTE    = 0
    NOTE_ON    = 1
    NOTE_OFF   = 2
    NOTE_HOLD  = 3

    # Special tokens
    PAD = 4
    # MASK = 5
    # BOS = 6
    # EOS = 7


class Vocabulary:
    """
    Flexible vocabulary supporting both core tokens and special tokens.
    Provides bidirectional mapping between token names and IDs.
    """
    
    def __init__(self, token_enum):
        """
        Args:
            token_enum: IntEnum class containing all tokens
            name: Optional name for the vocabulary (e.g., 'note', 'octave', 'event_type')
        """
        self.token_enum = token_enum
        self.size = len(self.token_enum)
        
        # Build Mappings
        self.token_to_id = {token.name: token.value for token in self.token_enum}
        self.id_to_token = {token.value: token.name for token in self.token_enum}

        # Specify Special Tokens
        if self.token_enum == EventTypeToken:
            self.special_tokens = ['PAD', 'MASK', 'BOS', 'EOS']
        else:
            self.special_tokens = ['NO_NOTE', 'PAD', 'MASK', 'BOS', 'EOS']
    
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
        return self.id_to_token[token_id] in self.special_tokens
    
    def decodable_tokens_range(self):
        """
        Get the start and end of the range of all non-special token IDs.
        
        Returns:
            start and end of the range of all non-special token IDs
        """
        decodable_token_indices = [token.value for token in self.token_enum if token.name not in self.special_tokens]

        return min(decodable_token_indices), max(decodable_token_indices) + 1   


# Global vocabulary instances
NOTE_VOCAB = Vocabulary(token_enum=NoteToken)
OCTAVE_VOCAB = Vocabulary(token_enum=OctaveToken)
EVENT_TYPE_VOCAB = Vocabulary(token_enum=EventTypeToken)


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
