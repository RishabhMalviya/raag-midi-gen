import os
from typing import Dict, Union

import numpy as np
from infmidi import Midi
from torch.utils.data import Dataset


class MIDIFilesDataset(Dataset):
    def __init__(self, _midi_files_dict: Dict[str, str]):
        self._midi_files_dict = _midi_files_dict
        self._midi_files_dict_keys = list(self._midi_files_dict.keys())

    def __getitem__(self, idx: Union[str, int]):
        if type(idx) in [int, str]:     
            if type(idx) is int: 
                if idx > len(self): raise IndexError
                else: str_idx = self._midi_files_dict_keys[idx]
                                    
            if type(idx) is str:
                if not idx in self._midi_files_dict: raise IndexError
                else: str_idx = idx

            return str_idx, self._midi_files_dict[str_idx]

        else: raise IndexError

    def __len__(self):
        return len(self._midi_files_dict) 
    

def get_dataset() -> MIDIFilesDataset:
    midi_files_dict = {}

    for i, os_walk_tuple in enumerate(os.walk('../../data/midi_files/')):
        files_list = os_walk_tuple[2]
        
        for filename in files_list:
            directory = os_walk_tuple[0]

            if filename[-3:] == 'mid':
                midi_files_dict[filename] = Midi.read(os.path.join(directory, filename))

    return MIDIFilesDataset(midi_files_dict)


def convert_midi_to_np_array(midi: Midi) -> np.array:
    ticks_per_beat = midi.ticks_per_beat
    
    array = np.zeros((midi.length*ticks_per_beat, 128), dtype=np.uint8)

    for note in midi.tracks[0].notes:
        note_start = int(note.location*ticks_per_beat)
        note_end = note_start + int(note.length*ticks_per_beat)

        array[note_start:note_end, note.value] = note.velocity

    return array
