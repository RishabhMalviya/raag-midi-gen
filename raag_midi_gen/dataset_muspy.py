import os
from typing import Dict, Union
from pathlib import Path

import muspy
import numpy as np
from torch.utils.data import Dataset


class MIDIFilesDataset(Dataset):
    def __init__(self, midi_files_dict: Dict[str, str]):
        self._midi_files_dict = midi_files_dict
        self._midi_files_dict_keys = list(midi_files_dict.keys())

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

    here = Path(__file__).parent
    data_dir = here / ".." / "data" / "midi_files"

    for i, os_walk_tuple in enumerate(os.walk(str(data_dir))):
        files_list = os_walk_tuple[2]
        
        for filename in files_list:
            directory = os_walk_tuple[0]

            if filename[-3:] == 'mid':
                midi_files_dict[filename] = muspy.read_midi(os.path.join(directory, filename))

    return MIDIFilesDataset(midi_files_dict)

