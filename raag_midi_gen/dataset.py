import os
from typing import Dict, Union, List
from pathlib import Path
from collections import defaultdict

import muspy
from torch.utils.data import Dataset, DataLoader

from raag_midi_gen.tokenization import tokenize


def list_all_midi_files(raag: str = 'Yaman'):
    data_dir = Path(__file__).parent / ".." / "data" / "midi_files" / raag

    midi_files_list = []
    for curr_dir, sub_dirs, files in os.walk(str(data_dir)):
        for filename in files:
            if filename[-3:] == 'mid':
                midi_files_list.append(os.path.join(os.path.abspath(curr_dir), filename))

    return midi_files_list


def map_plain_to_embellished_melodies(midi_files_list):
    """
    Organize the list of MIDI files so they're grouped together by the base melody
    
    Args:
        midi_files_list: A list of the file paths to all the MIDI files

    Returns:
        A python dictionary where each key is the base melody's file path (ending in `.0.mid`)
            The values are lists containing the embellished melody's file paths
            That means if a key has an empty list as it's values, then 
                there are no embellished melodies for that base melody.
    """
    files_by_melody = defaultdict(list)

    for file_path in midi_files_list:
        melody_name = file_path.split('/')[-1].split('.')[0]
        folder_path = '/'.join(file_path.split('/')[:-1])

        base_melody_file_path = folder_path + '/' + melody_name + '.0.mid'
        
        if not file_path == base_melody_file_path:
            files_by_melody[base_melody_file_path].append(file_path)
    
    return files_by_melody


def embellishment_dataset():
    class EmbellishmentDataset(Dataset):
        def __pair_plain_and_embellished_melodies(self, plain_to_embellished_melodies_mapping):
            self.melody_pairs = []

            for plain_melody_path, embellished_melodies_paths in plain_to_embellished_melodies_mapping.items():
                if embellished_melodies_paths:
                    for embellished_melody_path in embellished_melodies_paths:
                        self.melody_pairs.append((plain_melody_path, embellished_melody_path))

        def __init__(self, plain_to_embellished_melodies_mapping: Dict[str, List[str]]):
            self.__pair_plain_and_embellished_melodies(plain_to_embellished_melodies_mapping)

        def __getitem__(self, idx: int):
            idx = idx % len(self)

            plain_melody_file_path = self.melody_pairs[idx][0]
            embellished_melody_file_path = self.melody_pairs[idx][1]

            return tokenize(muspy.read_midi(plain_melody_file_path)), tokenize(muspy.read_midi(embellished_melody_file_path))

        def __len__(self):
            return len(self.melody_pairs)

    midi_files_list = list_all_midi_files()
    plain_to_embellished_melodies_mapping = map_plain_to_embellished_melodies(midi_files_list)

    return EmbellishmentDataset(plain_to_embellished_melodies_mapping)


if __name__ == "__main__":
    from pprint import pprint

    embellishment_dataset = embellishment_dataset()
    embellishment_dataloader = DataLoader(embellishment_dataset, batch_size=1, shuffle=True)

    plain_features, embellished_features = next(iter(embellishment_dataloader))
    print(plain_features['position'])
    print(embellished_features.keys())



def midi_files_dataset():
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