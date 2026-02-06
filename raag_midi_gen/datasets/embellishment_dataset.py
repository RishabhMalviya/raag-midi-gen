import os
from typing import Dict, List
from pathlib import Path
from collections import defaultdict

import muspy
from torch.utils.data import Dataset, DataLoader

from raag_midi_gen.tokenization.data_conversions.encoding.encode_inputs import encode_input, encode_target


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


class EmbellishmentDataset(Dataset):
    """
    This is a dataset class for loading pairs of (plain melody, embellished melody) MIDI files.

    Each `__getitiem__` call returns a tuple of (plain_melody_features: Dict, embellished_melody_features: Dict)

    1. `plain_melody_features` is the output of encode_input on a muspy.MIDI object corresponding to the plain melody
    2. `embellished_melody_features` is the output of encode_target on a muspy.MIDI object corresponding to the embellished melody
    """
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

        return encode_input(muspy.read_midi(plain_melody_file_path)), encode_target(muspy.read_midi(embellished_melody_file_path))

    def __len__(self):
        return len(self.melody_pairs)


def get_dataset():
    midi_files_list = list_all_midi_files()
    plain_to_embellished_melodies_mapping = map_plain_to_embellished_melodies(midi_files_list)

    return EmbellishmentDataset(plain_to_embellished_melodies_mapping)


if __name__ == "__main__":
    get_dataset = get_dataset()
    embellishment_dataloader = DataLoader(get_dataset, batch_size=1, shuffle=True)

    plain_features, embellished_features = next(iter(embellishment_dataloader))
    print(plain_features['position'])
    print(embellished_features.keys())
