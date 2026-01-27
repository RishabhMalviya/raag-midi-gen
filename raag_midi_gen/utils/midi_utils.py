from midi2audio import FluidSynth
from IPython.display import Audio, display
from midiutil import MIDIFile
import muspy



def play_midi_file(midi_file_path: str, audio_file_path: str = 'tmp.wav', sf2_path: str = '/usr/share/sounds/sf2/default-GM.sf2'):
    """
    Other possible SF2 File Paths are `TimGM6mb.sf2` and `FluidR3_GM.sf2`
    """

    fs = FluidSynth(sf2_path)
    fs.midi_to_audio(midi_file_path, audio_file_path)


    display(Audio(audio_file_path, autoplay=True))


def play_midiutil_output(midiutil_file: MIDIFile):
    with open('tmp.mid', 'wb') as output_file:
        midiutil_file.writeFile(output_file)

    play_midi_file('tmp.mid')


def play_muspy_music(muspy_music: muspy.Music):
    muspy.write_midi(path='tmp.mid', music=muspy_music)

    play_midi_file('tmp.mid')
