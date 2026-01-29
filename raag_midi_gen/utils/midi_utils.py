from midi2audio import FluidSynth
from IPython.display import Audio, display
import muspy



def play_midi_file(
    midi_file_path: str,
    audio_file_path: str = 'tmp.wav',
    sf2_file: str = 'default-GM.sf2'  # Other possible SF2 File Paths are `TimGM6mb.sf2` and `FluidR3_GM.sf2`
):
    sf2_path = f'/usr/share/sounds/sf2/{sf2_file}'

    fs = FluidSynth(sf2_path)
    fs.midi_to_audio(midi_file_path, audio_file_path)


    display(Audio(audio_file_path, autoplay=True))


def play_muspy_music(muspy_music: muspy.Music):
    muspy.write_midi(path='tmp.mid', music=muspy_music)

    play_midi_file('tmp.mid')
