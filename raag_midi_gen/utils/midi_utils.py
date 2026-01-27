from midi2audio import FluidSynth
from IPython.display import Audio, display
from midiutil import MIDIFile



def play_midi_file(midi_file):
    fs = FluidSynth('/usr/share/sounds/sf2/')
    fs.midi_to_audio(midi_file, 'tmp.wav')

    audio_file_path = 'output.wav' 

    display(Audio(audio_file_path, autoplay=True))


def play_midiutil_output(midiutil_file: MIDIFile):
    with open('tmp.mid', 'wb') as output_file:
        midiutil_file.writeFile(output_file)

    play_midi_file('tmp.mid')
