import sounddevice as sd
from scipy.io.wavfile import write
def yy():
    # 录制音频
    myrecording = sd.rec(int(4 * 44100), samplerate=44100, channels=2)
    sd.wait()  # 录制直至结束
    write('output/output12.wav', 44100, myrecording)
yy()
