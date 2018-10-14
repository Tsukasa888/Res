import pyaudio
import wave

def main() :
    REC_SEC = 10
    WAVE_OUTPUTS = "sample.wav"
    iDeviceIndex = 0

    FORMAT = pyaudio.paInt16
    chans = 1
    _rate = 44100
    _chunk = 2**11
    _audio = pyaudio.PyAudio()
    stream = _audio.open(format=FORMAT, channels=chans, rate=_rate, input=True, input_device_index = iDeviceIndex, frames_per_buffer=_chunk)

    print("rec...")
    frames = []
    sec = 0
    for i in range(0, int(_rate/ _chunk * REC_SEC)) :
        if 0 == i % int(_rate/ _chunk * 1) :
            print(f"sec : {sec}")
            sec += 1
        data = stream.read(_chunk)
        frames.append(data)

    print("rec end.")

    stream.stop_stream()
    stream.close()
    _audio.terminate()

    waveF = wave.open(WAVE_OUTPUTS, 'wb')
    waveF.setnchannels(chans)
    waveF.setsampwidth(_audio.get_sample_size(FORMAT))
    waveF.setframerate(_rate)
    waveF.writeframes(b''.join(frames))
    waveF.close()

if '__main__' == __name__:
    main()
