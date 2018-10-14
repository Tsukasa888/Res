from __future__ import print_function
import librosa
import librosa.display
import numpy as np
"""
import matplotlib
matplotlib.use('Agg')
"""
import matplotlib.pyplot as plt
import matplotlib.style as ms

def main():
    ms.use('seaborn-muted')

    filename = "E:\\myapp\\sample.wav"

    y, sr = librosa.load(filename)

    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    log_S = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(12,4))

    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    plt.title('spectrogram')

    plt.colorbar(format='%+02.0f dB')

    #plt.savefig('test.png')

    plt.show()

if '__main__' == __name__:
    main()
