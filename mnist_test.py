import numpy as np
import chainer
from chainer import Chain, Variable, serializers
import chainer.functions as F
import chainer.links as L
import librosa
import librosa.display

class MyCNN(chainer.Chain):
	def __init__(self, n_in=784, n_units=500, n_out=10):
		super(MyCNN, self).__init(
			l1=L.Linear(n_in, n_units),
			l2=L.Linear(n_units, n_units),
			l3=L.Linear(n_units, n_out),
		)
	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)

class Mymethods():
	def cvtWavtondArray(self, filename) -> np.ndarray :
		y, sr = librosa.load(filename)
		S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
		log_S = librosa.amplitude_to_db(S, ref=np.max)
		return log_S

	def check_accuracy(self, net, xs, ts) :
		ys = net(xs)
		loss = F.softmax_cross_entropy(ys, ts)
		ys = np.argmax(ys.data, axis=1)
		cors = (ys == ts)
		num_cors = sum(cors)
		accuracy = num_cors / ts.shape[0]
		return accuracy, loss
