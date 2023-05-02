import pickle
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

# feature extraction Audio-> Numbers


def extract_features(data):

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sample_rate).T, axis=0)
    return mel


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    return res1


# Execution starts here
data_path = pd.read_csv("./data_path.csv")
Features = pd.read_csv("./features.csv")

X = Features.iloc[:, :-1].values
Y = Features['labels'].values
X = preprocessing.normalize(X)

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, shuffle=True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# model retrieval
model_name = "model.pkl"
with open(model_name, 'rb') as file:
    model = pickle.load(file)


print("Accuracy of our model on test data : ",
      model.evaluate(x_test, y_test)[1]*100, "%")


# ---For Custom Input---

data, sample_rate = librosa.load("./data/test/YAF_base_neutral.wav")
input = extract_features(data)
testFile = np.array([input])
testFile = preprocessing.normalize(testFile)

# for plotting
samplingFrequency, signalData = wavfile.read(
    './data/test/YAF_base_neutral.wav')
librosa.display.waveshow(data)
plt.title("Waveform")
plt.show()
plt.specgram(input, Fs=samplingFrequency)
plt.title("Mel Spectogram")
plt.show()
r2 = np.array(range(0, 128, 1))
plt.scatter(r2, input)
plt.title("Feature Values")
plt.show()
plt.scatter(r2, testFile)
plt.title("Normalised Feature Values")
plt.show()


testFile = scaler.transform(testFile)
testFile = np.expand_dims(testFile, -1)
print(testFile.shape)
predArray = model.predict(testFile)
# print(predArray)

predictedEmotion = encoder.inverse_transform(predArray)
transformArr = [['happy'], ['angry'], ['sad'], ['fear'], [
    'disgust'], ['neutral'], ['calm'], ['surprise']]
here = encoder.transform(transformArr).toarray()
predictedResult = []
auxArr = []
for ele in here:
    aux = []
    for i in range(len(str(ele))):
        if str(ele)[i] == '0' or str(ele)[i] == '1':
            aux.append(str(ele)[i])
    auxArr.append(aux)

for ind1 in range(len(here)):
    for ind2 in range(len(ele)):
        if here[ind1][ind2] == 1:
            predictedResult.append([predArray[0][ind2], transformArr[ind1][0]])

predictedResult.sort(key=lambda x: x[0], reverse=1)

print("Predicted emotion: "+predictedEmotion[0][0])
print("\n----------- Prediction Summary --------------")
for ele in predictedResult:
    print(f"{ele[0]}  -> {ele[1]}")
