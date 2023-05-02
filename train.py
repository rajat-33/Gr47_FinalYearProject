import pandas as pd
import numpy as np

import os
import sys
import pickle

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from keras import optimizers

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


Ravdess = "./data/audio_speech_actors_01-24/"
Crema = "./data/AudioWAV/"
Tess = "./data/TESS Toronto emotional speech set data/"
Savee = "./data/ALL/"

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]  # [03-01-01-01-01-01-01, wav]
        part = part.split('-')  # [03, 01, 01, 01, 01, 01, 01]
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))  # [1]
        file_path.append(Ravdess + dir + '/' + file)
        # ["./data/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"]

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])  # [1, 2, 4, 3]

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])  # [p1, p2...]
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                            5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
Ravdess_df.head()


crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)


tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[-1]
        if part == 'ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]  # part = sa06.wav
    ele = part[:-6]  # ele = sa
    if ele == 'a':
        file_emotion.append('angry')
    elif ele == 'd':
        file_emotion.append('disgust')
    elif ele == 'f':
        file_emotion.append('fear')
    elif ele == 'h':
        file_emotion.append('happy')
    elif ele == 'n':
        file_emotion.append('neutral')
    elif ele == 'sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()


# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Ravdess_df, Tess_df, Crema_df, Savee_df], axis=0)
# data_path.to_csv("data_path.csv", index=False)


def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3))
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

# ------------ Emotion-> path (librosa-path)


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


# taking any example and checking for techniques.
path = np.array(data_path.Path)[1]
data, sample_rate = librosa.load(path)

plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=data, sr=sample_rate)
# plt.show() to show the plot
Audio(path)


x = noise(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
# plt.show()
Audio(x, rate=sample_rate)

x = librosa.effects.time_stretch(data, rate=0.8)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)

x = shift(data)
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
Audio(x, rate=sample_rate)


# pitch is not included
x = librosa.effects.pitch_shift(
    data, sr=sample_rate, n_steps=6)  # included with changes
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y=x, sr=sample_rate)
# plt.show()
Audio(x, rate=sample_rate)


# feature extraction Audio->numbers
def extract_features(data):
    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(
        y=data, sr=sample_rate).T, axis=0)
    return mel


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation -> mel-arr1
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise -> mel - arr2
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching -> mel - arr3
    new_data = librosa.effects.time_stretch(data, rate=0.8)
    data_stretch_pitch = librosa.effects.pitch_shift(
        data, sr=sample_rate, n_steps=6)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result
 # res =[[wa], [n], [ps]]


X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    print(len(X))
    feature = get_features(path)  # feature =[[wa], [n], [ps]]
    for ele in feature:
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        X.append(ele)
        Y.append(emotion)


Features = pd.DataFrame(X)
Features['labels'] = Y
# Features.to_csv('features.csv', index=False)


# Emotion -> feature data

# Data Normalization
X = Features.iloc[:, :-1].values
Y = Features['labels'].values  # Y = [[e1], ... , [e36k]]-string
X = preprocessing.normalize(X)  # X = normalized[[1],..., [36k]]

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
# Y = [[0-> []], ... , [36k-> []]]

# data is ready-----------------------------------

# splitting data 75-25
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Data is ready to train and test

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='softmax')
])

adam = optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, validation_data=(
    x_test, y_test), epochs=100, batch_size=64)

print("Accuracy of our model on test data : ",
      model.evaluate(x_test, y_test)[1]*100, "%")
