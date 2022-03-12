import sys
import numpy as np

import pyaudio

import pyqtgraph as pg
from PyQt5 import QtCore, uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import librosa

import torch
from torch import nn
import torch.nn.functional as F

SAMPLING_RATE = 22050  # 음성데이터의 샘플링 레이트
CHUNK_SIZE = 22050  # 음성데이터를 불러올 때 한번에 22050개의 정수를 불러옴
form_class = uic.loadUiType("22.ui")[0]


def feature_engineering_mel_spectrum(signal, sampling_rate, n_mels):
    cur_frame_temp = signal

    # Mel Spectrograme 추출
    mel_spectrum_temp = librosa.feature.melspectrogram(
        y=cur_frame_temp,
        sr=sampling_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512,
    )
    # power -> dB로 변환
    mel_spectrum_temp = librosa.core.power_to_db(mel_spectrum_temp)
    feature_vector = mel_spectrum_temp
    feature_vector = feature_vector[np.newaxis, :, :, np.newaxis]
    return feature_vector


class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()      # Pyaudio 인스턴스화
        # 음성 데이터 스트림 열기
        self.stream = self.p.open(
            format=pyaudio.paFloat32,   # 비트 깊이 = 32bit float
            channels=1,
            rate=SAMPLING_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

    def read(self):
        data = self.stream.read(CHUNK_SIZE, False)      # 음성 데이터를 문자열로 반환
        y = np.fromstring(data, 'float32')      # 문자열 -> 32bit float 넘파이 배열
        self.signal.emit(y)     # GUI로 값 전달

    # 스트림 종료
    def close(self):
        print('멈춤')
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class MyWindow(QMainWindow, form_class):
    read_collected = QtCore.pyqtSignal(np.ndarray)      # emit으로 GUI로 전달 될 값들의 매개변수 변수형을 기록

    def __init__(self, model):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.read_collected.connect(self.update)    # 시그널 및 슬롯 설정

        self.model = model

        # Bargraph
        pg.setConfigOptions(background='w', foreground='k')     # key-value 형태로 여러개 인자 사용 가능.

        self.pw1 = pg.PlotWidget(title="BarGraph")
        self.pw1.showGrid(x=True, y=True)

        self.graph_box.addWidget(self.pw1)      # pw1을 위젯으로 등록
        self.pw1.setGeometry(4, 1, 10, 5)  # x, y, width, height

        ticks = [list(zip(range(2), ('Environmental sound', 'Scream sound')))]
        xax = self.pw1.getAxis('bottom')        # bottom이라는 AxisItem 반환
        xax.setTicks(ticks)     # 표시할 눈금을 명시적으로 결정
        self.show()

    def update(self, chunk):
        x = np.arange(2)

        feature_vector = feature_engineering_mel_spectrum(chunk, SAMPLING_RATE, 64)
        feature_vector = torch.tensor(feature_vector).float()
        feature_vector = feature_vector.squeeze(3).unsqueeze(1)
        y_softmax = float(
            torch.sigmoid(self.model(feature_vector)).detach().numpy()
        )

        if y_softmax > 0.5:
            pixmap = QPixmap("img/scream.png")
            self.label_5.setPixmap(QPixmap(pixmap))
        else:
            pixmap = QPixmap("img/normal.png")
            self.label_5.setPixmap(QPixmap(pixmap))

        self.pw1.clear()
        barchart = pg.BarGraphItem(
            x=x, height=[1 - y_softmax, y_softmax], width=1, brush=(159, 191, 229)
        )
        self.pw1.addItem(barchart)


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        # Convolution Layer
        self.conv1 = nn.Conv2d(1, 32, (64, 1))
        self.conv2 = nn.Conv2d(32, 64, (1, 9), stride=4)

        # Nomalization Layer
        self.batch1 = nn.BatchNorm2d(32)
        self.batch2 = nn.BatchNorm2d(64)

        # fully connected layer
        self.fc1 = nn.Linear(64*1*9, 1)

        # 활성화 함수 ReLU
        self.relu = nn.ReLU()

    def forward(self, x):

        # Conv -> Nomalization -> ReLU -> Dropout
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu(x)
        x = F.dropout2d(x, p=0.3, training=self.training)

        # Conv -> Nomalization -> ReLU -> Dropout
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu(x)
        x = F.dropout2d(x, p=0.3, training=self.training)

        # Flatten -> Fully-connected
        x = x.view(-1, 64*1*9)
        x = self.fc1(x)

        return x


model_dir = './test.pth'

model = CNN_model()
model.load_state_dict(torch.load(model_dir, map_location ='cpu'))

app = QApplication(sys.argv)
myWindow = MyWindow(model=model)
mic = MicrophoneRecorder(myWindow.read_collected)

interval = SAMPLING_RATE / CHUNK_SIZE
t = QtCore.QTimer()
t.timeout.connect(mic.read)
t.start(500)

myWindow.show()
app.exec_()