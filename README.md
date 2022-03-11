# Scream_Detection
소리 데이터를 이해하고, Frame processing, Feature extraction을 통해 네트워크에 입력할 데이터를 만든다. CNN을 이용해 학습하며, Binary Classification으로 비명소리인지 아닌지를 구분하게 된다.

## 1. 사용 라이브러리
1) 소리 데이터 예시 코드
- Pandas, librosa : 소리 데이터 전처리
- Numpy : 벡터 생성
- sounddevice : 소리 재생
- matplotlib : 소리 데이터 시각화

2) 프로젝트 코드
- Pandas : 레이블 정보 모음 csv 파일 생성, 
- librosa : Frame processing, Feature extraction(Mel Spectrogram)
- Numpy : 벡터 생성
- Pytorch : Dataset & DataLoader 생성, CNN 구성
- Scikit-learn : 정확도, F1_score 사용
- matplotlib : Overfitting/Underfitting 그래프 확인
   
2-1) 프로젝트 데모 실행코드
- PyQT5, pyqtgraph
- pyaudio
- librosa
- Pytorch

## 2. 알고리즘 순서도

## 3. 네트워크 구성도
![detection](https://user-images.githubusercontent.com/86700191/157643069-3c3a71e5-31bc-4862-be1e-bde9611f4d0f.png)

## 4. 결과
- 레이블과 파일 리스트 정리

![label](https://user-images.githubusercontent.com/86700191/157155327-bb6f79cb-f9e3-460e-bce0-f71ab6617339.PNG)
![list](https://user-images.githubusercontent.com/86700191/157155333-d63f4d61-3c33-4a08-8a2a-96d925005284.PNG)

- Dataset 크기

![dataset](https://user-images.githubusercontent.com/86700191/157155455-f239bf8c-9c73-4701-9ea3-8d65cbc54558.PNG)

- 학습 진행 과정(Tensor 크기)

![train](https://user-images.githubusercontent.com/86700191/157013480-ec9b8a31-8e2f-4b7d-8281-ec21bf12acd0.PNG)

- epoch 과정(95~100)

![epoch](https://user-images.githubusercontent.com/86700191/157155669-88310d7c-33c8-4414-9e3e-920b4eb3149a.PNG)

- Train, Validation 손실값 및 정확도

![mat](https://user-images.githubusercontent.com/86700191/157155708-689a110d-bd8d-4881-9aa5-ca69cd8e1b47.PNG)

## 5. 유의점
- 해당 코드는 Google Colab으로 다루었지만, 소리데이터를 직접 들으면서 작업하는데에는 무리가 있다. 클라우드 기반이기 때문에 소리를 출력하는 기본장치가 없는 것으로 보인다.
- 파일 리스트가 정리된 xlsx 파일의 시트를 보면 정리는 잘 되어있지만 레이블의 한글명은 자음과 모음이 분리되어 있는 것을 볼 수있다. 이것은 압축을 풀기 전과 풀고 난 후 파일명에 차이점이 있는데 영어명에서 한글명으로 바뀌는 파일이 존재하고 아마 이 부분에서 인코딩의 차이가 있는 것으로 보인다.

## 6. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [Pandas 공식 설명](https://pandas.pydata.org/docs/reference/index.html)
- [librosa 공식 설명](https://librosa.org/doc/latest/index.html)
- [Base Code & data](https://github.com/bjpublic/DeepLearningProject)
- [Mel-Spectrgram 설명](https://newsight.tistory.com/294)
- [사이킷런 accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) / [사이킷런 f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
- [pyaudio API 문서](http://people.csail.mit.edu/hubert/pyaudio/docs/)
- [pyqtgraph API 문서](https://pyqtgraph.readthedocs.io/en/latest/apireference.html)
- [PyQT5 사용법](https://ybworld.tistory.com/10)