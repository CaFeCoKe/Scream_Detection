# Scream_Detection
소리 데이터를 이해하고, Frame processing, Feature extraction을 통해 네트워크에 입력할 데이터를 만든다. CNN을 이용해 학습하며, Binary Classification으로 비명소리인지 아닌지를 구분하게 된다.

## 1. 사용 라이브러리
1) 소리 데이터 예시 코드
- Pandas, librosa : 소리 데이터 전처리
- Numpy : 벡터 생성
- sounddevice : 소리 재생
- matplotlib : 소리 데이터 시각화

2) 프로젝트 코드
- Pandas : 레이블 정보 모음 xlsx 파일 생성 및 읽기
- librosa : Frame processing, Feature extraction(Mel Spectrogram)
- Numpy : 벡터 생성
- Pytorch : Dataset & DataLoader 생성, CNN 구성
- Scikit-learn : 정확도, F1_score 사용
- matplotlib : Overfitting/Underfitting 그래프 확인
   
3) 프로젝트 데모 실행코드
- PyQT5, pyqtgraph : 윈도우 ui 및 바그래프
- pyaudio : 마이크 입력
- librosa : Feature extraction(Mel Spectrogram)
- Pytorch : CNN 모델 불러오기 및 학습 진행

## 2. 알고리즘 순서도
![Algorithm_detection](https://user-images.githubusercontent.com/86700191/158113302-3a80dd2e-fc16-4fb8-975a-ffc493c00c6f.png)

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

- 데모 실행 결과
1) 비명이 아닌 소리 감지

![non_detection](https://user-images.githubusercontent.com/86700191/158050638-a0ef77f2-9e99-4f00-9487-fb2a4ff48712.PNG)

2) 비명 감지

![detection](https://user-images.githubusercontent.com/86700191/158050639-8ca6785f-302b-4ebf-94ec-f965ee2f1a54.PNG)

## 5. 유의점
- 해당 코드는 Google Colab으로 다루었지만, 소리데이터를 직접 들으면서 작업하는데에는 무리가 있다. 클라우드 기반이기 때문에 소리를 출력하는 기본장치가 없는 것으로 보인다.
- 데모를 실행 하려고 Colab에서 하려고 하면 안된다. Colab에는 디스플레이를 해줄수 있는 qt 모듈이 없다.
- 파일 리스트가 정리된 xlsx 파일의 시트를 보면 정리는 잘 되어있지만 레이블의 한글명은 자음과 모음이 분리되어 있는 것을 볼 수있다. 이것은 압축을 풀기 전과 풀고 난 후 파일명에 차이점이 있는데 영어명에서 한글명으로 바뀌는 파일이 존재하고 아마 이 부분에서 인코딩의 차이가 있는 것으로 보인다.
- 만약 데모 실행시 librosa에 대해 permissionError: (Errno 13)가 뜨게 된다면 파이썬 라이브러리들이 깔려있는 폴더 경로로 가서 librosa에 대해 쓰기 권한을 부여해줘야 한다.

  ![permission](https://user-images.githubusercontent.com/86700191/158050847-4bc01702-827b-42dd-a299-4a670cc1dc9e.PNG)
- 모델의 정확도와 달리 실제 데모를 실행할 때 비명 탐지가 잘 되지 않는 경우가 있다. 이것은 학습에 사용된 소리데이터가 실제 비명탐지에 적합하지 않았거나 개수의 부족 같은 문제가 있을 수 있고, 또한 마이크 기기에 따라 잡음이 비명처첨 처리 되는 경우도 있다.

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
