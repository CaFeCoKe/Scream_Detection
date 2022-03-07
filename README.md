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

## 2. 알고리즘 순서도

## 3. 네트워크 구성도

## 4. 결과
- 학습 진행 과정(Tensor 크기)

![train](https://user-images.githubusercontent.com/86700191/157013480-ec9b8a31-8e2f-4b7d-8281-ec21bf12acd0.PNG)

## 5. 유의점
- 해당 코드는 Google Colab으로 다루었지만, 소리데이터를 직접 들으면서 작업하는데에는 무리가 있다. 클라우드 기반이기 때문에 소리를 출력하는 기본장치가 없는 것으로 보인다.

## 6. 참고자료(사이트)
- [PyTorch 공식 설명](https://pytorch.org/docs/stable/index.html)
- [Pandas 공식 설명](https://pandas.pydata.org/docs/reference/index.html)
- [librosa 공식 설명](https://librosa.org/doc/latest/index.html)
- [Base Code & data](https://github.com/bjpublic/DeepLearningProject)
- [Mel-Spectrgram 설명](https://newsight.tistory.com/294)
- [사이킷런 accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) / [사이킷런 f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)