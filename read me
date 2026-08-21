Driver Drowsiness Detection은 웹캠을 이용해 운전자의 눈 상태를 실시간으로 분석하고, 일정 시간 이상 눈이 감겨 있으면 졸음 상태로 판단해 경고음을 발생시키는 프로젝트입니다.

눈 상태 분석에는 TensorFlow와 Keras로 학습한 CNN 모델을 사용했으며, 학습 데이터는 MRL Eye Dataset을 사용했습니다. 학습된 모델은 TensorFlow.js 형식으로 변환하여 웹브라우저에서 직접 실행할 수 있도록 구성했습니다.

웹에서는 MediaPipe Face Landmarker를 이용해 얼굴과 눈 주변의 랜드마크를 검출하고, 좌우 눈 영역을 추출합니다. 추출한 눈 이미지를 CNN 모델에 입력하여 Open Eyes와 Close Eyes 상태를 분류합니다.

CNN 결과만 사용하는 것이 아니라 눈의 가로 길이와 세로 길이 비율을 계산하는 EAR 방식도 함께 사용합니다. CNN 결과와 EAR 값을 조합하여 눈이 열려 있는지 감겨 있는지를 최종적으로 판단하며, 최근 여러 프레임의 결과도 함께 확인하여 순간적인 오검출을 줄였습니다.

양쪽 눈이 모두 감긴 상태가 2초 이상 지속되면 SLEEPING 상태로 판단하고 Web Audio API를 이용해 경고음을 발생시킵니다. 눈을 다시 뜨면 경고음은 자동으로 중지됩니다.

index.html은 카메라 화면, Start Camera 버튼, 상태 표시 영역 등 웹페이지의 기본 화면을 구성합니다.

app.js는 TensorFlow.js 모델과 MediaPipe 모델을 불러오고, 웹캠 실행, 얼굴 및 눈 검출, EAR 계산, CNN 예측, 졸음 판단, 경고음 발생 등 실제 실시간 분석 기능을 담당합니다.

best_eye_model.keras는 Google Colab에서 TensorFlow와 Keras를 이용해 학습한 원본 눈 상태 분류 모델입니다.

model.json과 group1-shard1of1.bin은 학습된 Keras 모델을 TensorFlow.js 형식으로 변환한 파일이며 웹브라우저에서 눈 상태를 예측하는 데 사용합니다.

face_landmarker.task는 MediaPipe Face Landmarker가 얼굴 랜드마크를 검출할 때 사용하는 모델 파일입니다.

사용 기술은 Python, TensorFlow, Keras, TensorFlow.js, JavaScript, HTML, MediaPipe Tasks Vision, Web Audio API, Google Colab입니다.

사용 데이터셋은 MRL Eye Dataset입니다.

MRL Eye Dataset
https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset

TensorFlow
https://www.tensorflow.org/

Keras
https://keras.io/

TensorFlow.js
https://www.tensorflow.org/js

MediaPipe
https://ai.google.dev/edge/mediapipe/

프로그램을 실행하면 먼저 TensorFlow.js 모델과 MediaPipe Face Landmarker 모델을 불러옵니다. 모델 로딩이 완료되면 Start Camera 버튼을 누르고 브라우저의 카메라 권한을 허용하면 실시간 분석이 시작됩니다.

웹캠에서 입력되는 사진이나 영상은 저장하지 않으며 실시간 눈 상태 분석에만 사용됩니다.
