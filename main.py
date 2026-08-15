import cv2
import time
import threading
import winsound
import numpy as np
import tensorflow as tf
import mediapipe as mp


# =========================================================
# 설정
# =========================================================

EYE_MODEL_PATH = "best_eye_model.keras"
FACE_MODEL_PATH = "face_landmarker.task"

IMG_SIZE = 64

CNN_OPEN_THRESHOLD = 0.55
EAR_CLOSED_THRESHOLD = 0.18

SLEEP_SECONDS = 2.0
SMOOTH_FRAMES = 5


# =========================================================
# 눈 OPEN / CLOSED CNN 모델
# =========================================================

print("눈 모델 로드 중...")

eye_model = tf.keras.models.load_model(
    EYE_MODEL_PATH,
    compile=False
)

print("눈 모델 로드 완료")


# =========================================================
# MediaPipe Face Landmarker
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)

print("Face Landmarker 로드 완료")


# =========================================================
# 눈 Landmark
# =========================================================

LEFT_EYE = [
    33, 133,
    159, 145,
    158, 153
]

RIGHT_EYE = [
    362, 263,
    386, 374,
    385, 380
]


# =========================================================
# 경고음
# =========================================================

alarm_running = False


def alarm_loop():
    global alarm_running

    while alarm_running:
        winsound.Beep(1100, 250)
        time.sleep(0.2)


def start_alarm():
    global alarm_running

    if alarm_running:
        return

    alarm_running = True

    threading.Thread(
        target=alarm_loop,
        daemon=True
    ).start()


def stop_alarm():
    global alarm_running
    alarm_running = False


# =========================================================
# 거리 계산
# =========================================================

def distance(p1, p2):
    return np.linalg.norm(
        np.array(p1, dtype=np.float32)
        - np.array(p2, dtype=np.float32)
    )


# =========================================================
# EAR 계산
# =========================================================

def calculate_ear(landmarks, indices, width, height):
    points = []

    for idx in indices:
        lm = landmarks[idx]

        points.append(
            (
                lm.x * width,
                lm.y * height
            )
        )

    horizontal = distance(
        points[0],
        points[1]
    )

    vertical1 = distance(
        points[2],
        points[3]
    )

    vertical2 = distance(
        points[4],
        points[5]
    )

    if horizontal == 0:
        return 0.0

    return (
        vertical1 + vertical2
    ) / (2.0 * horizontal)


# =========================================================
# 눈 영역 Crop
# =========================================================

def crop_eye(frame, landmarks, indices):
    height, width = frame.shape[:2]

    points = []

    for idx in indices:
        lm = landmarks[idx]

        points.append(
            (
                int(lm.x * width),
                int(lm.y * height)
            )
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    x2 = max(xs)

    y1 = min(ys)
    y2 = max(ys)

    eye_width = max(x2 - x1, 10)
    eye_height = max(y2 - y1, 10)

    margin_x = int(eye_width * 0.35)
    margin_y = int(eye_height * 1.2)

    x1 -= margin_x
    x2 += margin_x

    y1 -= margin_y
    y2 += margin_y

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    eye = frame[
        y1:y2,
        x1:x2
    ]

    return eye, (x1, y1, x2, y2)


# =========================================================
# CNN 눈 상태 예측
# =========================================================

def predict_eye(eye):
    if eye is None or eye.size == 0:
        return None, 0.0

    eye = cv2.resize(
        eye,
        (IMG_SIZE, IMG_SIZE)
    )

    eye = cv2.cvtColor(
        eye,
        cv2.COLOR_BGR2RGB
    )

    eye = np.expand_dims(
        eye,
        axis=0
    )

    # 모델 내부에 Rescaling(1./255)이 있으므로
    # 여기서 255로 나누지 않습니다.
    open_score = float(
        eye_model.predict(
            eye,
            verbose=0
        )[0][0]
    )

    # 학습 클래스가
    # ['close eyes', 'open eyes']
    # 였다면 0=CLOSED, 1=OPEN
    if open_score >= CNN_OPEN_THRESHOLD:
        return "OPEN", open_score

    return "CLOSED", 1.0 - open_score


# =========================================================
# CNN + EAR 통합 판정
# =========================================================

def combine_eye_result(cnn_state, cnn_confidence, ear):
    # EAR가 매우 낮으면 눈 감김으로 강하게 판단
    if ear < 0.15:
        return "CLOSED"

    # CNN과 EAR 둘 다 감김
    if (
        cnn_state == "CLOSED"
        and ear < EAR_CLOSED_THRESHOLD
    ):
        return "CLOSED"

    # CNN이 강하게 CLOSED
    if (
        cnn_state == "CLOSED"
        and cnn_confidence >= 0.80
    ):
        return "CLOSED"

    return "OPEN"


# =========================================================
# 웹캠
# =========================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not camera.isOpened():
    raise RuntimeError(
        "웹캠을 열 수 없습니다."
    )


# =========================================================
# 상태 변수
# =========================================================

closed_start = None

left_history = []
right_history = []

start_time = time.perf_counter()


print()
print("실시간 졸음 감지를 시작합니다.")
print("Q를 누르면 종료됩니다.")


# =========================================================
# Main Loop
# =========================================================

while True:
    ret, frame = camera.read()

    if not ret:
        print("카메라 프레임을 읽지 못했습니다.")
        break

    frame = cv2.flip(
        frame,
        1
    )

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        (
            time.perf_counter()
            - start_time
        ) * 1000
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # 기본 상태
    status = "FACE NOT DETECTED"
    status_color = (180, 180, 180)


    # =====================================================
    # 얼굴 검출 성공
    # =====================================================

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # EAR
        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE,
            width,
            height
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE,
            width,
            height
        )


        # 눈 영역 crop
        left_eye, left_box = crop_eye(
            frame,
            landmarks,
            LEFT_EYE
        )

        right_eye, right_box = crop_eye(
            frame,
            landmarks,
            RIGHT_EYE
        )


        # CNN 분석
        left_cnn, left_conf = predict_eye(
            left_eye
        )

        right_cnn, right_conf = predict_eye(
            right_eye
        )


        # CNN + EAR
        left_state = combine_eye_result(
            left_cnn,
            left_conf,
            left_ear
        )

        right_state = combine_eye_result(
            right_cnn,
            right_conf,
            right_ear
        )


        # =================================================
        # 최근 프레임 smoothing
        # =================================================

        left_history.append(left_state)
        right_history.append(right_state)

        if len(left_history) > SMOOTH_FRAMES:
            left_history.pop(0)

        if len(right_history) > SMOOTH_FRAMES:
            right_history.pop(0)


        left_closed_ratio = (
            left_history.count("CLOSED")
            / len(left_history)
        )

        right_closed_ratio = (
            right_history.count("CLOSED")
            / len(right_history)
        )


        left_final = (
            "CLOSED"
            if left_closed_ratio >= 0.6
            else "OPEN"
        )

        right_final = (
            "CLOSED"
            if right_closed_ratio >= 0.6
            else "OPEN"
        )


        # =================================================
        # 눈 BOX 표시
        # =================================================

        eye_results = [
            (
                "LEFT",
                left_final,
                left_box,
                left_ear,
                left_conf
            ),
            (
                "RIGHT",
                right_final,
                right_box,
                right_ear,
                right_conf
            )
        ]


        for name, state, box, ear, confidence in eye_results:
            x1, y1, x2, y2 = box

            if state == "OPEN":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} {state}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )


        # =================================================
        # 양쪽 눈 감김 확인
        # =================================================

        both_closed = (
            left_final == "CLOSED"
            and
            right_final == "CLOSED"
        )


        if both_closed:
            if closed_start is None:
                closed_start = time.time()

            elapsed = (
                time.time()
                - closed_start
            )

            if elapsed >= SLEEP_SECONDS:
                status = "SLEEPING"
                status_color = (0, 0, 255)

                start_alarm()

            else:
                status = "EYES CLOSED"
                status_color = (0, 165, 255)

                stop_alarm()

        else:
            closed_start = None

            stop_alarm()

            status = "AWAKE"
            status_color = (0, 255, 0)


        # =================================================
        # 디버깅 정보
        # =================================================

        cv2.putText(
            frame,
            f"L EAR: {left_ear:.3f}",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"R EAR: {right_ear:.3f}",
            (30, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"L CNN: {left_conf*100:.0f}%",
            (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"R CNN: {right_conf*100:.0f}%",
            (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )


    # =====================================================
    # 얼굴 미검출
    # =====================================================

    else:
        left_history.clear()
        right_history.clear()

        closed_start = None

        stop_alarm()


    # =====================================================
    # 메인 상태 표시
    # =====================================================

    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        status_color,
        3
    )


    if closed_start is not None:
        elapsed = (
            time.time()
            - closed_start
        )

        cv2.putText(
            frame,
            f"Closed: {elapsed:.1f}s",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            status_color,
            2
        )


    cv2.imshow(
        "Driver Drowsiness Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# 종료
# =========================================================

stop_alarm()

camera.release()

cv2.destroyAllWindows()

landmarker.close()