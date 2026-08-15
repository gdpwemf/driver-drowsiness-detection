import time

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf

from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="😴",
    layout="centered"
)

st.title("Driver Drowsiness Detection")

st.write(
    "카메라를 켜면 눈 상태를 실시간으로 분석합니다. "
    "눈이 일정 시간 이상 감겨 있으면 SLEEPING으로 표시합니다."
)


# =========================================================
# SETTINGS
# =========================================================

EAR_THRESHOLD = st.slider(
    "Eye closed threshold",
    min_value=0.10,
    max_value=0.30,
    value=0.18,
    step=0.01
)

SLEEP_SECONDS = st.slider(
    "Sleeping detection time (seconds)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5
)

CNN_OPEN_THRESHOLD = 0.55
IMG_SIZE = 64


# =========================================================
# LOAD CNN MODEL
# =========================================================

@st.cache_resource
def load_eye_model():
    return tf.keras.models.load_model(
        "best_eye_model.keras",
        compile=False
    )


eye_model = load_eye_model()


# =========================================================
# MEDIAPIPE
# =========================================================

@st.cache_resource
def load_face_landmarker():

    BaseOptions = mp.tasks.BaseOptions

    FaceLandmarker = (
        mp.tasks.vision.FaceLandmarker
    )

    FaceLandmarkerOptions = (
        mp.tasks.vision.FaceLandmarkerOptions
    )

    RunningMode = (
        mp.tasks.vision.RunningMode
    )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="face_landmarker.task"
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )

    return FaceLandmarker.create_from_options(
        options
    )


landmarker = load_face_landmarker()


# =========================================================
# EYE LANDMARKS
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
# STATE
# =========================================================

if "closed_start" not in st.session_state:
    st.session_state.closed_start = None


runtime = {
    "closed_start": None,
    "left_history": [],
    "right_history": []
}


# =========================================================
# FUNCTIONS
# =========================================================

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1, dtype=np.float32)
        - np.array(p2, dtype=np.float32)
    )


def calculate_ear(
    landmarks,
    indices,
    width,
    height
):

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
    ) / (2 * horizontal)


def crop_eye(
    frame,
    landmarks,
    indices
):

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

    eye_width = max(
        x2 - x1,
        10
    )

    eye_height = max(
        y2 - y1,
        10
    )

    margin_x = int(
        eye_width * 0.35
    )

    margin_y = int(
        eye_height * 1.2
    )

    x1 = max(
        0,
        x1 - margin_x
    )

    x2 = min(
        width,
        x2 + margin_x
    )

    y1 = max(
        0,
        y1 - margin_y
    )

    y2 = min(
        height,
        y2 + margin_y
    )

    eye = frame[
        y1:y2,
        x1:x2
    ]

    return eye, (
        x1,
        y1,
        x2,
        y2
    )


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

    open_score = float(
        eye_model.predict(
            eye,
            verbose=0
        )[0][0]
    )

    if open_score >= CNN_OPEN_THRESHOLD:

        return (
            "OPEN",
            open_score
        )

    return (
        "CLOSED",
        1.0 - open_score
    )


def combine_result(
    cnn_state,
    cnn_confidence,
    ear
):

    # EAR가 확실히 낮으면 CLOSED
    if ear < EAR_THRESHOLD - 0.03:
        return "CLOSED"

    # CNN과 EAR가 둘 다 CLOSED
    if (
        cnn_state == "CLOSED"
        and ear < EAR_THRESHOLD
    ):
        return "CLOSED"

    # CNN이 매우 확신하는 경우
    if (
        cnn_state == "CLOSED"
        and cnn_confidence >= 0.85
    ):
        return "CLOSED"

    return "OPEN"


def smooth_state(
    history,
    new_state
):

    history.append(
        new_state
    )

    if len(history) > 5:
        history.pop(0)

    closed_ratio = (
        history.count("CLOSED")
        / len(history)
    )

    if closed_ratio >= 0.6:
        return "CLOSED"

    return "OPEN"


# =========================================================
# VIDEO CALLBACK
# =========================================================

def process_frame(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    # 거울처럼 표시
    img = cv2.flip(
        img,
        1
    )

    height, width = img.shape[:2]

    rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(
        mp_image
    )

    status = "FACE NOT DETECTED"

    status_color = (
        180,
        180,
        180
    )


    # =====================================================
    # FACE FOUND
    # =====================================================

    if result.face_landmarks:

        landmarks = (
            result.face_landmarks[0]
        )


        # -------------------------------------------------
        # EAR
        # -------------------------------------------------

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


        # -------------------------------------------------
        # EYE CROP
        # -------------------------------------------------

        left_eye, left_box = crop_eye(
            img,
            landmarks,
            LEFT_EYE
        )

        right_eye, right_box = crop_eye(
            img,
            landmarks,
            RIGHT_EYE
        )


        # -------------------------------------------------
        # CNN
        # -------------------------------------------------

        left_cnn, left_conf = predict_eye(
            left_eye
        )

        right_cnn, right_conf = predict_eye(
            right_eye
        )


        # -------------------------------------------------
        # HYBRID DECISION
        # -------------------------------------------------

        left_state = combine_result(
            left_cnn,
            left_conf,
            left_ear
        )

        right_state = combine_result(
            right_cnn,
            right_conf,
            right_ear
        )


        # -------------------------------------------------
        # SMOOTHING
        # -------------------------------------------------

        left_final = smooth_state(
            runtime["left_history"],
            left_state
        )

        right_final = smooth_state(
            runtime["right_history"],
            right_state
        )


        # -------------------------------------------------
        # DRAW EYES
        # -------------------------------------------------

        eye_results = [

            (
                "LEFT",
                left_final,
                left_box,
                left_ear
            ),

            (
                "RIGHT",
                right_final,
                right_box,
                right_ear
            )
        ]


        for (
            name,
            state,
            box,
            ear
        ) in eye_results:

            x1, y1, x2, y2 = box

            if state == "OPEN":

                color = (
                    0,
                    255,
                    0
                )

            else:

                color = (
                    0,
                    0,
                    255
                )


            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


            cv2.putText(
                img,
                f"{name} {state} EAR:{ear:.2f}",
                (
                    x1,
                    max(y1 - 8, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )


        # -------------------------------------------------
        # DROWSINESS TIMER
        # -------------------------------------------------

        both_closed = (
            left_final == "CLOSED"
            and
            right_final == "CLOSED"
        )


        if both_closed:

            if runtime["closed_start"] is None:

                runtime["closed_start"] = (
                    time.time()
                )


            closed_time = (
                time.time()
                - runtime["closed_start"]
            )


            if closed_time >= SLEEP_SECONDS:

                status = "SLEEPING"

                status_color = (
                    0,
                    0,
                    255
                )

            else:

                status = "EYES CLOSED"

                status_color = (
                    0,
                    165,
                    255
                )


            cv2.putText(
                img,
                f"Closed: {closed_time:.1f}s",
                (25, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )


        else:

            runtime["closed_start"] = None

            status = "AWAKE"

            status_color = (
                0,
                255,
                0
            )


        # EAR values
        cv2.putText(
            img,
            f"L EAR {left_ear:.3f} | R EAR {right_ear:.3f}",
            (25, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


    else:

        runtime[
            "closed_start"
        ] = None

        runtime[
            "left_history"
        ].clear()

        runtime[
            "right_history"
        ].clear()


    # =====================================================
    # MAIN STATUS
    # =====================================================

    cv2.putText(
        img,
        status,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3
    )


    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


# =========================================================
# WEBRTC
# =========================================================

webrtc_streamer(
    key="drowsiness-camera",

    video_frame_callback=process_frame,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=False
)


st.caption(
    "눈을 2초 이상 감고 있으면 SLEEPING으로 표시됩니다."
)
