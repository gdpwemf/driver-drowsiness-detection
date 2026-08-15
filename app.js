import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304";


const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const ctx = canvas.getContext("2d");

const statusText = document.getElementById("status");
const infoText = document.getElementById("info");
const startButton = document.getElementById("startButton");


const IMG_SIZE = 64;

const CNN_OPEN_THRESHOLD = 0.55;
const EAR_CLOSED_THRESHOLD = 0.18;

const SLEEP_SECONDS = 2.0;

const SMOOTH_FRAMES = 5;


const LEFT_EYE = [
    33,
    133,
    159,
    145,
    158,
    153
];

const RIGHT_EYE = [
    362,
    263,
    386,
    374,
    385,
    380
];


let eyeModel = null;
let faceLandmarker = null;

let closedStart = null;

let leftHistory = [];
let rightHistory = [];

let running = false;


async function loadModels() {

    statusText.innerText = "LOADING MODEL...";

    eyeModel = await tf.loadLayersModel(
        "./model/model.json"
    );


    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm"
    );


    faceLandmarker = await FaceLandmarker.createFromOptions(
        vision,
        {
            baseOptions: {
                modelAssetPath:
                    "./face_landmarker.task",

                delegate: "GPU"
            },

            runningMode: "VIDEO",

            numFaces: 1,

            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        }
    );


    statusText.innerText = "READY";
}


function distance(p1, p2) {

    const dx =
        p1.x - p2.x;

    const dy =
        p1.y - p2.y;

    return Math.sqrt(
        dx * dx +
        dy * dy
    );
}


function calculateEAR(
    landmarks,
    indices
) {

    const p1 =
        landmarks[indices[0]];

    const p2 =
        landmarks[indices[1]];

    const p3 =
        landmarks[indices[2]];

    const p4 =
        landmarks[indices[3]];

    const p5 =
        landmarks[indices[4]];

    const p6 =
        landmarks[indices[5]];


    const horizontal =
        distance(
            p1,
            p2
        );


    const vertical1 =
        distance(
            p3,
            p4
        );


    const vertical2 =
        distance(
            p5,
            p6
        );


    if (horizontal === 0) {
        return 0;
    }


    return (
        vertical1 +
        vertical2
    ) / (
        2 * horizontal
    );
}


function getEyeBox(
    landmarks,
    indices,
    width,
    height
) {

    const points =
        indices.map(
            index => ({
                x:
                    landmarks[index].x
                    * width,

                y:
                    landmarks[index].y
                    * height
            })
        );


    const xs =
        points.map(p => p.x);

    const ys =
        points.map(p => p.y);


    let x1 =
        Math.min(...xs);

    let x2 =
        Math.max(...xs);

    let y1 =
        Math.min(...ys);

    let y2 =
        Math.max(...ys);


    const eyeWidth =
        Math.max(
            x2 - x1,
            10
        );

    const eyeHeight =
        Math.max(
            y2 - y1,
            10
        );


    const marginX =
        eyeWidth * 0.35;

    const marginY =
        eyeHeight * 1.2;


    x1 =
        Math.max(
            0,
            x1 - marginX
        );

    x2 =
        Math.min(
            width,
            x2 + marginX
        );

    y1 =
        Math.max(
            0,
            y1 - marginY
        );

    y2 =
        Math.min(
            height,
            y2 + marginY
        );


    return {
        x1,
        y1,
        x2,
        y2
    };
}


function predictEye(box) {

    return tf.tidy(() => {

        const width =
            Math.max(
                1,
                Math.round(
                    box.x2 -
                    box.x1
                )
            );


        const height =
            Math.max(
                1,
                Math.round(
                    box.y2 -
                    box.y1
                )
            );


        let image =
            tf.browser.fromPixels(
                video
            );


        image =
            tf.slice(
                image,
                [
                    Math.round(box.y1),
                    Math.round(box.x1),
                    0
                ],
                [
                    height,
                    width,
                    3
                ]
            );


        image =
            tf.image.resizeBilinear(
                image,
                [
                    IMG_SIZE,
                    IMG_SIZE
                ]
            );


        image =
            image.expandDims(0);


        const prediction =
            eyeModel.predict(
                image
            );


        return prediction.dataSync()[0];
    });
}


function combineResult(
    openScore,
    ear
) {

    const cnnState =
        openScore >= CNN_OPEN_THRESHOLD
            ? "OPEN"
            : "CLOSED";


    const closedConfidence =
        1 - openScore;


    if (ear < 0.15) {
        return "CLOSED";
    }


    if (
        cnnState === "CLOSED"
        &&
        ear < EAR_CLOSED_THRESHOLD
    ) {
        return "CLOSED";
    }


    if (
        cnnState === "CLOSED"
        &&
        closedConfidence >= 0.85
    ) {
        return "CLOSED";
    }


    return "OPEN";
}


function smoothState(
    history,
    state
) {

    history.push(state);


    if (
        history.length
        >
        SMOOTH_FRAMES
    ) {
        history.shift();
    }


    const closedCount =
        history.filter(
            x => x === "CLOSED"
        ).length;


    const ratio =
        closedCount
        /
        history.length;


    return ratio >= 0.6
        ? "CLOSED"
        : "OPEN";
}


function drawEye(
    box,
    state,
    ear
) {

    const color =
        state === "OPEN"
            ? "#00ff00"
            : "#ff0000";


    ctx.strokeStyle =
        color;

    ctx.lineWidth =
        3;


    ctx.strokeRect(
        box.x1,
        box.y1,
        box.x2 - box.x1,
        box.y2 - box.y1
    );


    ctx.fillStyle =
        color;

    ctx.font =
        "18px Arial";


    ctx.fillText(
        `${state} EAR:${ear.toFixed(2)}`,
        box.x1,
        Math.max(
            20,
            box.y1 - 8
        )
    );
}


async function processFrame() {

    if (!running) {
        return;
    }


    if (
        video.readyState
        <
        2
    ) {

        requestAnimationFrame(
            processFrame
        );

        return;
    }


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    const now =
        performance.now();


    const result =
        faceLandmarker.detectForVideo(
            video,
            now
        );


    if (
        result.faceLandmarks
        &&
        result.faceLandmarks.length > 0
    ) {

        const landmarks =
            result.faceLandmarks[0];


        const leftEAR =
            calculateEAR(
                landmarks,
                LEFT_EYE
            );


        const rightEAR =
            calculateEAR(
                landmarks,
                RIGHT_EYE
            );


        const leftBox =
            getEyeBox(
                landmarks,
                LEFT_EYE,
                canvas.width,
                canvas.height
            );


        const rightBox =
            getEyeBox(
                landmarks,
                RIGHT_EYE,
                canvas.width,
                canvas.height
            );


        const leftScore =
            predictEye(
                leftBox
            );


        const rightScore =
            predictEye(
                rightBox
            );


        const leftState =
            combineResult(
                leftScore,
                leftEAR
            );


        const rightState =
            combineResult(
                rightScore,
                rightEAR
            );


        const leftFinal =
            smoothState(
                leftHistory,
                leftState
            );


        const rightFinal =
            smoothState(
                rightHistory,
                rightState
            );


        drawEye(
            leftBox,
            leftFinal,
            leftEAR
        );


        drawEye(
            rightBox,
            rightFinal,
            rightEAR
        );


        const bothClosed =
            leftFinal === "CLOSED"
            &&
            rightFinal === "CLOSED";


        if (bothClosed) {

            if (
                closedStart === null
            ) {

                closedStart =
                    performance.now();
            }


            const elapsed =
                (
                    performance.now()
                    -
                    closedStart
                )
                /
                1000;


            if (
                elapsed
                >=
                SLEEP_SECONDS
            ) {

                statusText.innerText =
                    "SLEEPING";

                statusText.style.color =
                    "red";

            } else {

                statusText.innerText =
                    "EYES CLOSED";

                statusText.style.color =
                    "orange";
            }


            infoText.innerText =
                `Closed: ${elapsed.toFixed(1)} sec`;

        } else {

            closedStart =
                null;

            statusText.innerText =
                "AWAKE";

            statusText.style.color =
                "lime";

            infoText.innerText =
                "";
        }


        ctx.fillStyle =
            "white";

        ctx.font =
            "18px Arial";


        ctx.fillText(
            `L EAR: ${leftEAR.toFixed(3)}   R EAR: ${rightEAR.toFixed(3)}`,
            20,
            canvas.height - 20
        );


    } else {

        closedStart =
            null;

        leftHistory = [];
        rightHistory = [];


        statusText.innerText =
            "FACE NOT DETECTED";

        statusText.style.color =
            "white";

        infoText.innerText =
            "";
    }


    requestAnimationFrame(
        processFrame
    );
}


async function startCamera() {

    try {

        if (!eyeModel || !faceLandmarker) {

            statusText.innerText =
                "LOADING...";

            await loadModels();
        }


        const stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        facingMode:
                            "user",

                        width: {
                            ideal: 1280
                        },

                        height: {
                            ideal: 720
                        }
                    },

                    audio: false
                }
            );


        video.srcObject =
            stream;


        await video.play();


        running =
            true;


        startButton.style.display =
            "none";


        statusText.innerText =
            "AWAKE";

        statusText.style.color =
            "lime";


        processFrame();


    } catch (error) {

        console.error(
            error
        );


        statusText.innerText =
            "CAMERA ERROR";


        infoText.innerText =
            error.message;
    }
}


startButton.addEventListener(
    "click",
    startCamera
);


loadModels().catch(
    error => {

        console.error(
            error
        );

        statusText.innerText =
            "MODEL LOAD ERROR";

        infoText.innerText =
            error.message;
    }
);
