
import {
  ObjectDetector,
  FilesetResolver
} from "@mediapipe/tasks-vision";

const $ = (id) => document.getElementById(id);

const video = $("webcam");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const captureBtn = $("captureBtn");

let detector = null;
let stream = null;
let running = false;
let animationId = null;
let lastVideoTime = -1;
let lastInferenceTime = 0;
let previousInferenceTime = 0;

let fps = 0;
let totalObservations = 0;
let history = [];
let lastHistoryUpdate = 0;
let lastUiUpdate = 0;
let currentDetections = [];

const MAX_FPS = 15;
const MAX_HISTORY = 30;

function setStatus(message) {
  $("status").textContent = message;
}

function updateStats(detections) {
  $("objectCount").textContent = detections.length;
  $("fps").textContent = fps.toFixed(1);
  $("total").textContent = totalObservations;

  const average = detections.length
    ? detections.reduce((sum, d) =>
        sum + (d.categories[0]?.score || 0), 0
      ) / detections.length
    : 0;

  $("confidence").textContent =
    `${Math.round(average * 100)}%`;
}

function drawDetections(detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const detection of detections) {
    const box = detection.boundingBox;
    const category = detection.categories[0];

    if (!box || !category) continue;

    const x = box.originX;
    const y = box.originY;
    const w = box.width;
    const h = box.height;

    const label =
      `${category.categoryName} ${Math.round(category.score * 100)}%`;

    ctx.strokeStyle = "#24e0c4";
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    ctx.font = "bold 15px Arial";

    const textWidth = ctx.measureText(label).width + 16;
    const labelY = Math.max(0, y - 27);

    ctx.fillStyle = "#009e87";
    ctx.fillRect(x, labelY, textWidth, 26);

    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x + 8, labelY + 18);
  }
}

function createResultItem(name, score) {
  const item = document.createElement("div");
  item.className = "result-item";

  const top = document.createElement("div");
  top.className = "result-top";

  const title = document.createElement("strong");
  title.textContent = name;

  const percentage = document.createElement("span");
  percentage.textContent =
    `${Math.round(score * 100)}%`;

  top.append(title, percentage);

  const track = document.createElement("div");
  track.className = "progress-track";

  const fill = document.createElement("div");
  fill.className = "progress-fill";
  fill.style.width = `${score * 100}%`;

  track.appendChild(fill);
  item.append(top, track);

  return item;
}

function updateResults(detections) {
  const container = $("detections");
  container.replaceChildren();

  if (!detections.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No objects in frame";
    container.appendChild(empty);
    return;
  }

  for (const detection of detections) {
    const category = detection.categories[0];
    if (!category) continue;

    container.appendChild(
      createResultItem(
        category.categoryName,
        category.score
      )
    );
  }
}

function addHistory(detections, timestamp) {
  if (!detections.length) return;

  // Record a snapshot at most once every 2 seconds.
  if (timestamp - lastHistoryUpdate < 2000) return;

  lastHistoryUpdate = timestamp;
  const time = new Date().toLocaleTimeString();

  for (const detection of detections) {
    const category = detection.categories[0];
    if (!category) continue;

    history.unshift({
      name: category.categoryName,
      score: category.score,
      time
    });
  }

  // Keep only the latest 30 observations.
  history = history.slice(0, MAX_HISTORY);
  renderHistory();
}

function renderHistory() {
  const container = $("history");
  container.replaceChildren();

  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "History is empty";
    container.appendChild(empty);
    return;
  }

  for (const entry of history) {
    const row = document.createElement("div");
    row.className = "history-item";

    const name = document.createElement("strong");
    name.textContent = entry.name;

    const info = document.createElement("span");
    info.textContent =
      `${Math.round(entry.score * 100)}% · ${entry.time}`;

    row.append(name, info);
    container.appendChild(row);
  }
}

async function loadModel() {
  setStatus("Loading MediaPipe AI model...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
  );

  detector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite",
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    scoreThreshold: 0.45,
    maxResults: 10
  });
}

function detectLoop(timestamp) {
  if (!running) return;

  // Limit the rate of inference to reduce CPU load.
  if (
    timestamp - lastInferenceTime >= 1000 / MAX_FPS &&
    video.readyState >= 2 &&
    video.currentTime !== lastVideoTime
  ) {
    try {
      const result = detector.detectForVideo(
        video,
        performance.now()
      );

      currentDetections = result.detections;
      drawDetections(currentDetections);

      if (previousInferenceTime > 0) {
        const elapsed = performance.now() -
          previousInferenceTime;
        const instantaneousFPS = 1000 / elapsed;
        fps = fps === 0
          ? instantaneousFPS
          : fps * 0.7 + instantaneousFPS * 0.3;
      }

      previousInferenceTime = performance.now();
      lastInferenceTime = timestamp;
      lastVideoTime = video.currentTime;

      totalObservations += currentDetections.length;
      addHistory(currentDetections, timestamp);

      // Avoid rebuilding the dashboard every frame.
      if (timestamp - lastUiUpdate >= 200) {
        updateStats(currentDetections);
        updateResults(currentDetections);
        lastUiUpdate = timestamp;
      }
    } catch (error) {
      console.error(error);
      stopDetection();
      setStatus(`Detection error: ${error.message}`);
      return;
    }
  }

  animationId = requestAnimationFrame(detectLoop);
}

async function startDetection() {
  if (running) return;

  startBtn.disabled = true;

  try {
    if (!detector) await loadModel();

    setStatus("Connecting camera...");

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    video.srcObject = stream;
    await video.play();

    if (!video.videoWidth || !video.videoHeight) {
      throw new Error("Invalid camera dimensions");
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    lastVideoTime = -1;
    lastInferenceTime = 0;
    previousInferenceTime = 0;
    lastHistoryUpdate = 0;
    lastUiUpdate = 0;
    fps = 0;
    currentDetections = [];

    running = true;

    $("liveBadge").textContent = "● LIVE";
    $("liveBadge").classList.add("active");

    $("cameraStatus").textContent = "AI DETECTION ACTIVE";
    stopBtn.disabled = false;
    captureBtn.disabled = false;
    setStatus("AI running locally");

    animationId = requestAnimationFrame(detectLoop);
  } catch (error) {
    console.error(error);

    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }

    video.srcObject = null;
    startBtn.disabled = false;
    setStatus(`Camera error: ${error.message}`);
  }
}

function stopDetection() {
  running = false;

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  video.pause();
  video.srcObject = null;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  currentDetections = [];
  fps = 0;

  updateStats(currentDetections);
  updateResults(currentDetections);

  startBtn.disabled = false;
  stopBtn.disabled = true;
  captureBtn.disabled = true;

  $("liveBadge").textContent = "● OFFLINE";
  $("liveBadge").classList.remove("active");
  $("cameraStatus").textContent = "CAMERA INACTIVE";

  setStatus("Detection stopped");
}

function captureImage() {
  if (!running || !video.videoWidth) return;

  const image = document.createElement("canvas");
  image.width = video.videoWidth;
  image.height = video.videoHeight;

  const imageCtx = image.getContext("2d");

  // Mirror the video and labels together.
  imageCtx.translate(image.width, 0);
  imageCtx.scale(-1, 1);

  imageCtx.drawImage(
    video, 0, 0, image.width, image.height
  );

  imageCtx.drawImage(canvas, 0, 0);

  imageCtx.setTransform(1, 0, 0, 1, 0, 0);

  const link = document.createElement("a");
  link.download = `visionai-${Date.now()}.png`;
  link.href = image.toDataURL("image/png");
  link.click();
}

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);
captureBtn.addEventListener("click", captureImage);

$("clearBtn").addEventListener("click", () => {
  history = [];
  renderHistory();
});

window.addEventListener("beforeunload", () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
});
