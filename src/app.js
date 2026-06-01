import { ASLClassifier } from './classifier.js';
import defaultAlphabet from '../asl_alphabet.json';

// Elements
const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output-canvas');
const canvasCtx = canvasElement.getContext('2d');

const cameraStatus = document.getElementById('camera-status');
const modelStatus = document.getElementById('model-status');
const viewportOverlay = document.getElementById('viewport-overlay');
const overlayText = document.getElementById('overlay-text');
const handIndicator = document.getElementById('hand-indicator');
const handIndicatorText = document.getElementById('hand-indicator-text');
const debugCoords = document.getElementById('debug-coords');
const chkStrictMode = document.getElementById('chk-strict-mode');

// Controls
const toggleCameraBtn = document.getElementById('toggle-camera-btn');
const toggleSkeletonBtn = document.getElementById('toggle-skeleton-btn');
const resetCamBtn = document.getElementById('reset-cam-btn');

// Stats
const fpsVal = document.getElementById('fps-val');
const latencyVal = document.getElementById('latency-val');
const totalSamplesVal = document.getElementById('total-samples-val');

// Tabs
const tabTranslate = document.getElementById('tab-translate');
const tabTrain = document.getElementById('tab-train');
const panelTranslate = document.getElementById('panel-translate');
const panelTrain = document.getElementById('panel-train');

// Translate UI
const predText = document.getElementById('pred-text');
const predConfidenceFill = document.getElementById('pred-confidence-fill');
const predConfidenceText = document.getElementById('pred-confidence-text');
const sentenceOutput = document.getElementById('sentence-output');
const btnSpeak = document.getElementById('btn-speak');
const btnCopy = document.getElementById('btn-copy');
const btnSpace = document.getElementById('btn-space');
const btnBackspace = document.getElementById('btn-backspace');
const btnClear = document.getElementById('btn-clear');

// Train UI
const newGestureNameInput = document.getElementById('new-gesture-name');
const captureProgressBar = document.getElementById('capture-progress-bar');
const captureCountEl = document.getElementById('capture-count');
const btnCaptureSample = document.getElementById('btn-capture-sample');
const gestureListBody = document.getElementById('gesture-list-body');
const btnExportDataset = document.getElementById('btn-export-dataset');
const btnImportDatasetTrigger = document.getElementById('btn-import-dataset-trigger');
const importDatasetFile = document.getElementById('import-dataset-file');
const btnClearDataset = document.getElementById('btn-clear-dataset');

// App State
let activeTab = 'translate';
let isCameraActive = true;
let isSkeletonVisible = true;
let handsDetector = null;
let cameraManager = null;
let currentHandList = null;

// Performance
let lastFrameTime = performance.now();
let frameCount = 0;
let fps = 0;
let latencyHistory = [];

// Stability
const PREDICTION_BUFFER_SIZE = 12;
let predictionBuffer = [];
let lastStabilizedWord = '';
let stableCount = 0;
const STABILITY_LOCK_THRESHOLD = 20;

// Recording
let isCapturing = false;
let capturedSamplesCount = 0;
const REQUIRED_SAMPLES = 15;
let captureThrottleTime = 0;

// Initialize Classifier
const classifier = new ASLClassifier();

// Motion history buffer specifically for dynamic letters J and Z
let motionHistory = [];
let motionLockLabel = '';
let motionLockFrames = 0;

const getAlphabetArray = () => {
  if (Array.isArray(defaultAlphabet)) return defaultAlphabet;
  if (defaultAlphabet && Array.isArray(defaultAlphabet.default)) return defaultAlphabet.default;
  return [];
};
const alphabet = getAlphabetArray();

// Load pre-bundled alphabet if no custom gestures exist yet and user hasn't explicitly cleared them
const CURRENT_DB_VERSION = "v3_user_trained";
const storedDbVersion = localStorage.getItem("asl_db_version");
const isExplicitlyBlank = localStorage.getItem("asl_db_blank") === "true";

if (!isExplicitlyBlank && (classifier.samples.length === 0 || storedDbVersion !== CURRENT_DB_VERSION)) {
  classifier.samples = [...alphabet];
  classifier.saveToLocalStorage();
  localStorage.setItem("asl_db_version", CURRENT_DB_VERSION);
  localStorage.setItem("asl_db_blank", "false");
}

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [0, 9], [9, 10], [10, 11], [11, 12], // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17] // Palm Base
];

// Color palettes for multi-hand drawing
const HAND_COLORS = [
  {
    primary: 'hsl(185, 100%, 48%)',     // Cyan
    secondary: 'hsl(210, 100%, 65%)',   // Blue
    glow: 'rgba(0, 240, 240, 0.4)'
  },
  {
    primary: 'hsl(342, 95%, 62%)',      // Magenta
    secondary: 'hsl(255, 90%, 76%)',    // Purple
    glow: 'rgba(240, 50, 120, 0.4)'
  }
];

const playBeep = (freq = 440, duration = 0.15, type = 'sine') => {
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    oscillator.type = type;
    oscillator.frequency.value = freq;
    gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);
    oscillator.start();
    oscillator.stop(audioCtx.currentTime + duration);
  } catch (e) {
    console.warn("Audio Context error:", e);
  }
};

async function initApp() {
  overlayText.innerText = "Loading MediaPipe Models...";
  modelStatus.className = "status-indicator loading";
  modelStatus.querySelector('.status-label').innerText = "Loading Model...";

  try {
    handsDetector = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    // Configure MediaPipe for single hand tracking
    handsDetector.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.40,
      minTrackingConfidence: 0.40
    });

    handsDetector.onResults(onResults);
    modelStatus.className = "status-indicator online";
    modelStatus.querySelector('.status-label').innerText = "MediaPipe Ready";
  } catch (e) {
    console.error("MediaPipe load error:", e);
    overlayText.innerText = "Failed to load MediaPipe. Check connection.";
    modelStatus.className = "status-indicator offline";
    modelStatus.querySelector('.status-label').innerText = "Load Failed";
    return;
  }

  overlayText.innerText = "Requesting webcam access...";
  cameraStatus.className = "status-indicator loading";
  cameraStatus.querySelector('.status-label').innerText = "Starting Camera...";

  try {
    cameraManager = new Camera(videoElement, {
      onFrame: async () => {
        if (isCameraActive) {
          const startTime = performance.now();
          await handsDetector.send({ image: videoElement });
          
          const endTime = performance.now();
          const latency = Math.round(endTime - startTime);
          latencyHistory.push(latency);
          if (latencyHistory.length > 10) latencyHistory.shift();
          const avgLatency = Math.round(latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length);
          latencyVal.innerText = `${avgLatency} ms`;
        }
      },
      width: 640,
      height: 480
    });

    await cameraManager.start();
    cameraStatus.className = "status-indicator online";
    cameraStatus.querySelector('.status-label').innerText = "Camera Active";
    viewportOverlay.classList.add('hidden');
  } catch (e) {
    console.error("Camera error:", e);
    overlayText.innerText = "Camera access denied. Please grant permission.";
    cameraStatus.className = "status-indicator offline";
    cameraStatus.querySelector('.status-label').innerText = "Camera Denied";
  }

  if (!classifier.samples || classifier.samples.length === 0) {
    console.warn("Failsafe: classifier.samples was empty during initApp. Re-seeding from pre-bundled dataset...");
    classifier.samples = [...alphabet];
    classifier.saveToLocalStorage();
  }

  updateStats();
  renderGestureList();
  setupEventListeners();
}

function onResults(results) {
  if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // Filter out low-confidence hand detections to avoid background noise triggering 2-hand mode
  let validHands = [];
  let validHandedness = [];
  if (results.multiHandLandmarks && results.multiHandedness) {
    const rawHands = [];
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const score = results.multiHandedness[i].score;
      if (score >= 0.40) { // Strict score threshold for hand detection validity
        // MediaPipe's raw handedness assumes a mirrored selfie view. Since our camera stream
        // is unmirrored before it is processed, raw labels are inverted compared to physical hands.
        const rawLabel = results.multiHandedness[i].label; // "Left" or "Right"
        const physicalLabel = rawLabel === "Left" ? "Right" : "Left";
        rawHands.push({
          landmarks: results.multiHandLandmarks[i],
          label: physicalLabel,
          score: score
        });
      }
    }

    validHands = rawHands.map(h => h.landmarks);
    validHandedness = rawHands.map(h => h.label);
  }

  const numHands = validHands.length;

  if (numHands > 0) {
    currentHandList = validHands;
    
    // Update Hand presence indicator UI
    handIndicator.classList.remove('hidden');
    const handedness = validHandedness[0];
    handIndicatorText.innerText = `${handedness} Hand`;
    handIndicator.style.borderColor = HAND_COLORS[0].primary;
    handIndicator.style.boxShadow = `0 0 15px ${HAND_COLORS[0].glow}`;

    // Draw Skeleton for the hand with cyan color theme
    if (isSkeletonVisible) {
      drawHandSkeleton(currentHandList[0], HAND_COLORS[0]);
    }

    // Process predictions/training with handedness data
    if (activeTab === 'translate') {
      processTranslation(currentHandList, validHandedness);
    } else if (activeTab === 'train') {
      processTraining(currentHandList, validHandedness);
    }
  } else {
    currentHandList = null;
    handIndicator.classList.add('hidden');
    if (activeTab === 'translate') {
      handleNoHand();
    }
  }

  // Calculate FPS
  frameCount++;
  const currentTime = performance.now();
  if (currentTime - lastFrameTime >= 1000) {
    fps = Math.round((frameCount * 1000) / (currentTime - lastFrameTime));
    fpsVal.innerText = fps;
    frameCount = 0;
    lastFrameTime = currentTime;
  }
}

/**
 * Draw a single hand's landmarks and connectors with glowing theme.
 */
function drawHandSkeleton(landmarks, colors) {
  const width = canvasElement.width;
  const height = canvasElement.height;

  // Draw connectors
  canvasCtx.lineWidth = 4;
  canvasCtx.strokeStyle = colors.secondary;
  canvasCtx.shadowColor = colors.primary;
  canvasCtx.shadowBlur = 10;
  canvasCtx.lineCap = 'round';

  HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
    const pt1 = landmarks[startIdx];
    const pt2 = landmarks[endIdx];
    canvasCtx.beginPath();
    canvasCtx.moveTo(pt1.x * width, pt1.y * height);
    canvasCtx.lineTo(pt2.x * width, pt2.y * height);
    canvasCtx.stroke();
  });

  // Draw joint landmarks
  canvasCtx.shadowBlur = 12;
  landmarks.forEach((pt, index) => {
    canvasCtx.beginPath();
    const isTip = [4, 8, 12, 16, 20].includes(index);
    if (isTip) {
      canvasCtx.fillStyle = '#ffffff';
      canvasCtx.shadowColor = colors.primary;
      canvasCtx.arc(pt.x * width, pt.y * height, 6, 0, 2 * Math.PI);
    } else {
      canvasCtx.fillStyle = colors.primary;
      canvasCtx.shadowColor = colors.primary;
      canvasCtx.arc(pt.x * width, pt.y * height, 4, 0, 2 * Math.PI);
    }
    canvasCtx.fill();
  });

  canvasCtx.shadowBlur = 0;
}

function checkMotionGestures(handList, currentPred) {
  if (!handList || handList.length !== 1) {
    motionHistory = [];
    return null;
  }

  const landmarks = handList[0];
  const indexTip = landmarks[8];
  const pinkyTip = landmarks[20];

  motionHistory.push({
    index: { x: indexTip.x, y: indexTip.y },
    pinky: { x: pinkyTip.x, y: pinkyTip.y }
  });

  // Keep last 25 frames (~0.8 seconds at 30fps) for motion gesture checks
  if (motionHistory.length > 25) {
    motionHistory.shift();
  }

  if (motionHistory.length < 8) return null;

  // 1. Detect "J" (Starts in 'I' shape, draws a curved hook path down and left-up)
  if (currentPred === "I" || currentPred === "Y") {
    const ys = motionHistory.map(pt => pt.pinky.y);
    const xs = motionHistory.map(pt => pt.pinky.x);
    
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);

    const dy = maxY - minY;
    const dx = maxX - minX;

    // More forgiving curve size requirements (down from 0.15/0.08)
    if (dy > 0.08 && dx > 0.04) {
      const lowestPointIdx = ys.indexOf(maxY);
      // Lowest point should happen toward the middle-latter part of the gesture
      if (lowestPointIdx > 2 && lowestPointIdx < motionHistory.length - 1) {
        const yBefore = ys.slice(0, lowestPointIdx);
        const yAfter = ys.slice(lowestPointIdx);
        
        const movesDown = yBefore[yBefore.length - 1] > yBefore[0];
        const movesUp = yAfter[yAfter.length - 1] < yAfter[0];

        if (movesDown && movesUp) {
          motionHistory = []; // Reset trace
          return { label: "J", confidence: 0.95 };
        }
      }
    }
  }

  // 2. Detect "Z" (Starts in 'D'/'L'/'G' shape or temporary 'NO SIGN', traces a horizontal zig-zag)
  if (currentPred === "D" || currentPred === "L" || currentPred === "G" || currentPred === "U" || currentPred === "Z" || currentPred === "NO SIGN") {
    const xs = motionHistory.map(pt => pt.index.x);
    const ys = motionHistory.map(pt => pt.index.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const dx = maxX - minX;
    const dy = maxY - minY;

    // More forgiving box size requirements (down from 0.15/0.12)
    if (dx > 0.08 && dy > 0.06) {
      // Detect horizontal direction reversals
      let dirChanges = 0;
      let currentDir = 0;

      // Simple 3-frame average smoothing
      const smoothedXs = [];
      for (let i = 0; i < xs.length - 2; i++) {
        smoothedXs.push((xs[i] + xs[i+1] + xs[i+2]) / 3);
      }

      for (let i = 1; i < smoothedXs.length; i++) {
        const diff = smoothedXs[i] - smoothedXs[i - 1];
        if (Math.abs(diff) > 0.004) {
          const newDir = diff > 0 ? 1 : -1;
          if (currentDir !== 0 && newDir !== currentDir) {
            dirChanges++;
          }
          currentDir = newDir;
        }
      }

      // A standard zig-zag will have at least 2 directional shifts
      if (dirChanges >= 2 && dirChanges <= 5) {
        motionHistory = []; // Reset trace
        return { label: "Z", confidence: 0.95 };
      }
    }
  }

  return null;
}

function processTranslation(handList, handednessList = ["Right"]) {
  const width = videoElement.videoWidth || 640;
  const height = videoElement.videoHeight || 480;
  const aspectRatio = width / height;
  const isStrict = chkStrictMode ? chkStrictMode.checked : false;
  let prediction = classifier.classify(handList, handednessList, 5, aspectRatio, isStrict);
  
  // Check if we are currently locked in a motion prediction
  if (motionLockFrames > 0) {
    prediction = { label: motionLockLabel, confidence: 0.95 };
    motionLockFrames--;
  } else {
    // Inject motion tracing detection overrides for J & Z
    const motionPred = checkMotionGestures(handList, prediction.label);
    if (motionPred) {
      prediction = motionPred;
      motionLockLabel = motionPred.label;
      motionLockFrames = 12; // Hold J/Z for 12 frames to let the smoothing buffer stabilize and show it clearly
    }
  }
  
  if (debugCoords) {
    if (handList && handList.length > 0) {
      const lm = handList[0];
      const wrist = lm[0];
      const mcp = lm[9];
      const dxMcp = (mcp.x - wrist.x) * aspectRatio;
      const dyMcp = mcp.y - wrist.y;
      const rotationAngle = Math.atan2(dxMcp, -dyMcp);
      const angleDeg = Math.abs(rotationAngle * 180 / Math.PI);
      const rawYDiff = mcp.y - wrist.y;

      const dxIndex = (lm[8].x - lm[5].x) * aspectRatio;
      const dyIndex = lm[8].y - lm[5].y;
      const indexAngle = Math.atan2(dxIndex, -dyIndex);
      const indexAngleDeg = Math.abs(indexAngle * 180 / Math.PI);

      debugCoords.innerText = `Palm Angle: ${angleDeg.toFixed(1)}° | Finger Angle: ${indexAngleDeg.toFixed(1)}°\nPalm YDiff: ${rawYDiff.toFixed(3)} | Finger dy: ${dyIndex.toFixed(3)}`;
    } else {
      debugCoords.innerText = '';
    }
  }

  if (prediction.label !== "NO SIGN" && prediction.confidence > 0.45) {
    predText.innerText = prediction.label;
    predText.classList.remove('empty');
    const confPercent = Math.round(prediction.confidence * 100);
    predConfidenceFill.style.width = `${confPercent}%`;
    predConfidenceText.innerText = `${confPercent}%`;
  } else {
    predText.innerText = "NO SIGN";
    predText.classList.add('empty');
    predConfidenceFill.style.width = '0%';
    predConfidenceText.innerText = '0%';
  }

  // Smooth prediction
  predictionBuffer.push(prediction.label);
  if (predictionBuffer.length > PREDICTION_BUFFER_SIZE) {
    predictionBuffer.shift();
  }

  const counts = {};
  predictionBuffer.forEach(label => {
    counts[label] = (counts[label] || 0) + 1;
  });

  let dominantLabel = "NO SIGN";
  let maxCount = 0;
  Object.keys(counts).forEach(label => {
    if (counts[label] > maxCount) {
      maxCount = counts[label];
      dominantLabel = label;
    }
  });

  const stabilityRatio = maxCount / predictionBuffer.length;
  if (dominantLabel !== "NO SIGN" && stabilityRatio >= 0.75) {
    if (dominantLabel === lastStabilizedWord) {
      stableCount++;
      if (stableCount === STABILITY_LOCK_THRESHOLD) {
        appendWordToSentence(dominantLabel);
        playBeep(650, 0.05, 'triangle');
      }
    } else {
      lastStabilizedWord = dominantLabel;
      stableCount = 0;
    }
  } else {
    stableCount = 0;
  }
}

function handleNoHand() {
  motionHistory = []; // Clear motion tracker when hands leave screen
  motionLockFrames = 0; // Reset motion display lock
  motionLockLabel = '';
  predText.innerText = "NO SIGN";
  predText.classList.add('empty');
  predConfidenceFill.style.width = '0%';
  predConfidenceText.innerText = '0%';
  if (debugCoords) debugCoords.innerText = '';

  predictionBuffer.push("NO SIGN");
  if (predictionBuffer.length > PREDICTION_BUFFER_SIZE) {
    predictionBuffer.shift();
  }

  const noHandCount = predictionBuffer.filter(p => p === "NO SIGN").length;
  if (noHandCount > PREDICTION_BUFFER_SIZE / 2) {
    lastStabilizedWord = '';
    stableCount = 0;
  }
}

function appendWordToSentence(word) {
  const currentText = sentenceOutput.value.trim();
  if (word.length === 1) {
    sentenceOutput.value = currentText ? currentText + word : word;
  } else {
    sentenceOutput.value = currentText ? currentText + " " + word : word;
  }
  sentenceOutput.scrollTop = sentenceOutput.scrollHeight;
}

function processTraining(handList, handednessList = ["Right"]) {
  if (!isCapturing) return;

  const now = performance.now();
  if (now - captureThrottleTime < 150) return;
  captureThrottleTime = now;

  const label = newGestureNameInput.value.toUpperCase().trim();
  if (!label) {
    stopCapturing();
    alert("Please enter a gesture name first.");
    return;
  }

  const width = videoElement.videoWidth || 640;
  const height = videoElement.videoHeight || 480;
  const aspectRatio = width / height;
  const success = classifier.addSample(label, handList, handednessList, aspectRatio);
  if (success) {
    capturedSamplesCount++;
    captureCountEl.innerText = capturedSamplesCount;
    
    const dashOffset = 314.16 - (314.16 * (capturedSamplesCount / REQUIRED_SAMPLES));
    captureProgressBar.style.strokeDashoffset = dashOffset;
    
    playBeep(400 + (capturedSamplesCount * 25), 0.08, 'sine');

    if (capturedSamplesCount >= REQUIRED_SAMPLES) {
      stopCapturing();
      playBeep(880, 0.3, 'sine');
      setTimeout(() => {
        alert(`Successfully trained custom gesture: "${label}" (${handList.length} hand${handList.length > 1 ? 's' : ''})`);
      }, 50);
    }
  }
}

function startCapturing() {
  const label = newGestureNameInput.value.toUpperCase().trim();
  if (!label) {
    alert("Please enter a gesture name before capturing.");
    return;
  }

  isCapturing = true;
  capturedSamplesCount = 0;
  captureCountEl.innerText = '0';
  captureProgressBar.style.strokeDashoffset = 314.16;
  btnCaptureSample.classList.add('active');
  btnCaptureSample.querySelector('span:last-child').innerText = "Recording...";
}

function stopCapturing() {
  isCapturing = false;
  btnCaptureSample.classList.remove('active');
  btnCaptureSample.querySelector('span:last-child').innerText = "Capture Sample";
  renderGestureList();
  updateStats();
}

function renderGestureList() {
  gestureListBody.innerHTML = '';
  
  const grouped = {};
  classifier.samples.forEach(sample => {
    const key = sample.label;
    grouped[key] = {
      label: sample.label,
      count: (grouped[key]?.count || 0) + 1
    };
  });

  const keys = Object.keys(grouped);
  
  if (keys.length === 0) {
    gestureListBody.innerHTML = `
      <tr>
        <td colspan="3" class="empty-list-cell">No custom gestures trained yet.</td>
      </tr>
    `;
    return;
  }

  keys.forEach(key => {
    const item = grouped[key];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${item.label}</strong></td>
      <td>${item.count} samples</td>
      <td style="text-align: right;">
        <button class="btn btn-tertiary btn-icon btn-small delete-gesture-btn" data-label="${item.label}" style="color: var(--accent-danger);">
          <span class="material-icons-round">delete</span>
          <span>Delete</span>
        </button>
      </td>
    `;
    gestureListBody.appendChild(tr);
  });

  document.querySelectorAll('.delete-gesture-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const label = e.currentTarget.getAttribute('data-label');
      if (confirm(`Are you sure you want to delete the custom sign: "${label}"?`)) {
        classifier.deleteGesture(label);
        renderGestureList();
        updateStats();
      }
    });
  });
}

function updateStats() {
  totalSamplesVal.innerText = classifier.samples.length;
}

function setupEventListeners() {
  tabTranslate.addEventListener('click', () => {
    activeTab = 'translate';
    tabTranslate.classList.add('active');
    tabTrain.classList.remove('active');
    panelTranslate.classList.add('active');
    panelTrain.classList.remove('active');
  });

  tabTrain.addEventListener('click', () => {
    activeTab = 'train';
    tabTrain.classList.add('active');
    tabTranslate.classList.remove('active');
    panelTrain.classList.add('active');
    panelTranslate.classList.remove('active');
    renderGestureList();
  });

  toggleCameraBtn.addEventListener('click', () => {
    isCameraActive = !isCameraActive;
    if (isCameraActive) {
      toggleCameraBtn.classList.remove('active');
      toggleCameraBtn.querySelector('span:last-child').innerText = "Pause Cam";
      cameraStatus.className = "status-indicator online";
      cameraStatus.querySelector('.status-label').innerText = "Camera Active";
      viewportOverlay.classList.add('hidden');
    } else {
      toggleCameraBtn.classList.add('active');
      toggleCameraBtn.querySelector('span:last-child').innerText = "Resume Cam";
      cameraStatus.className = "status-indicator offline";
      cameraStatus.querySelector('.status-label').innerText = "Camera Paused";
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      overlayText.innerText = "Camera connection paused.";
      viewportOverlay.classList.remove('hidden');
    }
  });

  toggleSkeletonBtn.addEventListener('click', () => {
    isSkeletonVisible = !isSkeletonVisible;
    if (isSkeletonVisible) {
      toggleSkeletonBtn.classList.add('active');
    } else {
      toggleSkeletonBtn.classList.remove('active');
    }
  });

  resetCamBtn.addEventListener('click', async () => {
    if (cameraManager) {
      viewportOverlay.classList.remove('hidden');
      overlayText.innerText = "Restarting camera feed...";
      try {
        await cameraManager.stop();
        await cameraManager.start();
        isCameraActive = true;
        toggleCameraBtn.classList.remove('active');
        toggleCameraBtn.querySelector('span:last-child').innerText = "Pause Cam";
        cameraStatus.className = "status-indicator online";
        cameraStatus.querySelector('.status-label').innerText = "Camera Active";
        viewportOverlay.classList.add('hidden');
      } catch (e) {
        console.error("Camera reset failed:", e);
      }
    }
  });

  btnSpace.addEventListener('click', () => {
    appendWordToSentence(" ");
  });

  btnBackspace.addEventListener('click', () => {
    const words = sentenceOutput.value.trim().split(" ");
    words.pop();
    sentenceOutput.value = words.join(" ");
  });

  btnClear.addEventListener('click', () => {
    sentenceOutput.value = '';
    predictionBuffer = [];
    lastStabilizedWord = '';
    stableCount = 0;
  });

  btnCopy.addEventListener('click', () => {
    const text = sentenceOutput.value.trim();
    if (text) {
      navigator.clipboard.writeText(text).then(() => {
        playBeep(700, 0.1, 'sine');
        const originalText = btnCopy.innerHTML;
        btnCopy.innerHTML = `<span class="material-icons-round" style="color: var(--accent-success)">check</span>`;
        setTimeout(() => {
          btnCopy.innerHTML = originalText;
        }, 1500);
      });
    }
  });

  btnSpeak.addEventListener('click', () => {
    const text = sentenceOutput.value.trim();
    if (text && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  });

  newGestureNameInput.addEventListener('input', () => {
    const value = newGestureNameInput.value.trim();
    btnCaptureSample.disabled = value.length === 0;
  });

  window.addEventListener('keydown', (e) => {
    if (activeTab === 'train' && e.code === 'Space' && document.activeElement !== newGestureNameInput) {
      e.preventDefault();
      if (!isCapturing && newGestureNameInput.value.trim()) {
        startCapturing();
      }
    }
  });

  window.addEventListener('keyup', (e) => {
    if (activeTab === 'train' && e.code === 'Space') {
      if (isCapturing) {
        stopCapturing();
      }
    }
  });

  btnCaptureSample.addEventListener('mousedown', () => {
    if (newGestureNameInput.value.trim()) {
      startCapturing();
    }
  });

  btnCaptureSample.addEventListener('mouseup', () => {
    if (isCapturing) {
      stopCapturing();
    }
  });

  btnCaptureSample.addEventListener('mouseleave', () => {
    if (isCapturing) {
      stopCapturing();
    }
  });

  btnCaptureSample.addEventListener('touchstart', (e) => {
    e.preventDefault();
    if (newGestureNameInput.value.trim()) {
      startCapturing();
    }
  });

  btnCaptureSample.addEventListener('touchend', () => {
    if (isCapturing) {
      stopCapturing();
    }
  });

  btnExportDataset.addEventListener('click', () => {
    const json = classifier.exportDatasetJSON();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `asl-lens-dataset-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  btnImportDatasetTrigger.addEventListener('click', () => {
    importDatasetFile.click();
  });

  importDatasetFile.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const success = classifier.importDatasetJSON(evt.target.result);
      if (success) {
        alert("Custom ASL dataset imported successfully!");
        renderGestureList();
        updateStats();
      } else {
        alert("Failed to import dataset. Ensure file format is valid.");
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  });

  if (chkStrictMode) {
    chkStrictMode.checked = localStorage.getItem("asl_strict_mode") === "true";
    chkStrictMode.addEventListener("change", () => {
      localStorage.setItem("asl_strict_mode", chkStrictMode.checked);
    });
  }

  const btnWipeDataset = document.getElementById("btn-wipe-dataset");
  if (btnWipeDataset) {
    btnWipeDataset.addEventListener("click", () => {
      if (confirm("WARNING: This will wipe ALL signs (including default pre-bundled alphabet) leaving a completely blank slate. You will need to train all letters yourself. Proceed?")) {
        classifier.clearDataset();
        classifier.samples = [];
        classifier.saveToLocalStorage();
        localStorage.setItem("asl_db_blank", "true");
        renderGestureList();
        updateStats();
        playBeep(150, 0.5, "sawtooth");
      }
    });
  }

  btnClearDataset.addEventListener('click', () => {
    if (confirm("WARNING: This will wipe all custom gestures and reset the engine to default. This cannot be undone. Proceed?")) {
      classifier.clearDataset();
      // Reset to default pre-bundled alphabet
      classifier.samples = [...alphabet];
      classifier.saveToLocalStorage();
      localStorage.setItem("asl_db_version", CURRENT_DB_VERSION);
      localStorage.setItem("asl_db_blank", "false");
      renderGestureList();
      updateStats();
      playBeep(200, 0.4, 'sawtooth');
    }
  });
}

window.addEventListener('DOMContentLoaded', initApp);
