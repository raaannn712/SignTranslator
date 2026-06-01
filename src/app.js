import axios from 'axios';
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
const modeWordsRadio = document.getElementById('mode-words');
const modeLettersRadio = document.getElementById('mode-letters');

// Interpret mode: 'words' or 'letters'
let interpretMode = 'words';

if (modeWordsRadio && modeLettersRadio) {
  modeWordsRadio.addEventListener('change', () => { if (modeWordsRadio.checked) interpretMode = 'words'; });
  modeLettersRadio.addEventListener('change', () => { if (modeLettersRadio.checked) interpretMode = 'letters'; });
}

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
const trainingSaveStatus = document.getElementById('training-save-status');

// App State
let activeTab = 'translate';
let isCameraActive = true;
let isSkeletonVisible = true;
let handsDetector = null;
let cameraManager = null;
let currentHandList = null;
let faceMesh = null;
let currentFaceLandmarks = null;
let currentFaceEmotion = { label: 'NO FACE', confidence: 0 };

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
const STABILITY_LOCK_THRESHOLD = 10;

// Recording
let isCapturing = false;
let capturedSamplesCount = 0;
const REQUIRED_SAMPLES = 15;
let captureThrottleTime = 0;

// Initialize Classifier
const classifier = new ASLClassifier();

const ROBOFLOW_API_URL = 'https://serverless.roboflow.com';
const ROBOFLOW_MODEL_ID = 'hand-sign-tsze0/4';
const ROBOFLOW_REQUEST_INTERVAL_MS = 850;
const ROBOFLOW_RESULT_TTL_MS = 1200;
const ROBOFLOW_CROP_SIZE = 224;
const ROBOFLOW_MIN_CONFIDENCE = 0.38;
const BACKEND_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3000';
const UPLOADER_TAG_KEY = 'asl_uploader_tag';

let roboflowInFlight = false;
let lastRoboflowRequestAt = 0;
let latestRoboflowPrediction = null;
let latestRoboflowPredictionAt = 0;

function getUploaderTag() {
  let tag = localStorage.getItem(UPLOADER_TAG_KEY);
  if (!tag) {
    tag = (crypto?.randomUUID?.() || `anon-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    localStorage.setItem(UPLOADER_TAG_KEY, tag);
  }
  return tag;
}

async function saveTrainingSampleToBackend(label, handList, handednessList, aspectRatio, imagePath = null) {
  if (!handList || handList.length === 0) return null;

  const landmarks = handList[0];
  const handedness = handednessList?.[0] || 'Right';
  const features = classifier.normalize(landmarks, handedness, aspectRatio);

  if (!features) return null;

  const payload = {
    label,
    landmarks,
    features,
    numHands: handList.length,
    handedness,
    imagePath,
    source: 'webcam',
    quality: 100,
    approved: false,
    uploaderTag: getUploaderTag(),
    sessionId: null
  };

  if (trainingSaveStatus) {
    trainingSaveStatus.classList.remove('success', 'error');
    trainingSaveStatus.classList.add('saving');
    trainingSaveStatus.textContent = 'Supabase save status: saving sample...';
  }

  const response = await axios.post(`${BACKEND_BASE_URL}/api/training/sample`, payload, {
    headers: { 'Content-Type': 'application/json' }
  });

  if (trainingSaveStatus) {
    trainingSaveStatus.classList.remove('saving', 'error');
    trainingSaveStatus.classList.add('success');
    trainingSaveStatus.textContent = `Supabase save status: saved ${String(label || '').toUpperCase()}`;
  }

  return response.data;
}

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

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function getRoboflowApiKey() {
  return (localStorage.getItem('roboflow_api_key') || '').trim();
}

function buildHandCropBase64(handList) {
  if (!handList || handList.length === 0) return null;
  if (!videoElement.videoWidth || !videoElement.videoHeight) return null;

  const landmarks = handList[0];
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;

  landmarks.forEach((point) => {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });

  const handWidth = Math.max(maxX - minX, 0.12);
  const handHeight = Math.max(maxY - minY, 0.12);
  const padX = handWidth * 0.35;
  const padY = handHeight * 0.35;

  const sourceX = clamp((minX - padX) * videoElement.videoWidth, 0, videoElement.videoWidth - 1);
  const sourceY = clamp((minY - padY) * videoElement.videoHeight, 0, videoElement.videoHeight - 1);
  const sourceWidth = clamp((handWidth + padX * 2) * videoElement.videoWidth, 1, videoElement.videoWidth - sourceX);
  const sourceHeight = clamp((handHeight + padY * 2) * videoElement.videoHeight, 1, videoElement.videoHeight - sourceY);

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = ROBOFLOW_CROP_SIZE;
  cropCanvas.height = ROBOFLOW_CROP_SIZE;

  const cropCtx = cropCanvas.getContext('2d');
  if (!cropCtx) return null;

  cropCtx.drawImage(
    videoElement,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    ROBOFLOW_CROP_SIZE,
    ROBOFLOW_CROP_SIZE
  );

  const dataUrl = cropCanvas.toDataURL('image/jpeg', 0.88);
  return dataUrl.split(',')[1] || null;
}

function parseRoboflowPrediction(payload) {
  if (!payload) return null;

  if (typeof payload.top === 'string' && payload.top.trim()) {
    return {
      label: payload.top.trim().toUpperCase(),
      confidence: Number(payload.confidence ?? payload.top_confidence ?? 0)
    };
  }

  const predictions = Array.isArray(payload.predictions)
    ? payload.predictions
    : payload.predictions && typeof payload.predictions === 'object'
      ? Object.values(payload.predictions)
      : [];

  const topPrediction = predictions[0];
  if (!topPrediction) return null;

  const label = (topPrediction.class || topPrediction.label || topPrediction.name || '').trim();
  if (!label) return null;

  return {
    label: label.toUpperCase(),
    confidence: Number(topPrediction.confidence ?? topPrediction.class_confidence ?? 0)
  };
}

function getFreshRoboflowPrediction() {
  if (!latestRoboflowPrediction) return null;
  if (performance.now() - latestRoboflowPredictionAt > ROBOFLOW_RESULT_TTL_MS) return null;
  return latestRoboflowPrediction;
}

async function requestRoboflowPrediction(handList) {
  const now = performance.now();
  if (roboflowInFlight || now - lastRoboflowRequestAt < ROBOFLOW_REQUEST_INTERVAL_MS) {
    return;
  }

  const cropBase64 = buildHandCropBase64(handList);
  if (!cropBase64) return;

  lastRoboflowRequestAt = now;
  roboflowInFlight = true;

  try {
    const params = {
      format: 'json',
      confidence: 'default',
      image_type: 'base64',
      source: 'external'
    };

    const apiKey = getRoboflowApiKey();
    if (apiKey) {
      params.api_key = apiKey;
    }

    const response = await axios.post(
      `${ROBOFLOW_API_URL}/${ROBOFLOW_MODEL_ID}`,
      {
        image: cropBase64,
        image_type: 'base64'
      },
      { params }
    );

    const parsed = parseRoboflowPrediction(response.data);
    if (parsed && parsed.label && parsed.confidence >= ROBOFLOW_MIN_CONFIDENCE) {
      latestRoboflowPrediction = parsed;
      latestRoboflowPredictionAt = performance.now();
    }
  } catch (error) {
    console.warn('Roboflow inference failed, falling back to local classifier:', error);
  } finally {
    roboflowInFlight = false;
  }
}

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
      // Lowered a bit to be more permissive for low-light / low-res cams
      minDetectionConfidence: 0.25,
      minTrackingConfidence: 0.25
    });

    handsDetector.onResults(onResults);
    // FaceMesh optionally provides facial expression heuristics. Disabled by default
    // because some CDN builds trigger wasm/asset fetch errors on certain setups.
    const USE_FACEMESH = false;
    if (USE_FACEMESH) {
      try {
        faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: true,
          minDetectionConfidence: 0.45,
          minTrackingConfidence: 0.45
        });
        faceMesh.onResults(onFaceResults);
      } catch (fex) {
        console.warn('FaceMesh initialization failed:', fex);
        faceMesh = null;
      }
    } else {
      console.info('FaceMesh disabled to avoid wasm/asset CDN errors. Set USE_FACEMESH=true to enable.');
    }
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
          // Send image to both detectors; don't block one on the other
          const promises = [];
          if (handsDetector) promises.push(handsDetector.send({ image: videoElement }));
          if (faceMesh) promises.push(faceMesh.send({ image: videoElement }));
          await Promise.all(promises);

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

function onFaceResults(results) {
  if (!results || !results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    currentFaceLandmarks = null;
    currentFaceEmotion = { label: 'NO FACE', confidence: 0 };
    const fi = document.getElementById('face-indicator');
    const fit = document.getElementById('face-indicator-text');
    if (fi) fi.classList.add('hidden');
    if (fit) fit.innerText = 'No Face';
    return;
  }

  currentFaceLandmarks = results.multiFaceLandmarks[0];
  // Simple heuristic expressions using a few landmark indices
  const lm = currentFaceLandmarks;
  const get = (i) => ({ x: lm[i].x, y: lm[i].y, z: lm[i].z });
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  // Indices for common points (MediaPipe FaceMesh):
  const MOUTH_LEFT = 61, MOUTH_RIGHT = 291, MOUTH_TOP = 13, MOUTH_BOTTOM = 14;
  const LEFT_EYE_TOP = 159, LEFT_EYE_BOTTOM = 145, RIGHT_EYE_TOP = 386, RIGHT_EYE_BOTTOM = 374;

  const minX = Math.min(...lm.map(p => p.x));
  const maxX = Math.max(...lm.map(p => p.x));
  const minY = Math.min(...lm.map(p => p.y));
  const maxY = Math.max(...lm.map(p => p.y));
  const faceWidth = Math.max(0.001, maxX - minX);
  const faceHeight = Math.max(0.001, maxY - minY);

  const mouthWidth = dist(get(MOUTH_LEFT), get(MOUTH_RIGHT)) / faceWidth;
  const mouthHeight = dist(get(MOUTH_TOP), get(MOUTH_BOTTOM)) / faceHeight;
  const leftEyeOpen = dist(get(LEFT_EYE_TOP), get(LEFT_EYE_BOTTOM)) / faceHeight;
  const rightEyeOpen = dist(get(RIGHT_EYE_TOP), get(RIGHT_EYE_BOTTOM)) / faceHeight;

  // Heuristics
  let emotion = 'NEUTRAL';
  let conf = 0.0;
  if (mouthHeight > 0.30) { emotion = 'SURPRISED'; conf = clamp(mouthHeight, 0.3, 1); }
  else if (mouthWidth > 0.36 && mouthHeight < 0.12) { emotion = 'SMILE'; conf = clamp(mouthWidth, 0.36, 1); }
  else if ((leftEyeOpen < 0.02 && rightEyeOpen < 0.02)) { emotion = 'EYES_CLOSED'; conf = 0.9; }
  else { emotion = 'NEUTRAL'; conf = 0.5; }

  currentFaceEmotion = { label: emotion, confidence: conf };

  const fi = document.getElementById('face-indicator');
  const fit = document.getElementById('face-indicator-text');
  if (fi) fi.classList.remove('hidden');
  if (fit) fit.innerText = `${emotion}`;
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
  if (results.multiHandLandmarks) {
    const rawHands = [];

    // If handedness data is present, use it; otherwise fall back to a default label
    const hasHandedness = Array.isArray(results.multiHandedness) && results.multiHandedness.length === results.multiHandLandmarks.length;

    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      let score = 0.5;
      let rawLabel = 'Right';

      if (hasHandedness) {
        const handednessEntry = results.multiHandedness[i];
        if (handednessEntry) {
          if (typeof handednessEntry.score === 'number') {
            score = handednessEntry.score;
          } else if (handednessEntry.classification && handednessEntry.classification[0] && typeof handednessEntry.classification[0].score === 'number') {
            score = handednessEntry.classification[0].score;
          }
          rawLabel = handednessEntry.label || (handednessEntry.classification && handednessEntry.classification[0] && handednessEntry.classification[0].label) || rawLabel;
        }
      }

      // Accept lower scores to be permissive; if handedness is missing we still accept landmarks
      if (score >= 0.20 || !hasHandedness) {
        const physicalLabel = rawLabel === "Left" ? "Right" : "Left";
        rawHands.push({ landmarks: results.multiHandLandmarks[i], label: physicalLabel, score });
      }
    }

    validHands = rawHands.map(h => h.landmarks);
    validHandedness = rawHands.map(h => h.label);

    // Debug: if landmarks arrived but were previously filtered, show counts
    if (debugCoords) {
      debugCoords.innerText = `Detected hands: ${validHands.length} | Handedness data: ${hasHandedness}`;
    }
    console.debug('onResults: multiHandLandmarks count=', results.multiHandLandmarks.length, 'validHands=', validHands.length, 'hasHandedness=', hasHandedness);
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

function checkMotionGestures(handList, currentPred, interpretMode = 'words') {
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

    // Optimized curve size requirements (dy > 0.12, dx > 0.05) to prevent drift false-positives
    if (dy > 0.12 && dx > 0.05) {
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

  // 0. Detect "HELLO" (wave): starts from open-hand (B-like) and shows lateral oscillation
  // Only detect HELLO when interpretMode === 'words' to avoid conflicts with letters
  if (interpretMode === 'words' && (currentPred === "B" || currentPred === "HELLO")) {
    const xs = motionHistory.map(pt => pt.index.x);
    const ys = motionHistory.map(pt => pt.index.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const dx = maxX - minX;
    const dy = maxY - minY;

    if (dx > 0.10 && dy < 0.12) {
      // Count direction changes using 3-frame smoothing
      const smoothed = [];
      for (let i = 0; i < xs.length - 2; i++) smoothed.push((xs[i] + xs[i+1] + xs[i+2]) / 3);
      let dirChanges = 0;
      let currentDir = 0;
      for (let i = 1; i < smoothed.length; i++) {
        const diff = smoothed[i] - smoothed[i-1];
        if (Math.abs(diff) > 0.006) {
          const newDir = diff > 0 ? 1 : -1;
          if (currentDir !== 0 && newDir !== currentDir) dirChanges++;
          currentDir = newDir;
        }
      }
      if (dirChanges >= 2) {
        motionHistory = [];
        return { label: "HELLO", confidence: 0.95 };
      }
    }
  }

  // 2. Detect "Z" (Starts in 'D'/'L'/'G'/'U'/'Z' pointing shapes to avoid triggering on NO SIGN transitions)
  if (currentPred === "D" || currentPred === "L" || currentPred === "G" || currentPred === "U" || currentPred === "Z") {
    const xs = motionHistory.map(pt => pt.index.x);
    const ys = motionHistory.map(pt => pt.index.y);

    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const dx = maxX - minX;
    const dy = maxY - minY;

    // Optimized size requirements (dx > 0.11, dy > 0.08)
    if (dx > 0.11 && dy > 0.08) {
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
        // Filter out slow drift and jitter by requiring larger frame-to-frame change (diff > 0.007)
        if (Math.abs(diff) > 0.007) {
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
  const stabilityConfidenceGate = isStrict ? 0.72 : 0.58;
  // Only use Roboflow/web-hosted model when user selected Interpret Words mode
  if (interpretMode === 'words') {
    void requestRoboflowPrediction(handList);
  }

  const localPrediction = classifier.classify(handList, handednessList, 5, aspectRatio, isStrict);
  const roboflowPrediction = (interpretMode === 'words') ? getFreshRoboflowPrediction() : null;
  let prediction = roboflowPrediction || localPrediction;
  console.debug('processTranslation: local=', localPrediction, 'roboflow=', roboflowPrediction, 'chosen=', prediction);
  
  // Check if we are currently locked in a motion prediction
  if (motionLockFrames > 0) {
    prediction = { label: motionLockLabel, confidence: 0.95 };
    motionLockFrames--;
  } else {
    // Inject motion tracing detection overrides for J & Z
    const motionPred = checkMotionGestures(handList, prediction.label, interpretMode);
    if (motionPred) {
      prediction = motionPred;
      motionLockLabel = motionPred.label;
      motionLockFrames = 12; // Hold J/Z for 12 frames to let the smoothing buffer stabilize and show it clearly
    }
  }

  // If user selected Words mode, suppress single-letter predictions to avoid
  // letters being appended when the user wants words only. This lets Roboflow
  // or motion/word rules drive multi-letter outputs. Single-letter motions
  // (like J/Z) will be suppressed here; switch to Letters mode to enable them.
  if (interpretMode === 'words' && prediction && typeof prediction.label === 'string') {
    const lbl = prediction.label.trim().toUpperCase();
    if (/^[A-Z]$/.test(lbl)) {
      prediction = { label: 'NO SIGN', confidence: 0 };
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
  console.debug('processTranslation: displayed=', predText.innerText, 'confidence=', predConfidenceText.innerText);

  // Smooth prediction
  const bufferedLabel = prediction.label !== "NO SIGN" && prediction.confidence >= stabilityConfidenceGate
    ? prediction.label
    : "NO SIGN";

  predictionBuffer.push(bufferedLabel);
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
          // Respect user's interpret mode: in 'letters' mode, do not accumulate multi-letter words
          const isMultiLetter = typeof dominantLabel === 'string' && dominantLabel.trim().length > 1;
          if (!(interpretMode === 'letters' && isMultiLetter)) {
            appendWordToSentence(dominantLabel);
            playBeep(650, 0.05, 'triangle');
          } else {
            // Provide subtle feedback that a word was detected but not accumulated in Letters mode
            playBeep(220, 0.06, 'sine');
          }
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
  latestRoboflowPrediction = null;
  latestRoboflowPredictionAt = 0;
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
  const filipinoMap = {
    'HELLO': 'KUMUSTA',
    'YES': 'OO',
    'NO': 'HINDI',
    'I LOVE YOU': 'MAHAL KITA',
    'THANK YOU': 'SALAMAT',
    'THANKS': 'SALAMAT',
    'GOOD': 'MABUTI',
    'PLEASE': 'PAWANG',
    'BYE': 'PAALAM',
    'HELLO (WAVE)': 'KUMUSTA'
  };

  const normalized = (word || '').toString().trim().toUpperCase();
  const translated = (interpretMode === 'words' && normalized.length > 1 && filipinoMap[normalized]) ? filipinoMap[normalized] : word;

  const currentText = sentenceOutput.value.trim();
  if ((translated || '').length === 1) {
    sentenceOutput.value = currentText ? currentText + translated : translated;
  } else {
    // Optionally append detected facial expression for context when in Words mode
    const emotionTag = (interpretMode === 'words' && currentFaceEmotion && currentFaceEmotion.label && currentFaceEmotion.label !== 'NEUTRAL' && currentFaceEmotion.label !== 'NO FACE')
      ? ` (${currentFaceEmotion.label})`
      : '';
    sentenceOutput.value = currentText ? currentText + " " + translated + emotionTag : translated + emotionTag;
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
    void saveTrainingSampleToBackend(label, handList, handednessList, aspectRatio)
      .catch((error) => {
        if (trainingSaveStatus) {
          trainingSaveStatus.classList.remove('saving', 'success');
          trainingSaveStatus.classList.add('error');
          trainingSaveStatus.textContent = 'Supabase save status: save failed';
        }
        console.warn('Backend sample save failed:', error);
      });

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
