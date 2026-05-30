import { ASLClassifier } from './classifier.js';

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

const classifier = new ASLClassifier();

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

    // Configure MediaPipe for up to 2 hands
    handsDetector.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.78,
      minTrackingConfidence: 0.78
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
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const score = results.multiHandedness[i].score;
      if (score >= 0.85) { // Strict score threshold for hand detection validity
        validHands.push(results.multiHandLandmarks[i]);
        validHandedness.push(results.multiHandedness[i]);
      }
    }
  }

  const numHands = validHands.length;

  if (numHands > 0) {
    currentHandList = validHands;
    
    // Update Hand presence indicator UI
    handIndicator.classList.remove('hidden');
    if (numHands === 1) {
      const handedness = validHandedness[0].label;
      handIndicatorText.innerText = `${handedness} Hand`;
      handIndicator.style.borderColor = HAND_COLORS[0].primary;
      handIndicator.style.boxShadow = `0 0 15px ${HAND_COLORS[0].glow}`;
    } else {
      handIndicatorText.innerText = "2 Hands Detected";
      handIndicator.style.borderColor = 'var(--accent-purple)';
      handIndicator.style.boxShadow = `0 0 15px rgba(200, 100, 255, 0.4)`;
    }

    // Draw Skeletons for each hand with distinct color sets
    if (isSkeletonVisible) {
      currentHandList.forEach((landmarks, idx) => {
        const colorPalette = HAND_COLORS[idx % HAND_COLORS.length];
        drawHandSkeleton(landmarks, colorPalette);
      });
    }

    // Process predictions/training
    if (activeTab === 'translate') {
      processTranslation(currentHandList);
    } else if (activeTab === 'train') {
      processTraining(currentHandList);
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

function processTranslation(handList) {
  const prediction = classifier.classify(handList);
  
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
  predText.innerText = "NO SIGN";
  predText.classList.add('empty');
  predConfidenceFill.style.width = '0%';
  predConfidenceText.innerText = '0%';

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

function processTraining(handList) {
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

  const success = classifier.addSample(label, handList);
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
    const key = `${sample.label} (${sample.numHands} Hand${sample.numHands > 1 ? 's' : ''})`;
    grouped[key] = {
      label: sample.label,
      count: (grouped[key]?.count || 0) + 1,
      numHands: sample.numHands
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
      <td><strong>${item.label}</strong> <span class="badge" style="background: rgba(255,255,255,0.05); color: var(--text-secondary); font-size:10px; padding:2px 6px; border-radius:4px; margin-left:8px;">${item.numHands} Hand${item.numHands > 1 ? 's' : ''}</span></td>
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

  btnClearDataset.addEventListener('click', () => {
    if (confirm("WARNING: This will wipe all custom gestures and reset the engine to default. This cannot be undone. Proceed?")) {
      classifier.clearDataset();
      renderGestureList();
      updateStats();
      playBeep(200, 0.4, 'sawtooth');
    }
  });
}

window.addEventListener('DOMContentLoaded', initApp);
