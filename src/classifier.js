/**
 * ASL Gesture Classifier Engine (Multi-Hand Support)
 * Combines Rule-Based heuristic detection for common signs/letters
 * with a dynamic 3D Landmark K-Nearest Neighbors (KNN) classifier for custom gestures.
 * Supports both single-hand and dual-hand signs.
 */

export class ASLClassifier {
  constructor() {
    this.samples = []; // Array of { label: "NAME", numHands: 1|2, features: [...] }
    this.loadDefaultDataset();
  }

  /**
   * Helper to calculate hand scale factor.
   */
  getScaleFactor(landmarks) {
    const wrist = landmarks[0];
    const dxScale = landmarks[9].x - wrist.x;
    const dyScale = landmarks[9].y - wrist.y;
    const dzScale = landmarks[9].z - wrist.z;
    return Math.sqrt(dxScale * dxScale + dyScale * dyScale + dzScale * dzScale) || 1.0;
  }

  /**
   * Normalize 21 3D landmarks to be translation-invariant (relative to wrist)
   * and scale-invariant (relative to hand size).
   * @param {Array} landmarks - 21 coordinates of {x, y, z}
   * @returns {Array} 63-dimensional flat float array
   */
  normalize(landmarks) {
    if (!landmarks || landmarks.length !== 21) return null;

    const wrist = landmarks[0];
    const scaleFactor = this.getScaleFactor(landmarks);

    const normalized = [];
    for (let i = 0; i < 21; i++) {
      const lm = landmarks[i];
      // Translate to wrist origin, then divide by scale factor
      normalized.push((lm.x - wrist.x) / scaleFactor);
      normalized.push((lm.y - wrist.y) / scaleFactor);
      normalized.push((lm.z - wrist.z) / scaleFactor);
    }
    return normalized;
  }

  /**
   * Normalize two hands and compute their relative position.
   * @returns {Array} 129-dimensional flat float array
   */
  normalizeDual(hand1, hand2) {
    const feat1 = this.normalize(hand1);
    const feat2 = this.normalize(hand2);
    if (!feat1 || !feat2) return null;

    const s1 = this.getScaleFactor(hand1);
    const s2 = this.getScaleFactor(hand2);
    const sAvg = (s1 + s2) / 2;

    // Relative offset of hand 2's wrist from hand 1's wrist, normalized by hand sizes
    const dx = (hand2[0].x - hand1[0].x) / sAvg;
    const dy = (hand2[0].y - hand1[0].y) / sAvg;
    const dz = (hand2[0].z - hand1[0].z) / sAvg;

    return [...feat1, ...feat2, dx, dy, dz];
  }

  /**
   * Extract high-level boolean features of finger extensions and gestures.
   * @param {Array} landmarks - 21 coordinates of {x, y, z}
   */
  getFingerStates(landmarks) {
    const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    const wrist = landmarks[0];

    const indexExtended = dist(landmarks[8], wrist) > dist(landmarks[6], wrist);
    const middleExtended = dist(landmarks[12], wrist) > dist(landmarks[10], wrist);
    const ringExtended = dist(landmarks[16], wrist) > dist(landmarks[14], wrist);
    const pinkyExtended = dist(landmarks[20], wrist) > dist(landmarks[18], wrist);

    const thumbExtended = dist(landmarks[4], wrist) > dist(landmarks[2], wrist) && 
                          dist(landmarks[4], landmarks[9]) > dist(landmarks[2], landmarks[9]);

    const thumbIndexDist = dist(landmarks[4], landmarks[8]);
    const thumbMiddleDist = dist(landmarks[4], landmarks[12]);

    return {
      thumb: thumbExtended,
      index: indexExtended,
      middle: middleExtended,
      ring: ringExtended,
      pinky: pinkyExtended,
      thumbIndexDist,
      thumbMiddleDist,
      indexMiddleDist: dist(landmarks[8], landmarks[12]),
      ringPinkyDist: dist(landmarks[16], landmarks[20])
    };
  }

  /**
   * Heuristic rules for common ASL letters and basic signs.
   * @param {Array} landmarks - 21 coordinates
   */
  detectHeuristics(landmarks) {
    const states = this.getFingerStates(landmarks);
    const wrist = landmarks[0];
    const thumbTip = landmarks[4];

    // HELLO (Open Hand)
    if (states.thumb && states.index && states.middle && states.ring && states.pinky) {
      return { label: "HELLO", confidence: 0.95 };
    }

    // I LOVE YOU
    if (states.thumb && states.index && !states.middle && !states.ring && states.pinky) {
      return { label: "I LOVE YOU", confidence: 0.95 };
    }

    // YES (Thumbs Up)
    if (states.thumb && !states.index && !states.middle && !states.ring && !states.pinky) {
      if (thumbTip.y < landmarks[2].y) {
        return { label: "YES", confidence: 0.95 };
      }
      return { label: "A", confidence: 0.85 };
    }

    // NO (ASL "no")
    if (!states.ring && !states.pinky && states.index && states.middle) {
      if (states.thumbIndexDist < 0.12 && states.indexMiddleDist < 0.08) {
        return { label: "NO", confidence: 0.90 };
      }
    }

    // Letter B
    if (!states.thumb && states.index && states.middle && states.ring && states.pinky) {
      if (states.indexMiddleDist < 0.05) {
        return { label: "B", confidence: 0.92 };
      }
    }

    // Letter L
    if (states.thumb && states.index && !states.middle && !states.ring && !states.pinky) {
      return { label: "L", confidence: 0.95 };
    }

    // Letter V
    if (!states.thumb && states.index && states.middle && !states.ring && !states.pinky) {
      if (states.indexMiddleDist > 0.08) {
        return { label: "V", confidence: 0.95 };
      }
      return { label: "U", confidence: 0.90 };
    }

    // Letter W
    if (!states.thumb && states.index && states.middle && states.ring && !states.pinky) {
      return { label: "W", confidence: 0.90 };
    }

    // Letter Y
    if (states.thumb && !states.index && !states.middle && !states.ring && states.pinky) {
      return { label: "Y", confidence: 0.95 };
    }

    // OK Sign / Letter F
    if (states.thumbIndexDist < 0.05 && states.middle && states.ring && states.pinky) {
      return { label: "F", confidence: 0.95 };
    }

    // Letter D
    if (states.index && !states.middle && !states.ring && !states.pinky && states.thumbIndexDist > 0.08) {
      if (states.thumbMiddleDist < 0.06) {
        return { label: "D", confidence: 0.90 };
      }
    }

    return null;
  }

  /**
   * Classify hand landmarks using KNN + Heuristics.
   * Runs rule-based classification first, then KNN.
   * Supports both 1-hand and 2-hand inputs.
   * @param {Array} handList - Array of hands landmarks, e.g. [hand1] or [hand1, hand2]
   */
  classify(handList, k = 5) {
    if (!handList || handList.length === 0) {
      return { label: "NO SIGN", confidence: 0 };
    }

    const numHands = handList.length;

    // 1. For single hand, run rule-based check first
    if (numHands === 1) {
      const heuristicMatch = this.detectHeuristics(handList[0]);
      if (heuristicMatch) {
        return heuristicMatch;
      }
    }

    // 2. Query KNN Classifier
    const activeSamples = this.samples.filter(s => s.numHands === numHands);
    if (activeSamples.length === 0) {
      return { label: "NO SIGN", confidence: 0 };
    }

    // Normalize test features
    let testFeature;
    if (numHands === 1) {
      testFeature = this.normalize(handList[0]);
    } else {
      testFeature = this.normalizeDual(handList[0], handList[1]);
    }

    if (!testFeature) return { label: "NO SIGN", confidence: 0 };

    // Calculate Euclidean distances
    const dim = testFeature.length;
    const distances = activeSamples.map(sample => {
      let sumSq = 0;
      for (let i = 0; i < dim; i++) {
        const diff = testFeature[i] - sample.features[i];
        sumSq += diff * diff;
      }
      return {
        label: sample.label,
        distance: Math.sqrt(sumSq)
      };
    });

    // Sort by distance ascending
    distances.sort((a, b) => a.distance - b.distance);

    // Get top K nearest neighbors
    const nearest = distances.slice(0, Math.min(k, distances.length));

    // Threshold check: Higher dimensions (129 for 2 hands) naturally produce larger Euclidean distances.
    // Scale distance limit appropriately.
    const threshold = numHands === 2 ? 1.75 : 1.20;
    if (nearest[0].distance > threshold) {
      return { label: "NO SIGN", confidence: 0 };
    }

    // Count class votes
    const votes = {};
    nearest.forEach(neighbor => {
      votes[neighbor.label] = (votes[neighbor.label] || 0) + 1;
    });

    // Find class with majority votes
    let bestLabel = "NO SIGN";
    let maxVotes = 0;
    Object.keys(votes).forEach(label => {
      if (votes[label] > maxVotes) {
        maxVotes = votes[label];
        bestLabel = label;
      }
    });

    const totalNearest = nearest.length;
    const labelVotes = votes[bestLabel];
    
    let confidence = labelVotes / totalNearest;

    // Adjust confidence based on match proximity
    const avgDistance = nearest
      .filter(n => n.label === bestLabel)
      .reduce((sum, n) => sum + n.distance, 0) / labelVotes;
      
    const distancePenalty = Math.max(0, 1 - avgDistance / threshold);
    confidence = (confidence * 0.7) + (distancePenalty * 0.3);

    return {
      label: bestLabel,
      confidence: Math.min(1.0, Math.max(0.1, confidence))
    };
  }

  /**
   * Save a hand gesture sample (1 or 2 hands) to the training database.
   */
  addSample(label, handList) {
    if (!handList || handList.length === 0) return false;
    
    const numHands = handList.length;
    let features;

    if (numHands === 1) {
      features = this.normalize(handList[0]);
    } else {
      features = this.normalizeDual(handList[0], handList[1]);
    }

    if (features) {
      this.samples.push({
        label: label.toUpperCase().trim(),
        numHands: numHands,
        features: features
      });
      this.saveToLocalStorage();
      return true;
    }
    return false;
  }

  /**
   * Remove all samples of a specific gesture.
   */
  deleteGesture(label) {
    const initialLength = this.samples.length;
    this.samples = this.samples.filter(sample => sample.label !== label.toUpperCase());
    this.saveToLocalStorage();
    return initialLength - this.samples.length;
  }

  /**
   * Clear the entire custom dataset.
   */
  clearDataset() {
    this.samples = [];
    localStorage.removeItem("asl_gesture_samples");
  }

  /**
   * Save training dataset to browser LocalStorage.
   */
  saveToLocalStorage() {
    try {
      localStorage.setItem("asl_gesture_samples", JSON.stringify(this.samples));
    } catch (e) {
      console.error("Failed to save dataset to local storage:", e);
    }
  }

  /**
   * Load dataset from LocalStorage or initialize default dataset.
   */
  loadDefaultDataset() {
    try {
      const stored = localStorage.getItem("asl_gesture_samples");
      if (stored) {
        const parsed = JSON.parse(stored);
        // Map and support legacy models lacking the numHands key
        this.samples = parsed.map(item => ({
          label: item.label,
          numHands: item.numHands || (item.features.length === 63 ? 1 : 2),
          features: item.features
        }));
      }
    } catch (e) {
      console.error("Failed to load local dataset:", e);
    }
  }

  /**
   * Export database as JSON content.
   */
  exportDatasetJSON() {
    return JSON.stringify(this.samples, null, 2);
  }

  /**
   * Import database from JSON content.
   */
  importDatasetJSON(jsonString) {
    try {
      const parsed = JSON.parse(jsonString);
      if (Array.isArray(parsed)) {
        // Validate items contain label and correct length features
        const valid = parsed.every(item => 
          item.label && 
          Array.isArray(item.features) && 
          (item.features.length === 63 || item.features.length === 129)
        );
        if (valid) {
          this.samples = parsed.map(item => ({
            label: item.label,
            numHands: item.numHands || (item.features.length === 63 ? 1 : 2),
            features: item.features
          }));
          this.saveToLocalStorage();
          return true;
        }
      }
      return false;
    } catch (e) {
      console.error("Import failed:", e);
      return false;
    }
  }
}
