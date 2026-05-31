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
  getScaleFactor(landmarks, aspectRatio = 1.3333) {
    const wrist = landmarks[0];
    const dxScale = (landmarks[9].x - wrist.x) * aspectRatio;
    const dyScale = landmarks[9].y - wrist.y;
    const dzScale = (landmarks[9].z - wrist.z) * aspectRatio;
    return Math.sqrt(dxScale * dxScale + dyScale * dyScale + dzScale * dzScale) || 1.0;
  }

  /**
   * Normalize 21 3D landmarks to be translation-invariant (relative to wrist)
   * and scale-invariant (relative to hand size).
   * @param {Array} landmarks - 21 coordinates of {x, y, z}
   * @returns {Array} 63-dimensional flat float array
   */
  normalize(landmarks, handedness = "Right", aspectRatio = 1.3333) {
    if (!landmarks || landmarks.length !== 21) return null;

    const wrist = landmarks[0];
    const scaleFactor = this.getScaleFactor(landmarks, aspectRatio);

    // Calculate rotation angle to align the wrist-to-middle-knuckle (0 -> 9) vector straight up
    const mcp = landmarks[9];
    const dxMcp = (mcp.x - wrist.x) * aspectRatio;
    const dyMcp = mcp.y - wrist.y;
    // Angle relative to straight up (which is dx=0, dy < 0)
    const angle = Math.atan2(dxMcp, -dyMcp);
    const cosA = Math.cos(-angle);
    const sinA = Math.sin(-angle);

    const normalized = [];
    for (let i = 0; i < 21; i++) {
      const lm = landmarks[i];
      const dx = (lm.x - wrist.x) * aspectRatio;
      const dy = lm.y - wrist.y;
      const dz = (lm.z - wrist.z) * aspectRatio;
      
      // Rotate dx and dy around the wrist
      const rotatedDx = dx * cosA - dy * sinA;
      const rotatedDy = dx * sinA + dy * cosA;

      // Mirror the X-axis for Left Hand input to match the Right Hand baseline templates
      const finalDx = handedness === "Left" ? -rotatedDx : rotatedDx;

      normalized.push(finalDx / scaleFactor);
      normalized.push(rotatedDy / scaleFactor);
      normalized.push(dz / scaleFactor);
    }
    return normalized;
  }

  /**
   * Extract high-level boolean features of finger extensions and gestures.
   * @param {Array} landmarks - 21 coordinates of {x, y, z}
   */
  getFingerStates(landmarks) {
    const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);

    // Calculate straight-line tip-to-knuckle distance vs segment sum to detect true finger extension
    const isStraight = (tip, dip, pip, mcp) => {
      const straightDist = dist(landmarks[tip], landmarks[mcp]);
      const segmentSum = dist(landmarks[tip], landmarks[dip]) + 
                         dist(landmarks[dip], landmarks[pip]) + 
                         dist(landmarks[pip], landmarks[mcp]);
      return straightDist > 0.82 * segmentSum; // Calibrated straightness threshold
    };

    const indexExtended = isStraight(8, 7, 6, 5);
    const middleExtended = isStraight(12, 11, 10, 9);
    const ringExtended = isStraight(16, 15, 14, 13);
    const pinkyExtended = isStraight(20, 19, 18, 17);

    // Thumb extended check: straight length vs segment sum, pointing outward from palm index base (9)
    const thumbStraight = dist(landmarks[4], landmarks[1]) > 0.85 * (
      dist(landmarks[4], landmarks[3]) + dist(landmarks[3], landmarks[2]) + dist(landmarks[2], landmarks[1])
    );
    const thumbExtended = thumbStraight && dist(landmarks[4], landmarks[9]) > dist(landmarks[2], landmarks[9]);

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
   * Classify hand landmarks using 100% data-driven KNN.
   * Compares the hand joint distances against pre-trained patterns.
   * Supports single-hand (63 features) inputs.
   * @param {Array} handList - Array of hands landmarks, e.g. [hand1]
   * @param {Array} handednessList - Array of hand labels, e.g. ["Right"]
   */
  classify(handList, handednessList = ["Right"], k = 5, aspectRatio = 1.3333) {
    if (!handList || handList.length === 0) {
      return { label: "NO SIGN", confidence: 0 };
    }

    const landmarks = handList[0];
    const handedness = handednessList[0] || "Right";

    // Query KNN Classifier
    const activeSamples = this.samples;
    if (activeSamples.length === 0) {
      return { label: "NO SIGN", confidence: 0 };
    }

    // Normalize test features
    const testFeature = this.normalize(landmarks, handedness, aspectRatio);
    if (!testFeature) return { label: "NO SIGN", confidence: 0 };

    // 1. Try Rule-Based classification first for core fingerspelling letters
    const ruleLabel = this.classifyRuleBased(testFeature);
    if (ruleLabel) {
      return {
        label: ruleLabel,
        confidence: 0.95,
        nearestLabel: ruleLabel,
        nearestDistance: 0.0,
        testFeature: testFeature,
        nearest3D: [ruleLabel + " (Rule)"],
        nearest2D: [ruleLabel + " (Rule)"]
      };
    }

    // Calculate 3D Euclidean distances
    const dim = testFeature.length;
    const distances = activeSamples.map(sample => {
      let sumSq = 0;
      const sampleDim = sample.features.length;
      // Safeguard in case there are dual hand samples in legacy local storage
      const maxDim = Math.min(dim, sampleDim);
      for (let i = 0; i < maxDim; i++) {
        const diff = testFeature[i] - sample.features[i];
        sumSq += diff * diff;
      }
      return {
        label: sample.label,
        distance: Math.sqrt(sumSq)
      };
    });

    // Calculate 2D Euclidean distances
    const distances2D = activeSamples.map(sample => {
      let sumSq = 0;
      const sampleDim = sample.features.length;
      const maxDim = Math.min(dim, sampleDim);
      for (let i = 0; i < maxDim; i++) {
        if (i % 3 === 2) continue; // Skip Z
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
    distances2D.sort((a, b) => a.distance - b.distance);

    // Get top K nearest neighbors
    const nearest = distances.slice(0, Math.min(k, distances.length));

    // Threshold check
    const threshold = 0.90;
    const isBelowThreshold = nearest[0].distance <= threshold;

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
      label: isBelowThreshold ? bestLabel : "NO SIGN",
      confidence: isBelowThreshold ? Math.min(1.0, Math.max(0.1, confidence)) : 0,
      nearestLabel: nearest[0].label,
      nearestDistance: nearest[0].distance,
      testFeature: testFeature,
      nearest3D: nearest.slice(0, 3).map(n => `${n.label} (${n.distance.toFixed(2)})`),
      nearest2D: distances2D.slice(0, 3).map(n => `${n.label} (${n.distance.toFixed(2)})`)
    };
  }

  /**
   * Rule-Based heuristic classifier for core fingerspelling letters.
   * Utilizes finger extension thresholds on rotated normalized 3D landmarks.
   */
  classifyRuleBased(features) {
    const lm = [];
    for (let i = 0; i < 21; i++) {
      lm.push({
        x: features[i * 3],
        y: features[i * 3 + 1],
        z: features[i * 3 + 2]
      });
    }

    // Helper to calculate the straightness ratio of a finger (1.0 = fully straight, <0.8 = bent/curved)
    const getStraightness = (tip, dip, pip, mcp) => {
      const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      const straightDist = d(lm[tip], lm[mcp]);
      const segmentSum = d(lm[tip], lm[dip]) + d(lm[dip], lm[pip]) + d(lm[pip], lm[mcp]);
      return straightDist / (segmentSum || 1.0);
    };

    const isStraightIndex = getStraightness(8, 7, 6, 5) > 0.82;
    const isStraightMiddle = getStraightness(12, 11, 10, 9) > 0.82;
    const isStraightRing = getStraightness(16, 15, 14, 13) > 0.82;
    const isStraightPinky = getStraightness(20, 19, 18, 17) > 0.80;

    // Calculate finger extensions using Y differences in normalized space
    const extIndex = (lm[5].y - lm[8].y) > 0.35;
    const extMiddle = (lm[9].y - lm[12].y) > 0.35;
    const extRing = (lm[13].y - lm[16].y) > 0.35;
    const extPinky = (lm[17].y - lm[20].y) > 0.28;
    
    const thumbExtended = lm[4].x > 0.55;

    // 1. B & C: Index, Middle, Ring, Pinky extended
    if (extIndex && extMiddle && extRing && extPinky) {
      if (!isStraightIndex && !isStraightMiddle) {
        return "C";
      }
      if (isStraightIndex && isStraightMiddle && isStraightRing && isStraightPinky && !thumbExtended) {
        return "B";
      }
    }
    
    // 2. F: Middle, Ring, Pinky extended (straight), Index folded
    if (!extIndex && extMiddle && extRing && extPinky && isStraightMiddle && isStraightRing && isStraightPinky) {
      return "F";
    }
    
    // 3. W: Index, Middle, Ring extended (straight), Pinky folded
    if (extIndex && extMiddle && extRing && !extPinky && isStraightIndex && isStraightMiddle && isStraightRing) {
      return "W";
    }
    
    // 4. Y: Pinky extended (straight), Thumb extended, Index/Middle folded
    if (!extIndex && !extMiddle && !extRing && extPinky && isStraightPinky && thumbExtended) {
      return "Y";
    }
    
    // 5. I: Pinky extended (straight), Thumb folded, Index/Middle folded
    if (!extIndex && !extMiddle && !extRing && extPinky && isStraightPinky && !thumbExtended) {
      return "I";
    }
    
    // 6. L: Index extended (straight), Thumb extended, others folded
    if (extIndex && isStraightIndex && !extMiddle && !extRing && !extPinky && thumbExtended) {
      return "L";
    }
    
    // 7. D: Index extended (straight), Thumb folded, others folded
    if (extIndex && isStraightIndex && !extMiddle && !extRing && !extPinky && !thumbExtended) {
      return "D";
    }
    
    // 8. V, U, R: Index and Middle extended vertically (straight), Ring and Pinky folded
    if (extIndex && extMiddle && isStraightIndex && isStraightMiddle && !extRing && !extPinky) {
      const indexMiddleDist = Math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y);
      if (lm[8].x < lm[12].x) {
        return "R"; // Crossed
      }
      if (indexMiddleDist > 0.38) {
        return "V"; // Separated
      } else {
        return "U"; // Close together
      }
    }

    // 9. I LOVE YOU: Thumb, Index, Pinky extended (straight), Middle and Ring folded
    if (extIndex && isStraightIndex && !extMiddle && !extRing && extPinky && isStraightPinky && thumbExtended) {
      return "I LOVE YOU";
    }

    return null; // Fallback to KNN
  }

  /**
   * Save a hand gesture sample to the training database.
   */
  addSample(label, handList, handednessList = ["Right"], aspectRatio = 1.3333) {
    if (!handList || handList.length === 0) return false;
    
    const landmarks = handList[0];
    const handedness = handednessList[0] || "Right";
    const features = this.normalize(landmarks, handedness, aspectRatio);

    if (features) {
      this.samples.push({
        label: label.toUpperCase().trim(),
        numHands: 1,
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
