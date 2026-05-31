# Handy Talk - Real-time Sign Language Translator 🤟

Handy Talk is a premium, client-side web application designed to translate **American Sign Language (ASL)** into text in real-time. Built using vanilla JS, custom CSS, and Google's **MediaPipe Hands**, it operates 100% in the browser with sub-millisecond classification latency and absolute data privacy.

![Premium UI Dark Theme Mockup](https://raw.githubusercontent.com/google/mediapipe/master/docs/images/mobile/hand_crops.png) *(Placeholder caption: Active Hand Joint Mesh Visualization)*

## Features ✨

*   **Dual-Hand Skeletal Mesh Overlay:** Real-time 3D landmark skeleton tracking utilizing MediaPipe. Skeletons are custom-rendered with glowing cyberpunk color palettes (Cyan/Blue for the first hand, Magenta/Pink for the second).
*   **Dynamic Dimensionality Routing:**
    *   *1 Hand Detected:* Normalizes the hand landmarks into a 63-dimensional coordinate space, running heuristic rules (for immediate letters/basic vocabulary) and KNN classification.
    *   *2 Hands Detected:* Normalizes both hands independently, calculates the scale-invariant relative distance between wrists, and routes the search to a combined 129-dimensional KNN classification space.
*   **Custom Sign Training Tool:** Name new signs (1-hand or 2-hands) and record 15-frame spatial samples directly in the browser. Gestures are persisted locally via `localStorage` and can be exported/imported as JSON files.
*   **Sentence Builder:** Automatically accumulates recognized signs over time with space, backspace, and clear controls.
*   **Text-to-Speech (TTS):** Read accumulated sentences aloud using native browser speech synthesis.
*   **Anti-Ghosting Filtering:** Advanced detection parameters (thresholds set to `0.78` and handedness certainty filtered at `>= 0.85`) to prevent background shadows or noise from triggering false dual-hand modes.

## Tech Stack 🛠️

*   **Core:** HTML5, CSS3, JavaScript (ES6 Modules)
*   **AI/ML Tracking:** Google MediaPipe Hands (Tasks Vision API via CDN)
*   **ML Engine:** Custom-built K-Nearest Neighbors (KNN) 3D coordinate vector classifier
*   **Tooling:** Vite (Development Server & Production Bundler)

## Getting Started 🚀

### Prerequisites

*   [Node.js](https://nodejs.org/) (v16 or higher recommended)
*   A webcam connected to your computer

### Installation & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/asl-translator.git
   cd asl-translator
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open the local address printed in your terminal (usually `http://localhost:5173/` or `http://localhost:5174/`) in a secure browser.

## Built-In Default Gestures 📖

The application identifies several gestures out of the box using geometric rule-based algorithms:

*   **Words:** `HELLO` (flat open hand), `YES` (thumbs up or fist), `NO` (snapping fingers), `I LOVE YOU` (thumb, index, and pinky extended).
*   **Letters:** `A`, `B`, `D`, `F` (OK sign), `L`, `U`, `V` (peace sign), `W`, `Y`.

Go to the **Train Custom Signs** tab in the app to teach it new words and letters!

## License 📄

This project is open-source and available under the [MIT License](LICENSE).
