Care for Your Brain: A Browser-Based Early Cognitive Decline Assessment Tool

Overview:

Care for Your Brain is an open-source, browser-based web application designed to assist in the early detection of cognitive decline (such as dementia) by analyzing speech patterns and eye-tracking data. The tool generates an interpretable report using GPT-4 and provides visual feedback to users on how their scores compare to healthy baselines.

Features
Speech Analysis

Measures pitch variability, volume consistency, tone stability, and fluency.

Scores each metric from 0–100%, where 100% indicates optimal performance (close to healthy baseline).

Uses either real-time audio analysis or simulated metrics if no audio is provided.

Plots performance visually with color-coded status indicators (green/orange/red).

Eye Tracking Tests

Saccadic Test: Measures eye movement responsiveness by following a moving object.

Fixation Test: Measures stability of gaze on a static object.

Pupil Test: Simulates or collects average pupil size data.

Scores each metric and computes an aggregate “Eye Score”.

Report Generation

GPT-4 generates a clear, two-table summary of all results.

Indicates whether results suggest low, moderate, or high cognitive risk.

No diagnosis—just early insights and a recommendation to seek clinical evaluation if needed.

Frontend & Backend Integration

Frontend uses MediaPipe JS for real-time facial landmark tracking.

Backend is built with Flask, uses Librosa and NumPy for signal processing, and connects to OpenAI’s API for reporting.

Tech Stack
Frontend: HTML5, JavaScript, MediaPipe Face Mesh (JS), Web Audio API

Backend: Python 3, Flask, NumPy, Librosa, Matplotlib, GPT-4 (OpenAI API)

Visualization: Matplotlib plots returned as base64-encoded images

Speech Input: Browser-based audio + optional speech recognition

Session Handling: Flask-Session

Community Contribution Guide
This project is designed for collaborative development. Contributors are welcome to:

Add more biometric tests (e.g. drawing, memory recall)

Improve speech analysis (e.g. add more features like speech rate or pauses)

Replace simulated inputs with real webcam/audio-based metrics

Improve UI/UX for accessibility and elderly-friendly usage

Localize the app into other languages

Enhance the GPT prompt or replace it with open-source LLM alternatives

 Want to contribute? Fork the repo, add a module or suggest an enhancement via issues or pull requests!

📄 License
This project is open-sourced under the MIT License. Use it, build on it, and help make early detection more accessible and affordable for everyone.
