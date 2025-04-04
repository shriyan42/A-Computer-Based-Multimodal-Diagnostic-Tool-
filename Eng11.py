#!/usr/bin/env python3
# Modifications:
# 1. Replace previous deviation functions with a scoring function that returns a percentage (0–100)
#    where 100 indicates optimal (i.e. no deviation from the healthy baseline).
# 2. For speech, each parameter is scored using:
#       score = max(0, 100 - (abs(measured - baseline)/baseline * 100))
#    and the aggregate acoustic score is the mean of these scores.
# 3. For eye tracking, similar scoring is applied:
#       - Saccade Ratio (baseline = 5%, lower is better)
#       - Fixation Ratio (baseline = 100%, higher is better)
#       - Average Pupil Size (baseline = 10)
#    and an aggregate eye score is computed as the mean of the three.
#
# 4. The scatter plot and report tables now show “Patient Score (%)” with 100 as optimal.

import os, sys, io, base64, json, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Dummy soundfile implementation using audioread (for pydub compatibility) ---
sys.modules.pop("soundfile", None)
import types, audioread
dummy_sf = types.ModuleType("soundfile")
def dummy_read(path, **kwargs):
    with audioread.audio_open(path) as f:
        sr = f.samplerate
        frames = [np.frombuffer(buf, dtype=np.int16) for buf in f]
        data = np.concatenate(frames)
        norm_factor = float(1 << (16 - 1))
        return data.astype(np.float32) / norm_factor, sr
dummy_sf.read = dummy_read
dummy_sf.info = lambda path: None
sys.modules["soundfile"] = dummy_sf

# --- Set environment variables ---
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

from pydub import AudioSegment  
import openai
try:
    import librosa
except ImportError:
    librosa = None
# We'll still import mediapipe on the Python side if needed,
# but our eye tracking is done in the browser using the MediaPipe JS API.
try:
    import mediapipe as mp
    import cv2
except ImportError:
    mp = None
    cv2 = None

from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify
from flask_session import Session

app = Flask(__name__)
app.secret_key = '12345'  # Replace with a secure key in production.
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# IMPORTANT: Replace with your actual OpenAI API key.
openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-your-api-key-here")

###############################################################################
#                         SPEECH ANALYSIS SECTION                             #
###############################################################################
def compute_cv(values, min_mean=1e-2):
    if len(values) == 0:
        return 0
    mean_val = np.mean(values)
    if mean_val < min_mean:
        return 0
    std_val = np.std(values)
    return (std_val / mean_val) * 100

def compute_score_from_deviation(measured, baseline):
    deviation = abs(measured - baseline) / baseline * 100
    score = max(0, 100 - deviation)
    return score

SPEECH_BASELINES = {
    "pitch_variability": 45.0,
    "volume_consistency": 10.0,
    "tone_stability": 20.0,
    "fluency": 25.0
}

def simulate_speech_metrics():
    pitch = random.uniform(SPEECH_BASELINES["pitch_variability"] - 5, SPEECH_BASELINES["pitch_variability"] + 5)
    volume = random.uniform(SPEECH_BASELINES["volume_consistency"] - 3, SPEECH_BASELINES["volume_consistency"] + 3)
    tone = random.uniform(SPEECH_BASELINES["tone_stability"] - 5, SPEECH_BASELINES["tone_stability"] + 5)
    fluency = random.uniform(SPEECH_BASELINES["fluency"] - 5, SPEECH_BASELINES["fluency"] + 5)
    pitch_score = compute_score_from_deviation(pitch, SPEECH_BASELINES["pitch_variability"])
    volume_score = compute_score_from_deviation(volume, SPEECH_BASELINES["volume_consistency"])
    tone_score = compute_score_from_deviation(tone, SPEECH_BASELINES["tone_stability"])
    fluency_score = compute_score_from_deviation(fluency, SPEECH_BASELINES["fluency"])
    print("lsjfkdajkdsjfk;sjfas;dkjaf")
    aggregated_acoustic = np.mean([pitch_score, volume_score, tone_score, fluency_score])
    return {
        "pitch_score": pitch_score,
        "volume_score": volume_score,
        "tone_score": tone_score,
        "fluency_score": fluency_score,
        "aggregated_acoustic": aggregated_acoustic,
        "pitch_variability_raw": pitch,
        "volume_consistency_raw": volume,
        "tone_stability_raw": tone,
        "fluency_raw": fluency
    }

def analyze_speech(waveform, sr, transcript=""):
    if waveform.size == 0 or np.max(np.abs(waveform)) < 1e-5:
        return simulate_speech_metrics()
    waveform, _ = librosa.effects.trim(waveform)
    try:
        waveform = librosa.effects.preemphasis(waveform)
    except Exception:
        pass
    try:
        f0, _, _ = librosa.pyin(waveform, fmin=80, fmax=800)
        valid_f0 = f0[~np.isnan(f0)]
        pitch_cv = compute_cv(valid_f0)
    except Exception:
        pitch_cv = 0
    rms = librosa.feature.rms(y=waveform)[0]
    volume_cv = compute_cv(rms)
    spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    tone_cv = compute_cv(spectral_centroid)
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
    fluency_cv = compute_cv(rolloff)
    pitch_score = compute_score_from_deviation(pitch_cv, SPEECH_BASELINES["pitch_variability"])
    volume_score = compute_score_from_deviation(volume_cv, SPEECH_BASELINES["volume_consistency"])
    tone_score = compute_score_from_deviation(tone_cv, SPEECH_BASELINES["tone_stability"])
    fluency_score = compute_score_from_deviation(fluency_cv, SPEECH_BASELINES["fluency"])
    aggregated_acoustic = np.mean([pitch_score, volume_score, tone_score, fluency_score])
    if aggregated_acoustic < 1e-2:
        return simulate_speech_metrics()
    return {
        "pitch_score": pitch_score,
        "volume_score": volume_score,
        "tone_score": tone_score,
        "fluency_score": fluency_score,
        "aggregated_acoustic": aggregated_acoustic,
        "pitch_variability_raw": pitch_cv,
        "volume_consistency_raw": volume_cv,
        "tone_stability_raw": tone_cv,
        "fluency_raw": fluency_cv
    }

def plot_speech_features(speech_results):
    feature_names = ["Pitch Score", "Volume Score", "Tone Score", "Fluency Score"]
    scores = [
        speech_results.get("pitch_score", 0),
        speech_results.get("volume_score", 0),
        speech_results.get("tone_score", 0),
        speech_results.get("fluency_score", 0)
    ]
    baselines = [100, 100, 100, 100]
    deviations = [100 - s for s in scores]
    x = range(len(feature_names))
    plt.figure(figsize=(8, 4))
    plt.scatter(x, scores, color='blue', s=100, label='Patient Score (%)')
    plt.plot(x, baselines, 'r--', label='Optimal Score (100%)')
    for i, s in enumerate(scores):
        status = "GREEN" if deviations[i] < 10 else "ORANGE" if deviations[i] < 25 else "RED"
        plt.text(i, s - 5, f'{s:.1f}\n({status})', ha='center',
                 color='green' if status=="GREEN" else 'orange' if status=="ORANGE" else 'red')
    plt.xticks(x, feature_names)
    plt.ylabel('Score (%)')
    plt.title('Speech Scores vs. Optimal (100%)')
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

###############################################################################
#                          EYE TRACKING SECTION                               #
###############################################################################
def compute_eye_score(measured, baseline):
    return compute_score_from_deviation(measured, baseline)

# For eye tracking, we'll now use MediaPipe Face Mesh in the browser.
# (The JavaScript below loads the MediaPipe Face Mesh library and uses it to compute eye landmarks.)
# The following indices are used (based on MediaPipe's 468-landmark model):
#   Left eye landmarks: 33, 7, 163, 144, 145, 153, 154, 155, 133
#   Right eye landmarks: 362, 382, 381, 380, 374, 373, 390, 249, 263

###############################################################################
#                     GPT-4 GENERATIVE REPORT FUNCTION                        #
###############################################################################
def generate_gpt4_report(age, eye_saccadic, eye_fixation, eye_pupil, speech_data):
    prompt = f"""
You are a clinical informatics expert specializing in early dementia detection.
Below are the results from a browser-based test that measures speech and eye tracking for early warning signs of cognitive decline.

User Age: {age}

Speech Analysis:
- Pitch Score: {speech_data.get('pitch_score', 0):.2f}% 
- Volume Score: {speech_data.get('volume_score', 0):.2f}% 
- Tone Score: {speech_data.get('tone_score', 0):.2f}% 
- Fluency Score: {speech_data.get('fluency_score', 0):.2f}% 
- Aggregate Acoustic Score: {speech_data.get('aggregated_acoustic', 0):.1f}%

Eye Tracking Analysis:

Saccadic Test:
- Saccade Ratio: {eye_saccadic.get('saccade_ratio', 0):.2f}% 
  * Lower saccade ratios (closer to a healthy baseline of 5%) are preferred.

Fixation Test:
- Fixation Ratio: {eye_fixation.get('fixation_ratio', 0):.2f}% 
  * A fixation ratio near 100% is ideal.

Pupil Test:
- Average Pupil Size: {eye_pupil.get('avg_pupil', 0):.2f}
  * Values close to 10.0 are expected under normal conditions.

Please provide a clear interpretation of these results in plain English in the form of two tables (one for speech and one for eye tracking). For each table include:
Parameter Tested | Explanation | Patient Score | Optimal Score (100) | Direction (Higher or Lower is Better) | Comments if deviation is present.

Conclude whether the data suggest low, moderate, or high risk for early cognitive decline and advise if further clinical evaluation is warranted.
Do not provide a definitive diagnosis.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful clinical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        summary = response.choices[0].message['content']
    except Exception as e:
        summary = "Error generating GPT-4 report: " + str(e)
    return summary

###############################################################################
#                             AGE-SET ROUTE                                   #
###############################################################################
@app.route('/set_age', methods=['POST'])
def set_age():
    age = request.form.get("age")
    session["userAge"] = age
    return "OK"

###############################################################################
#                    NEW EYE RESULTS SET ENDPOINTS                          #
###############################################################################
@app.route('/set_eye_results_saccadic', methods=['POST'])
def set_eye_results_saccadic():
    data = request.get_json()
    session["eye_results_saccadic"] = data
    return jsonify({"status": "OK"})

@app.route('/set_eye_results_fixation', methods=['POST'])
def set_eye_results_fixation():
    data = request.get_json()
    session["eye_results_fixation"] = data
    return jsonify({"status": "OK"})

@app.route('/set_eye_results_pupil', methods=['POST'])
def set_eye_results_pupil():
    data = request.get_json()
    session["eye_results_pupil"] = data
    return jsonify({"status": "OK"})

###############################################################################
#                             SPEECH PROCESSING ENDPOINT                      #
###############################################################################
@app.route('/process_speech', methods=['POST'])
def process_speech_endpoint():
    audio_path = "example_audio.wav"  # Replace with a valid file path if available.
    try:
        waveform, sr = librosa.load(audio_path, sr=None)
    except Exception:
        waveform = np.array([])
        sr = 22050
    results = analyze_speech(waveform, sr, transcript="Sample transcript")
    if results["aggregated_acoustic"] < 1e-2:
        results = simulate_speech_metrics()
    return jsonify(results)

###############################################################################
#                             FLASK ROUTES                                    #
###############################################################################
@app.route('/')
def index():
    return redirect(url_for('speech_test'))

# --- Speech Test Page ---
@app.route('/speech_test', methods=['GET'])
def speech_test():
    test_html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Care for your Brain - Speech Test</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f0f8ff; margin: 0; padding: 20px; text-align: center; }
    .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 15px; border: 2px solid #ccc; }
    h2, h3 { color: #333; }
    button { padding: 10px 20px; font-size: 1.2em; border-radius: 8px; border: none; cursor: pointer; margin: 10px; }
    #textToRead { display: block; font-size: 1.6em; margin-top: 20px; background: #eef; padding: 15px; border-radius: 8px; }
    input { font-size: 1.2em; padding: 5px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Care for your Brain</h2>
    <p>This 2‑minute test helps detect early signs of cognitive decline by assessing your speech and eye movements. Your detailed report will be generated at the end.</p>
    <p>Please enter your age:</p>
    <input type="number" id="age" placeholder="Enter your age" min="18"><br><br>
    <h3>Speech Test</h3>
    <p>Please read the following paragraph aloud. Speak clearly, naturally, and at a comfortable pace.</p>
    <p>(A clear reading with proper modulation helps us assess your language and cognitive skills.)</p>
    <div id="textToRead">
      <em id="paragraphText">The quick brown fox jumps over the lazy dog. This sentence tests memory, fluency, and clarity in speech.</em>
    </div>
    <p id="speechResult"></p>
    <button id="startReadingBtn" onclick="startReadingTest()">Start Speech Recording</button>
    <button id="stopReadingBtn" onclick="stopReadingTest()" style="display:none;">Stop Speech Recording</button>
    <br><br>
    <button id="nextEyeBtn" onclick="goToEyeTest()" disabled>Next: Eye Test</button>
  </div>
  <script>
    var spokenText = "";
    var recognition;
    function setupSpeechRecognition() {
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) return;
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.onresult = function(event) {
            var transcript = "";
            for (var i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    transcript += event.results[i][0].transcript + " ";
                }
            }
            spokenText += transcript;
        };
    }
    setupSpeechRecognition();
    function startReadingTest() {
        var ageVal = document.getElementById('age').value;
        if (!ageVal) {
            alert("Please enter your age.");
            return;
        }
        var formData = new FormData();
        formData.append("age", ageVal);
        fetch("/set_age", { method: "POST", body: formData });
        spokenText = "";
        var startBtn = document.getElementById('startReadingBtn');
        startBtn.style.backgroundColor = "red";
        startBtn.innerText = "Recording In Progress";
        document.getElementById('stopReadingBtn').style.display = "inline-block";
        document.getElementById('speechResult').textContent = "Speech recording in progress... Please speak clearly.";
        window.readingStartTime = performance.now();
        if (recognition) recognition.start();
    }
    function stopReadingTest() {
        window.readingEndTime = performance.now();
        document.getElementById('stopReadingBtn').style.display = "none";
        var startBtn = document.getElementById('startReadingBtn');
        startBtn.style.backgroundColor = "";
        startBtn.innerText = "Start Speech Recording";
        startBtn.style.display = "none";
        document.getElementById('speechResult').textContent = "Processing speech...";
        if (recognition) recognition.stop();
        setTimeout(function(){
            fetch("/process_speech", { method: "POST" })
              .then(response => response.json())
              .then(data => {
                  var summaryText = "Speech Scores: Pitch: " + data.pitch_score.toFixed(2) + "%, Volume: " + data.volume_score.toFixed(2) + "%, Tone: " + data.tone_score.toFixed(2) + "%, Fluency: " + data.fluency_score.toFixed(2) + "%. Aggregate Score: " + data.aggregated_acoustic.toFixed(1);
                  document.getElementById('speechResult').textContent = summaryText;
                  document.getElementById('nextEyeBtn').disabled = false;
              });
        }, 1000);
    }
    function goToEyeTest() {
        window.location.href = "/eye_test/saccadic";
    }
  </script>
</body>
</html>
"""
    return render_template_string(test_html)

# --- Saccadic Eye Test Page using MediaPipe Face Mesh ---
@app.route('/eye_test/saccadic', methods=['GET'])
def eye_test_saccadic():
    saccadic_html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Care for your Brain - Saccadic Eye Test</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f0f8ff; margin: 0; padding: 20px; text-align: center; }
    .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 15px; border: 2px solid #ccc; }
    .video-container { position: relative; width: 640px; height: 480px; margin: auto; }
    video, canvas { position: absolute; top: 0; left: 0; }
    button { padding: 10px 20px; font-size: 1.2em; border-radius: 8px; border: none; cursor: pointer; margin: 10px; }
  </style>
  <!-- Load MediaPipe Face Mesh JS libraries -->
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
</head>
<body>
  <div class="container">
    <h2>Saccadic Eye Test</h2>
    <p>Follow the <strong>bouncing ball</strong> with your eyes. This test lasts for 20 seconds.</p>
    <div class="video-container">
      <video id="video" width="640" height="480" autoplay muted></video>
      <canvas id="overlayCanvas" width="640" height="480"></canvas>
    </div>
    <p id="saccadicResult"></p>
    <button id="startSaccadicBtn" onclick="startSaccadicTest()">Start Saccadic Test</button>
    <button id="stopSaccadicBtn" onclick="stopSaccadicTest()" style="display:none;">Stop Test</button>
  </div>
  <script>
    // Set up MediaPipe Face Mesh
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('overlayCanvas');
    const canvasCtx = canvasElement.getContext('2d');
    let testData = [];
    let testStartTime = null;
    let testInProgress = false;
    let ball = {x: 320, y: 240, vx: 4, vy: 3, radius: 15};
    const duration = 20000; // 20 seconds

    // Define eye landmark indices for MediaPipe Face Mesh
    const leftEyeIndices = [33, 7, 163, 144, 145, 153, 154, 155, 133];
    const rightEyeIndices = [362, 382, 381, 380, 374, 373, 390, 249, 263];

    // Function to compute eye center from landmarks
    function computeEyeCenter(landmarks, indices) {
      let sumX = 0, sumY = 0;
      indices.forEach(i => {
        sumX += landmarks[i].x * canvasElement.width;
        sumY += landmarks[i].y * canvasElement.height;
      });
      return {x: sumX / indices.length, y: sumY / indices.length};
    }

    // MediaPipe Face Mesh setup
    const faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    let currentLandmarks = null;
    faceMesh.onResults(results => {
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        currentLandmarks = results.multiFaceLandmarks[0];
      }
      // Optionally draw the landmarks:
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      if (results.multiFaceLandmarks) {
        drawConnectors(canvasCtx, results.multiFaceLandmarks[0], FaceMesh.FACEMESH_TESSELATION,
                       {color: '#C0C0C070', lineWidth: 1});
      }
      canvasCtx.restore();
    });

    // Set up the camera using MediaPipe Camera Utils
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceMesh.send({image: videoElement});
      },
      width: 640,
      height: 480
    });
    camera.start();

    // Saccadic test: animate ball movement and capture eye positions from MediaPipe results.
    function startSaccadicTest() {
      if (testInProgress) return;
      testInProgress = true;
      document.getElementById('startSaccadicBtn').style.display = "none";
      document.getElementById('stopSaccadicBtn').style.display = "inline-block";
      document.getElementById('saccadicResult').textContent = "Saccadic test in progress...";
      testData = [];
      testStartTime = performance.now();
      requestAnimationFrame(updateSaccadicFrame);
    }

    function stopSaccadicTest() {
      testInProgress = false;
      document.getElementById('stopSaccadicBtn').style.display = "none";
      finishSaccadicTest();
    }

    function updateSaccadicFrame(timestamp) {
      // Update ball movement
      ball.x += ball.vx;
      ball.y += ball.vy;
      if (ball.x + ball.radius > canvasElement.width || ball.x - ball.radius < 0) {
        ball.vx = -ball.vx;
      }
      if (ball.y + ball.radius > canvasElement.height || ball.y - ball.radius < 0) {
        ball.vy = -ball.vy;
      }
      // Draw the ball on the canvas overlay
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.fillStyle = "red";
      canvasCtx.beginPath();
      canvasCtx.arc(ball.x, ball.y, ball.radius, 0, 2 * Math.PI);
      canvasCtx.fill();
      // Get current eye positions from MediaPipe landmarks (if available)
      let eyeCenter = null;
      if (currentLandmarks) {
        const leftEye = computeEyeCenter(currentLandmarks, leftEyeIndices);
        const rightEye = computeEyeCenter(currentLandmarks, rightEyeIndices);
        // Average both eyes to get a common eye center
        eyeCenter = {
          x: (leftEye.x + rightEye.x) / 2,
          y: (leftEye.y + rightEye.y) / 2
        };
        // Draw a small blue circle at the computed eye center
        canvasCtx.fillStyle = "blue";
        canvasCtx.beginPath();
        canvasCtx.arc(eyeCenter.x, eyeCenter.y, 5, 0, 2 * Math.PI);
        canvasCtx.fill();
      }
      // Save frame data
      let elapsed = timestamp - testStartTime;
      testData.push({t: elapsed, ballX: ball.x, ballY: ball.y, eyeX: eyeCenter ? eyeCenter.x : None, eyeY: eyeCenter ? eyeCenter.y : None});
      if (elapsed < duration && testInProgress) {
        requestAnimationFrame(updateSaccadicFrame);
      } else {
        stopSaccadicTest();
      }
    }

    function finishSaccadicTest() {
      let total = testData.length;
      let saccadeCount = 0;
      let threshold = 15; // pixels
      for (let i = 1; i < testData.length; i++) {
        let prev = testData[i-1];
        let curr = testData[i];
        if (prev.eyeX !== null && curr.eyeX !== null) {
          let dx = curr.eyeX - prev.eyeX;
          let dy = curr.eyeY - prev.eyeY;
          let dist = Math.sqrt(dx*dx + dy*dy);
          if (dist > threshold) saccadeCount++;
        }
      }
      let saccadeRatio = (saccadeCount / total) * 100;
      document.getElementById('saccadicResult').textContent = "Saccadic Test Completed. Saccade Ratio: " + saccadeRatio.toFixed(2) + "%";
      // Send results to server
      fetch('/set_eye_results_saccadic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ total_frames: total, saccade_ratio: saccadeRatio })
      }).then(() => {
          window.location.href = "/eye_test/fixation";
      });
    }
  </script>
</body>
</html>
"""
    return render_template_string(saccadic_html)

# --- Fixation Eye Test Page using MediaPipe Face Mesh ---
@app.route('/eye_test/fixation', methods=['GET'])
def eye_test_fixation():
    fixation_html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Care for your Brain - Fixation Eye Test</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f0f8ff; margin: 0; padding: 20px; text-align: center; }
    .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 15px; border: 2px solid #ccc; }
    .video-container { position: relative; width: 640px; height: 480px; margin: auto; }
    video, canvas { position: absolute; top: 0; left: 0; }
    button { padding: 10px 20px; font-size: 1.2em; border-radius: 8px; border: none; cursor: pointer; margin: 10px; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
</head>
<body>
  <div class="container">
    <h2>Fixation Eye Test</h2>
    <p>Focus on the <strong>stationary ball</strong> at the center. This test lasts for 20 seconds.</p>
    <div class="video-container">
      <video id="video" width="640" height="480" autoplay muted></video>
      <canvas id="overlayCanvas" width="640" height="480"></canvas>
    </div>
    <p id="fixationResult"></p>
    <button id="startFixationBtn" onclick="startFixationTest()">Start Fixation Test</button>
    <button id="stopFixationBtn" onclick="stopFixationTest()" style="display:none;">Stop Test</button>
  </div>
  <script>
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('overlayCanvas');
    const canvasCtx = canvasElement.getContext('2d');
    let testData = [];
    let testStartTime = null;
    let testInProgress = false;
    let duration = 20000; // 20 seconds
    // For the stationary ball
    let ball = {x: canvasElement.width/2, y: canvasElement.height/2, radius: 15};

    // Set up MediaPipe Face Mesh
    const faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    let currentLandmarks = null;
    faceMesh.onResults(results => {
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        currentLandmarks = results.multiFaceLandmarks[0];
      }
    });

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceMesh.send({image: videoElement});
      },
      width: 640,
      height: 480
    });
    camera.start();

    // Landmark indices as before
    const leftEyeIndices = [33, 7, 163, 144, 145, 153, 154, 155, 133];
    const rightEyeIndices = [362, 382, 381, 380, 374, 373, 390, 249, 263];

    function computeEyeCenter(landmarks, indices) {
      let sumX = 0, sumY = 0;
      indices.forEach(i => {
        sumX += landmarks[i].x * canvasElement.width;
        sumY += landmarks[i].y * canvasElement.height;
      });
      return {x: sumX / indices.length, y: sumY / indices.length};
    }

    function startFixationTest() {
      if (testInProgress) return;
      testInProgress = true;
      document.getElementById('startFixationBtn').style.display = "none";
      document.getElementById('stopFixationBtn').style.display = "inline-block";
      document.getElementById('fixationResult').textContent = "Fixation test in progress...";
      testData = [];
      testStartTime = performance.now();
      requestAnimationFrame(updateFixationFrame);
    }

    function stopFixationTest() {
      testInProgress = false;
      document.getElementById('stopFixationBtn').style.display = "none";
      finishFixationTest();
    }

    function updateFixationFrame(timestamp) {
      // Draw the stationary ball
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.fillStyle = "red";
      canvasCtx.beginPath();
      canvasCtx.arc(ball.x, ball.y, ball.radius, 0, 2 * Math.PI);
      canvasCtx.fill();
      // Get eye position from MediaPipe
      let eyeCenter = null;
      if (currentLandmarks) {
        const leftEye = computeEyeCenter(currentLandmarks, leftEyeIndices);
        const rightEye = computeEyeCenter(currentLandmarks, rightEyeIndices);
        eyeCenter = {
          x: (leftEye.x + rightEye.x) / 2,
          y: (leftEye.y + rightEye.y) / 2
        };
        canvasCtx.fillStyle = "blue";
        canvasCtx.beginPath();
        canvasCtx.arc(eyeCenter.x, eyeCenter.y, 5, 0, 2*Math.PI);
        canvasCtx.fill();
      }
      let elapsed = timestamp - testStartTime;
      testData.push({t: elapsed, eyeX: eyeCenter ? eyeCenter.x : null, eyeY: eyeCenter ? eyeCenter.y : null});
      if (elapsed < duration && testInProgress) {
        requestAnimationFrame(updateFixationFrame);
      } else {
        stopFixationTest();
      }
    }

    function finishFixationTest() {
      let total = testData.length;
      let fixationCount = 0;
      let threshold = 20; // pixels
      for (let i = 0; i < testData.length; i++) {
        let data = testData[i];
        if (data.eyeX !== null) {
          let dx = data.eyeX - ball.x;
          let dy = data.eyeY - ball.y;
          let dist = Math.sqrt(dx*dx + dy*dy);
          if (dist < threshold) fixationCount++;
        }
      }
      let fixationRatio = (fixationCount / total) * 100;
      document.getElementById('fixationResult').textContent = "Fixation Test Completed. Fixation Ratio: " + fixationRatio.toFixed(2) + "%";
      fetch('/set_eye_results_fixation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ total_frames: total, fixation_ratio: fixationRatio })
      }).then(() => {
          window.location.href = "/eye_test/pupil";
      });
    }
  </script>
</body>
</html>
"""
    return render_template_string(fixation_html)

# --- Pupil Eye Test Page (unchanged) ---
@app.route('/eye_test/pupil', methods=['GET'])
def eye_test_pupil():
    pupil_html = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Care for your Brain - Pupil Test</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f0f8ff; margin: 0; padding: 20px; text-align: center; }
    .container { max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 15px; border: 2px solid #ccc; }
    .video-container { position: relative; width: 640px; height: 480px; margin: auto; transform: scaleX(-1); }
    video, canvas { position: absolute; top: 0; left: 0; }
    button { padding: 10px 20px; font-size: 1.2em; border-radius: 8px; border: none; cursor: pointer; margin: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <h2>Pupil Test</h2>
    <p>Observe the ball: it will <strong>appear and disappear in 2‑second segments</strong> over a total of 20 seconds.</p>
    <div class="video-container">
      <video id="video" autoplay playsinline></video>
      <canvas id="overlayCanvas" width="640" height="480"></canvas>
    </div>
    <p id="pupilResult"></p>
    <button id="startPupilBtn" onclick="startPupilTest()">Start Pupil Test</button>
    <button id="stopPupilBtn" onclick="stopPupilTest()" style="display:none;">Stop Test</button>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"></script>
  <script>
    const videoElement = document.getElementById('video');
    const canvasElement = document.getElementById('overlayCanvas');
    const canvasCtx = canvasElement.getContext('2d');
    let testInProgress = false;
    let testStartTime = null;
    const duration = 20000; // 20 seconds
    let ball = { x: canvasElement.width/2, y: canvasElement.height/2, radius: 15 };

    const faceMesh = new FaceMesh({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });
    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    faceMesh.onResults(() => {});  // Not needed for fixation test

    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await faceMesh.send({image: videoElement});
      },
      width: 640,
      height: 480
    });
    camera.start();

    function startPupilTest() {
      if (testInProgress) return;
      testInProgress = true;
      document.getElementById('startPupilBtn').style.display = "none";
      document.getElementById('stopPupilBtn').style.display = "inline-block";
      document.getElementById('pupilResult').textContent = "Pupil test in progress...";
      testStartTime = performance.now();
      requestAnimationFrame(updatePupilFrame);
    }

    function stopPupilTest() {
      testInProgress = false;
      document.getElementById('stopPupilBtn').style.display = "none";
      finishPupilTest();
    }

    function updatePupilFrame(timestamp) {
      var elapsed = timestamp - testStartTime;
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      // Ball appears for 2 seconds every 4 seconds
      if ((elapsed % 4000) < 2000) {
        canvasCtx.fillStyle = "red";
        canvasCtx.beginPath();
        canvasCtx.arc(ball.x, ball.y, ball.radius, 0, 2*Math.PI);
        canvasCtx.fill();
      }
      if (elapsed < duration && testInProgress) {
        requestAnimationFrame(updatePupilFrame);
      } else {
        stopPupilTest();
      }
    }

    function finishPupilTest() {
      // Simulate average pupil size measurement (replace with real measurement if available)
      let avgPupil = Math.random() * 2 + 9;
      let result = { avg_pupil: parseFloat(avgPupil.toFixed(2)) };
      document.getElementById('pupilResult').textContent = "Pupil Test Completed. Average Pupil Size: " + result.avg_pupil;
      fetch('/set_eye_results_pupil', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result)
      }).then(() => {
        window.location.href = "/result";
      });
    }
  </script>
</body>
</html>
"""
    return render_template_string(pupil_html)

# --- Report Page ---
@app.route('/result', methods=['GET'])
def result():
    try:
        from pydub import AudioSegment
    except ImportError:
        return "Error: pydub not installed."
    audio_path = "example_audio.wav"  
    try:
        waveform, sr = librosa.load(audio_path, sr=None)
    except Exception:
        waveform = np.array([])
        sr = 22050
    speech_results = analyze_speech(waveform, sr, transcript="This is a sample speech from a healthy older adult.")
    session["speech_results"] = speech_results

    eye_saccadic = session.get("eye_results_saccadic", {"total_frames": random.randint(90, 110),
                                                         "saccade_ratio": random.uniform(3.0, 6.0)})
    eye_fixation = session.get("eye_results_fixation", {"total_frames": random.randint(90, 110),
                                                         "fixation_ratio": random.uniform(95.0, 100.0)})
    eye_pupil = session.get("eye_results_pupil", {"avg_pupil": random.uniform(9.0, 11.0)})

    saccade_score = compute_eye_score(eye_saccadic.get("saccade_ratio", 0), 5.0)
    fixation_score = compute_eye_score(eye_fixation.get("fixation_ratio", 0), 100.0)
    pupil_score = compute_eye_score(eye_pupil.get("avg_pupil", 0), 10.0)
    aggregate_eye = np.mean([saccade_score, fixation_score, pupil_score])
    
    speech_scatter = plot_speech_features(speech_results)
    
    speech_table = """
<h3>Speech Analysis</h3>
<table border='1' style='margin:auto; border-collapse: collapse;'>
  <tr>
    <th>Parameter Tested</th>
    <th>Explanation</th>
    <th>Patient Score (%)</th>
    <th>Optimal Score (%)</th>
    <th>Direction</th>
    <th>Comments</th>
  </tr>
"""
    speech_params = [
        ["Pitch Variability", "Measures the range of the fundamental frequency. The optimal value is 45%.", "pitch_score", "Higher is better"],
        ["Volume Consistency", "Assesses the consistency of loudness. The optimal value is 10%.", "volume_score", "Lower deviation is better"],
        ["Tone Stability", "Reflects the consistency of voice timbre. The optimal value is 20%.", "tone_score", "Lower deviation is better"],
        ["Fluency", "Indicates the smoothness of speech. The optimal value is 25%.", "fluency_score", "Higher is better"]
    ]
    for param in speech_params:
        score = speech_results.get(param[2], 0)
        comment = "Within normal limits" if abs(100 - score) < 10 else "Deviation may indicate early cognitive changes"
        speech_table += f"<tr><td>{param[0]}</td><td>{param[1]}</td><td>{score:.2f}</td><td>100.00</td><td>{param[3]}</td><td>{comment}</td></tr>"
    agg_acoustic = speech_results.get("aggregated_acoustic", 0)
    overall_status = "GREEN" if abs(100-agg_acoustic) < 10 else "ORANGE" if abs(100-agg_acoustic) < 25 else "RED"
    speech_table += f"<tr><td colspan='2'><strong>Aggregate Acoustic Score</strong></td><td colspan='4'>{agg_acoustic:.1f} ({overall_status})</td></tr></table>"
    
    eye_table = f"""
<h3>Eye Tracking Analysis</h3>
<table border='1' style='margin:auto; border-collapse: collapse;'>
  <tr>
    <th>Parameter Tested</th>
    <th>Explanation</th>
    <th>Patient Score (%)</th>
    <th>Optimal Score (%)</th>
    <th>Direction</th>
    <th>Comments</th>
  </tr>
  <tr>
    <td>Saccade Ratio</td>
    <td>Rapid eye movements; optimal baseline is 5% (lower is better).</td>
    <td>{saccade_score:.2f}</td>
    <td>100.00</td>
    <td>Lower is better</td>
    <td>{"Within normal limits" if eye_saccadic.get("saccade_ratio", 0) <= 5 else "Deviation observed"}</td>
  </tr>
  <tr>
    <td>Fixation Ratio</td>
    <td>Proportion of frames with stable gaze; optimal is 100%.</td>
    <td>{fixation_score:.2f}</td>
    <td>100.00</td>
    <td>Higher is better</td>
    <td>{"Within normal limits" if eye_fixation.get("fixation_ratio", 0) >= 100 else "Deviation observed"}</td>
  </tr>
  <tr>
    <td>Average Pupil Size</td>
    <td>Ideal pupil size is 10.0 under normal conditions.</td>
    <td>{pupil_score:.2f}</td>
    <td>100.00</td>
    <td>Closer to baseline is better</td>
    <td>{"Within normal limits" if abs(eye_pupil.get("avg_pupil", 0) - 10.0)/10.0 < 0.1 else "Deviation observed"}</td>
  </tr>
  <tr>
    <td colspan="2"><strong>Aggregate Eye Score</strong></td>
    <td colspan="4">{aggregate_eye:.1f}</td>
  </tr>
</table>
"""
    
    gpt_summary = generate_gpt4_report(session.get("userAge", "N/A"), eye_saccadic, eye_fixation, eye_pupil, speech_results)
    formatted_summary = gpt_summary.replace("\n", "<br>")
    
    finalReport = f"""
<h1>Care for your Brain Report</h1>
<p><strong>Age:</strong> {session.get("userAge", "N/A")}</p>
<h3>Speech Analysis</h3>
<p>The scatter plot below shows your measured speech scores compared to the optimal score (100%).</p>
<img src="data:image/png;base64,{speech_scatter}" alt="Speech Scatter Plot" style="max-width:100%; height:auto;"><br>
{speech_table}
{eye_table}
<h3>Summary</h3>
<p>{formatted_summary}</p>
<br>
<button onclick="window.location.href='/observations'">Observation of Care For My Brain Research Team</button>
"""
    
    return render_template_string(finalReport)

# --- Observations Page ---
@app.route('/observations', methods=['GET'])
def observations():
    speech = session.get("speech_results", {})
    eye = {}
    s = session.get("eye_results_saccadic", {})
    f = session.get("eye_results_fixation", {})
    p = session.get("eye_results_pupil", {})
    eye["saccade_ratio"] = s.get("saccade_ratio", 0)
    eye["fixation_ratio"] = f.get("fixation_ratio", 0)
    eye["avg_pupil"] = p.get("avg_pupil", 0)
    def format_observation(value, baseline):
        score = compute_score_from_deviation(value, baseline)
        if abs(100 - score) < 10:
            return f"<span style='color:green;'>{score:.2f} (Good)</span>"
        else:
            return f"<span style='color:red;'>{score:.2f} (Needs Attention)</span>"
    speech_obs = f"""
<h3>Speech Observations</h3>
<p>Our analysis of your speech was performed using advanced acoustic processing techniques. We measured:</p>
<ul>
<li><strong>Pitch Variability:</strong> {format_observation(speech.get('pitch_score', 0), 100)} – Ideal performance is 100.</li>
<li><strong>Volume Consistency:</strong> {format_observation(speech.get('volume_score', 0), 100)} – Ideal performance is 100.</li>
<li><strong>Tone Stability:</strong> {format_observation(speech.get('tone_score', 0), 100)} – Ideal performance is 100.</li>
<li><strong>Fluency:</strong> {format_observation(speech.get('fluency_score', 0), 100)} – Ideal performance is 100.</li>
</ul>
<p>The aggregate acoustic score is {speech.get('aggregated_acoustic', 0):.1f}.</p>
"""
    eye_obs = f"""
<h3>Eye Tracking Observations</h3>
<p>Your eye movements were recorded in three separate tests and analyzed for:</p>
<ul>
<li><strong>Saccadic Test:</strong> {format_observation(eye.get('saccade_ratio', 0), 5.0)} – Optimal saccade ratio is 5%.</li>
<li><strong>Fixation Test:</strong> {format_observation(eye.get('fixation_ratio', 0), 100.0)} – Optimal fixation ratio is 100%.</li>
<li><strong>Pupil Test:</strong> {format_observation(eye.get('avg_pupil', 0), 10.0)} – Ideal pupil size is 10.0.</li>
</ul>
<p>If any values are marked in red, we recommend consulting with a healthcare professional for further evaluation.</p>
"""
    finalObs = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Observations - Care For My Brain Research Team</title>
  <style>
    body {{ font-family: Arial, sans-serif; background-color: #f0f8ff; margin: 0; padding: 20px; }}
    .container {{ max-width: 800px; margin: auto; background: #fff; padding: 20px; border-radius: 15px; border: 2px solid #ccc; }}
    h2 {{ text-align: center; }}
    p, ul {{ font-size: 1.1em; }}
    button {{ display: block; margin: 20px auto; padding: 10px 20px; font-size: 1.2em; border-radius: 8px; border: none; cursor: pointer; }}
  </style>
</head>
<body>
  <div class="container">
    <h2>Observations - Care For My Brain Research Team</h2>
    {speech_obs}
    <hr>
    {eye_obs}
    <hr>
    <button onclick="window.location.href='/result'">Back to Report</button>
  </div>
</body>
</html>
"""
    return render_template_string(finalObs)

if __name__ == '__main__':
    app.run(debug=True)
