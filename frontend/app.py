# frontend/app.py
"""
CoheRex-Integrity — Forensic Video Analysis Dashboard v2

Clean two-box UI → Upload → Analyze → Results + Synced Video Player
"""

import os
import sys
import json
import subprocess
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import streamlit.components.v1 as components

from coherex.detection.yolo_detector import YOLODetector
from coherex.tracking.manager import TrackManager
from coherex.trajectory.store import TrajectoryStore
from coherex.coherence.mcv import MotionCoherenceEngine
from coherex.coherence.interpreter import MotionInterpreter
from coherex.output.visualizer import Visualizer
from coherex.integrity.motion_agent import MotionAgent
from coherex.integrity.continuity_agent import ContinuityAgent
from coherex.integrity.crowd_agent import CrowdAgent
from coherex.integrity.fusion_engine import IntegrityFusionEngine
from coherex.integrity.segment_aggregator import SegmentIntegrityAggregator
from coherex.integrity.reliability import ReliabilityEstimator
from ultralytics import YOLO
from coherex.config import CONFIG
from coherex.meta.classifier import MetaClassifier
from coherex.meta.feature_extractor import build_feature_dict

# Singleton AI Classifier
_meta_clf = MetaClassifier()

# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoheRex-Integrity",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  .stApp {
    background: #0a0e17;
    color: #e6edf3;
    font-family: 'Inter', -apple-system, sans-serif;
  }
  [data-testid="stSidebar"] { display: none; }
  header[data-testid="stHeader"] { background: transparent; }

  /* Header */
  .main-header {
    text-align: center;
    padding: 24px 0 8px 0;
  }
  .main-header h1 {
    font-size: 1.8rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #21d4fd, #b721ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .main-header p { color: #7d8590; font-size: 0.82rem; margin: 4px 0 0 0; }

  /* Upload / Control boxes */
  .panel-box {
    background: #111827;
    border: 1.5px solid #1e293b;
    border-radius: 14px;
    padding: 32px 24px;
    min-height: 320px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    transition: border-color 0.25s;
  }
  .panel-box:hover { border-color: #21d4fd44; }
  .panel-box .box-title {
    font-size: 0.72rem; font-weight: 600; color: #7d8590;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 18px;
  }
  .panel-box .box-icon { font-size: 2.5rem; margin-bottom: 12px; }

  /* Metric cards */
  .metric-card {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 10px; padding: 14px 18px; text-align: center;
    transition: border-color .2s;
  }
  .metric-card:hover { border-color: #21d4fd55; }
  .metric-val { font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
  .metric-lbl {
    font-size: 0.7rem; color: #7d8590; text-transform: uppercase;
    letter-spacing: .08em;
  }

  /* Section header */
  .sec-hdr {
    font-size: 0.72rem; font-weight: 600; color: #7d8590;
    text-transform: uppercase; letter-spacing: .1em;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px; margin: 28px 0 14px 0;
  }

  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: #0a0e17; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
OUTPUT_DIR  = ROOT / "data" / "dashboard_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BAND_HIGH, BAND_MODERATE = CONFIG.integrity.verdict_thresholds
CLR_HIGH      = "#3fb950"
CLR_MOD       = "#e3b341"
CLR_COMP      = "#f85149"

def clr(s):
    if s >= BAND_HIGH: return CLR_HIGH
    if s >= BAND_MODERATE: return CLR_MOD
    return CLR_COMP

# ── Session state ──────────────────────────────────────────────────────────
for key, default in [
    ("phase", "upload"),      # upload | analyzing | results
    ("report", None),
    ("annotated_path", None),
    ("input_path", None),
    ("stop", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def convert_h264(src, dst):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ff, "-y", "-i", str(src),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def run_pipeline(video_path, window_size, progress_placeholder):
    detector         = YOLODetector()
    pose_model       = YOLO("yolov8n-pose.pt")
    tracker          = TrackManager()
    traj_store       = TrajectoryStore()
    mcv_engine       = MotionCoherenceEngine()
    interpreter_     = MotionInterpreter()
    visualizer       = Visualizer()
    motion_agent     = MotionAgent()
    continuity_agent = ContinuityAgent()
    # CrowdAgent is disabled per perfect execution instructions
    crowd_agent      = None 
    # Reliability is disabled for classification decision
    reliability_est  = None 
    fusion           = IntegrityFusionEngine(reliability_estimator=None)
    aggregator       = SegmentIntegrityAggregator(window_size=window_size)

    cap = cv2.VideoCapture(str(video_path))
    fps    = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_out = OUTPUT_DIR / "annotated_raw.mp4"
    writer  = cv2.VideoWriter(str(raw_out),
                              cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_scores, segment_scores = [], []
    motion_rels, cont_rels, crowd_rels = [], [], []
    
    # Hybrid feature accumulators
    all_motion_scores_vid = []
    all_continuity_scores_vid = []
    all_raw_mcvs = []
    all_ssim_scores = []
    all_flow_variances = []
    all_track_accelerations = []
    all_crowd_scores_vid = []
    frames_with_anomaly = 0
    current_burst = 0
    max_burst = 0
    prev_gray = None

    bar = progress_placeholder.progress(0, text="Starting pipeline...")
    fid = 0

    while True:
        if st.session_state.get("stop"):
            break
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        centers, boxes = [], []
        for d in dets:
            if d["class_id"] == CONFIG.detection.person_class_id:
                x1, y1, x2, y2 = d["bbox"]
                centers.append(((x1+x2)/2, (y1+y2)/2))
                boxes.append(d["bbox"])

        pose_res = pose_model(frame, verbose=False)
        pose_kp = {}
        for r in pose_res:
            if r.keypoints is None: continue
            for pi, kp in enumerate(r.keypoints.xy):
                pose_kp[pi] = kp.cpu().numpy()

        tracker.current_frame = frame
        tracker.current_pose = pose_kp
        tracks = tracker.update(detections=centers, bboxes=boxes, frame_id=fid)
        traj_store.update_from_tracks(tracks)

        ms, cs = [], []
        frame_has_physics_anomaly = False

        for t in tracks:
            state = traj_store.get_state(t.track_id)
            if state is None: continue
            
            mcv = mcv_engine.compute(state, t.confidence)
            raw_mcv_abs = abs(mcv)
            all_raw_mcvs.append(raw_mcv_abs)
            
            if raw_mcv_abs > 2.0:
                frame_has_physics_anomaly = True

            mst = interpreter_.interpret(mcv)
            if fid <= t.tamper_until_frame: mst = "INCONSISTENT"
            
            m_score = motion_agent.evaluate(state, t.confidence, track=t, current_frame=fid)
            c_score = continuity_agent.evaluate(t, fid)
            
            ms.append(m_score)
            cs.append(c_score)
            
            if len(state.accelerations) > 0:
                all_track_accelerations.append(abs(state.accelerations[-1]))

            bbox, md = None, float("inf")
            for bx in boxes:
                cx, cy = (bx[0]+bx[2])/2, (bx[1]+bx[3])/2
                dd = (t.x-cx)**2 + (t.y-cy)**2
                if dd < md: md, bbox = dd, bx
            if bbox:
                visualizer.draw(frame, t, bbox, mst,
                                motion_score=m_score,
                                continuity_score=c_score)

        if ms: all_motion_scores_vid.extend(ms)
        if cs: all_continuity_scores_vid.extend(cs)
        
        if frame_has_physics_anomaly:
            frames_with_anomaly += 1
            current_burst += 1
            max_burst = max(max_burst, current_burst)
        else:
            current_burst = 0

        # Optical flow variance + SSIM drift
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            all_flow_variances.append(float(np.var(flow_mag)))
            try:
                from skimage.metrics import structural_similarity as ssim_fn
                s, _ = ssim_fn(prev_gray, gray, full=True)
                all_ssim_scores.append(float(s))
            except Exception:
                pass
        prev_gray = gray

        crd = 1.0 # Crowd signal disabled
        all_crowd_scores_vid.append(crd)
        
        isc = fusion.fuse(ms, cs, crd, tracks=tracks, trajectory_store=traj_store)
        rm, rc, rcr = fusion.get_last_reliabilities()
        motion_rels.append(rm)
        cont_rels.append(rc)
        crowd_rels.append(rcr)

        aggregator.update(isc)
        frame_scores.append(isc)
        segment_scores.append(aggregator.get_segment_score())

        lbl = fusion.interpret(isc)
        _fw = (fusion.w_motion, fusion.w_continuity, fusion.w_crowd)
        visualizer.draw_frame_overlay(
            frame,
            frame_score=isc,
            segment_score=aggregator.get_segment_score(),
            verdict=lbl,
            fusion_weights=_fw,
            ai_result=None,   # AI result only available after full pass
        )

        writer.write(frame)
        fid += 1
        if total > 0:
            bar.progress(min(fid/total, 1.0),
                         text=f"Frame {fid}/{total} - Integrity: {isc:.2f}")

    cap.release()
    writer.release()
    bar.progress(1.0, text=f"Complete - {fid} frames processed")

    h264_out = OUTPUT_DIR / "annotated.mp4"
    try:
        convert_h264(raw_out, h264_out)
    except Exception:
        h264_out = raw_out

    snap = aggregator.get_snapshot()
    events = []
    in_ev, start = False, 0
    for i, s in enumerate(frame_scores):
        if s < BAND_MODERATE and not in_ev: in_ev, start = True, i
        elif s >= BAND_MODERATE and in_ev:
            in_ev = False
            events.append({"start_frame":start, "end_frame":i-1,
                           "start_time":round(start/fps,2), "end_time":round((i-1)/fps,2),
                           "min_score":round(min(frame_scores[start:i]),4)})
    if in_ev:
        events.append({"start_frame":start, "end_frame":len(frame_scores)-1,
                       "start_time":round(start/fps,2),
                       "end_time":round((len(frame_scores)-1)/fps,2),
                       "min_score":round(min(frame_scores[start:]),4)})

    # Score with Meta-Classifier
    try:
        from scipy.stats import kurtosis as scipy_kurtosis
        accel_kurt = float(scipy_kurtosis(all_track_accelerations)) if len(all_track_accelerations) > 4 else 0.0
    except Exception:
        accel_kurt = 0.0

    feat_dict = build_feature_dict(
        motion_mean        = sum(all_motion_scores_vid)/len(all_motion_scores_vid) if all_motion_scores_vid else 1.0,
        motion_min         = min(all_motion_scores_vid) if all_motion_scores_vid else 1.0,
        min_integrity      = min(frame_scores) if frame_scores else 1.0,
        volatility         = snap["volatility"],
        anomaly_density    = frames_with_anomaly / fid if fid > 0 else 0.0,
        perc_below_06      = sum(1 for s in frame_scores if s < 0.6) / len(frame_scores) if frame_scores else 0.0,
        max_mcv            = max(all_raw_mcvs) if all_raw_mcvs else 0.0,
    )
    ai_result = _meta_clf.score_dict(feat_dict)

    report = {
        "video":video_path.name, "fps":fps, "total_frames":len(frame_scores),
        "duration_sec":round(len(frame_scores)/fps,2), "window_size":window_size,
        "mean_score":snap["segment_score"], "min_score":snap["min_score"],
        "volatility":snap["volatility"], "verdict":fusion.interpret(snap["segment_score"]),
        "frame_scores":[round(s,4) for s in frame_scores],
        "segment_scores":[round(s,4) for s in segment_scores],
        "anomaly_events":events,
        "mean_reliability": {
            "motion": round(sum(motion_rels)/len(motion_rels),3) if motion_rels else 1.0,
            "continuity": round(sum(cont_rels)/len(cont_rels),3) if cont_rels else 1.0,
            "crowd": round(sum(crowd_rels)/len(crowd_rels),3) if crowd_rels else 1.0,
        },
        "ai_classifier": ai_result,
        "hybrid_features": {
            "flow_var_mean":    round(sum(all_flow_variances)/len(all_flow_variances), 4) if all_flow_variances else 0.0,
            "ssim_mean":        round(sum(all_ssim_scores)/len(all_ssim_scores), 4) if all_ssim_scores else 1.0,
            "accel_kurtosis":   round(accel_kurt, 4),
            "max_mcv":          round(max(all_raw_mcvs), 4) if all_raw_mcvs else 0.0,
            "burst_length":     max_burst,
            "anomaly_density":  round(frames_with_anomaly / fid, 4) if fid > 4 else 0.0,
            "motion_min":       round(min(all_motion_scores_vid), 4) if all_motion_scores_vid else 1.0,
            "volatility":       round(snap["volatility"], 4),
        },
        "config":CONFIG.to_dict(),
    }
    rp = REPORTS_DIR / f"{video_path.stem}_report.json"
    with open(rp,"w") as f: json.dump(report, f, indent=2)

    return str(h264_out), report


# ═══════════════════════════════════════════════════════════════════════════
# Charts
# ═══════════════════════════════════════════════════════════════════════════

def chart_integrity(rpt):
    fs, ss, fps = rpt["frame_scores"], rpt["segment_scores"], rpt["fps"]
    t = [i/fps for i in range(len(fs))]
    fig, ax = plt.subplots(figsize=(14,3.2), facecolor="#0a0e17")
    ax.set_facecolor("#0a0e17")
    ax.axhspan(BAND_HIGH,1.05, alpha=0.06, color=CLR_HIGH)
    ax.axhspan(BAND_MODERATE,BAND_HIGH, alpha=0.06, color=CLR_MOD)
    ax.axhspan(-0.05,BAND_MODERATE, alpha=0.06, color=CLR_COMP)
    ax.axhline(BAND_HIGH, color=CLR_HIGH, lw=0.7, alpha=0.4, ls="--")
    ax.axhline(BAND_MODERATE, color=CLR_MOD, lw=0.7, alpha=0.4, ls="--")
    ax.plot(t, fs, color="#21d4fd", lw=0.5, alpha=0.3, label="Frame")
    ax.plot(t, ss, color="#21d4fd", lw=1.8, alpha=0.9, label=f"Segment ({rpt['window_size']}fr)")
    for ev in rpt.get("anomaly_events",[]):
        ax.axvspan(ev["start_time"],ev["end_time"], color=CLR_COMP, alpha=0.12)
    ax.set_xlim(0, max(t) if t else 1); ax.set_ylim(0,1)
    ax.set_xlabel("Time (s)", color="#7d8590", fontsize=9)
    ax.set_ylabel("Integrity", color="#7d8590", fontsize=9)
    ax.tick_params(colors="#7d8590", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
    ax.legend(loc="lower left", fontsize=8, facecolor="#111827",
              edgecolor="#1e293b", labelcolor="#e6edf3")
    fig.tight_layout(pad=0.5)
    return fig

def chart_histogram(rpt):
    sc = rpt["frame_scores"]
    fig, ax = plt.subplots(figsize=(6,2.6), facecolor="#0a0e17")
    ax.set_facecolor("#0a0e17")
    bins = np.linspace(0,1,21)
    counts, edges = np.histogram(sc, bins=bins)
    cen = (edges[:-1]+edges[1:])/2
    cols = [CLR_HIGH if c>=BAND_HIGH else CLR_MOD if c>=BAND_MODERATE else CLR_COMP for c in cen]
    ax.bar(cen, counts, width=0.047, color=cols, alpha=0.8)
    ax.set_xlabel("Score", color="#7d8590", fontsize=9)
    ax.set_ylabel("Frames", color="#7d8590", fontsize=9)
    ax.tick_params(colors="#7d8590", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
    fig.tight_layout(pad=0.5)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Synced Player HTML (with zoom mirror)
# ═══════════════════════════════════════════════════════════════════════════

def build_synced_player(input_b64, annotated_b64):
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0e17; font-family:'Inter',-apple-system,sans-serif; color:#e6edf3; }}

  .player-container {{ width:100%; }}

  .video-row {{ display:flex; gap:10px; margin-bottom:12px; }}

  .vid-panel {{
    flex:1; position:relative; overflow:hidden;
    border:1.5px solid #1e293b; border-radius:10px;
    cursor:crosshair; background:#000;
  }}
  .vid-panel video {{
    width:100%; display:block;
    transform-origin: 0 0;
    transition: transform 0.1s ease-out;
  }}
  .vid-label {{
    position:absolute; top:8px; left:10px; z-index:10;
    font-size:0.65rem; font-weight:600; color:#e6edf3;
    background:rgba(10,14,23,0.75); padding:3px 10px;
    border-radius:6px; text-transform:uppercase; letter-spacing:0.08em;
  }}
  .zoom-badge {{
    position:absolute; bottom:8px; right:10px; z-index:10;
    font-size:0.62rem; color:#7d8590;
    background:rgba(10,14,23,0.75); padding:2px 8px;
    border-radius:4px; display:none;
  }}

  .controls {{
    display:flex; align-items:center; gap:10px;
    padding:10px 16px;
    background:#111827; border:1.5px solid #1e293b; border-radius:12px;
  }}

  .ctrl-btn {{
    width:36px; height:36px; border-radius:50%;
    border:none; cursor:pointer; display:flex;
    align-items:center; justify-content:center;
    transition: transform 0.12s, background 0.15s;
    flex-shrink:0;
  }}
  .ctrl-btn:hover {{ transform:scale(1.1); }}
  .ctrl-btn svg {{ fill:#fff; width:14px; height:14px; }}

  .play-btn {{ background:linear-gradient(135deg,#21d4fd,#b721ff); }}
  .skip-btn {{ background:#1e293b; }}
  .skip-btn:hover {{ background:#2d3748; }}

  .seek {{
    flex:1; height:5px; -webkit-appearance:none; appearance:none;
    background:#1e293b; border-radius:3px; outline:none; cursor:pointer;
  }}
  .seek::-webkit-slider-thumb {{
    -webkit-appearance:none; width:13px; height:13px;
    border-radius:50%; background:#21d4fd; cursor:pointer;
  }}

  .time-lbl {{ font-size:0.72rem; color:#7d8590; white-space:nowrap;
               font-variant-numeric:tabular-nums; flex-shrink:0; }}

  .speed-btn {{
    background:#1e293b; border:1px solid #2d3748; color:#e6edf3;
    border-radius:6px; padding:4px 10px; font-size:0.68rem;
    cursor:pointer; flex-shrink:0; transition:border-color 0.15s;
  }}
  .speed-btn:hover {{ border-color:#21d4fd; }}

  .zoom-btn {{
    background:#1e293b; border:1px solid #2d3748; color:#e6edf3;
    border-radius:6px; padding:4px 10px; font-size:0.68rem;
    cursor:pointer; flex-shrink:0;
  }}
  .zoom-btn:hover {{ border-color:#b721ff; }}
</style>
</head>
<body>
<div class="player-container">

  <div class="video-row">
    <div class="vid-panel" id="panel1">
      <div class="vid-label">📹 Input</div>
      <div class="zoom-badge" id="zoomBadge1">1.0x</div>
      <video id="v1" preload="auto" muted playsinline>
        <source src="data:video/mp4;base64,{input_b64}" type="video/mp4">
      </video>
    </div>
    <div class="vid-panel" id="panel2">
      <div class="vid-label">🔍 Annotated</div>
      <div class="zoom-badge" id="zoomBadge2">1.0x</div>
      <video id="v2" preload="auto" muted playsinline>
        <source src="data:video/mp4;base64,{annotated_b64}" type="video/mp4">
      </video>
    </div>
  </div>

  <div class="controls">
    <button class="ctrl-btn skip-btn" onclick="skip(-2)" title="Back 2s">
      <svg viewBox="0 0 24 24"><path d="M11 18V6l-8.5 6 8.5 6zm.5-6l8.5 6V6l-8.5 6z"/></svg>
    </button>

    <button class="ctrl-btn play-btn" id="playBtn" onclick="togglePlay()" title="Play/Pause">
      <svg id="pIcon" viewBox="0 0 24 24"><polygon points="6,3 20,12 6,21"/></svg>
    </button>

    <button class="ctrl-btn skip-btn" onclick="skip(2)" title="Forward 2s">
      <svg viewBox="0 0 24 24"><path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"/></svg>
    </button>

    <input type="range" class="seek" id="seekBar" min="0" max="10000" value="0"
           oninput="seekTo(this.value)">

    <span class="time-lbl" id="timeLbl">0:00 / 0:00</span>
    <button class="speed-btn" id="spdBtn" onclick="cycleSpeed()">1x</button>
    <button class="zoom-btn" id="zoomBtn" onclick="cycleZoom()">🔎 1x</button>
  </div>

</div>

<script>
  const v1 = document.getElementById('v1');
  const v2 = document.getElementById('v2');
  const pIcon = document.getElementById('pIcon');
  const seekBar = document.getElementById('seekBar');
  const timeLbl = document.getElementById('timeLbl');
  const spdBtn = document.getElementById('spdBtn');
  const zoomBtn = document.getElementById('zoomBtn');
  const panel1 = document.getElementById('panel1');
  const panel2 = document.getElementById('panel2');
  const zb1 = document.getElementById('zoomBadge1');
  const zb2 = document.getElementById('zoomBadge2');

  let playing = false;
  const speeds = [0.25, 0.5, 1, 1.5, 2];
  let spdIdx = 2;
  const zooms = [1, 1.5, 2, 3, 4];
  let zoomIdx = 0;
  let curZoom = 1;

  function fmt(t) {{
    if (isNaN(t)) return '0:00';
    const m = Math.floor(t/60);
    const s = Math.floor(t%60);
    return m + ':' + (s<10?'0':'') + s;
  }}

  async function togglePlay() {{
    if (playing) {{
      v1.pause();
      v2.pause();
      playing = false;
      pIcon.innerHTML = '<polygon points="6,3 20,12 6,21"/>';
    }} else {{
      v2.currentTime = v1.currentTime;
      try {{
        await Promise.all([v1.play(), v2.play()]);
        playing = true;
        pIcon.innerHTML = '<rect x="5" y="3" width="4" height="18"/><rect x="15" y="3" width="4" height="18"/>';
      }} catch(err) {{
        console.warn('Playback error:', err);
        playing = false;
      }}
    }}
  }}

  function skip(sec) {{
    const t = Math.max(0, Math.min(v1.duration || 0, v1.currentTime + sec));
    v1.currentTime = t;
    v2.currentTime = t;
  }}

  function seekTo(val) {{
    const dur = v1.duration || 1;
    const t = (val / 10000) * dur;
    v1.currentTime = t;
    v2.currentTime = t;
  }}

  function cycleSpeed() {{
    spdIdx = (spdIdx + 1) % speeds.length;
    v1.playbackRate = speeds[spdIdx];
    v2.playbackRate = speeds[spdIdx];
    spdBtn.textContent = speeds[spdIdx] + 'x';
  }}

  function cycleZoom() {{
    zoomIdx = (zoomIdx + 1) % zooms.length;
    curZoom = zooms[zoomIdx];
    zoomBtn.textContent = '🔎 ' + curZoom + 'x';
    if (curZoom === 1) {{
      applyZoom(v1, 0, 0);
      applyZoom(v2, 0, 0);
      zb1.style.display = 'none';
      zb2.style.display = 'none';
    }} else {{
      zb1.style.display = 'block'; zb1.textContent = curZoom + 'x';
      zb2.style.display = 'block'; zb2.textContent = curZoom + 'x';
    }}
  }}

  function applyZoom(vid, px, py) {{
    if (curZoom === 1) {{
      vid.style.transform = 'scale(1)';
      vid.style.transformOrigin = '0 0';
    }} else {{
      vid.style.transformOrigin = px + '% ' + py + '%';
      vid.style.transform = 'scale(' + curZoom + ')';
    }}
  }}

  // Mouse-driven zoom pan (synced)
  function handleMouseMove(e, panel, vid, otherVid) {{
    if (curZoom <= 1) return;
    const rect = panel.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * 100;
    const py = ((e.clientY - rect.top) / rect.height) * 100;
    applyZoom(vid, px, py);
    applyZoom(otherVid, px, py);
  }}

  panel1.addEventListener('mousemove', (e) => handleMouseMove(e, panel1, v1, v2));
  panel2.addEventListener('mousemove', (e) => handleMouseMove(e, panel2, v2, v1));

  // Reset zoom on mouse leave
  panel1.addEventListener('mouseleave', () => {{ if(curZoom>1){{ applyZoom(v1,50,50); applyZoom(v2,50,50); }} }});
  panel2.addEventListener('mouseleave', () => {{ if(curZoom>1){{ applyZoom(v1,50,50); applyZoom(v2,50,50); }} }});

  // Time update
  v1.addEventListener('timeupdate', () => {{
    const dur = v1.duration || 1;
    seekBar.value = (v1.currentTime / dur) * 10000;
    timeLbl.textContent = fmt(v1.currentTime) + ' / ' + fmt(dur);
  }});

  v1.addEventListener('ended', () => {{
    playing = false;
    pIcon.innerHTML = '<polygon points="6,3 20,12 6,21"/>';
  }});

  // Drift correction
  setInterval(() => {{
    if (playing && Math.abs(v1.currentTime - v2.currentTime) > 0.12) {{
      v2.currentTime = v1.currentTime;
    }}
  }}, 400);

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {{
    if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
    if (e.code === 'ArrowLeft') skip(-2);
    if (e.code === 'ArrowRight') skip(2);
  }});
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ═══════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔬 CoheRex-Integrity</h1>
  <p>Multi-Agent Temporal Consistency Verification Framework</p>
</div>
""", unsafe_allow_html=True)

phase = st.session_state["phase"]

# ═══════════════════════════════════════════════════════════════════════════
# PHASE: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════

if phase == "upload":
    left, right = st.columns(2, gap="medium")

    with left:
        inp_path_existing = st.session_state.get("input_path")

        if inp_path_existing and Path(inp_path_existing).exists():
            # ── Video loaded: show it at the top of the panel ─────────
            st.video(inp_path_existing)
            fname = Path(inp_path_existing).name.replace("input_", "", 1)
            st.success(f"✅ **{fname}**  — ready to analyze")
            # Small re-upload option
            with st.expander("🔄 Replace video"):
                new_upload = st.file_uploader(
                    "Select a new video file",
                    type=["mp4", "avi", "mov", "mkv"],
                    label_visibility="collapsed",
                )
                if new_upload:
                    inp = OUTPUT_DIR / f"input_{new_upload.name}"
                    with open(inp, "wb") as f:
                        f.write(new_upload.getbuffer())
                    st.session_state["input_path"] = str(inp)
                    st.rerun()
        else:
            # ── Nothing loaded yet: show icon placeholder + uploader ───
            st.markdown("""
            <div class="panel-box">
              <div class="box-icon">📁</div>
              <div class="box-title">Input Video</div>
            </div>
            """, unsafe_allow_html=True)

            uploaded = st.file_uploader(
                "Select a video file",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed",
            )

            if uploaded:
                inp = OUTPUT_DIR / f"input_{uploaded.name}"
                with open(inp, "wb") as f:
                    f.write(uploaded.getbuffer())
                st.session_state["input_path"] = str(inp)
                st.rerun()  # triggers re-render showing video at top


    with right:
        st.markdown("""
        <div class="panel-box">
          <div class="box-icon">⚙️</div>
          <div class="box-title">Analysis Settings</div>
        </div>
        """, unsafe_allow_html=True)

        window = st.slider("Segment window (frames)", 30, 600,
                           CONFIG.integrity.segment_window_frames, 30,
                           help=f"{CONFIG.integrity.segment_window_frames} ≈ 5 sec @ 30 FPS")

        mode = st.radio("Analysis mode",
                        ["Full Pipeline (YOLO + Agents)", "Lightweight (Optical Flow)"],
                        index=0, horizontal=True)

        st.markdown("---")

        if st.button("▶  ANALYZE", type="primary", width="stretch"):
            if st.session_state.get("input_path") is None:
                st.error("⚠ Upload a video first!")
            else:
                st.session_state["phase"] = "analyzing"
                st.session_state["window"] = window
                st.session_state["stop"] = False
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE: ANALYZING
# ═══════════════════════════════════════════════════════════════════════════

elif phase == "analyzing":
    inp = st.session_state["input_path"]

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown("**📹 Input Video**")
        with open(inp, "rb") as f:
            st.video(f.read())

    with right:
        st.markdown("**🔍 Processing…**")
        progress_area = st.empty()

        if st.button("⏹  STOP", type="secondary", width="stretch"):
            st.session_state["stop"] = True

    # Run pipeline
    ann_path, report = run_pipeline(
        Path(inp),
        st.session_state.get("window", 150),
        progress_area if 'progress_area' in dir() else st.empty(),
    )

    st.session_state["annotated_path"] = ann_path
    st.session_state["report"] = report
    st.session_state["phase"] = "results"
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PHASE: RESULTS
# ═══════════════════════════════════════════════════════════════════════════

elif phase == "results":
    report = st.session_state["report"]
    ann_path = st.session_state["annotated_path"]
    inp_path = st.session_state["input_path"]

    if report is None:
        st.session_state["phase"] = "upload"
        st.rerun()

    # ── Metrics row ────────────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Forensic Metrics</p>', unsafe_allow_html=True)

    verdict  = report["verdict"]
    mean_s   = report["mean_score"]
    min_s    = report["min_score"]
    vol      = report["volatility"]
    n_ev     = len(report.get("anomaly_events",[]))
    dur      = report["duration_sec"]
    vc       = clr(mean_s)

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Verdict</div>'
                    f'<div class="metric-val" style="font-size:1.1rem;color:{vc}">{verdict}</div></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Mean</div>'
                    f'<div class="metric-val" style="color:{vc}">{mean_s:.3f}</div></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Min</div>'
                    f'<div class="metric-val" style="color:{clr(min_s)}">{min_s:.3f}</div></div>',
                    unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Volatility</div>'
                    f'<div class="metric-val" style="color:{CLR_COMP if vol>0.15 else CLR_MOD if vol>0.05 else CLR_HIGH}">{vol:.3f}</div></div>',
                    unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Events</div>'
                    f'<div class="metric-val" style="color:{CLR_COMP if n_ev>0 else CLR_HIGH}">{n_ev}</div></div>',
                    unsafe_allow_html=True)
    with k6:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Duration</div>'
                    f'<div class="metric-val" style="color:#e6edf3;font-size:1.3rem">{dur:.1f}s</div></div>',
                    unsafe_allow_html=True)

    # ── Reliability row ────────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Agent Reliability Estimation (Dynamic Weights)</p>', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    rel = report.get("mean_reliability", {"motion":1.0, "continuity":1.0, "crowd":1.0})
    
    with r1:
        rm = rel["motion"]
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Motion Reliab.</div>'
                    f'<div class="metric-val" style="color:{CLR_HIGH if rm>0.8 else CLR_MOD if rm>0.5 else CLR_COMP}">{rm:.2f}</div></div>',
                    unsafe_allow_html=True)
    with r2:
        rc = rel["continuity"]
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Continuity Reliab.</div>'
                    f'<div class="metric-val" style="color:{CLR_HIGH if rc>0.8 else CLR_MOD if rc>0.5 else CLR_COMP}">{rc:.2f}</div></div>',
                    unsafe_allow_html=True)
    with r3:
        rw = rel["crowd"]
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Crowd Reliab.</div>'
                    f'<div class="metric-val" style="color:{CLR_HIGH if rw>0.8 else CLR_MOD if rw>0.5 else CLR_COMP}">{rw:.2f}</div></div>',
                    unsafe_allow_html=True)
    with r4:
        # Calc overall avg reliability
        ravg = (rel["motion"] + rel["continuity"] + rel["crowd"]) / 3
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">System Trust</div>'
                    f'<div class="metric-val" style="color:{CLR_HIGH if ravg>0.8 else CLR_MOD if ravg>0.5 else CLR_COMP}">{ravg:.2f}</div></div>',
                    unsafe_allow_html=True)


    # -- AI Meta-Classifier Panel ----------------------------------------------
    st.markdown('<p class="sec-hdr">&#129302; AI Decision Confidence (Random Forest Meta-Classifier)</p>',
                unsafe_allow_html=True)

    ai = report.get("ai_classifier", {})
    hf = report.get("hybrid_features", {})

    if ai and ai.get("available"):
        conf       = ai["confidence"]
        ai_verdict = ai["verdict"]

        verdict_color = {
            "AUTHENTIC":  CLR_HIGH,
            "SUSPICIOUS": CLR_MOD,
            "TAMPERED":   CLR_COMP,
        }.get(ai_verdict, "#e6edf3")

        ai1, ai2, ai3 = st.columns(3)

        with ai1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-lbl">AI Verdict</div>'
                f'<div class="metric-val" style="font-size:1.1rem;color:{verdict_color}">'
                f'{ai_verdict}</div></div>',
                unsafe_allow_html=True)

        with ai2:
            bar_w   = int(conf * 100)
            bar_clr = CLR_COMP if conf >= 0.65 else CLR_MOD if conf >= 0.40 else CLR_HIGH
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-lbl">Tamper Probability</div>'
                f'<div class="metric-val" style="color:{bar_clr}">{conf:.1%}</div>'
                f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:8px;">'
                f'<div style="background:{bar_clr};width:{bar_w}%;height:6px;border-radius:4px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True)

        with ai3:
            phys_s = report["mean_score"]
            agree  = (conf >= 0.5 and phys_s < 0.7) or (conf < 0.5 and phys_s >= 0.7)
            agree_txt   = "Agreement" if agree else "Divergent"
            agree_icon  = "&#9989;"   if agree else "&#9888;"
            agree_color = CLR_HIGH    if agree else CLR_MOD
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-lbl">Physics vs AI</div>'
                f'<div class="metric-val" style="font-size:1rem;color:{agree_color}">'
                f'{agree_icon} {agree_txt}</div>'
                f'<div style="font-size:0.68rem;color:#7d8590;margin-top:4px;">'
                f'AI {conf:.2f} | Physics {phys_s:.2f}</div></div>',
                unsafe_allow_html=True)

        if hf:
            _rows = "".join([
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:3px 0;border-bottom:1px solid #1e293b32;">'
                f'<span style="color:#7d8590;font-size:0.76rem;">{_k}</span>'
                f'<span style="color:#e6edf3;font-size:0.76rem;font-weight:600;">{_v}</span></div>'
                for _k, _v in [
                    ("Optical Flow Variance",  f'{hf.get("flow_var_mean", 0):.4f}'),
                    ("SSIM Drift (mean)",       f'{hf.get("ssim_mean", 1.0):.4f}'),
                    ("Accel. Kurtosis",         f'{hf.get("accel_kurtosis", 0):.4f}'),
                    ("Max MCV",                 f'{hf.get("max_mcv", 0):.4f}'),
                    ("Burst Length (frames)",   str(hf.get("burst_length", 0))),
                    ("Anomaly Density",         f'{hf.get("anomaly_density", 0):.4f}'),
                    ("Motion Min (agent)",      f'{hf.get("motion_min", 1.0):.4f}'),
                    ("Volatility (sigma)",      f'{hf.get("volatility", 0):.4f}'),
                ]
            ])
            st.markdown(
                f'<div style="margin-top:12px;padding:14px 18px;background:#111827;'
                f'border:1px solid #1e293b;border-radius:10px;">'
                f'<div style="font-size:0.68rem;color:#7d8590;text-transform:uppercase;'
                f'letter-spacing:.1em;margin-bottom:10px;">Hybrid Feature Signal</div>'
                f'{_rows}</div>',
                unsafe_allow_html=True)
    else:
        st.info(
            "**AI classifier not trained yet** -- physics score is your forensic signal.\n\n"
            "To enable AI Decision Confidence:\n\n"
            "```\n"
            "python scripts/evaluate_dataset.py\n"
            "python scripts/train_meta_classifier.py\n"
            "```\n\n"
            "Restart the dashboard after training."
        )

    # ── Synced Video Player ────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Video Comparison — Synchronized Player</p>',
                unsafe_allow_html=True)

    has_inp = inp_path and Path(inp_path).exists()
    has_ann = ann_path and Path(ann_path).exists()

    if has_inp and has_ann:
        with open(inp_path, "rb") as f:
            ib64 = base64.b64encode(f.read()).decode()
        with open(ann_path, "rb") as f:
            ann_bytes = f.read()
            ab64 = base64.b64encode(ann_bytes).decode()

        components.html(build_synced_player(ib64, ab64), height=500, scrolling=False)

        st.download_button("⬇ Download Annotated Video", data=ann_bytes,
                           file_name="coherex_annotated.mp4", mime="video/mp4", width="stretch")
    else:
        st.warning("Video files not available.")

    # ── Integrity Graph ────────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Temporal Integrity Graph</p>', unsafe_allow_html=True)
    st.pyplot(chart_integrity(report))

    # ── Histogram + breakdown ──────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Score Distribution</p>', unsafe_allow_html=True)
    hc, sc = st.columns([2,1])
    with hc:
        st.pyplot(chart_histogram(report))
    with sc:
        total_count = report.get("total_frames", 0)
        total = max(1, total_count)
        hp = sum(1 for s in report["frame_scores"] if s>=BAND_HIGH)/total*100
        mp = sum(1 for s in report["frame_scores"] if BAND_MODERATE<=s<BAND_HIGH)/total*100
        cp = sum(1 for s in report["frame_scores"] if s<BAND_MODERATE)/total*100
        st.markdown(f"""
**Frame Breakdown**

| Band | Frames | % |
|:---|---:|---:|
| 🟢 High | {int(hp*total/100)} | {hp:.1f}% |
| 🟡 Moderate | {int(mp*total/100)} | {mp:.1f}% |
| 🔴 Compromised | {int(cp*total/100)} | {cp:.1f}% |

**Total:** {total} frames &nbsp;|&nbsp; **FPS:** {report['fps']}
""")

    # ── Anomaly Event Log ──────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Anomaly Event Log</p>', unsafe_allow_html=True)
    events = report.get("anomaly_events",[])
    if events:
        for i, ev in enumerate(events, 1):
            sev = "🔴 CRITICAL" if ev["min_score"]<0.4 else "🟡 WARNING"
            with st.expander(
                f"Event {i}  |  {ev['start_time']:.1f}s → {ev['end_time']:.1f}s  "
                f"|  min={ev['min_score']:.3f}  |  {sev}",
                expanded=(i==1),
            ):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Start", f"{ev['start_time']:.2f}s")
                c2.metric("End", f"{ev['end_time']:.2f}s")
                c3.metric("Duration", f"{ev['end_time']-ev['start_time']:.2f}s")
                c4.metric("Min Score", f"{ev['min_score']:.3f}")
    else:
        st.success("✅ No anomaly events detected — video integrity is clean.")

    # ── Export ─────────────────────────────────────────────────────────
    st.markdown('<p class="sec-hdr">Export</p>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇ JSON Report", data=json.dumps(report,indent=2),
                           file_name=f"{report['video'].split('.')[0]}_report.json",
                           mime="application/json", width="stretch")
    with e2:
        csv = ["frame,time_sec,frame_score,segment_score"]
        for i,(fs,ss) in enumerate(zip(report["frame_scores"],report["segment_scores"])):
            csv.append(f"{i},{i/report['fps']:.3f},{fs},{ss}")
        st.download_button("⬇ Scores CSV", data="\n".join(csv),
                           file_name=f"{report['video'].split('.')[0]}_scores.csv",
                           mime="text/csv", width="stretch")

    # ── New analysis button ────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄  New Analysis", width="stretch"):
        st.session_state["phase"] = "upload"
        st.session_state["report"] = None
        st.session_state["annotated_path"] = None
        st.session_state["input_path"] = None
        st.rerun()
