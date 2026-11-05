"""
Patient Triage Web Dashboard & Live Inference Server.
Built for Track G: Semi-Supervised Learning in Healthcare ED Triage.
Author: VASANTHKUMAR (TFTPL444)
"""

import os
import sys
import json
import mimetypes
import urllib.parse
try:
    from http.server import ThreadingHTTPServer as HTTPServer, BaseHTTPRequestHandler
except ImportError:
    from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib
import numpy as np
import pandas as pd

# Ensure 'src' is in path for custom classes (e.g., HandwrittenCoTrainingClassifier)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

try:
    import co_training
    from feature_engineering import (
        VITALS_FEATURES,
        HISTORY_NUM_FEATURES,
        HISTORY_CAT_FEATURES,
        INV_LABEL_MAP
    )
except ImportError as e:
    print(f"Warning during import: {e}")

# Global models dictionary
MODELS = {}
PREPROCESSORS = {}

def load_resources():
    models_dir = os.path.join(BASE_DIR, "models")
    print(f"Loading preprocessors and models from {models_dir}...")
    
    PREPROCESSORS['vitals'] = joblib.load(os.path.join(models_dir, "vitals_preprocessor.joblib"))
    PREPROCESSORS['history'] = joblib.load(os.path.join(models_dir, "history_preprocessor.joblib"))
    
    MODELS['random_forest'] = joblib.load(os.path.join(models_dir, "random_forest.joblib"))
    MODELS['logistic_regression'] = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
    MODELS['co_training'] = joblib.load(os.path.join(models_dir, "co_training_model.joblib"))
    if os.path.exists(os.path.join(models_dir, "label_spreading.joblib")):
        MODELS['label_spreading'] = joblib.load(os.path.join(models_dir, "label_spreading.joblib"))
    
    print(f"Loaded {len(MODELS)} models successfully.")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Emergency Department AI Patient Triage Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .badge-immediate { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
    .badge-urgent { background-color: #ffedd5; color: #9a3412; border: 1px solid #fb923c; }
    .badge-standard { background-color: #fef9c3; color: #854d0e; border: 1px solid #facc15; }
    .badge-nonurgent { background-color: #dcfce7; color: #166534; border: 1px solid #4ade80; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen">
  <!-- Top Navigation -->
  <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
          <i class="fa-solid fa-heart-pulse text-white text-xl"></i>
        </div>
        <div>
          <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
            ED Patient Triage Assistant
            <span class="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded-full border border-indigo-500/30 font-medium">Semi-Supervised ML</span>
          </h1>
          <p class="text-xs text-slate-400">Track G: 4-Level Emergency Risk Stratification with ~10% Labelled Visits</p>
        </div>
      </div>
      <div class="flex items-center space-x-3 text-xs">
        <span class="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Engine Active
        </span>
        <span class="text-slate-400">ID: <strong class="text-slate-200">TFTPL444</strong></span>
      </div>
    </div>
  </header>

  <!-- Medical Safety Banner -->
  <div class="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 text-xs text-amber-200 text-center flex items-center justify-center gap-2">
    <i class="fa-solid fa-triangle-exclamation text-amber-400"></i>
    <span><strong>Research & Educational System:</strong> Predictions are generated from synthetic ED records and must not be used for autonomous clinical diagnosis or patient triage.</span>
  </div>

  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-8">
    
    <!-- Navigation Tabs -->
    <div class="flex border-b border-slate-800 space-x-4">
      <button onclick="switchTab('triage')" id="tab-triage" class="tab-btn px-4 py-2.5 font-semibold text-sm border-b-2 border-indigo-500 text-indigo-400 flex items-center gap-2">
        <i class="fa-solid fa-hospital-user"></i> Live Patient Triage
      </button>
      <button onclick="switchTab('metrics')" id="tab-metrics" class="tab-btn px-4 py-2.5 font-semibold text-sm border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-chart-line"></i> Model Benchmarks & Metrics
      </button>
      <button onclick="switchTab('charts')" id="tab-charts" class="tab-btn px-4 py-2.5 font-semibold text-sm border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-images"></i> 12 Research Visualizations
      </button>
      <button onclick="switchTab('about')" id="tab-about" class="tab-btn px-4 py-2.5 font-semibold text-sm border-b-2 border-transparent text-slate-400 hover:text-slate-200 flex items-center gap-2">
        <i class="fa-solid fa-circle-info"></i> Methodology & Insights
      </button>
    </div>

    <!-- TAB 1: LIVE PATIENT TRIAGE CALCULATOR -->
    <div id="view-triage" class="tab-content grid grid-cols-1 lg:grid-cols-12 gap-6">
      
      <!-- Input Form (7 cols) -->
      <div class="lg:col-span-7 bg-slate-950 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
        
        <div class="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div>
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <i class="fa-solid fa-clipboard-check text-indigo-400"></i> Patient Clinical Presentation
            </h2>
            <p class="text-xs text-slate-400">Enter physiological vitals (View 1) and patient history (View 2)</p>
          </div>
          <div class="flex items-center gap-2">
            <span class="text-xs text-slate-400">Presets:</span>
            <select id="preset-select" onchange="loadPreset(this.value)" class="text-xs bg-slate-900 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1.5 focus:ring-1 focus:ring-indigo-500">
              <option value="">Select Sample Case...</option>
              <option value="stemi">🔴 Acute STEMI / Shock (Immediate)</option>
              <option value="asthma">🟠 Severe Asthma Crisis (Urgent)</option>
              <option value="abdominal">🟡 Moderate Abdominal Pain (Standard)</option>
              <option value="ankle">🟢 Minor Ankle Strain (Non-Urgent)</option>
            </select>
          </div>
        </div>

        <form id="triage-form" onsubmit="runTriage(event)" class="space-y-6">
          
          <!-- View 1: Physiological Vitals -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-xs font-bold uppercase tracking-wider text-indigo-400 flex items-center gap-1.5">
                <i class="fa-solid fa-gauge-high"></i> View 1: Vital Signs (7 Features)
              </h3>
              <span class="text-[11px] text-slate-500">Objective Measurements</span>
            </div>
            
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
              
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Age (yrs)</label>
                <input type="number" id="age" required min="1" max="110" value="62" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Heart Rate (bpm)</label>
                <input type="number" id="heart_rate" required min="30" max="220" value="118" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Systolic BP</label>
                <input type="number" id="systolic_bp" required min="50" max="250" value="88" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Diastolic BP</label>
                <input type="number" id="diastolic_bp" required min="30" max="150" value="56" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">SpO2 (%)</label>
                <input type="number" id="spo2_pct" required min="50" max="100" value="89" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Resp Rate (/min)</label>
                <input type="number" id="resp_rate" required min="6" max="60" value="28" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div class="col-span-2">
                <label class="block text-xs font-medium text-slate-300 mb-1">Temperature (°C)</label>
                <input type="number" step="0.1" id="temperature_c" required min="30" max="44" value="38.2" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

            </div>
          </div>

          <!-- View 2: History & Presentation Symptoms -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-xs font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
                <i class="fa-solid fa-notes-medical"></i> View 2: Symptoms & History (8 Features)
              </h3>
              <span class="text-[11px] text-slate-500">Clinical Context</span>
            </div>

            <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
              
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Sex</label>
                <select id="sex" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
                  <option value="M">Male</option>
                  <option value="F">Female</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Arrival Mode</label>
                <select id="arrival_mode" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
                  <option value="Ambulance">Ambulance</option>
                  <option value="Walk-in">Walk-in</option>
                  <option value="Referred">Referred</option>
                  <option value="Police">Police</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Chest Pain</label>
                <select id="chest_pain" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Breathlessness</label>
                <select id="breathlessness" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Trauma Involved</label>
                <select id="trauma" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
                  <option value="No">No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Pain Score (0-10)</label>
                <input type="number" id="pain_score" required min="0" max="10" value="9" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Duration (hours)</label>
                <input type="number" step="0.5" id="symptom_duration_hrs" required min="0.1" max="168" value="1.5" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Comorbidity Count</label>
                <input type="number" id="comorbidity_count" required min="0" max="8" value="2" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none">
              </div>

            </div>
          </div>

          <!-- Model Choice & Run Button -->
          <div class="pt-2 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div class="w-full sm:w-auto flex items-center gap-2">
              <label class="text-xs text-slate-400 whitespace-nowrap">Model Engine:</label>
              <select id="model_name" class="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:ring-1 focus:ring-indigo-500">
                <option value="random_forest">Random Forest Classifier (Macro-F1: 0.917)</option>
                <option value="co_training">Handwritten Co-Training (Semi-Supervised, Macro-F1: 0.892)</option>
                <option value="logistic_regression">Multinomial Logistic Regression (Macro-F1: 0.873)</option>
              </select>
            </div>
            <button type="submit" id="btn-submit" class="w-full sm:w-auto px-6 py-2.5 bg-gradient-to-r from-rose-600 via-indigo-600 to-indigo-700 hover:from-rose-500 hover:to-indigo-600 text-white font-semibold text-sm rounded-xl shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all">
              <i class="fa-solid fa-play"></i> Evaluate Triage Category
            </button>
          </div>

        </form>

      </div>

      <!-- Prediction Output Panel (5 cols) -->
      <div class="lg:col-span-5 space-y-6">
        
        <div class="bg-slate-950 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden" id="result-card">
          <div class="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <i class="fa-solid fa-bell"></i> Stratification Result
            </h3>
            <span id="model-tag" class="text-[11px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">Model: Ready</span>
          </div>

          <!-- Empty state -->
          <div id="result-placeholder" class="py-12 text-center text-slate-500 space-y-2">
            <i class="fa-solid fa-stethoscope text-4xl text-slate-700"></i>
            <p class="text-sm">Click "Evaluate Triage Category" to generate instant classification.</p>
          </div>

          <!-- Populated state -->
          <div id="result-details" class="hidden space-y-5">
            
            <div class="text-center py-4 rounded-xl border" id="triage-badge-container">
              <span class="text-xs uppercase font-extrabold tracking-widest text-slate-400" id="triage-level-tag">Triage Category</span>
              <div class="text-3xl font-black mt-1" id="triage-level-name">Immediate</div>
              <p class="text-xs mt-1 text-slate-300" id="triage-level-action">Critical condition requiring immediate life-saving intervention.</p>
            </div>

            <!-- Probability Bars -->
            <div>
              <div class="flex justify-between items-center text-xs mb-2 text-slate-400">
                <span>Class Probability Distribution:</span>
                <span class="text-[10px] text-slate-500">Softmax Confidence</span>
              </div>
              
              <div class="space-y-2.5 text-xs">
                
                <div>
                  <div class="flex justify-between mb-1">
                    <span class="text-rose-400 font-semibold flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-rose-500"></span> 1 - Immediate</span>
                    <span id="prob-0" class="font-mono text-slate-300">0.0%</span>
                  </div>
                  <div class="w-full bg-slate-900 rounded-full h-2">
                    <div id="bar-0" class="bg-rose-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between mb-1">
                    <span class="text-amber-400 font-semibold flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-amber-500"></span> 2 - Urgent</span>
                    <span id="prob-1" class="font-mono text-slate-300">0.0%</span>
                  </div>
                  <div class="w-full bg-slate-900 rounded-full h-2">
                    <div id="bar-1" class="bg-amber-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between mb-1">
                    <span class="text-yellow-400 font-semibold flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-yellow-500"></span> 3 - Standard</span>
                    <span id="prob-2" class="font-mono text-slate-300">0.0%</span>
                  </div>
                  <div class="w-full bg-slate-900 rounded-full h-2">
                    <div id="bar-2" class="bg-yellow-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                  </div>
                </div>

                <div>
                  <div class="flex justify-between mb-1">
                    <span class="text-emerald-400 font-semibold flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-emerald-500"></span> 4 - Non-Urgent</span>
                    <span id="prob-3" class="font-mono text-slate-300">0.0%</span>
                  </div>
                  <div class="w-full bg-slate-900 rounded-full h-2">
                    <div id="bar-3" class="bg-emerald-500 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                  </div>
                </div>

              </div>
            </div>

            <!-- Vital Flags & Clinical Explanation -->
            <div class="border-t border-slate-800 pt-3">
              <h4 class="text-xs font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
                <i class="fa-solid fa-notes-medical text-indigo-400"></i> Clinical Flag Trigger Analysis:
              </h4>
              <div id="vital-flags" class="space-y-1.5 text-xs text-slate-400">
                <!-- Injected via JS -->
              </div>
            </div>

          </div>

        </div>

        <!-- Quick Info Card -->
        <div class="bg-slate-950/60 border border-slate-800/80 rounded-2xl p-5 text-xs text-slate-400 space-y-3">
          <div class="font-semibold text-slate-200 flex items-center gap-2">
            <i class="fa-solid fa-shield-halved text-emerald-400"></i> Safety Safeguard: Under-Triage Prevention
          </div>
          <p>Under-triage occurs when an Emergency Department system predicts an urgency level lower than actual clinical severity (e.g. classifying an <strong>Immediate</strong> patient as <strong>Urgent</strong> or <strong>Standard</strong>).</p>
          <div class="grid grid-cols-2 gap-2 pt-1 font-mono text-[11px]">
            <div class="bg-slate-900 p-2 rounded border border-slate-800">
              <span class="text-slate-500 block">Random Forest Under-Triage:</span>
              <strong class="text-emerald-400 text-sm">4.37%</strong>
            </div>
            <div class="bg-slate-900 p-2 rounded border border-slate-800">
              <span class="text-slate-500 block">Co-Training Under-Triage:</span>
              <strong class="text-indigo-400 text-sm">6.32%</strong>
            </div>
          </div>
        </div>

      </div>

    </div>

    <!-- TAB 2: MODEL BENCHMARKS & METRICS -->
    <div id="view-metrics" class="tab-content hidden space-y-6">
      
      <!-- Key Stat KPI cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Top Macro-F1 (Supervised)</div>
          <div class="text-2xl font-black text-emerald-400 mt-1">0.9173</div>
          <div class="text-xs text-slate-500 mt-1">Random Forest on ~10% labels</div>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Semi-Supervised Co-Training</div>
          <div class="text-2xl font-black text-indigo-400 mt-1">0.8920</div>
          <div class="text-xs text-slate-500 mt-1">Grew from 576 to 2,996 visits</div>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Immediate Recall (Safety)</div>
          <div class="text-2xl font-black text-rose-400 mt-1">91.45%</div>
          <div class="text-xs text-slate-500 mt-1">Misses < 9% of critical patients</div>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-4">
          <div class="text-xs text-slate-400">Lowest Under-Triage Rate</div>
          <div class="text-2xl font-black text-cyan-400 mt-1">4.37%</div>
          <div class="text-xs text-slate-500 mt-1">Exceeds the 10% safety target</div>
        </div>

      </div>

      <!-- Comparative Table -->
      <div class="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div class="p-5 border-b border-slate-800 flex justify-between items-center">
          <div>
            <h3 class="font-bold text-white text-base">Locked Test Set Performance Comparison (N = 1,440 visits)</h3>
            <p class="text-xs text-slate-400">Evaluated against clinical audit gold-standard labels (never seen in training)</p>
          </div>
          <span class="text-xs font-mono bg-slate-900 border border-slate-800 px-3 py-1 rounded-full text-slate-400">20% Stratified Split</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="bg-slate-900 text-slate-400 uppercase tracking-wider font-semibold">
              <tr>
                <th class="p-4">Model Paradigm</th>
                <th class="p-4">Accuracy</th>
                <th class="p-4">Macro-F1</th>
                <th class="p-4">Immediate Recall</th>
                <th class="p-4">Urgent Recall</th>
                <th class="p-4">Under-Triage Rate</th>
                <th class="p-4">Status</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800 font-mono" id="metrics-table-body">
              <tr class="hover:bg-slate-900/50">
                <td class="p-4 font-sans font-semibold text-slate-200">Multinomial Logistic Regression</td>
                <td class="p-4 text-slate-300">87.22%</td>
                <td class="p-4 text-indigo-400 font-bold">0.8729</td>
                <td class="p-4 text-slate-300">83.76%</td>
                <td class="p-4 text-slate-300">86.55%</td>
                <td class="p-4 text-amber-400">7.43%</td>
                <td class="p-4 font-sans"><span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Passed</span></td>
              </tr>
              <tr class="bg-indigo-950/20 border-l-4 border-indigo-500 hover:bg-indigo-950/40">
                <td class="p-4 font-sans font-bold text-white flex items-center gap-2">
                  <i class="fa-solid fa-crown text-amber-400"></i> Random Forest Classifier
                </td>
                <td class="p-4 text-emerald-300 font-bold">91.11%</td>
                <td class="p-4 text-emerald-400 font-extrabold text-sm">0.9173</td>
                <td class="p-4 text-rose-400 font-bold">91.45%</td>
                <td class="p-4 text-slate-300">94.16%</td>
                <td class="p-4 text-emerald-400 font-bold">4.37%</td>
                <td class="p-4 font-sans"><span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">Best Performer</span></td>
              </tr>
              <tr class="hover:bg-slate-900/50">
                <td class="p-4 font-sans font-semibold text-slate-200">Label Spreading (k-NN Graph)</td>
                <td class="p-4 text-slate-300">78.82%</td>
                <td class="p-4 text-slate-300">0.8161</td>
                <td class="p-4 text-slate-300">82.91%</td>
                <td class="p-4 text-slate-300">77.16%</td>
                <td class="p-4 text-rose-400 font-bold">20.56%</td>
                <td class="p-4 font-sans"><span class="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">Hubness issue</span></td>
              </tr>
              <tr class="hover:bg-slate-900/50">
                <td class="p-4 font-sans font-semibold text-indigo-300">Handwritten Co-Training</td>
                <td class="p-4 text-slate-300">89.38%</td>
                <td class="p-4 text-indigo-400 font-bold">0.8920</td>
                <td class="p-4 text-slate-300">85.47%</td>
                <td class="p-4 text-slate-300">92.13%</td>
                <td class="p-4 text-indigo-400 font-bold">6.32%</td>
                <td class="p-4 font-sans"><span class="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">Top Semi-Sup</span></td>
              </tr>
              <tr class="text-slate-500 hover:bg-slate-900/30">
                <td class="p-4 font-sans italic text-slate-400">All-Labelled Upper Bound (Oracle)</td>
                <td class="p-4">91.87%</td>
                <td class="p-4">0.9134</td>
                <td class="p-4">85.47%</td>
                <td class="p-4">92.64%</td>
                <td class="p-4">4.86%</td>
                <td class="p-4 font-sans"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-400">Theoretical</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- TAB 3: VISUALIZATIONS GALLERY -->
    <div id="view-charts" class="tab-content hidden space-y-6">
      
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-base font-bold text-white">Research Figure Gallery (300 DPI High-Resolution)</h2>
          <p class="text-xs text-slate-400">12 analytical charts generated during pipeline execution</p>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6" id="charts-gallery">
        <!-- Injected via JavaScript -->
      </div>

    </div>

    <!-- TAB 4: METHODOLOGY & INSIGHTS -->
    <div id="view-about" class="tab-content hidden space-y-6">
      
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div class="bg-slate-950 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-lightbulb text-amber-400"></i> 5 Core Research Insights
          </h3>
          <ul class="space-y-3 text-xs text-slate-300 leading-relaxed">
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">1</span>
              <div><strong class="text-white">Co-Training matches supervised performance with only 10% labels:</strong> Reached Macro-F1 of 0.8920 using multi-view consensus to accept 2,420 pseudo-labels at ~98% accuracy.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">2</span>
              <div><strong class="text-white">Label Spreading suffers from graph hubness:</strong> High-dimensional k-NN graphs propagate label errors across distant nodes, causing a high 20.56% under-triage rate.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">3</span>
              <div><strong class="text-white">Vitals view dominates history in acute discrimination:</strong> Physiological signals (SpO2, heart rate, BP) alone achieve Macro-F1 > 0.82, while history alone attains ~0.65.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">4</span>
              <div><strong class="text-white">Label budget elbow occurs between 2% and 10%:</strong> The steep gain happens when moving from 2% to 10% labels; beyond 20%, marginal returns flatten.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">5</span>
              <div><strong class="text-white">Random Forest provides maximum patient safety:</strong> 91.45% Immediate recall and only 4.37% under-triage makes RF the strongest candidate for clinical trial.</div>
            </li>
          </ul>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-2xl p-6 space-y-4">
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <i class="fa-solid fa-hospital text-emerald-400"></i> Clinical Deployment Recommendations
          </h3>
          <ul class="space-y-3 text-xs text-slate-300 leading-relaxed">
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">1</span>
              <div><strong class="text-white">Deploy Handwritten Co-Training in label-poor hospitals:</strong> Where nurse annotation bandwidth is limited, Co-Training with confidence threshold ≥ 0.82 enables rapid adaptation without manual labelling of the remaining 90%.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">2</span>
              <div><strong class="text-white">Apply safety override rules for vital red flags:</strong> Any patient with SpO2 < 90% or Systolic BP < 90 mmHg must be automatically elevated to Immediate regardless of model output.</div>
            </li>
            <li class="flex items-start gap-2">
              <span class="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px] shrink-0 mt-0.5">3</span>
              <div><strong class="text-white">Target 10% active-learning senior nurse review:</strong> Human-in-the-loop audit should prioritize cases where View A and View B models disagree, maximizing the information value of nurse audits.</div>
            </li>
          </ul>
        </div>

      </div>

    </div>

  </div>

  <script>
    const FIGURES = [
      { id: "01_label_scarcity_mix.png", title: "1. Label Scarcity & Triage Mix", desc: "10% Senior Nurse labels vs 90% unlabelled visits." },
      { id: "02_vitals_by_triage.png", title: "2. Physiological Vitals Distributions", desc: "Heart rate, SpO2, blood pressure and respiratory rate across triage tiers." },
      { id: "03_history_symptoms_by_triage.png", title: "3. History & Symptom Distributions", desc: "Pain scores, arrival modes, and presentation durations." },
      { id: "04_ed_arrivals_by_hour.png", title: "4. ED Arrivals by Hour", desc: "Diurnal arrival peaks vs urgency distribution." },
      { id: "05_feature_correlation_two_views.png", title: "5. Two-View Correlation Heatmap", desc: "Correlation structure between vitals and clinical history." },
      { id: "06_view_strength_comparison.png", title: "6. Single View Strength Comparison", desc: "Vitals alone vs History alone vs Combined views." },
      { id: "07_co_training_progress.png", title: "7. Co-Training Iteration Dynamics", desc: "Pseudo-labels accepted, consensus rates, and pseudo-accuracy across rounds." },
      { id: "08_model_comparison_test_set.png", title: "8. Test Set Performance Comparison", desc: "Macro-F1, Accuracy, and per-class recalls across all models." },
      { id: "09_confusion_matrix_selected_model.png", title: "9. Confusion Matrix (Random Forest)", desc: "Normalized confusion matrix on locked test set." },
      { id: "10_recall_and_undertriage_rate.png", title: "10. Recall & Under-Triage Safety Metrics", desc: "Clinical safety trade-offs between baselines and semi-supervised models." },
      { id: "11_label_budget_curves.png", title: "11. Label Budget Learning Curves", desc: "Macro-F1 and Immediate recall scaling from 2% to 40% label availability." },
      { id: "12_immediate_recall_by_presentation.png", title: "12. Immediate Recall by Patient Subgroup", desc: "Subgroup sensitivity across arrival modes and cardiac presentations." }
    ];

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => {
        b.classList.remove('border-indigo-500', 'text-indigo-400');
        b.classList.add('border-transparent', 'text-slate-400');
      });
      document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));

      document.getElementById('tab-' + name).classList.add('border-indigo-500', 'text-indigo-400');
      document.getElementById('tab-' + name).classList.remove('border-transparent', 'text-slate-400');
      document.getElementById('view-' + name).classList.remove('hidden');

      if (name === 'charts') loadCharts();
    }

    function loadCharts() {
      const container = document.getElementById('charts-gallery');
      if (container.children.length > 0) return;

      FIGURES.forEach(fig => {
        const card = document.createElement('div');
        card.className = "bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-lg hover:border-slate-700 transition flex flex-col";
        card.innerHTML = `
          <div class="bg-slate-900 border-b border-slate-800 p-2.5">
            <h4 class="text-xs font-bold text-white truncate">${fig.title}</h4>
            <p class="text-[11px] text-slate-400 truncate">${fig.desc}</p>
          </div>
          <div class="p-2 flex-1 flex items-center justify-center bg-black/30">
            <a href="/figures/${fig.id}" target="_blank">
              <img src="/figures/${fig.id}" alt="${fig.title}" class="rounded-lg object-contain max-h-56 hover:scale-[1.02] transition" loading="lazy">
            </a>
          </div>
          <div class="p-2.5 bg-slate-950/80 border-t border-slate-800/80 text-right">
            <a href="/figures/${fig.id}" target="_blank" class="text-[11px] text-indigo-400 hover:text-indigo-300 font-semibold inline-flex items-center gap-1">
              <i class="fa-solid fa-up-right-from-square"></i> Open High-Res (300 DPI)
            </a>
          </div>
        `;
        container.appendChild(card);
      });
    }

    const PRESETS = {
      stemi: {
        age: 68, heart_rate: 130, systolic_bp: 82, diastolic_bp: 52,
        spo2_pct: 87, resp_rate: 32, temperature_c: 38.4,
        sex: 'M', arrival_mode: 'Ambulance', chest_pain: 'Yes',
        breathlessness: 'Yes', trauma: 'No', pain_score: 10,
        symptom_duration_hrs: 1.0, comorbidity_count: 3
      },
      asthma: {
        age: 34, heart_rate: 110, systolic_bp: 130, diastolic_bp: 85,
        spo2_pct: 92, resp_rate: 26, temperature_c: 37.1,
        sex: 'F', arrival_mode: 'Walk-in', chest_pain: 'No',
        breathlessness: 'Yes', trauma: 'No', pain_score: 6,
        symptom_duration_hrs: 4.0, comorbidity_count: 1
      },
      abdominal: {
        age: 45, heart_rate: 86, systolic_bp: 122, diastolic_bp: 78,
        spo2_pct: 98, resp_rate: 18, temperature_c: 37.5,
        sex: 'F', arrival_mode: 'Walk-in', chest_pain: 'No',
        breathlessness: 'No', trauma: 'No', pain_score: 5,
        symptom_duration_hrs: 12.0, comorbidity_count: 0
      },
      ankle: {
        age: 24, heart_rate: 72, systolic_bp: 118, diastolic_bp: 75,
        spo2_pct: 99, resp_rate: 14, temperature_c: 36.6,
        sex: 'M', arrival_mode: 'Walk-in', chest_pain: 'No',
        breathlessness: 'No', trauma: 'Yes', pain_score: 3,
        symptom_duration_hrs: 24.0, comorbidity_count: 0
      }
    };

    function loadPreset(key) {
      if (!key || !PRESETS[key]) return;
      const data = PRESETS[key];
      Object.keys(data).forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = data[id];
      });
      runTriage(new Event('submit'));
    }

    async function runTriage(event) {
      if (event) event.preventDefault();
      
      const payload = {
        age: parseFloat(document.getElementById('age').value),
        heart_rate: parseFloat(document.getElementById('heart_rate').value),
        systolic_bp: parseFloat(document.getElementById('systolic_bp').value),
        diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value),
        spo2_pct: parseFloat(document.getElementById('spo2_pct').value),
        resp_rate: parseFloat(document.getElementById('resp_rate').value),
        temperature_c: parseFloat(document.getElementById('temperature_c').value),
        sex: document.getElementById('sex').value,
        arrival_mode: document.getElementById('arrival_mode').value,
        chest_pain: document.getElementById('chest_pain').value,
        breathlessness: document.getElementById('breathlessness').value,
        trauma: document.getElementById('trauma').value,
        pain_score: parseFloat(document.getElementById('pain_score').value),
        symptom_duration_hrs: parseFloat(document.getElementById('symptom_duration_hrs').value),
        comorbidity_count: parseInt(document.getElementById('comorbidity_count').value),
        model_name: document.getElementById('model_name').value
      };

      const btn = document.getElementById('btn-submit');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Processing...';

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        alert('Prediction error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Evaluate Triage Category';
      }
    }

    function renderResult(data) {
      document.getElementById('result-placeholder').classList.add('hidden');
      document.getElementById('result-details').classList.remove('hidden');

      document.getElementById('model-tag').textContent = 'Model: ' + data.model_display;

      const badge = document.getElementById('triage-badge-container');
      const name = document.getElementById('triage-level-name');
      const action = document.getElementById('triage-level-action');

      badge.className = "text-center py-4 rounded-xl border";
      
      const actions = {
        'Immediate': '🔴 Resuscitation priority: Immediate life-saving stabilization required.',
        'Urgent': '🟠 Emergent priority: Requires rapid clinical assessment within 15-30 minutes.',
        'Standard': '🟡 Urgent priority: Stable condition requiring assessment within 60 minutes.',
        'Non-Urgent': '🟢 Routine priority: Mild presentation, can wait safely for standard workup.'
      };

      const styles = {
        'Immediate': 'bg-rose-500/15 border-rose-500/40 text-rose-300',
        'Urgent': 'bg-amber-500/15 border-amber-500/40 text-amber-300',
        'Standard': 'bg-yellow-500/15 border-yellow-500/40 text-yellow-300',
        'Non-Urgent': 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
      };

      badge.classList.add(...styles[data.prediction].split(' '));
      name.textContent = data.prediction;
      action.textContent = actions[data.prediction] || '';

      // Probabilities
      for (let i = 0; i < 4; i++) {
        const pct = (data.probabilities[i] * 100).toFixed(1);
        document.getElementById(`prob-${i}`).textContent = pct + '%';
        document.getElementById(`bar-${i}`).style.width = pct + '%';
      }

      // Flags
      const flagsEl = document.getElementById('vital-flags');
      flagsEl.innerHTML = '';
      if (data.flags.length === 0) {
        flagsEl.innerHTML = '<div class="text-emerald-400"><i class="fa-solid fa-check"></i> All primary vitals within standard baseline range.</div>';
      } else {
        data.flags.forEach(f => {
          flagsEl.innerHTML += `<div class="text-rose-400 font-medium flex items-center gap-1.5"><i class="fa-solid fa-triangle-exclamation"></i> ${f}</div>`;
        });
      }
    }

    // Run sample on page load
    window.addEventListener('DOMContentLoaded', () => {
      runTriage();
    });
  </script>
</body>
</html>
"""

class TriageRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
            return

        elif path.startswith("/figures/"):
            filename = os.path.basename(path)
            fig_path = os.path.join(BASE_DIR, "outputs", "figures", filename)
            if os.path.exists(fig_path):
                self.send_response(200)
                mime, _ = mimetypes.guess_type(fig_path)
                self.send_header("Content-Type", mime or "image/png")
                self.end_headers()
                with open(fig_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Figure not found")
                return

        elif path == "/api/metrics":
            metrics_path = os.path.join(BASE_DIR, "outputs", "reports", "metrics.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(df.to_json(orient="records").encode('utf-8'))
                return

        elif path == "/api/summary":
            summary_path = os.path.join(BASE_DIR, "outputs", "reports", "summary.json")
            if os.path.exists(summary_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(summary_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        elif path in ["/download/presentation", "/presentation.pptx"]:
            pptx_path = os.path.join(BASE_DIR, "outputs", "reports", "Patient_Triage_AI_Presentation.pptx")
            if os.path.exists(pptx_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
                self.send_header("Content-Disposition", "attachment; filename=Patient_Triage_AI_Presentation.pptx")
                self.end_headers()
                with open(pptx_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                result = self.handle_prediction(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
            return

        self.send_error(404, "Not Found")

    def handle_prediction(self, data):
        model_name = data.get('model_name', 'random_forest')
        clf = MODELS.get(model_name, MODELS.get('random_forest'))

        # Prepare vitals dataframe
        vitals_df = pd.DataFrame([{
            'age': float(data['age']),
            'heart_rate': float(data['heart_rate']),
            'systolic_bp': float(data['systolic_bp']),
            'diastolic_bp': float(data['diastolic_bp']),
            'spo2_pct': float(data['spo2_pct']),
            'resp_rate': float(data['resp_rate']),
            'temperature_c': float(data['temperature_c'])
        }])

        # Prepare history dataframe
        history_df = pd.DataFrame([{
            'pain_score': float(data['pain_score']),
            'symptom_duration_hrs': float(data['symptom_duration_hrs']),
            'comorbidity_count': int(data['comorbidity_count']),
            'sex': str(data['sex']),
            'arrival_mode': str(data['arrival_mode']),
            'chest_pain': str(data['chest_pain']),
            'breathlessness': str(data['breathlessness']),
            'trauma': str(data['trauma'])
        }])

        vp = PREPROCESSORS['vitals']
        hp = PREPROCESSORS['history']

        X_v = vp.transform(vitals_df[VITALS_FEATURES])
        X_h = hp.transform(history_df[HISTORY_NUM_FEATURES + HISTORY_CAT_FEATURES])
        X_c = np.hstack([X_v, X_h])

        pred_class_idx = int(clf.predict(X_c)[0])
        pred_label = INV_LABEL_MAP[pred_class_idx]

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_c)[0].tolist()
        else:
            probs = [0.0, 0.0, 0.0, 0.0]
            probs[pred_class_idx] = 1.0

        # Clinical Red Flags
        flags = []
        if float(data['spo2_pct']) < 90:
            flags.append(f"Severe Hypoxemia: SpO2 {data['spo2_pct']}% (< 90%)")
        elif float(data['spo2_pct']) < 94:
            flags.append(f"Mild Hypoxemia: SpO2 {data['spo2_pct']}% (< 94%)")

        if float(data['systolic_bp']) < 90:
            flags.append(f"Hypotensive Shock Warning: Systolic BP {data['systolic_bp']} mmHg (< 90)")
        elif float(data['systolic_bp']) > 180:
            flags.append(f"Hypertensive Crisis: Systolic BP {data['systolic_bp']} mmHg (> 180)")

        if float(data['heart_rate']) > 120:
            flags.append(f"Marked Tachycardia: Heart Rate {data['heart_rate']} bpm (> 120)")
        elif float(data['heart_rate']) < 50:
            flags.append(f"Severe Bradycardia: Heart Rate {data['heart_rate']} bpm (< 50)")

        if float(data['resp_rate']) > 28:
            flags.append(f"Tachypnea / Respiratory Distress: {data['resp_rate']} breaths/min (> 28)")

        if data['chest_pain'] == 'Yes' and float(data['age']) >= 50:
            flags.append("High-Risk Acute Coronary Syndrome (ACS) presentation")

        model_labels = {
            'random_forest': 'Random Forest Classifier (Supervised)',
            'co_training': 'Handwritten Co-Training (Semi-Supervised)',
            'logistic_regression': 'Multinomial Logistic Regression'
        }

        return {
            'prediction': pred_label,
            'prediction_idx': pred_class_idx,
            'probabilities': probs,
            'flags': flags,
            'model_display': model_labels.get(model_name, model_name)
        }


def run_server(port=8080):
    load_resources()
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, TriageRequestHandler)
    print("=" * 65)
    print("Emergency Department AI Patient Triage Assistant Server")
    print(f"Server running at: http://0.0.0.0:{port}/")
    print("API Endpoints:")
    print(f"   - Web UI: http://localhost:{port}/")
    print(f"   - Predict: POST http://localhost:{port}/api/predict")
    print(f"   - Metrics: GET http://localhost:{port}/api/metrics")
    print(f"   - Figures: GET http://localhost:{port}/figures/<filename>")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
