<h1 align="center">AfterShoot ML Challenge</h1>

<p align="center">
  <strong>Automatic Temperature & Tint Prediction from Images + Metadata</strong><br>
  Submission-ready Machine Learning Pipeline
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-CNN%20%2B%20Metadata-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Language-Python-green?style=flat-square" />
</p>

<hr>

<h2>📌 Overview</h2>
<p>
This project was developed for the <strong>AfterShoot Machine Learning Challenge</strong>.
The goal is to predict White Balance <strong>Temperature</strong> and <strong>Tint</strong> slider values applied to RAW images.
</p>
<ul>
  <li>Input: TIFF Image + Metadata (camera, lens, WB info)</li>
  <li>Output: Rounded Temperature & Tint predictions (integer)</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
Aftershoot-ML-Challenge/
├── data/
│   ├── Train/
│   └── Validation/
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── model_def.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>🧠 Model Architecture</h2>
<ul>
  <li>EfficientNetB0 CNN for visual features</li>
  <li>Metadata encoded via StandardScaler + OneHotEncoder</li>
  <li>Multi-input fusion for final Temperature & Tint prediction</li>
</ul>

<p align="center">
  <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="110"/>
</p>

<hr>

<h2>⚙️ Installation</h2>

<pre><code>pip install -r requirements.txt</code></pre>

<hr>

<h2>🚀 Train the Model</h2>

<pre><code>python -m src.train</code></pre>

<p>This will:</p>
<ul>
  <li>Load training data</li>
  <li>Fit metadata preprocessors</li>
  <li>Train fusion network</li>
  <li>Save best model to models/best_model.h5</li>
</ul>

<hr>

<h2>🔍 Generate Predictions</h2>

<pre><code>python -m src.predict</code></pre>

<p>It will:</p>
<ul>
  <li>Load validation images + metadata</li>
  <li>Predict Temperature & Tint</li>
  <li>Round values to nearest integer</li>
  <li>Save <strong>results/submission.csv</strong></li>
</ul>

<hr>

<h2>✔ Submission Format Requirements</h2>

<table>
<tr><th>Requirement</th><th>Status</th></tr>
<tr><td>CSV file & No index column</td><td>✔</td></tr>
<tr><td>493 x 3 shape</td><td>✔</td></tr>
<tr><td>Columns: id_global, Temperature, Tint</td><td>✔</td></tr>
<tr><td>Rounded integer predictions</td><td>✔</td></tr>
</table>

<p><strong>Example Final Output:</strong></p>

<pre>
id_global,Temperature,Tint
EB5BEE31-8D4F-450A-8BDD-27C762C75AA6,4780,12
DE666E1F-0433-4958-AEC0-9A0CC0F81036,5214,9
...
(493 rows)
</pre>

<hr>

<h2>📦 Requirements</h2>

<p>Major libraries:</p>
<ul>
  <li>TensorFlow</li>
  <li>Pandas</li>
  <li>NumPy</li>
  <li>Scikit-learn</li>
  <li>Pillow</li>
</ul>

<p>All listed in <code>requirements.txt</code></p>

<hr>

<h2>📝 Submission Instructions</h2>
<ol>
  <li>Upload <strong>results/submission.csv</strong> under Upload File</li>
  <li>Upload Source Code (<strong>.ipynb</strong> or <strong>src/</strong> folder)</li>
  <li>Add explanation in <strong>Your Answer</strong> section</li>
  <li>Click Submit</li>
</ol>

<hr>

<h2>🙌 Credits</h2>
<p>
Made for the <strong>AfterShoot Machine Learning Challenge</strong>.
</p>

<p align="center">
  Built with ❤️, Python, and TensorFlow
</p>

