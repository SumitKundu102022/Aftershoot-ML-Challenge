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
  <li>Input: TIFF Images + Metadata (camera, lens, WB info)</li>
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
├── venv/
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>🧠 Model Architecture</h2>

<ul>
  <li>EfficientNetB0 CNN for image feature extraction</li>
  <li>Metadata encoded using StandardScaler + OneHotEncoder</li>
  <li>Fusion of image + metadata → Multi-output regression</li>
</ul>

<p align="center">
  <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" width="110"/>
</p>

<hr>

<h2>⚙️ Setup</h2>

<p><strong>Step 1:</strong> Create and activate virtual environment</p>

<pre><code>python -m venv venv
venv\Scripts\activate  (Windows)
source venv/bin/activate  (Linux/Mac)
</code></pre>

<p><strong>Step 2:</strong> Install dependencies</p>

<pre><code>pip install -r requirements.txt</code></pre>

<hr>

<h2>🚀 Train the Model</h2>

<pre><code>python -m src.train</code></pre>

<ul>
  <li>Loads training data</li>
  <li>Fits metadata preprocessor</li>
  <li>Trains deep fusion model</li>
  <li>Saves best model to <strong>models/best_model.h5</strong></li>
</ul>

<hr>

<h2>🔍 Generate Predictions</h2>

<pre><code>python -m src.predict</code></pre>

<ul>
  <li>Loads validation images + metadata</li>
  <li>Predicts Temperature & Tint</li>
  <li>Rounds values to nearest integer</li>
  <li>Outputs <strong>results/submission.csv</strong></li>
</ul>

<hr>

<h2>✔ Submission Format Requirements</h2>

<table>
<tr><th>Requirement</th><th>Status</th></tr>
<tr><td>CSV format, 493 x 3</td><td>✔</td></tr>
<tr><td>Columns: id_global, Temperature, Tint</td><td>✔</td></tr>
<tr><td>Integer predictions</td><td>✔</td></tr>
<tr><td>No index column</td><td>✔</td></tr>
</table>

<strong>Example:</strong>
<pre>
id_global,Temperature,Tint
EB5BEE31-8D4F-450A-8BDD-27C762C75AA6,4780,12
DE666E1F-0433-4958-AEC0-9A0CC0F81036,5214,9
...
(493 rows)
</pre>

<hr>

<h2>📦 Requirements</h2>

<ul>
  <li>TensorFlow</li>
  <li>Pandas</li>
  <li>NumPy</li>
  <li>Scikit-learn</li>
  <li>Pillow</li>
</ul>

<p>All libraries listed in <code>requirements.txt</code></p>

<hr>

<h2>📝 What to Write in “Your Answer” Section</h2>

<p>You can write the following:</p>

<pre>
I have uploaded the submission.csv file under the Upload File section.
I have also uploaded the source code project (.ipynb and src folder zip) under Upload Source Code.
A virtual environment (venv) was created before installing dependencies.
All setup instructions and implementation details are available in the README.md file of the project.
</pre>

<hr>

<h2>🙌 Credits</h2>

<p>
This project was developed as part of the <strong>AfterShoot Machine Learning Challenge</strong>.
</p>

<p align="center">
  Built with ❤️, Python, and TensorFlow
</p>
