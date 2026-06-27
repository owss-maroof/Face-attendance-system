# Face Recognition Attendance System

A deep learning attendance system that identifies students in real time through a webcam and logs attendance automatically. Built from scratch — no off-the-shelf face recognition libraries used.

## How It Works

The system runs in three stages:

**Stage 1 — Learn faces in general**  
A custom CNN is trained from scratch on VGGFace2 (480 people, 176K images). The model learns general face features — eye shapes, jawlines, skin texture, lighting variation — before ever seeing a single student.

**Stage 2 — Collect student data**  
A webcam script captures 80 face images per student. Students slowly move their head left, right, up and down so the model sees their face from multiple angles.

**Stage 3 — Fine-tune on students**  
The pre-trained encoder is fine-tuned on student images using a two-phase approach:
- Phase 1: encoder frozen, only the new student head trains
- Phase 2: last conv block unfrozen, everything fine-tuned at a very low learning rate

This prevents catastrophic forgetting while adapting the model to your specific students, camera, and lighting.

**Attendance**  
The final model runs in real time on webcam feed. Recognised students are logged with timestamps to a daily CSV file. A cooldown prevents the same student being marked twice within 30 seconds.

---

## Project Structure

```
face-attendance-system/
├── 01_train_vggface2.py      # Train CNN on VGGFace2 dataset
├── 02_collect_students.py    # Collect student face images via webcam
├── 03_finetune_students.py   # Fine-tune model on student data
├── 04_attendance.py          # Real-time attendance marking
├── requirements.txt
└── README.md
```

---

## Architecture

```
Input (160x160 RGB)
        ↓
Block 1 — Conv 32   edges and textures
        ↓
Block 2 — Conv 64   eyes, nose, mouth
        ↓
Block 3 — Conv 128  facial structure
        ↓
Block 4 — Conv 256  identity features
        ↓
GlobalAveragePooling
        ↓
Dense 512
        ↓
Embedding 128-d (L2 normalised)   face fingerprint
        ↓
Softmax — student classes
```

Total parameters: ~1.58M — lightweight enough for real-time inference on a laptop CPU.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Face Detection | OpenCV Haar Cascades |
| Training Dataset | VGGFace2 (480 people, 176K images) |
| Training Hardware | Google Colab Tesla T4 GPU |
| Face Embeddings | 128-d L2 normalised vectors |
| Attendance Log | CSV with timestamps |

---

## Setup and Usage

**1. Clone the repo**
```bash
git clone https://github.com/owss-maroof/face-attendance-system.git
cd face-attendance-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train on VGGFace2** (Google Colab recommended — needs GPU)
```bash
python 01_train_vggface2.py --data_dir dataset/vggface2 --epochs 30
```

**4. Collect student faces** (webcam required)
```bash
python 02_collect_students.py
# Enter student name when prompted
# Repeat for each student
```

**5. Fine-tune on students**
```bash
python 03_finetune_students.py
```

**6. Start attendance system**
```bash
python 04_attendance.py
```

Attendance is saved daily to `attendance/YYYY-MM-DD_attendance.csv`

---

## Key Design Decisions

**Why train from scratch?**  
The goal was to understand face recognition at a low level — not just call an API. Every architectural decision (filter sizes, embedding dimension, L2 normalisation) has a reason behind it.

**Why two-phase fine-tuning?**  
Fine-tuning all layers at once on a small student dataset leads to catastrophic forgetting — the model loses what it learned from VGGFace2. Freezing the encoder first lets the new head stabilise, then the conv layers are carefully adjusted at a very low learning rate (1e-5).

**Why L2-normalised embeddings?**  
Cosine similarity between normalised vectors equals their dot product — fast to compute and robust to lighting differences between images.

**Why OpenCV Haar Cascades for detection?**  
Fast enough for real-time use on CPU. The recognition model handles the harder job of identifying who the face belongs to.

---

## Results

- Stage 1 training in progress on Google Colab (Tesla T4 GPU)
- Will update with final accuracy after training completes

---

## Author

**Mohammad Owais Maroof**  
B.Tech CSE (Data Science) — Jamia Millia Islamia  
[LinkedIn](https://linkedin.com/in/owais-maroof) • [GitHub](https://github.com/owss-maroof)
