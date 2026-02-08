import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import sys

print("🔵 Live ISL started")

# ================= CONFIG =================
MODEL_PATH = "isl_cnn_bilstm_126.h5"
MEAN_PATH  = "mean.npy"
STD_PATH   = "std.npy"
LABEL_PATH = "labels.npy"

SEQ_LEN = 30
CONF_THRESH = 0.6
SMOOTHING = 7

# ================= LOAD =================
model  = tf.keras.models.load_model(MODEL_PATH)
mean   = np.load(MEAN_PATH)
std    = np.load(STD_PATH)
LABELS = np.load(LABEL_PATH)

print("✅ Model & stats loaded")

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================= BUFFERS =================
sequence = deque(maxlen=SEQ_LEN)
pred_buffer = deque(maxlen=SMOOTHING)

# ================= FEATURE EXTRACTION =================
def extract_features(results):
    features = []

    if not results.multi_hand_landmarks:
        return np.zeros(126, dtype=np.float32)

    for hand in results.multi_hand_landmarks[:2]:
        wrist = hand.landmark[0]

        for lm in hand.landmark:
            features.extend([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

    while len(features) < 126:
        features.extend([0, 0, 0])

    return np.array(features, dtype=np.float32)

# ================= PAD WITH LAST FRAME =================
def pad_sequence(seq):
    if len(seq) >= SEQ_LEN:
        return list(seq)

    last = seq[-1]
    while len(seq) < SEQ_LEN:
        seq.append(last)

    return list(seq)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera error")
    sys.exit()

print("🟢 Webcam running (Press Q to quit)")

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    feat = extract_features(results)
    sequence.append(feat)

    # Draw hands
    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

    # ---------- PREDICTION ----------
    if len(sequence) >= 5:
        seq = pad_sequence(deque(sequence))
        X = np.expand_dims(seq, axis=0)

        # same normalization as training
        X = (X - mean) / std

        preds = model.predict(X, verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]

        if conf > CONF_THRESH:
            pred_buffer.append(idx)
            final_idx = max(set(pred_buffer), key=pred_buffer.count)
            word = LABELS[final_idx]

            cv2.putText(
                frame,
                f"{word} ({conf:.2f})",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3
            )

    cv2.imshow("ISL Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🔴 Live stopped")
