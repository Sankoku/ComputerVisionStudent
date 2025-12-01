from ultralytics import YOLO
import cv2
import numpy as np
import csv
from collections import deque, defaultdict
import time
import torch
import matplotlib.pyplot as plt

# ================== НАЛАШТУВАННЯ (на основі досліджень) ==================
VIDEO_PATH = "video3.mp4"
MODEL_PATH = "yolov8m-pose.pt"
CONF = 0.3

# ===== НОВІ ПАРАМЕТРИ =====
MAX_DURATION_MINUTES = 30  # Скільки хвилин обробляти (None = все відео)
SPEED_MULTIPLIER = 3      # Прискорення: 1=normal, 2=2x швидше, 3=3x швидше
                          # При прискоренні обробляється кожен N-й кадр
# ===========================

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"🔍 Device: {device}")

# Калібрування baseline
BASELINE_FRAMES = 90  # ~3 секунди при 30 FPS (було 50)
EMA_ALPHA = 0.3  # Згладжування

# Пороги уваги 
SCORE_ATTENTIVE = 0.70  # Уважний
SCORE_NEUTRAL = 0.50    # Нейтральний
SCORE_DISTRACTED = 0.30 # Відволікся

# Часові пороги
INATTENTIVE_SECONDS = 3.0  # Скільки секунд неуважності = проблема
HAND_RAISED_MIN_FRAMES = 15  # Мінімум кадрів з піднятою рукою
FPS = 30.0

# Трекінг
IOU_MATCH_THRESH = 0.15 
MAX_MISSED_FRAMES = 150  # ~5 секунд

# ================== ІНІЦІАЛІЗАЦІЯ ==================
model = YOLO(MODEL_PATH)
model.to(device)

cap = cv2.VideoCapture(VIDEO_PATH)
if cap.get(cv2.CAP_PROP_FPS) > 0:
    FPS = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Збереження відео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_scientific.mp4', fourcc, int(FPS), (w, h))

students = {}
next_id = 0
activity_log = []
frame_id = 0
processed_frames = 0  # Лічильник оброблених кадрів

# Обчислюємо максимальну кількість кадрів
max_frames = None
if MAX_DURATION_MINUTES is not None:
    max_frames = int(MAX_DURATION_MINUTES * 60 * FPS)

print(f"  Video: {w}x{h} @ {FPS:.1f} FPS")
print(f"  Baseline: {BASELINE_FRAMES} frames ({BASELINE_FRAMES/FPS:.1f}s)")
print(f"  IOU threshold: {IOU_MATCH_THRESH}")
print(f"  Duration limit: {MAX_DURATION_MINUTES} min" if MAX_DURATION_MINUTES else "⏱️  Duration: Full video")
print(f"  Speed multiplier: {SPEED_MULTIPLIER}x (processing every {SPEED_MULTIPLIER} frame)")
if max_frames:
    print(f" Will process ~{max_frames // SPEED_MULTIPLIER} frames total")

# ================== ФУНКЦІЇ ==================

def bbox_from_kp(kp):
    """Створює bounding box з keypoints"""
    x_min = int(np.min(kp[:,0]))
    x_max = int(np.max(kp[:,0]))
    y_min = int(np.min(kp[:,1]))
    y_max = int(np.max(kp[:,1]))
    return (x_min, y_min, x_max, y_max)

def iou(boxA, boxB):
    """Intersection over Union для матчінгу"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1, (boxA[2]-boxA[0])*(boxA[3]-boxA[1]))
    boxBArea = max(1, (boxB[2]-boxB[0])*(boxB[3]-boxB[1]))
    return interArea / (boxAArea + boxBArea - interArea + 1e-9)

def match_detections(prev_students, det_bboxes, iou_thresh=IOU_MATCH_THRESH):
    """Матчить попередні детекції з новими"""
    matches = {}
    unmatched_prev = set(prev_students.keys())
    unmatched_det = set(range(len(det_bboxes)))
    ious = {}
    
    for pid, s in prev_students.items():
        for j, bbox in enumerate(det_bboxes):
            try:
                ious[(pid,j)] = iou(s['bbox'], bbox)
            except Exception:
                ious[(pid,j)] = 0.0
    
    # Greedy matching
    while ious:
        (pid,j), best = max(ious.items(), key=lambda x: x[1])
        if best < iou_thresh:
            break
        matches[pid] = j
        unmatched_prev.discard(pid)
        unmatched_det.discard(j)
        keys_to_del = [k for k in ious if k[0]==pid or k[1]==j]
        for k in keys_to_del:
            del ious[k]
    
    return matches, unmatched_prev, unmatched_det

def eye_aspect_ratio(eye_points):
    """
    EAR (Eye Aspect Ratio) - для детекції закритих очей
    На основі досліджень Soukupová and Čech (2016)
    """
    if eye_points.shape[0] < 2:
        return 1.0
    # Вертикальна відстань між точками ока
    vertical = np.linalg.norm(eye_points[0] - eye_points[1])
    # Горизонтальна відстань
    horizontal = np.linalg.norm(eye_points[0] - eye_points[-1]) + 1e-6
    ear = vertical / horizontal
    return ear

def compute_features(kp):
    """
    Обчислює розширений набір features на основі наукових досліджень:
    - Head pose (pitch, yaw)
    - Hand position
    - Body posture
    - Eye aspect ratio (для втоми/сну)
    """
    if kp is None or kp.shape[0] < 13:
        return None
    
    # Keypoints COCO format
    nose = kp[0]
    left_eye = kp[1]
    right_eye = kp[2]
    left_ear = kp[3]
    right_ear = kp[4]
    left_shoulder = kp[5]
    right_shoulder = kp[6]
    left_elbow = kp[7]
    right_elbow = kp[8]
    left_wrist = kp[9]
    right_wrist = kp[10]
    left_hip = kp[11]
    right_hip = kp[12]
    
    # Базові точки
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    mid_hip = (left_hip + right_hip) / 2.0
    mid_eyes = (left_eye + right_eye) / 2.0
    
    # Довжина торсу для нормалізації
    torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-6
    
    # 1. HEAD POSE (ключовий індикатор за дослідженнями)
    head_pitch = float((nose[1] - mid_shoulder[1]) / torso_len)  # Нахил вперед/назад
    head_yaw = float((nose[0] - mid_eyes[0]) / torso_len)  # Поворот вліво/вправо
    
    # 2. EYE ASPECT RATIO (для визначення закритих очей)
    # Спрощена версія - відстань між очима та носом
    eye_openness = float(np.linalg.norm(mid_eyes - nose) / torso_len)
    
    # 3. HAND POSITION ANALYSIS (за дослідженнями - підняті руки = активна участь)
    hands_up = float(
        (left_wrist[1] < left_shoulder[1] - 0.2*torso_len) or 
        (right_wrist[1] < right_shoulder[1] - 0.2*torso_len)
    )
    
    # Руки внизу (можливо на столі або розслаблений)
    hands_below = float(
        (left_wrist[1] > mid_hip[1] + 0.1*torso_len) and 
        (right_wrist[1] > mid_hip[1] + 0.1*torso_len)
    )
    
    # Відстань між руками (для детекції жестикуляції)
    hands_distance = float(np.linalg.norm(left_wrist - right_wrist) / torso_len)
    
    # 4. BODY POSTURE (сутулість/втома)
    # Відстань між вухами та плечима (індикатор сутулості)
    mid_ears = (left_ear + right_ear) / 2.0
    slouch_factor = float((mid_shoulder[1] - mid_ears[1]) / torso_len)
    
    # 5. MOVEMENT/ACTIVITY LEVEL
    # Буде обчислюватись при порівнянні з baseline
    
    return {
        "head_pitch": head_pitch,
        "head_yaw": head_yaw,
        "eye_openness": eye_openness,
        "hands_up": hands_up,
        "hands_below": hands_below,
        "hands_distance": hands_distance,
        "slouch_factor": slouch_factor,
        "torso_len": torso_len
    }

def student_attention_score(features, baseline):
    """
    Покращена формула на основі досліджень про engagement detection
    
    Базується на:
    - Canedo et al. (2018) - Head pose estimation
    - Whitehill et al. (2014) - Facial features for engagement
    - Raca et al. (2015) - Body language indicators
    """
    baseline = baseline or {}
    
    # Нормалізовані відхилення від baseline
    pitch_dev = features["head_pitch"] - baseline.get("head_pitch", 0.0)
    yaw_dev = abs(features["head_yaw"] - baseline.get("head_yaw", 0.0))
    
    hands_up = features["hands_up"]
    hands_below = features["hands_below"]
    eye_openness = features["eye_openness"]
    slouch_dev = features["slouch_factor"] - baseline.get("slouch_factor", 0.0)
    
    #Score components (ваги на основі досліджень)
    score = 0.0
    
    # 1. Підняті руки - сильний позитивний сигнал (+2.0)
    score += 2.0 * hands_up
    
    # 2. Орієнтація голови (найважливіший фактор)
    # Pitch: голова дивиться вгору = добре, вниз = погано
    score += 1.2 * max(0.0, 1.0 - abs(pitch_dev * 1.5))
    score -= 1.5 * max(0.0, pitch_dev * 1.0)  # Штраф за нахил вниз
    
    # Yaw: голова повернута вбік = неуважність
    score -= 1.0 * min(1.0, yaw_dev * 2.0)
    
    # 3. Руки внизу - можливо відволікся або на телефоні
    score -= 0.8 * hands_below
    
    # 4. Очі (закриті або сонні)
    baseline_eye = baseline.get("eye_openness", 0.5)
    if eye_openness < baseline_eye * 0.6:  # Очі майже закриті
        score -= 1.5
    
    # 5. Сутулість (зміна пози може означати втому)
    if slouch_dev > 0.15:  # Сильна сутулість відносно baseline
        score -= 0.7
    
    # Sigmoid для нормалізації в [0, 1]
    return float(1.0 / (1.0 + np.exp(-score)))

# ================== ОСНОВНИЙ ЦИКЛ ==================
print("\nПочинаємо обробку... (Натисни 'q' для зупинки)\n")
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Перевірка ліміту часу
    if max_frames is not None and frame_id >= max_frames:
        print(f"\n  Досягнуто ліміт часу: {MAX_DURATION_MINUTES} хвилин")
        break
    
    # Прискорення: пропускаємо кадри
    if frame_id % SPEED_MULTIPLIER != 0:
        frame_id += 1
        continue
    
    processed_frames += 1

    # YOLO детекція з GPU
    results = model(frame, conf=CONF, verbose=False, device=device)
    people = results[0].keypoints
    
    det_bboxes = []
    det_kps = []
    det_feats = []

    # Збираємо детекції
    for p in people:
        kp = p.xy[0].cpu().numpy() if hasattr(p, 'xy') else None
        if kp is None or kp.shape[0] < 13:
            continue
        
        bbox = bbox_from_kp(kp)
        feats = compute_features(kp)
        if feats is None:
            continue
        
        det_bboxes.append(bbox)
        det_kps.append(kp)
        det_feats.append(feats)

    # Matching з попередніми студентами
    matches, unmatched_prev, unmatched_det = match_detections(
        students, det_bboxes, IOU_MATCH_THRESH
    )
    
    updated_ids = set()

    # Оновлюємо matched студентів
    for pid, j in matches.items():
        kp = det_kps[j]
        bbox = det_bboxes[j]
        feats = det_feats[j]
        s = students[pid]

        s['bbox'] = bbox
        s['last_seen'] = frame_id

        # Baseline калібрування
        if s['baseline_count'] < BASELINE_FRAMES:
            # Акумулюємо features для baseline
            for k in feats:
                s['baseline'][k] = (
                    s['baseline'][k] * s['baseline_count'] + feats[k]
                ) / (s['baseline_count'] + 1)
            s['baseline_count'] += 1
        else:
            # Обчислюємо attention score
            base = s.get('baseline', {})
            raw_score = student_attention_score(feats, base)
            
            # EMA згладжування
            s['ema'] = EMA_ALPHA * raw_score + (1 - EMA_ALPHA) * s.get('ema', raw_score)
            
            # Історія
            s.setdefault('history', deque(maxlen=int(FPS*10))).append(s['ema'])
            
            # Лічильник неуважності
            if s['ema'] < SCORE_NEUTRAL:
                s['inattentive_frames'] += 1
            else:
                s['inattentive_frames'] = 0
            
            # Детекція підняття руки (з фільтрацією)
            if feats['hands_up']:
                s['hand_raised_frames'] = s.get('hand_raised_frames', 0) + 1
            else:
                s['hand_raised_frames'] = 0

        updated_ids.add(pid)

        # ВІЗУАЛІЗАЦІЯ
        x1, y1, x2, y2 = bbox
        
        if s.get('baseline_count', 0) < BASELINE_FRAMES:
            # Калібрування
            color = (180, 180, 180)
            progress = int(100 * s['baseline_count'] / BASELINE_FRAMES)
            label = f"ID{pid}: Calibrating {progress}%"
        else:
            # Класифікація уваги
            if s['ema'] >= SCORE_ATTENTIVE:
                color = (0, 255, 0)  # Зелений - уважний
                label = f"ID{pid}: Attentive {s['ema']:.2f}"
            elif s['ema'] >= SCORE_NEUTRAL:
                color = (0, 255, 255)  # Жовтий - нейтральний
                label = f"ID{pid}: Neutral {s['ema']:.2f}"
            elif s['ema'] >= SCORE_DISTRACTED:
                color = (0, 165, 255)  # Помаранчевий - відволікся
                label = f"ID{pid}: Distracted {s['ema']:.2f}"
            else:
                needed = int(INATTENTIVE_SECONDS * FPS)
                is_problem = s['inattentive_frames'] >= needed
                color = (0, 0, 255) if is_problem else (0, 100, 255)  # Червоний
                label = f"ID{pid}: Inattentive {s['ema']:.2f}"
            
            # Індикація підняття руки
            if s.get('hand_raised_frames', 0) >= HAND_RAISED_MIN_FRAMES:
                cv2.putText(frame, "HAND UP!", (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1-8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Створюємо нових студентів
    for j in unmatched_det:
        kp = det_kps[j]
        bbox = det_bboxes[j]
        feats = det_feats[j]
        
        pid = next_id
        next_id += 1
        
        students[pid] = {
            'bbox': bbox,
            'baseline': feats.copy(),
            'baseline_count': 1,
            'ema': 0.5,
            'history': deque(maxlen=int(FPS*10)),
            'inattentive_frames': 0,
            'hand_raised_frames': 0,
            'last_seen': frame_id
        }
        
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
        cv2.putText(frame, f"New ID{pid}", (x1, max(0, y1-8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)

    # Видаляємо старі треки
    to_delete = []
    for pid, s in list(students.items()):
        if frame_id - s.get('last_seen', -99999) > MAX_MISSED_FRAMES:
            to_delete.append(pid)
    for pid in to_delete:
        del students[pid]

    # ЗАГАЛЬНА СТАТИСТИКА
    people_visible = len(det_bboxes)
    
    # Підрахунок по категоріях (тільки після калібрування)
    calibrated = [s for s in students.values() if s.get('baseline_count', 0) >= BASELINE_FRAMES]
    
    attentive_cnt = sum(1 for s in calibrated if s.get('ema', 0) >= SCORE_ATTENTIVE)
    neutral_cnt = sum(1 for s in calibrated if SCORE_NEUTRAL <= s.get('ema', 0) < SCORE_ATTENTIVE)
    distracted_cnt = sum(1 for s in calibrated if SCORE_DISTRACTED <= s.get('ema', 0) < SCORE_NEUTRAL)
    inattentive_cnt = sum(1 for s in calibrated if s.get('ema', 0) < SCORE_DISTRACTED 
                         and s.get('inattentive_frames', 0) >= int(INATTENTIVE_SECONDS*FPS))
    
    hands_up_cnt = sum(1 for s in students.values() 
                       if s.get('hand_raised_frames', 0) >= HAND_RAISED_MIN_FRAMES)

    # ІНДЕКС УВАГИ КЛАСУ (0-1)
    total = max(1, len(calibrated))
    attention_index = (
        attentive_cnt + 0.5 * neutral_cnt + 0.8 * hands_up_cnt - 0.5 * inattentive_cnt
    ) / total
    attention_index = max(0.0, min(1.0, attention_index))

    # OVERLAY ПАНЕЛЬ
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (550, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_pos = 35
    cv2.putText(frame, f"Students: {people_visible}", (20, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(frame, f"Attentive: {attentive_cnt} | Neutral: {neutral_cnt}", 
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_pos += 30
    cv2.putText(frame, f"Distracted: {distracted_cnt} | Inattentive: {inattentive_cnt}", 
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    y_pos += 30
    cv2.putText(frame, f"Hands up: {hands_up_cnt}", 
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_pos += 30
    
    # Індекс уваги з кольором
    attn_color = (0, 255, 0) if attention_index > 0.7 else \
                 (0, 255, 255) if attention_index > 0.5 else (0, 0, 255)
    cv2.putText(frame, f"Class Attention Index: {attention_index:.2f}", 
               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, attn_color, 2)

    # Лог
    current_time = frame_id / FPS  # Реальний час у відео
    activity_log.append({
        "frame": frame_id,
        "time_sec": round(current_time, 2),
        "visible": people_visible,
        "attentive": attentive_cnt,
        "neutral": neutral_cnt,
        "distracted": distracted_cnt,
        "inattentive": inattentive_cnt,
        "hands_up": hands_up_cnt,
        "attention_index": round(attention_index, 3)
    })

    cv2.imshow("Scientific Monitor", frame)
    out.write(frame)
    
    frame_id += 1
    
    # Прогрес кожні 60 оброблених кадрів
    if processed_frames % 60 == 0:
        elapsed = time.time() - t0
        fps_actual = processed_frames / elapsed
        current_time = frame_id / FPS
        time_min = int(current_time // 60)
        time_sec = int(current_time % 60)
        print(f"⏱️  Frame {frame_id} ({time_min}:{time_sec:02d}) | "
              f"Processing: {fps_actual:.1f} FPS | "
              f"Students: {people_visible} | Attention: {attention_index:.2f}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ЗБЕРЕЖЕННЯ CSV
with open("activity_scientific.csv", "w", newline="", encoding='utf-8') as f:
    fieldnames = ["frame", "time_sec", "visible", "attentive", "neutral", 
                  "distracted", "inattentive", "hands_up", "attention_index"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(activity_log)

print("\n Завершено!")
print(f" Оброблено {processed_frames} кадрів (з {frame_id} у відео) за {time.time()-t0:.1f}с")
print(f" Реальна швидкість обробки: {processed_frames/(time.time()-t0):.1f} FPS")
print(f"  Проаналізовано {frame_id/FPS/60:.1f} хвилин відео")
print("  Збережено: activity_scientific.csv, output_scientific.mp4")

# ===== ВІЗУАЛІЗАЦІЯ СТАТИСТИКИ =====
if len(activity_log) > 0:
    print("\n Генерація графіків...")

    times = [d['time_sec'] for d in activity_log]
    visible = [d['visible'] for d in activity_log]
    attentive = [d['attentive'] for d in activity_log]
    neutral = [d['neutral'] for d in activity_log]
    distracted = [d['distracted'] for d in activity_log]
    inattentive = [d['inattentive'] for d in activity_log]
    hands_up = [d['hands_up'] for d in activity_log]
    attention_index = [d['attention_index'] for d in activity_log]

    # Створюємо фігуру з 6 графіками (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Науковий аналіз навчальної активності учнів', 
                 fontsize=16, fontweight='bold')

    # Графік 1: Кількість учнів
    axes[0, 0].plot(times, visible, color='blue', linewidth=2, marker='o', markersize=2)
    axes[0, 0].set_title('Кількість виявлених учнів', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Час (сек)')
    axes[0, 0].set_ylabel('Кількість')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(bottom=0)

    # Графік 2: Підняті руки
    axes[0, 1].plot(times, hands_up, color='green', linewidth=2, marker='o', markersize=2)
    axes[0, 1].fill_between(times, hands_up, alpha=0.3, color='green')
    axes[0, 1].set_title('Підняті руки (активна участь)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Час (сек)')
    axes[0, 1].set_ylabel('Кількість')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(bottom=0)

    # Графік 3: Індекс уваги класу
    axes[0, 2].plot(times, attention_index, color='purple', linewidth=2.5)
    axes[0, 2].fill_between(times, attention_index, alpha=0.3, color='purple')
    axes[0, 2].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Attentive')
    axes[0, 2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Neutral')
    axes[0, 2].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Distracted')
    axes[0, 2].set_title('Індекс уваги класу (Class Attention)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Час (сек)')
    axes[0, 2].set_ylabel('Індекс (0-1)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend(loc='lower right', fontsize=8)

    # Графік 4: Уважні учні
    axes[1, 0].plot(times, attentive, color='green', linewidth=2, marker='o', markersize=2)
    axes[1, 0].fill_between(times, attentive, alpha=0.3, color='green')
    axes[1, 0].set_title('Уважні учні (Attentive ≥0.70)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Час (сек)')
    axes[1, 0].set_ylabel('Кількість')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(bottom=0)

    # Графік 5: Неуважні та відволікаються
    axes[1, 1].plot(times, inattentive, color='red', linewidth=2, marker='o', markersize=2, label='Inattentive')
    axes[1, 1].plot(times, distracted, color='orange', linewidth=2, marker='o', markersize=2, label='Distracted')
    axes[1, 1].fill_between(times, inattentive, alpha=0.2, color='red')
    axes[1, 1].fill_between(times, distracted, alpha=0.2, color='orange')
    axes[1, 1].set_title('Неуважність і відволікання', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Час (сек)')
    axes[1, 1].set_ylabel('Кількість')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].legend(loc='upper right', fontsize=9)

    # Графік 6: Розподіл станів (stacked area)
    axes[1, 2].stackplot(times, attentive, neutral, distracted, inattentive,
                        labels=['Attentive', 'Neutral', 'Distracted', 'Inattentive'],
                        colors=['green', 'yellow', 'orange', 'red'],
                        alpha=0.7)
    axes[1, 2].set_title('Розподіл станів уваги в часі', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Час (сек)')
    axes[1, 2].set_ylabel('Кількість учнів')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig('activity_statistics_scientific.png', dpi=300, bbox_inches='tight')
    print(" Збережено: activity_statistics_scientific.png")

    # ===== ПІДСУМКОВА СТАТИСТИКА =====
    print("\n" + "="*70)
    print("📊 ПІДСУМКОВА СТАТИСТИКА")
    print("="*70)
    
    total_time = max(times)
    total_minutes = int(total_time // 60)
    total_seconds = int(total_time % 60)
    
    print(f"  Тривалість відео: {total_time:.1f} секунд ({total_minutes}:{total_seconds:02d})")
    print(f" Оброблено кадрів: {processed_frames}")
    print(f" Середній FPS обробки: {processed_frames/total_time:.1f}")
    print(f" Середня кількість учнів: {np.mean(visible):.1f}")
    print(f" Максимум учнів на екрані: {max(visible)}")
    print()
    print(f" Середня кількість уважних: {np.mean(attentive):.2f}")
    print(f" Середня кількість нейтральних: {np.mean(neutral):.2f}")
    print(f" Середня кількість відволіканих: {np.mean(distracted):.2f}")
    print(f" Середня кількість неуважних: {np.mean(inattentive):.2f}")
    print()
    print(f" Всього підняттів рук: {sum(hands_up)}")
    print(f" Середня кількість піднятих рук: {np.mean(hands_up):.2f}")
    print(f" Максимум піднятих рук одночасно: {max(hands_up)}")
    print()
    print(f" Середній індекс уваги класу: {np.mean(attention_index):.3f}")
    print(f" Мінімальний індекс уваги: {min(attention_index):.3f}")
    print(f" Максимальний індекс уваги: {max(attention_index):.3f}")
    print()
    
    # Відсоток часу в різних станах
    if max(visible) > 0:
        total_student_time = sum(visible)
        pct_attentive = 100 * sum(attentive) / total_student_time
        pct_neutral = 100 * sum(neutral) / total_student_time
        pct_distracted = 100 * sum(distracted) / total_student_time
        pct_inattentive = 100 * sum(inattentive) / total_student_time
        
        print(" РОЗПОДІЛ ЧАСУ УВАГИ:")
        print(f"   Уважні: {pct_attentive:.1f}%")
        print(f"   Нейтральні: {pct_neutral:.1f}%")
        print(f"   Відволікаються: {pct_distracted:.1f}%")
        print(f"   Неуважні: {pct_inattentive:.1f}%")
    
    print("="*70)

    plt.show()
else:
    print("  Недостатньо даних для графіків")


print("\n🎓 Аналіз завершено! Використовуй графіки для дипломної роботи.")

