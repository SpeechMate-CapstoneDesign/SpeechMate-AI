import cv2
import mediapipe as mp
import time
import os
import math
import numpy as np


# Holistic을 사용한 동영상 분석 코드 (FaceMesh 및 Blink 통합)

# 2D 픽셀 거리 계산 함수 (Holistic, FaceMesh 공용)
def get_distance_2d(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x1 - x2, y1 - y2)


# --- 설정값 ---
VIDEO_PATH = "unbal_vid.mp4"  # 분석할 동영상 경로
OUTPUT_VIDEO_PATH = "unbalanced_res.mp4"  # 결과 동영상 저장 경로
OUTPUT_LOG_PATH = "behavior_straight_final_log.txt"  # 행동 기록 로그 파일

# --- MediaPipe 모델 초기화 ---
# 1. Holistic (포즈, 손 감지용)
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# 2. FaceMesh (얼굴, 눈 깜빡임 감지용)
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # [중요] 눈 깜빡임용 478개 랜드마크 활성화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils

# --- 파라미터 설정 ---
VISIBILITY_THRESHOLD = 0.3  # 랜드마크 감지 임계값

# === 1. Head Pose (NPR, HPR) 설정 ===
NPR_CEILING_THRESHOLD = 0.20
NPR_FLOOR_THRESHOLD = 0.44
HPR_LEFT_THRESHOLD = 0.40
HPR_RIGHT_THRESHOLD = 0.60
POSE_FRAME_COUNT = 20  # (아래에서 fps 기준으로 재설정됨)

# === 2. 입술 깨물기 (Lip Bite) 설정 ===
LIP_BITE_RATIO_THRESHOLD = 0.25
LIP_BITE_FRAME_COUNT = 15

# === 3. 손 제스처 (Hand Near Face) 설정 ===
HAND_NEAR_FACE_THRESHOLD = 0.15  # 얼굴(코)-손(검지) 상대 거리
HAND_GESTURE_FRAME_COUNT = 15

# === 4. 비스듬한 자세 (Slant) 설정 ===
SLANT_RATIO_THRESHOLD = 0.03  # [수정] sin(수직/대각선) 기준 임계값
SLANT_FRAME_COUNT = 30

# === 5. 경직된 차려 (Rigid Attention) 설정 (비율 기반) ===
RIGID_FRAME_COUNT = 60
ATTENTION_DIST_RATIO = 0.5  # 손목-엉덩이 거리 < (어깨너비 * 0.5)
RIGID_MOVEMENT_RATIO = 0.02  # 손목 움직임 std < (어깨너비 * 0.02)

# === 6. [신규] 눈 깜빡임 (Blink) 설정 ===
EAR_THRESHOLD = 0.20  # 눈 감음 임계값 (튜닝 필요)
BLINK_FRAME_COUNT = 2  # EAR_THRESHOLD 미만을 2프레임 유지 시 '감음' 상태로 인정

# --- 상태 변수 ---
pose_status = "Forward";
pose_counter = 0;
last_pose_event = ""
lip_bite_status = False;
lip_bite_counter = 0;
last_lip_bite_event = False
hand_near_face_status = False;
hand_near_face_counter = 0;
last_hand_near_face_event = False
slant_status = False;
slant_counter = 0;
last_slant_event = False
rigid_status = False;
rigid_counter = 0;
last_rigid_event = False
left_wrist_history = [];
right_wrist_history = []
blink_status_closed = False;
blink_counter = 0;
last_blink_event = False  # [신규]

# --- 이벤트별 카운터 ---
total_floor_events = 0
total_ceiling_events = 0
total_left_events = 0
total_right_events = 0
total_lip_bite_events = 0
total_hand_near_face_events = 0
total_slant_events = 0
total_rigid_events = 0
total_blink_events = 0  # [신규]

behavior_logs = []

# --- 동영상 읽기 ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 동영상을 열 수 없습니다: {VIDEO_PATH}");
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# [수정] 프레임 카운트를 fps 기준으로 재설정 (2초, 1.5초 등)
RIGID_FRAME_COUNT = max(fps * 2, 60)  # 최소 60프레임 혹은 2초
SLANT_FRAME_COUNT = int(fps * 1.5)
POSE_FRAME_COUNT = int(fps * 0.8)
LIP_BITE_FRAME_COUNT = int(fps * 0.5)
HAND_GESTURE_FRAME_COUNT = int(fps * 0.5)
BLINK_FRAME_COUNT = max(int(fps * 0.05), 2)  # 최소 2프레임

print(f"📹 동영상 정보: {width}x{height} @ {fps}fps, 총 {total_frames}프레임")
print(f"설정된 경직 프레임: {RIGID_FRAME_COUNT} (약 2초)")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# --- 메인 루프 ---
frame_count = 0
start_time = time.time()

while True:
    success, img = cap.read()
    if not success: break

    frame_count += 1
    current_time_sec = frame_count / fps
    current_time_str = time.strftime('%H:%M:%S', time.gmtime(current_time_sec))

    if frame_count % (fps * 2) == 0:
        progress = (frame_count / total_frames) * 100
        print(f"처리 중... {progress:.1f}% ({frame_count}/{total_frames})")

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- [수정] 1. Holistic 실행 (포즈, 손) ---
    results_holistic = holistic.process(imgRGB)
    # --- [수정] 2. FaceMesh 실행 (얼굴, 눈) ---
    results_facemesh = face_mesh.process(imgRGB)

    # 상태 초기화
    current_pose_direction = "Forward"
    current_lip_bite = False
    current_hand_near_face = False
    current_slant_pose = False
    current_rigid_pose = False
    current_blink = False  # [신규]

    # 튜닝용 변수 초기화
    avg_npr = 0.5;
    avg_hpr = 0.5;
    avg_lip_ratio = 1.0;
    avg_ear = 0.5
    slant_ratio = 0.0
    dist_l_wrist_hip = 1000.0;
    dist_r_wrist_hip = 1000.0
    movement_std = 100.0
    shoulder_width_px = 0.0
    dynamic_attention_thresh = 0.0
    dynamic_rigid_thresh = 0.0

    # === [수정] A. 얼굴 랜드마크 분석 (FaceMesh 결과 사용) ===
    # (Blink, Head Pose, Lip Bite 통합)
    if results_facemesh.multi_face_landmarks:
        face_landmarks = results_facemesh.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark  # 478개 랜드마크

        try:
            # --- 1. Head Pose (NPR, HPR) ---
            # (로직 동일)
            eye_center_y = landmarks[6].y
            nose_tip_y = landmarks[1].y
            chin_tip_y = landmarks[152].y
            face_vertical_height = chin_tip_y - eye_center_y
            if face_vertical_height > 0.01:
                avg_npr = (nose_tip_y - eye_center_y) / face_vertical_height
                if avg_npr > NPR_FLOOR_THRESHOLD:
                    current_pose_direction = "Floor"
                elif avg_npr < NPR_CEILING_THRESHOLD:
                    current_pose_direction = "Ceiling"

            if current_pose_direction == "Forward":
                face_left_x = landmarks[234].x
                face_right_x = landmarks[454].x
                nose_tip_x = landmarks[1].x
                face_horizontal_width = face_right_x - face_left_x
                if face_horizontal_width > 0.1:
                    avg_hpr = (nose_tip_x - face_left_x) / face_horizontal_width
                    if avg_hpr > HPR_RIGHT_THRESHOLD:
                        current_pose_direction = "Right"
                    elif avg_hpr < HPR_LEFT_THRESHOLD:
                        current_pose_direction = "Left"

            # --- 2. 입술 깨물기 (Lip Bite) ---
            # (로직 동일)
            upper_outer_lip_y = landmarks[12].y
            upper_inner_lip_y = landmarks[13].y
            lower_inner_lip_y = landmarks[14].y
            lower_outer_lip_y = landmarks[17].y
            total_lip_height = abs(lower_outer_lip_y - upper_outer_lip_y)
            inner_lip_height = lower_inner_lip_y - upper_inner_lip_y
            if total_lip_height > 0.005:
                avg_lip_ratio = inner_lip_height / total_lip_height
                if avg_lip_ratio < LIP_BITE_RATIO_THRESHOLD:
                    current_lip_bite = True

            # --- 3. [신규] 눈 깜빡임 (Blink) ---
            # (로직 동일)
            left_v_top = landmarks[386];
            left_v_bottom = landmarks[374]
            left_h_left = landmarks[362];
            left_h_right = landmarks[263]
            right_v_top = landmarks[159];
            right_v_bottom = landmarks[145]
            right_h_left = landmarks[133];
            right_h_right = landmarks[33]

            left_vertical_dist = get_distance_2d(left_v_top, left_v_bottom, width, height)
            left_horizontal_dist = get_distance_2d(left_h_left, left_h_right, width, height)
            right_vertical_dist = get_distance_2d(right_v_top, right_v_bottom, width, height)
            right_horizontal_dist = get_distance_2d(right_h_left, right_h_right, width, height)

            if left_horizontal_dist > 0 and right_horizontal_dist > 0:
                left_ear = left_vertical_dist / left_horizontal_dist
                right_ear = right_vertical_dist / right_horizontal_dist
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear < EAR_THRESHOLD:
                    current_blink = True

            # --- 4. Head Pose 이벤트 처리 ---
            # (로직 동일)
            if current_pose_direction != "Forward":
                if current_pose_direction == pose_status:
                    pose_counter += 1
                else:
                    pose_status = current_pose_direction;
                    pose_counter = 1;
                    last_pose_event = ""
                if pose_counter >= POSE_FRAME_COUNT and last_pose_event != pose_status:
                    last_pose_event = pose_status
                    if pose_status == "Floor":
                        total_floor_events += 1;
                        log_msg = f"[{current_time_str}] 바닥 보기 (총: {total_floor_events}회)"
                    elif pose_status == "Ceiling":
                        total_ceiling_events += 1;
                        log_msg = f"[{current_time_str}] 천장 보기 (총: {total_ceiling_events}회)"
                    elif pose_status == "Left":
                        total_left_events += 1;
                        log_msg = f"[{current_time_str}] 왼쪽 보기 (총: {total_left_events}회)"
                    elif pose_status == "Right":
                        total_right_events += 1;
                        log_msg = f"[{current_time_str}] 오른쪽 보기 (총: {total_right_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
            else:
                pose_counter = 0;
                pose_status = "Forward";
                last_pose_event = ""

            # --- 5. Lip Bite 이벤트 처리 ---
            # (로직 동일)
            if current_lip_bite:
                lip_bite_status = True;
                lip_bite_counter += 1
                if lip_bite_counter >= LIP_BITE_FRAME_COUNT and not last_lip_bite_event:
                    last_lip_bite_event = True;
                    total_lip_bite_events += 1
                    log_msg = f"[{current_time_str}] 입술 깨물기 감지 (총: {total_lip_bite_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
            else:
                lip_bite_status = False;
                lip_bite_counter = 0;
                last_lip_bite_event = False

            # --- 6. [신규] Blink 이벤트 처리 ---
            # (로직 동일)
            if current_blink:  # 눈 감음 (EAR < Threshold)
                blink_counter += 1
                blink_status_closed = True  # (시각화용)
            else:  # 눈 뜸
                if blink_counter >= BLINK_FRAME_COUNT:  # 감은 상태가 일정 프레임 유지되었었다면
                    total_blink_events += 1
                    log_msg = f"[{current_time_str}] 눈 깜빡임 감지 (총: {total_blink_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
                blink_counter = 0
                blink_status_closed = False

        except Exception as e:
            # print(f"얼굴 분석 오류: {e}")
            pass

    # === [수정] B. 포즈 랜드마크 분석 (Holistic 결과 사용) ===
    # (Slant, Rigid)
    if results_holistic.pose_landmarks:
        landmarks = results_holistic.pose_landmarks.landmark

        try:
            # --- 1. [수정] '비스듬히' 및 '경직' 로직 ---
            left_shoulder = landmarks[mpHolistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mpHolistic.PoseLandmark.RIGHT_SHOULDER.value]

            if (left_shoulder.visibility > VISIBILITY_THRESHOLD and
                    right_shoulder.visibility > VISIBILITY_THRESHOLD):

                # (A) '경직' 로직을 위한 '대각선' 어깨너비 계산
                shoulder_width_px = get_distance_2d(left_shoulder, right_shoulder, width, height)

                # (B) [수정] '비스듬히' 로직을 위한 '수직' 차이 계산
                # (이전 'right_shoulder.y' 중복 사용 버그 수정)
                shoulder_y_diff_px = abs(int(left_shoulder.y * height) - int(right_shoulder.y * height))

                # --- 1-A. [수정] 비스듬한 자세 (Slant) 로직 (sin 기반 - 요청사항) ---
                if shoulder_width_px > 0:
                    # (수직 차이 / 대각선 거리)
                    slant_ratio = shoulder_y_diff_px / shoulder_width_px
                else:
                    slant_ratio = 0.0

                if slant_ratio > SLANT_RATIO_THRESHOLD:
                    current_slant_pose = True

            # --- 2. 경직된 차려 자세 (Rigid Attention) 로직 ---
            # (이하 로직은 'shoulder_width_px'를 사용하므로 동일)
            if shoulder_width_px > 10:  # 어깨가 감지되었을 때만
                left_wrist = landmarks[mpHolistic.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mpHolistic.PoseLandmark.RIGHT_WRIST.value]
                left_hip = landmarks[mpHolistic.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mpHolistic.PoseLandmark.RIGHT_HIP.value]

                if (left_wrist.visibility > VISIBILITY_THRESHOLD and right_wrist.visibility > VISIBILITY_THRESHOLD and
                        left_hip.visibility > VISIBILITY_THRESHOLD and right_hip.visibility > VISIBILITY_THRESHOLD):

                    dynamic_attention_thresh = shoulder_width_px * ATTENTION_DIST_RATIO
                    dynamic_rigid_thresh = shoulder_width_px * RIGID_MOVEMENT_RATIO

                    dist_l_wrist_hip = get_distance_2d(left_wrist, left_hip, width, height)
                    dist_r_wrist_hip = get_distance_2d(right_wrist, right_hip, width, height)

                    is_attention_pose = (dist_l_wrist_hip < dynamic_attention_thresh and
                                         dist_r_wrist_hip < dynamic_attention_thresh)

                    lw_x_px, lw_y_px = int(left_wrist.x * width), int(left_wrist.y * height)
                    rw_x_px, rw_y_px = int(right_wrist.x * width), int(right_wrist.y * height)
                    left_wrist_history.append((lw_x_px, lw_y_px))
                    right_wrist_history.append((rw_x_px, rw_y_px))

                    if len(left_wrist_history) > RIGID_FRAME_COUNT:
                        left_wrist_history.pop(0)
                        right_wrist_history.pop(0)

                    is_rigid = False
                    if len(left_wrist_history) == RIGID_FRAME_COUNT:
                        l_std_x = np.std([pos[0] for pos in left_wrist_history])
                        l_std_y = np.std([pos[1] for pos in left_wrist_history])
                        r_std_x = np.std([pos[0] for pos in right_wrist_history])
                        r_std_y = np.std([pos[1] for pos in right_wrist_history])
                        movement_std = (l_std_x + l_std_y + r_std_x + r_std_y) / 4.0

                        if movement_std < dynamic_rigid_thresh:
                            is_rigid = True

                    if is_attention_pose and is_rigid:
                        current_rigid_pose = True

            # --- 3. Slant 이벤트 처리 ---
            if current_slant_pose:
                slant_status = True;
                slant_counter += 1
                if slant_counter >= SLANT_FRAME_COUNT and not last_slant_event:
                    last_slant_event = True;
                    total_slant_events += 1
                    log_msg = f"[{current_time_str}] 비스듬한 자세 감지 (총: {total_slant_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
            else:
                slant_status = False;
                slant_counter = 0;
                last_slant_event = False

            # --- 4. [FIXED] Rigid 이벤트 처리 (이중 카운트 버그 수정) ---
            if current_rigid_pose:
                rigid_status = True;
                rigid_counter += 1
                if not last_rigid_event:  # [수정] 2초(RIGID_FRAME_COUNT)가 막 지났을 때 1회만
                    last_rigid_event = True;
                    total_rigid_events += 1
                    log_msg = f"[{current_time_str}] 경직된 차려 자세 감지 (총: {total_rigid_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
            else:
                rigid_status = False;
                rigid_counter = 0;
                last_rigid_event = False

        except Exception as e:
            # print(f"포즈 분석 오류: {e}")
            pass

    # === [수정] C. 손-얼굴 상호작용 분석 (Holistic + FaceMesh) ===
    # (로직 동일)
    if results_facemesh.multi_face_landmarks and (
            results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks):
        try:
            nose_x = results_facemesh.multi_face_landmarks[0].landmark[1].x
            nose_y = results_facemesh.multi_face_landmarks[0].landmark[1].y

            hand_detected_near_face = False
            if results_holistic.left_hand_landmarks:
                left_hand_tip = results_holistic.left_hand_landmarks.landmark[8]
                dist_left = ((left_hand_tip.x - nose_x) ** 2 + (left_hand_tip.y - nose_y) ** 2) ** 0.5
                if dist_left < HAND_NEAR_FACE_THRESHOLD:
                    hand_detected_near_face = True
            if results_holistic.right_hand_landmarks:
                right_hand_tip = results_holistic.right_hand_landmarks.landmark[8]
                dist_right = ((right_hand_tip.x - nose_x) ** 2 + (right_hand_tip.y - nose_y) ** 2) ** 0.5
                if dist_right < HAND_NEAR_FACE_THRESHOLD:
                    hand_detected_near_face = True

            if hand_detected_near_face:
                hand_near_face_status = True;
                hand_near_face_counter += 1
                if hand_near_face_counter >= HAND_GESTURE_FRAME_COUNT and not last_hand_near_face_event:
                    last_hand_near_face_event = True;
                    total_hand_near_face_events += 1
                    log_msg = f"[{current_time_str}] 얼굴 근처 손 제스처 감지 (총: {total_hand_near_face_events}회)"
                    behavior_logs.append(log_msg);
                    print(log_msg)
            else:
                hand_near_face_status = False;
                hand_near_face_counter = 0;
                last_hand_near_face_event = False
        except Exception as e:
            # print(f"손-얼굴 분석 오류: {e}")
            pass

    # === D. 시각화 ===
    # (랜드마크 그리기 로직 동일)
    if results_holistic.left_hand_landmarks:
        mpDraw.draw_landmarks(img, results_holistic.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    if results_holistic.right_hand_landmarks:
        mpDraw.draw_landmarks(img, results_holistic.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    if results_holistic.pose_landmarks:
        mpDraw.draw_landmarks(img, results_holistic.pose_landmarks, mpHolistic.POSE_CONNECTIONS)
    if results_facemesh.multi_face_landmarks:
        mpDraw.draw_landmarks(
            image=img, landmark_list=results_facemesh.multi_face_landmarks[0],
            connections=mpFaceMesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1))
        mpDraw.draw_landmarks(
            image=img, landmark_list=results_facemesh.multi_face_landmarks[0],
            connections=mpFaceMesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1))

    # (정보 텍스트 로직 동일)
    cv2.putText(img, f"Time: {current_time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_status = 70
    if pose_status != "Forward":
        cv2.putText(img, f"Pose: {pose_status} ({pose_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        y_status += 40
    if lip_bite_status:
        cv2.putText(img, f"Lip Biting ({lip_bite_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        y_status += 40
    if hand_near_face_status:
        cv2.putText(img, f"Hand Near Face ({hand_near_face_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)
        y_status += 40
    if slant_status:
        cv2.putText(img, f"Slanted ({slant_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_status += 40
    if rigid_status:
        cv2.putText(img, f"Rigid Attention ({rigid_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        y_status += 40
    if blink_status_closed:
        cv2.putText(img, f"Blink Closed ({blink_counter})", (10, y_status), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # (카운터 표시 로직 동일)
    y_pos = height - 200
    cv2.putText(img, f"Floor: {total_floor_events}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Ceiling: {total_ceiling_events}", (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                2)
    cv2.putText(img, f"Left: {total_left_events}", (10, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Right: {total_right_events}", (10, y_pos + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Lip Bites: {total_lip_bite_events}", (10, y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2)
    cv2.putText(img, f"Hand Gestures: {total_hand_near_face_events}", (200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2)
    cv2.putText(img, f"Slanted: {total_slant_events}", (200, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                2)
    cv2.putText(img, f"Rigid: {total_rigid_events}", (200, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f"Blinks: {total_blink_events}", (200, y_pos + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0),
                2)

    # [수정] 튜닝용 값 시각화 (우측 상단)
    info_x = width - 350
    cv2.putText(img, f"EAR: {avg_ear:.3f} (T: {EAR_THRESHOLD})", (info_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    cv2.putText(img, f"Shoulder Width: {shoulder_width_px:.1f} px", (info_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)
    # [수정] "tan" -> "sin"
    cv2.putText(img, f"Slant Ratio (sin): {slant_ratio:.3f} (T: {SLANT_RATIO_THRESHOLD})", (info_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Wrist-Hip L: {dist_l_wrist_hip:.1f} (T: {dynamic_attention_thresh:.1f})", (info_x, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Wrist-Hip R: {dist_r_wrist_hip:.1f}", (info_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                2)
    cv2.putText(img, f"Movement Std: {movement_std:.2f} (T: {dynamic_rigid_thresh:.2f})", (info_x, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 결과 동영상에 프레임 저장
    out.write(img)

# --- 정리 및 로그 저장 ---
cap.release()
out.release()
holistic.close()
face_mesh.close()

end_time = time.time()
processing_time = end_time - start_time

print("\n" + "=" * 60)
print("📊 분석 완료!")
print("=" * 60)
# (이하 로그 출력 부분은 동일)
print(f"⏱️  처리 시간: {processing_time:.2f}초")
print(f"📹 결과 동영상: {OUTPUT_VIDEO_PATH}")
print(f"📝 로그 파일: {OUTPUT_LOG_PATH}")
print("\n=== 비언어 행동 통계 ===")
print(f"바닥 보기 (고개 숙임): {total_floor_events}회")
print(f"천장 보기 (고개 듦): {total_ceiling_events}회")
print(f"왼쪽 보기: {total_left_events}회")
print(f"오른쪽 보기: {total_right_events}회")
print(f"입술 깨물기: {total_lip_bite_events}회")
print(f"얼굴 근처 손 제스처: {total_hand_near_face_events}회")
print(f"비스듬한 자세: {total_slant_events}회")
print(f"경직된 차려 자세: {total_rigid_events}회")
print(f"눈 깜빡임: {total_blink_events}회")
print("=" * 60)

with open(OUTPUT_LOG_PATH, 'w', encoding='utf-8') as f:
    f.write("=== 비언어 행동 분석 결과 ===\n\n")
    f.write(f"동영상: {VIDEO_PATH}\n")
    f.write(f"처리 시간: {processing_time:.2f}초\n")
    f.write(f"총 프레임: {total_frames}\n\n")
    f.write("=== 통계 ===\n")
    f.write(f"바닥 보기: {total_floor_events}회\n")
    f.write(f"천장 보기: {total_ceiling_events}회\n")
    f.write(f"왼쪽 보기: {total_left_events}회\n")
    f.write(f"오른쪽 보기: {total_right_events}회\n")
    f.write(f"입술 깨물기: {total_lip_bite_events}회\n")
    f.write(f"얼굴 근처 손 제스처: {total_hand_near_face_events}회\n")
    f.write(f"비스듬한 자세: {total_slant_events}회\n")
    f.write(f"경직된 차려 자세: {total_rigid_events}회\n")
    f.write(f"눈 깜빡임: {total_blink_events}회\n")
    f.write("\n=== 상세 로그 ===\n")
    for log in behavior_logs:
        f.write(log + "\n")

print(f"\n✅ 모든 결과가 저장되었습니다!")