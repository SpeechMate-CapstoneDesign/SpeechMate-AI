import cv2
import mediapipe as mp
import time
import math

#무화과 잎 자세 판단.
#손목사이의 거리, 손목이 엉덩이보다 아래에 있는가?(Hip Y가 Hand Y보다 작은경우일때)

# 두 랜드마크 사이의 2D 거리를 계산하는 함수
def get_distance_2d(lm1, lm2, w, h):
    """
    두 MediaPipe 랜드마크 사이의 픽셀 기준 2D 거리를 계산합니다.
    """
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x1 - x2, y1 - y2)


# --- 설정값 ---
cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose(min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# === '무화과 잎 자세' (Fig Leaf) 설정 ===
HANDS_CLOSE_THRESHOLD = 100  # [튜닝] Hand Dist (픽셀)
POSE_FRAME_COUNT = 30  # 1초간 지속되어야 인정
# [튜닝] 이 값은 필요 없고, 'Hand Y > Hip Y' 조건으로 바로 비교합니다.

# === 상태 변수 ===
fig_leaf_status = False
fig_leaf_counter = 0
last_fig_leaf_event = False
total_fig_leaf_events = 0

# --- 메인 루프 ---
while cv2.waitKey(33) < 0:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_shape = img.shape
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    current_fig_leaf_pose = False
    dist_hands = 0.0
    hip_center_y = 0.0  # [수정] 엉덩이 Y좌표 (초기화)
    hands_center_y = 0.0  # [수정] 손목 Y좌표 (초기화)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        try:
            # --- 🌟 '무화과 잎 자세' 감지 로직 🌟 ---

            left_wrist = landmarks[mpPose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[mpPose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mpPose.PoseLandmark.RIGHT_HIP.value]

            if (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
                    left_hip.visibility > 0.5 and right_hip.visibility > 0.5):

                dist_hands = get_distance_2d(left_wrist, right_wrist, w, h)

                # [수정] Y좌표 계산
                hip_center_y = (left_hip.y + right_hip.y) / 2.0
                hands_center_y = (left_wrist.y + right_wrist.y) / 2.0

                # 4. 자세 판정 (두 조건)
                if dist_hands < HANDS_CLOSE_THRESHOLD:
                    if hands_center_y > hip_center_y:  # ❗ [조건 2] 손이 엉덩이보다 아래 (y값이 큼)
                        current_fig_leaf_pose = True

            # --- 🌟 상태 지속 시간 카운트 🌟 ---
            if current_fig_leaf_pose:
                fig_leaf_status = True
                fig_leaf_counter += 1

                if fig_leaf_counter >= POSE_FRAME_COUNT and not last_fig_leaf_event:
                    last_fig_leaf_event = True
                    total_fig_leaf_events += 1
                    current_time = time.strftime('%H:%M:%S')
                    print(f"[{current_time}] '무화과 잎 자세' 감지 (총: {total_fig_leaf_events}회)")
            else:
                fig_leaf_status = False
                fig_leaf_counter = 0
                last_fig_leaf_event = False

            # --- 🌟 시각화 🌟 ---
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            if fig_leaf_status:
                cv2.putText(img, f"Fig Leaf Pose ({fig_leaf_counter})", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

            cv2.putText(img, f"Fig Leaf: {total_fig_leaf_events}", (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # [수정] 튜닝을 위한 Y좌표 및 거리 표시
            cv2.putText(img, f"Hand Dist: {dist_hands:.0f}", (400, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Hip Y: {hip_center_y:.2f}", (400, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Hand Y: {hands_center_y:.2f}", (400, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        except Exception as e:
            pass

    cv2.imshow("Speechmate Pose Analysis", img)

# --- 정리 ---
cap.release()
cv2.destroyAllWindows()