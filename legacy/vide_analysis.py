import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import timedelta

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 전역 변수
nose_positions = []
head_angles = []
hand_face_distances = []

# 행동 기록
detected_actions = []

def detect_head_shake(landmarks, image_width):
    """고개 흔들기 감지"""
    nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
    nose_x = nose.x * image_width

    nose_positions.append(nose_x)

    if len(nose_positions) > 50:
        nose_positions.pop(0)

    if len(nose_positions) < 30:
        return False

    movements = [nose_positions[i + 1] - nose_positions[i]
                 for i in range(len(nose_positions) - 1)]

    significant_movements = [m for m in movements if abs(m) > 4]

    if len(significant_movements) < 10:
        return False

    direction_changes = 0
    prev_direction = 0

    for movement in significant_movements:
        current_direction = 1 if movement > 0 else -1

        if prev_direction != 0 and current_direction != prev_direction:
            direction_changes += 1

        prev_direction = current_direction

    movement_range = max(nose_positions) - min(nose_positions)

    if direction_changes >= 4 and movement_range > image_width * 0.1:
        nose_positions.clear()
        return True

    return False

def detect_head_bow(landmarks):
    """고개 숙이기 감지"""
    nose = landmarks[mp_holistic.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    head_position = nose.y - shoulder_mid_y

    head_angles.append(head_position)

    if len(head_angles) > 15:
        head_angles.pop(0)

    if len(head_angles) < 10:
        return False

    current_angle = head_angles[-1]

    if current_angle > -0.15:
        recent_bowing = sum(1 for angle in head_angles[-5:] if angle > -0.15)
        if recent_bowing >= 3:
            head_angles.clear()
            return True

    return False

def detect_hand_to_face(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
    """손-얼굴 접촉 감지"""
    if not pose_landmarks or not face_landmarks:
        return False

    nose_tip = face_landmarks.landmark[1]
    forehead = face_landmarks.landmark[10]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]
    chin = face_landmarks.landmark[152]

    face_center_x = nose_tip.x
    face_center_y = nose_tip.y

    face_width = abs(right_cheek.x - left_cheek.x)
    face_height = abs(forehead.y - chin.y)

    hand_detected = False

    if left_hand_landmarks:
        finger_tips = [
            left_hand_landmarks.landmark[8],
            left_hand_landmarks.landmark[12],
            left_hand_landmarks.landmark[16],
            left_hand_landmarks.landmark[20],
            left_hand_landmarks.landmark[4]
        ]

        for finger in finger_tips:
            distance_x = abs(finger.x - face_center_x)
            distance_y = abs(finger.y - face_center_y)

            if distance_x < face_width * 1.2 and distance_y < face_height * 1.2:
                hand_detected = True
                break

    if right_hand_landmarks and not hand_detected:
        finger_tips = [
            right_hand_landmarks.landmark[8],
            right_hand_landmarks.landmark[12],
            right_hand_landmarks.landmark[16],
            right_hand_landmarks.landmark[20],
            right_hand_landmarks.landmark[4]
        ]

        for finger in finger_tips:
            distance_x = abs(finger.x - face_center_x)
            distance_y = abs(finger.y - face_center_y)

            if distance_x < face_width * 1.2 and distance_y < face_height * 1.2:
                hand_detected = True
                break

    hand_face_distances.append(hand_detected)

    if len(hand_face_distances) > 15:
        hand_face_distances.pop(0)

    if len(hand_face_distances) < 10:
        return False

    recent_detections = sum(hand_face_distances[-5:])

    if recent_detections >= 3:
        hand_face_distances.clear()
        return True

    return False

def format_timestamp(seconds):
    """초를 MM:SS 형식으로 변환"""
    return str(timedelta(seconds=int(seconds)))

# ============================================
# 메인 코드
# ============================================

video_source = "s.mp4"

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"📹 비디오 FPS: {fps}")
print(f"📹 총 프레임: {total_frames}")
print(f"⏱️  영상 길이: {format_timestamp(duration)}")
print(f"\n처리 시작...\n")

# 출력 동영상 설정
save_output = True
if save_output:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../output_result.mp4', fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        frame_count += 1
        current_time = frame_count / fps  # 현재 영상 시간 (초)

        # 진행률 표시
        elapsed = time.time() - start_time
        progress = (frame_count / total_frames) * 100
        fps_processing = frame_count / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0

        print(f"진행: {progress:.1f}% | 프레임: {frame_count}/{total_frames} | "
              f"처리속도: {fps_processing:.1f} fps | 남은시간: {int(eta)}초", end='\r')

        image_height, image_width, _ = image.shape

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

        # 제스처 감지
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 손-얼굴 접촉 감지
            if detect_hand_to_face(
                    results.pose_landmarks,
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                    results.face_landmarks
            ):
                detected_actions.append({
                    'action': '손-얼굴 접촉',
                    'time': current_time,
                    'frame': frame_count
                })
                cv2.putText(image, "Hand to Face!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 고개 흔들기 감지
            if detect_head_shake(landmarks, image_width):
                detected_actions.append({
                    'action': '고개 흔들기',
                    'time': current_time,
                    'frame': frame_count
                })
                cv2.putText(image, "Head Shake!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # 고개 숙이기 감지
            if detect_head_bow(landmarks):
                detected_actions.append({
                    'action': '고개 숙이기',
                    'time': current_time,
                    'frame': frame_count
                })
                cv2.putText(image, "Head Bow!", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # 현재 시간 표시
        cv2.putText(image, f"Time: {format_timestamp(current_time)}",
                   (50, image_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(image, f"Frame: {frame_count}/{total_frames}",
                   (50, image_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if save_output:
            out.write(image)

        cv2.imshow('Processing...', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

# 처리 완료
total_time = time.time() - start_time
print(f"\n\n✅ 처리 완료!")
print(f"⏱️  총 처리 시간: {int(total_time // 60)}분 {int(total_time % 60)}초")
print(f"⚡ 평균 처리 속도: {frame_count / total_time:.1f} fps")
print(f"📊 실시간 대비: {(duration / total_time):.2f}배")

# 행동 요약
print(f"\n{'='*60}")
print("📋 행동 감지 요약")
print(f"{'='*60}")

if detected_actions:
    # 행동별로 그룹화
    action_groups = {}
    for action in detected_actions:
        action_type = action['action']
        if action_type not in action_groups:
            action_groups[action_type] = []
        action_groups[action_type].append(action)

    for action_type, actions in action_groups.items():
        print(f"\n🔹 {action_type}: 총 {len(actions)}회")
        for i, action in enumerate(actions[:10], 1):  # 최대 10개만 표시
            print(f"   {i}. {format_timestamp(action['time'])} (프레임: {action['frame']})")
        if len(actions) > 10:
            print(f"   ... 외 {len(actions) - 10}개 더")
else:
    print("감지된 행동이 없습니다.")

print(f"\n💾 결과 저장: output_result.mp4")