import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


nose_positions = [] # 코 위치
head_angles = [] # 머리 위치
hand_face_distances = [] # 손과 얼굴 사이의 거리 (Boolean 리스트)

def detect_head_shake(landmarks, image_width):
    """고개 흔들기 감지 - 덜 민감한 버전"""
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

    if current_angle > -0.2:
        recent_bowing = sum(1 for angle in head_angles[-5:] if angle > -0.2)
        if recent_bowing >= 3:
            head_angles.clear()
            return True

    return False


def detect_hand_to_face(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
    """손이 어느 부위에 있는지 구분해서 출력"""

    if not pose_landmarks or not face_landmarks:
        return False

    # Face Mesh의 얼굴 주요 포인트
    nose_tip = face_landmarks.landmark[1]
    forehead = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]

    # Pose의 귀 좌표
    left_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR.value]
    right_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR.value]

    # 얼굴 중심은 코 기준
    face_center_x = nose_tip.x
    face_center_y = nose_tip.y

    # 얼굴 크기 계산
    face_width_half = abs(right_ear.x - left_ear.x) / 2
    face_height_top = abs(nose_tip.y - forehead.y)
    face_height_bottom = abs(chin.y - nose_tip.y)

    # 주요 y 좌표들
    forehead_y = forehead.y
    chin_y = chin.y

    # 귀 영역 정의 (귀는 코보다 옆에 있음)
    left_ear_x = left_ear.x
    right_ear_x = right_ear.x
    ear_zone_threshold = face_width_half * 0.7  # 귀 영역 판정 기준

    hand_detected = False
    detected_region = None  # 감지된 부위 저장

    def check_hand_region(hand_landmarks, hand_name):
        """손의 위치를 판별하고 어느 부위인지 반환"""
        if not hand_landmarks:
            return None

        # 21개 모든 랜드마크 체크
        for idx in range(21):
            point = hand_landmarks.landmark[idx]

            # 턱보다 아래면 스킵
            if point.y > chin_y:
                continue

            distance_x = abs(point.x - face_center_x)
            distance_y_from_nose = point.y - face_center_y

            # 좌우 범위 체크
            x_in_range = distance_x < face_width_half * 1.2

            # 상하 범위 체크
            if distance_y_from_nose < 0:  # 이마 쪽
                y_in_range = abs(distance_y_from_nose) < face_height_top * 1.35
            else:  # 턱 쪽
                y_in_range = distance_y_from_nose < face_height_bottom * 1.15

            if x_in_range and y_in_range:
                # 어느 부위인지 판별
                region = classify_region(point, hand_name)
                return region

        return None

    def classify_region(point, hand_name):
        """포인트가 어느 부위에 속하는지 분류"""

        # 1. 이마 위 (머리/헤어 영역)
        if point.y < forehead_y:
            return f"👤 {hand_name} → 머리/이마 위"

        # 2. 귀 영역 (얼굴 중심에서 멀리 떨어진 좌우)
        distance_from_center = abs(point.x - face_center_x)

        if distance_from_center > ear_zone_threshold:
            if point.x < face_center_x:  # 왼쪽
                return f"👂 {hand_name} → 왼쪽 귀"
            else:  # 오른쪽
                return f"👂 {hand_name} → 오른쪽 귀"

        # 3. 얼굴 영역 (세부 구분)
        # 이마 영역
        if point.y < nose_tip.y - face_height_top * 0.5:
            return f"😊 {hand_name} → 이마"

        # 눈/코 영역 (코 위아래)
        elif point.y < nose_tip.y + face_height_bottom * 0.3:
            return f"👃 {hand_name} → 눈/코"

        # 입/턱 영역
        else:
            return f"😮 {hand_name} → 입/턱"

    # 왼손 체크
    left_region = check_hand_region(left_hand_landmarks, "왼손")
    if left_region:
        hand_detected = True
        detected_region = left_region

    # 오른손 체크
    if not hand_detected:
        right_region = check_hand_region(right_hand_landmarks, "오른손")
        if right_region:
            hand_detected = True
            detected_region = right_region

    # 감지 결과 출력
    if hand_detected and detected_region:
        print(detected_region)

    # 히스토리 관리
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

# ============================================
# 메인 코드: 웹캠 사용
# ============================================

cap = cv2.VideoCapture(0)

# 카메라 설정 (옵션)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("📹 웹캠 시작!")
print("ESC 키를 누르면 종료됩니다.")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 찾을 수 없습니다.")
            continue

        image_height, image_width, _ = image.shape

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())

        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())

        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())

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
                cv2.putText(image, "Hand to Face!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("✋ 손-얼굴 접촉 감지!")

            # 고개 흔들기 감지
            if detect_head_shake(landmarks, image_width):
                cv2.putText(image, "Head Shake!", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                print("🔄 고개 흔들기 감지!")

            # 고개 숙이기 감지
            if detect_head_bow(landmarks):
                cv2.putText(image, "Head Bow!", (50, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("🙇 고개 숙이기 감지!")

        # 좌우 반전
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

        # ESC 키로 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("👋 프로그램 종료")