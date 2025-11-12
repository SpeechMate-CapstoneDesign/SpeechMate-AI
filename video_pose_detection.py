import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 전역 변수
nose_positions = []
head_angles = []
hand_face_distances = []

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

    if current_angle > -0.15:
        recent_bowing = sum(1 for angle in head_angles[-5:] if angle > -0.15)
        if recent_bowing >= 3:
            head_angles.clear()
            return True

    return False


def detect_hand_to_face(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
    """Holistic을 활용한 정밀한 손-얼굴 접촉 감지"""

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

    # 왼손 체크
    if left_hand_landmarks:
        finger_tips = [
            left_hand_landmarks.landmark[8],   # 검지
            left_hand_landmarks.landmark[12],  # 중지
            left_hand_landmarks.landmark[16],  # 약지
            left_hand_landmarks.landmark[20],  # 새끼
            left_hand_landmarks.landmark[4]    # 엄지
        ]

        for finger in finger_tips:
            distance_x = abs(finger.x - face_center_x)
            distance_y = abs(finger.y - face_center_y)

            if distance_x < face_width * 1.2 and distance_y < face_height * 1.2:
                hand_detected = True
                break

    # 오른손 체크
    if right_hand_landmarks and not hand_detected:
        finger_tips = [
            right_hand_landmarks.landmark[8],
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


# ============================================
# 메인 코드: 웹캠 또는 동영상 파일 선택
# ============================================

# 옵션 1: 웹캠 사용
# video_source = 0

# 옵션 2: mp4 파일 사용
video_source = "hand_to_face.mp4"  # 여기에 파일 경로 입력

cap = cv2.VideoCapture(video_source)

# 동영상 정보 확인
if not cap.isOpened():
    print("❌ 비디오를 열 수 없습니다. 파일 경로를 확인하세요.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"📹 비디오 FPS: {fps}")
print(f"📹 총 프레임 수: {total_frames}")

# 출력 동영상 설정 (선택사항)
save_output = True  # True로 설정하면 결과 저장
if save_output:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_result.mp4', fourcc, fps, (width, height))

frame_count = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("✅ 동영상 처리 완료!")
            break

        frame_count += 1
        print(f"처리 중... {frame_count}/{total_frames} 프레임", end='\r')

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
                print(f"\n✋ 손-얼굴 접촉 감지! (프레임: {frame_count})")

            # 고개 흔들기 감지
            if detect_head_shake(landmarks, image_width):
                print(f"\n🔄 고개 흔들기 감지! (프레임: {frame_count})")

            # 고개 숙이기 감지
            if detect_head_bow(landmarks):
                print(f"\n🙇 고개 숙이기 감지! (프레임: {frame_count})")

        # 프레임 번호 표시
        cv2.putText(image, f"Frame: {frame_count}/{total_frames}", (50, image_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 결과 저장
        if save_output:
            out.write(image)

        # 화면에 표시 (옵션)
        cv2.imshow('MediaPipe Holistic - Video', image)

        # 'q' 키로 중단 가능
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏸️ 사용자가 중단했습니다.")
            break

cap.release()
if save_output:
    out.release()
    print(f"\n💾 결과 저장 완료: output_result.mp4")

cv2.destroyAllWindows()