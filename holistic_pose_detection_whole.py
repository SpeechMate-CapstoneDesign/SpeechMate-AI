import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque


class GestureDetector:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 히스토리 (deque 사용으로 효율성 향상)
        self.nose_positions = deque(maxlen=50)
        self.head_angles = deque(maxlen=15)
        self.hand_face_distances = deque(maxlen=15)
        self.wrist_angles_left = deque(maxlen=10)
        self.wrist_angles_right = deque(maxlen=10)
        self.scratching_history = {'left': deque(maxlen=20), 'right': deque(maxlen=20)}
        self.rubbing_history = deque(maxlen=30)

        # 감지된 제스처 저장
        self.detected_gestures = []

    def process_frame(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks, image_width):
        """한 프레임의 모든 제스처 감지"""
        self.detected_gestures = []

        if not pose_landmarks:
            return self.detected_gestures

        landmarks = pose_landmarks.landmark

        # === 머리 동작 ===
        self._detect_head_gestures(landmarks, image_width)

        # === 팔 동작 ===
        self._detect_arm_gestures(pose_landmarks)

        # === 손 동작 ===
        self._detect_hand_gestures(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks)

        return self.detected_gestures

    # 머리 동작
    def _detect_head_gestures(self, landmarks, image_width):
        """머리 관련 제스처 감지"""
        # 고개 흔들기
        if self._detect_head_shake(landmarks, image_width):
            self.detected_gestures.append('머리동작/흔들기')

        # 고개 숙이기
        if self._detect_head_bow(landmarks):
            self.detected_gestures.append('머리동작/숙이기')

    def _detect_head_shake(self, landmarks, image_width):
        """고개 흔들기"""
        nose = landmarks[self.mp_holistic.PoseLandmark.NOSE.value]
        nose_x = nose.x * image_width

        self.nose_positions.append(nose_x)

        if len(self.nose_positions) < 30:
            return False

        movements = [self.nose_positions[i + 1] - self.nose_positions[i]
                     for i in range(len(self.nose_positions) - 1)]

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

        movement_range = max(self.nose_positions) - min(self.nose_positions)

        if direction_changes >= 4 and movement_range > image_width * 0.1:
            self.nose_positions.clear()
            return True

        return False

    def _detect_head_bow(self, landmarks):
        """고개 숙이기"""
        nose = landmarks[self.mp_holistic.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        head_position = nose.y - shoulder_mid_y

        self.head_angles.append(head_position)

        if len(self.head_angles) < 10:
            return False

        current_angle = self.head_angles[-1]

        if current_angle > -0.2:
            recent_bowing = sum(1 for angle in list(self.head_angles)[-5:] if angle > -0.2)
            if recent_bowing >= 3:
                self.head_angles.clear()
                return True

        return False


    # 팔 동작
    def _detect_arm_gestures(self, pose_landmarks):
        """팔 관련 제스처 감지"""
        landmarks = pose_landmarks.landmark
        # 뒷짐
        if self._detect_hands_behind_back(landmarks):
            self.detected_gestures.append('팔동작/뒷짐')


    def _detect_hands_behind_back(self, landmarks):
        """뒷짐"""
        left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]

        center_z = (left_hip.z + right_hip.z) / 2
        hip_y = (left_hip.y + right_hip.y) / 2

        left_behind = left_wrist.z > center_z + 0.05
        right_behind = right_wrist.z > center_z + 0.05

        left_at_hip = abs(left_wrist.y - hip_y) < 0.15
        right_at_hip = abs(right_wrist.y - hip_y) < 0.15

        return left_behind and right_behind and left_at_hip and right_at_hip

    # 손 동작

    def _detect_hand_gestures(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        """손 관련 제스처 감지"""

        # 손-얼굴 접촉 (부위별)
        face_region = self._detect_hand_to_face_region(pose_landmarks, left_hand_landmarks,
                                                       right_hand_landmarks, face_landmarks)
        if face_region:
            self.detected_gestures.append(face_region)

        # 손 비비기
        if self._detect_hand_rubbing(left_hand_landmarks, right_hand_landmarks):
            self.detected_gestures.append('손 비비기')


        # 몸 긁기
        scratch_location = self._detect_body_scratching(pose_landmarks, left_hand_landmarks,
                                                        right_hand_landmarks)
        if scratch_location:
            self.detected_gestures.append(f'손동작/몸 긁기/{scratch_location}')

    def _detect_hand_to_face_region(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        """손-얼굴 접촉 (부위별 분류)"""

        if not pose_landmarks or not face_landmarks:
            return None

        nose_tip = face_landmarks.landmark[1]
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        left_ear = pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR.value]
        right_ear = pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR.value]

        face_center_x = nose_tip.x
        face_center_y = nose_tip.y

        face_width_half = abs(right_ear.x - left_ear.x) / 2
        face_height_top = abs(nose_tip.y - forehead.y)
        face_height_bottom = abs(chin.y - nose_tip.y)

        forehead_y = forehead.y
        chin_y = chin.y
        ear_zone_threshold = face_width_half * 0.7

        def check_hand(hand_landmarks, hand_name):
            if not hand_landmarks:
                return None

            for idx in range(21):
                point = hand_landmarks.landmark[idx]

                if point.y > chin_y:
                    continue

                distance_x = abs(point.x - face_center_x)
                distance_y_from_nose = point.y - face_center_y

                x_in_range = distance_x < face_width_half * 1.2

                if distance_y_from_nose < 0:
                    y_in_range = abs(distance_y_from_nose) < face_height_top * 1.35
                else:
                    y_in_range = distance_y_from_nose < face_height_bottom * 1.15

                if x_in_range and y_in_range:
                    # 부위 분류
                    if point.y < forehead_y:
                        return '손동작/머리'

                    distance_from_center = abs(point.x - face_center_x)
                    if distance_from_center > ear_zone_threshold:
                        if point.x < face_center_x:
                            return '손동작/왼쪽 귀'
                        else:
                            return '손동작/오른쪽 귀'

                    if point.y < nose_tip.y - face_height_top * 0.3:
                        return '손동작/이마'
                    elif point.y < nose_tip.y + face_height_bottom * 0.2:
                        return '손동작/코'
                    else:
                        return '손동작/턱'

            return None

        # 왼손 체크
        region = check_hand(left_hand_landmarks, "왼손")
        if region:
            self.hand_face_distances.append(True)
        else:
            # 오른손 체크
            region = check_hand(right_hand_landmarks, "오른손")
            if region:
                self.hand_face_distances.append(True)
            else:
                self.hand_face_distances.append(False)

        if len(self.hand_face_distances) >= 10:
            recent = sum(list(self.hand_face_distances)[-5:])
            if recent >= 3:
                self.hand_face_distances.clear()
                return region

        return None

    def _detect_hand_rubbing(self, left_hand_landmarks, right_hand_landmarks):
        """양손 비비기"""

        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        # 양손 손바닥 중심
        def get_palm_center(hand_landmarks):
            palm_points = [0, 5, 9, 13, 17]
            x = sum(hand_landmarks.landmark[i].x for i in palm_points) / len(palm_points)
            y = sum(hand_landmarks.landmark[i].y for i in palm_points) / len(palm_points)
            return (x, y)

        left_palm = get_palm_center(left_hand_landmarks)
        right_palm = get_palm_center(right_hand_landmarks)

        distance = math.sqrt((left_palm[0] - right_palm[0]) ** 2 +
                             (left_palm[1] - right_palm[1]) ** 2)

        self.rubbing_history.append({
            'left': left_palm,
            'right': right_palm,
            'distance': distance
        })

        if len(self.rubbing_history) < 20:
            return False

        # 양손이 가까운지
        recent_distances = [f['distance'] for f in list(self.rubbing_history)[-20:]]
        avg_distance = sum(recent_distances) / len(recent_distances)

        if avg_distance > 0.15:
            return False

        # 반대 방향 움직임
        opposite_count = 0
        history_list = list(self.rubbing_history)

        for i in range(1, len(history_list)):
            prev = history_list[i - 1]
            curr = history_list[i]

            left_dx = curr['left'][0] - prev['left'][0]
            right_dx = curr['right'][0] - prev['right'][0]

            if left_dx * right_dx < 0:
                opposite_count += 1

        return opposite_count >= 6

    def _detect_body_scratching(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        """몸 긁기"""

        if not pose_landmarks:
            return None

        landmarks = pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]

        torso_left = min(left_shoulder.x, left_hip.x)
        torso_right = max(right_shoulder.x, right_hip.x)
        torso_top = min(left_shoulder.y, right_shoulder.y)
        torso_bottom = max(left_hip.y, right_hip.y)

        def check_scratching(hand_landmarks, hand_key):
            if not hand_landmarks:
                return None

            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]

            hand_x = (wrist.x + index_tip.x) / 2
            hand_y = (wrist.y + index_tip.y) / 2

            in_torso = (torso_left < hand_x < torso_right and
                        torso_top < hand_y < torso_bottom)

            if not in_torso:
                return None

            self.scratching_history[hand_key].append((hand_x, hand_y))

            if len(self.scratching_history[hand_key]) < 15:
                return None

            positions = list(self.scratching_history[hand_key])

            # 속도 계산
            velocities = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i - 1][0]
                dy = positions[i][1] - positions[i - 1][1]
                velocity = math.sqrt(dx ** 2 + dy ** 2)
                velocities.append(velocity)

            avg_velocity = sum(velocities) / len(velocities)

            # 방향 전환
            direction_changes = 0
            for i in range(2, len(positions)):
                prev_dy = positions[i - 1][1] - positions[i - 2][1]
                curr_dy = positions[i][1] - positions[i - 1][1]

                if prev_dy * curr_dy < 0:
                    direction_changes += 1

            # 범위
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            is_scratching = (avg_velocity > 0.015 and
                             x_range < 0.15 and y_range < 0.2 and
                             direction_changes >= 3)

            if is_scratching:
                # 부위 특정
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_y = (left_hip.y + right_hip.y) / 2

                if hand_y < shoulder_y + (hip_y - shoulder_y) * 0.3:
                    return "가슴"
                elif hand_y < shoulder_y + (hip_y - shoulder_y) * 0.7:
                    return "배"
                else:
                    return "배아래"

            return None

        # 왼손 체크
        location = check_scratching(left_hand_landmarks, 'left')
        if location:
            return location

        # 오른손 체크
        location = check_scratching(right_hand_landmarks, 'right')
        if location:
            return location

        return None

    def get_detected_gestures(self):
        """감지된 제스처 목록 반환"""
        return self.detected_gestures


# ========================================
# 메인 실행 코드
# ========================================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    print("📹 웹캠 시작!")
    print("ESC 키를 누르면 종료됩니다.")

    # 제스처 감지기 초기화
    detector = GestureDetector()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image_height, image_width, _ = image.shape

            # MediaPipe 처리
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
                gestures = detector.process_frame(
                    results.pose_landmarks,
                    results.left_hand_landmarks,
                    results.right_hand_landmarks,
                    results.face_landmarks,
                    image_width
                )

                # 감지된 제스처 화면에 표시
                y_offset = 50
                for gesture in gestures:
                    print(f"✅ {gesture}")
                    y_offset += 40

            cv2.imshow('Gesture Detection', cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 프로그램 종료")


if __name__ == "__main__":
    main()