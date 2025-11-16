import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque


class GestureDetector:
    """제스처 감지 통합 클래스 - 개선 버전"""

    def __init__(self):
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 히스토리
        self.nose_positions = deque(maxlen=50)
        self.head_angles = deque(maxlen=15)
        self.hand_face_distances = deque(maxlen=15)

        # 팔 관련
        self.arms_crossed_history = deque(maxlen=20)
        self.behind_back_history = deque(maxlen=20)
        self.arm_movement_history = deque(maxlen=30)  # 무의미 반동

        # 손 관련
        self.hand_distance_history = deque(maxlen=30)  # 손 비비기 개선
        self.scratching_history = {'left': deque(maxlen=25), 'right': deque(maxlen=25)}
        self.wrist_angles = {'left': deque(maxlen=15), 'right': deque(maxlen=15)}  # 손목 꺾기

        # 감지된 제스처
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
        self._detect_arm_gestures(pose_landmarks, left_hand_landmarks, right_hand_landmarks)

        # === 손 동작 ===
        self._detect_hand_gestures(pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks)

        return self.detected_gestures

    # ========================================
    # 머리 동작
    # ========================================

    def _detect_head_gestures(self, landmarks, image_width):
        """머리 관련 제스처 감지"""
        if self._detect_head_shake(landmarks, image_width):
            self.detected_gestures.append('머리동작/흔들기')

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

    # ========================================
    # 팔 동작 (개선)
    # ========================================

    def _detect_arm_gestures(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        """팔 관련 제스처 감지"""
        landmarks = pose_landmarks.landmark

        # 팔짱 끼기 (개선)
        if self._detect_arms_crossed_improved(landmarks):
            self.detected_gestures.append('팔짱끼기')

        # 뒷짐 (개선)
        if self._detect_hands_behind_back_improved(landmarks, left_hand_landmarks, right_hand_landmarks):
            self.detected_gestures.append('팔동작/뒷짐')

        # 무의미 반동
        if self._detect_arm_fidgeting(landmarks):
            self.detected_gestures.append('팔동작/무의미반동')

    def _detect_arms_crossed_improved(self, landmarks):
        """
        팔짱 끼기 개선 버전

        로직:
        1. 양 손목이 반대편으로 교차
        2. 양 손목이 가슴 높이 (어깨-엉덩이 사이 상위 50%)
        3. 양 팔꿈치가 몸통 옆에 있음 (벌어지지 않음)
        4. 양 손목 사이 거리가 적당히 가까움 (팔을 교차시킴)
        """
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]

        # 1. 손목 교차 확인 (더 엄격하게)
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)

        # 왼손목이 중심선보다 오른쪽으로, 어깨 너비의 10% 이상
        left_wrist_crossed = left_wrist.x > shoulder_center_x + shoulder_width * 0.1
        # 오른손목이 중심선보다 왼쪽으로, 어깨 너비의 10% 이상
        right_wrist_crossed = right_wrist.x < shoulder_center_x - shoulder_width * 0.1

        # 2. 높이 확인 (가슴 높이)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        chest_y_top = shoulder_y + (hip_y - shoulder_y) * 0.2
        chest_y_bottom = shoulder_y + (hip_y - shoulder_y) * 0.6

        left_at_chest = chest_y_top < left_wrist.y < chest_y_bottom
        right_at_chest = chest_y_top < right_wrist.y < chest_y_bottom

        # 3. 팔꿈치가 몸통 가까이
        left_elbow_close = abs(left_elbow.x - left_shoulder.x) < shoulder_width * 0.5
        right_elbow_close = abs(right_elbow.x - right_shoulder.x) < shoulder_width * 0.5

        # 4. 양 손목 거리 (너무 멀면 팔짱이 아님)
        wrist_distance = math.sqrt(
            (left_wrist.x - right_wrist.x) ** 2 +
            (left_wrist.y - right_wrist.y) ** 2
        )
        wrists_close = wrist_distance < shoulder_width * 1.2

        # 모든 조건 만족
        is_crossed = (
                left_wrist_crossed and right_wrist_crossed and
                left_at_chest and right_at_chest and
                left_elbow_close and right_elbow_close and
                wrists_close
        )

        self.arms_crossed_history.append(is_crossed)

        if len(self.arms_crossed_history) >= 15:
            recent = sum(list(self.arms_crossed_history)[-10:])
            # 10프레임 중 8프레임 이상 (더 엄격)
            if recent >= 8:
                return True

        return False

    def _detect_hands_behind_back_improved(self, landmarks, left_hand_landmarks, right_hand_landmarks):
        """
        뒷짐 개선 버전

        로직:
        1. 양 손목이 엉덩이보다 뒤 (z축)
        2. 양 손목이 엉덩이 높이 근처
        3. 양 손목이 몸 중심 근처 (x축)
        4. 손이 감지되지 않거나 visibility 낮음 (뒤에 가려짐)
        """
        left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

        # 1. z축 확인 (뒤에 있음)
        center_z = (left_hip.z + right_hip.z) / 2
        left_behind = left_wrist.z > center_z + 0.03
        right_behind = right_wrist.z > center_z + 0.03

        # 2. 높이 확인 (엉덩이 근처)
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        torso_height = abs(hip_y - shoulder_y)

        left_at_hip = abs(left_wrist.y - hip_y) < torso_height * 0.3
        right_at_hip = abs(right_wrist.y - hip_y) < torso_height * 0.3

        # 3. x축 확인 (몸 중심 근처)
        center_x = (left_hip.x + right_hip.x) / 2
        hip_width = abs(right_hip.x - left_hip.x)

        left_centered = abs(left_wrist.x - center_x) < hip_width * 0.8
        right_centered = abs(right_wrist.x - center_x) < hip_width * 0.8

        # 4. 손 visibility 확인 (선택적)
        hands_hidden = (left_hand_landmarks is None) or (right_hand_landmarks is None)

        is_behind = (
                left_behind and right_behind and
                left_at_hip and right_at_hip and
                left_centered and right_centered
        )

        self.behind_back_history.append(is_behind)

        if len(self.behind_back_history) >= 15:
            recent = sum(list(self.behind_back_history)[-10:])
            # 10프레임 중 7프레임 이상
            if recent >= 7:
                return True

        return False

    def _detect_arm_fidgeting(self, landmarks):
        """
        팔 무의미 반동

        로직:
        1. 손목의 빠른 움직임 (작은 범위 내에서)
        2. 방향 전환이 많음 (왔다갔다)
        3. 손목이 몸통 근처 (공중에서 흔드는 것)
        4. 지속적인 패턴
        """
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

        # 양 손목의 평균 위치
        avg_wrist_x = (left_wrist.x + right_wrist.x) / 2
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2

        self.arm_movement_history.append((avg_wrist_x, avg_wrist_y))

        if len(self.arm_movement_history) < 25:
            return False

        positions = list(self.arm_movement_history)

        # 1. 속도 계산
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            velocity = math.sqrt(dx ** 2 + dy ** 2)
            velocities.append(velocity)

        avg_velocity = sum(velocities) / len(velocities)

        # 2. 방향 전환 횟수
        direction_changes = 0
        for i in range(2, len(positions)):
            # x 방향 전환
            prev_dx = positions[i - 1][0] - positions[i - 2][0]
            curr_dx = positions[i][0] - positions[i - 1][0]

            if prev_dx * curr_dx < 0 and abs(prev_dx) > 0.005 and abs(curr_dx) > 0.005:
                direction_changes += 1

        # 3. 움직임 범위 (좁은 범위)
        x_coords = [p[0] for p in positions[-20:]]
        y_coords = [p[1] for p in positions[-20:]]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # 4. 어깨 근처 (손이 몸통 주변)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        near_body = abs(avg_wrist_y - shoulder_y) < 0.3

        # 무의미 반동 조건:
        # - 적당한 속도
        # - 많은 방향 전환
        # - 좁은 범위
        # - 몸통 근처
        is_fidgeting = (
                0.008 < avg_velocity < 0.04 and
                direction_changes >= 5 and
                x_range < 0.2 and y_range < 0.2 and
                near_body
        )

        return is_fidgeting

    # ========================================
    # 손 동작 (개선)
    # ========================================

    def _detect_hand_gestures(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        """손 관련 제스처 감지"""

        # 손-얼굴 접촉 (개선)
        face_region = self._detect_hand_to_face_improved(pose_landmarks, left_hand_landmarks,
                                                         right_hand_landmarks, face_landmarks)
        if face_region:
            self.detected_gestures.append(face_region)

        # 손 비비기 (개선)
        if self._detect_hand_rubbing_improved(left_hand_landmarks, right_hand_landmarks):
            self.detected_gestures.append('손 비비기')

        # 몸 긁기 (개선)
        scratch_location = self._detect_body_scratching_improved(pose_landmarks, left_hand_landmarks,
                                                                 right_hand_landmarks)
        if scratch_location:
            self.detected_gestures.append(f'손동작/몸 긁기/{scratch_location}')

        # 손목 꺾기 (새로 추가)
        wrist_bend = self._detect_wrist_bending(left_hand_landmarks, right_hand_landmarks)
        if wrist_bend:
            self.detected_gestures.append(f'손목 꺾기/{wrist_bend}')

    def _detect_hand_to_face_improved(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        """
        손-얼굴 접촉 개선 버전

        로직:
        1. 얼굴 영역을 더 정확하게 정의 (귀 포함)
        2. 손가락 끝 5개만 체크 (손바닥 제외)
        3. 턱보다 위에만 감지
        4. 부위별 명확한 구분
        """
        if not pose_landmarks or not face_landmarks:
            return None

        # 얼굴 랜드마크
        nose_tip = face_landmarks.landmark[1]
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        # Pose 랜드마크
        left_ear = pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR.value]
        right_ear = pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR.value]

        # 얼굴 기준점
        face_center_x = nose_tip.x
        face_center_y = nose_tip.y

        # 얼굴 크기
        face_width = abs(right_ear.x - left_ear.x)
        face_height_top = abs(nose_tip.y - forehead.y)
        face_height_bottom = abs(chin.y - nose_tip.y)

        # 경계값
        forehead_y = forehead.y
        chin_y = chin.y
        left_ear_x = left_ear.x
        right_ear_x = right_ear.x

        def check_hand(hand_landmarks, hand_name):
            if not hand_landmarks:
                return None

            # 손가락 끝만 체크
            fingertips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 새끼

            for tip_idx in fingertips:
                point = hand_landmarks.landmark[tip_idx]

                # 턱보다 아래면 스킵
                if point.y > chin_y:
                    continue

                # 1. 머리/이마 위 (forehead보다 위)
                if point.y < forehead_y - face_height_top * 0.2:
                    # 얼굴 폭 안에 있어야 함
                    if left_ear_x - face_width * 0.3 < point.x < right_ear_x + face_width * 0.3:
                        return '손동작/머리'
                    continue

                # 2. 귀 영역 (얼굴 중심에서 멀리)
                distance_from_center = abs(point.x - face_center_x)

                if distance_from_center > face_width * 0.4:
                    # 귀 높이 (이마 ~ 코)
                    if forehead_y < point.y < nose_tip.y + face_height_bottom * 0.2:
                        if point.x < face_center_x:
                            return '손동작/왼쪽 귀'
                        else:
                            return '손동작/오른쪽 귀'
                    continue

                # 3. 얼굴 중앙 영역 (세부 구분)
                # 얼굴 폭 안에 있는지 확인
                if distance_from_center > face_width * 0.5:
                    continue

                # 이마 (forehead ~ 코 위 30%)
                if point.y < nose_tip.y - face_height_top * 0.3:
                    return '손동작/이마'

                # 코 (코 위 30% ~ 코 아래 20%)
                elif point.y < nose_tip.y + face_height_bottom * 0.2:
                    return '손동작/코'

                # 턱 (코 아래 20% ~ 턱)
                else:
                    return '손동작/턱'

            return None

        # 왼손 체크
        region = check_hand(left_hand_landmarks, "왼손")
        if not region:
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

    def _detect_hand_rubbing_improved(self, left_hand_landmarks, right_hand_landmarks):
        """
        양손 비비기 개선 버전

        로직:
        1. 양 손바닥이 가까이 (거리 < 0.1)
        2. 반대 방향 움직임 (왼손 →, 오른손 ←)
        3. 적당한 속도
        4. 손가락이 펴져 있음 (깍지 X)
        """
        if not left_hand_landmarks or not right_hand_landmarks:
            return False

        # 손바닥 중심
        def get_palm_center(hand_landmarks):
            palm_points = [0, 5, 9, 13, 17]
            x = sum(hand_landmarks.landmark[i].x for i in palm_points) / len(palm_points)
            y = sum(hand_landmarks.landmark[i].y for i in palm_points) / len(palm_points)
            z = sum(hand_landmarks.landmark[i].z for i in palm_points) / len(palm_points)
            return (x, y, z)

        # 손가락이 펴져 있는지 확인 (깍지 감지)
        def fingers_extended(hand_landmarks):
            # 중지 끝과 손목 거리
            wrist = hand_landmarks.landmark[0]
            middle_tip = hand_landmarks.landmark[12]

            distance = math.sqrt(
                (middle_tip.x - wrist.x) ** 2 +
                (middle_tip.y - wrist.y) ** 2
            )

            # 손가락이 펴져 있으면 거리가 큼
            return distance > 0.15

        left_palm = get_palm_center(left_hand_landmarks)
        right_palm = get_palm_center(right_hand_landmarks)

        # 양손 거리 (3D)
        distance = math.sqrt(
            (left_palm[0] - right_palm[0]) ** 2 +
            (left_palm[1] - right_palm[1]) ** 2 +
            (left_palm[2] - right_palm[2]) ** 2
        )

        # 손가락 펴짐 확인
        left_extended = fingers_extended(left_hand_landmarks)
        right_extended = fingers_extended(right_hand_landmarks)

        self.hand_distance_history.append({
            'left': left_palm,
            'right': right_palm,
            'distance': distance,
            'extended': left_extended and right_extended
        })

        if len(self.hand_distance_history) < 20:
            return False

        history_list = list(self.hand_distance_history)

        # 1. 양손이 가까운지
        recent_distances = [h['distance'] for h in history_list[-15:]]
        avg_distance = sum(recent_distances) / len(recent_distances)

        if avg_distance > 0.12:  # 너무 멀면 비비기 아님
            return False

        # 2. 손가락이 펴져 있는지 (깍지 제외)
        extended_count = sum(1 for h in history_list[-10:] if h['extended'])
        if extended_count < 7:  # 10프레임 중 7프레임 이상 펴져 있어야
            return False

        # 3. 반대 방향 움직임
        opposite_count = 0

        for i in range(1, len(history_list)):
            prev = history_list[i - 1]
            curr = history_list[i]

            left_dx = curr['left'][0] - prev['left'][0]
            right_dx = curr['right'][0] - prev['right'][0]

            # 반대 방향이고 움직임이 충분히 큰 경우
            if left_dx * right_dx < 0 and abs(left_dx) > 0.003 and abs(right_dx) > 0.003:
                opposite_count += 1

        # 4. 속도 확인 (너무 빠르거나 느리면 안됨)
        velocities = []
        for i in range(1, len(history_list)):
            prev = history_list[i - 1]
            curr = history_list[i]

            left_vel = math.sqrt(
                (curr['left'][0] - prev['left'][0]) ** 2 +
                (curr['left'][1] - prev['left'][1]) ** 2
            )
            velocities.append(left_vel)

        avg_velocity = sum(velocities) / len(velocities)

        # 비비기 조건:
        # - 가까움
        # - 손가락 펴짐
        # - 반대 방향 움직임 많음
        # - 적당한 속도
        is_rubbing = (
                avg_distance < 0.12 and
                extended_count >= 7 and
                opposite_count >= 8 and
                0.01 < avg_velocity < 0.05
        )

        return is_rubbing

    def _detect_body_scratching_improved(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        """
        몸 긁기 개선 버전

        로직:
        1. 손이 몸통 영역에 있음
        2. 빠른 왕복 운동 (속도 > 0.02)
        3. 좁은 범위 (< 0.1)
        4. 방향 전환 많음 (>= 4회)
        """
        if not pose_landmarks:
            return None

        landmarks = pose_landmarks.landmark
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]

        # 몸통 영역 (약간 여유 있게)
        torso_left = min(left_shoulder.x, left_hip.x) - 0.05
        torso_right = max(right_shoulder.x, right_hip.x) + 0.05
        torso_top = min(left_shoulder.y, right_shoulder.y)
        torso_bottom = max(left_hip.y, right_hip.y)

        def check_scratching(hand_landmarks, hand_key):
            if not hand_landmarks:
                return None

            # 손가락 끝들의 평균 위치
            fingertips = [8, 12, 16]  # 검지, 중지, 약지

            hand_x = sum(hand_landmarks.landmark[i].x for i in fingertips) / len(fingertips)
            hand_y = sum(hand_landmarks.landmark[i].y for i in fingertips) / len(fingertips)

            # 몸통 영역 확인
            in_torso = (torso_left < hand_x < torso_right and
                        torso_top < hand_y < torso_bottom)

            if not in_torso:
                # 영역 밖이면 히스토리 초기화
                if len(self.scratching_history[hand_key]) > 0:
                    self.scratching_history[hand_key].clear()
                return None

            self.scratching_history[hand_key].append((hand_x, hand_y))

            if len(self.scratching_history[hand_key]) < 18:
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

            # 방향 전환 (y축 - 위아래로 긁기)
            direction_changes = 0
            for i in range(2, len(positions)):
                prev_dy = positions[i - 1][1] - positions[i - 2][1]
                curr_dy = positions[i][1] - positions[i - 1][1]

                # 방향 전환이고 움직임이 충분한 경우
                if prev_dy * curr_dy < 0 and abs(prev_dy) > 0.003 and abs(curr_dy) > 0.003:
                    direction_changes += 1

            # 범위
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)

            # 긁기 조건 (더 엄격)
            is_scratching = (
                    avg_velocity > 0.02 and  # 빠른 속도
                    x_range < 0.1 and  # 좁은 x 범위
                    y_range < 0.15 and  # 좁은 y 범위
                    direction_changes >= 4  # 많은 왕복
            )

            if is_scratching:
                # 부위 특정
                shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_y = (left_hip.y + right_hip.y) / 2
                center_x = (left_shoulder.x + right_shoulder.x) / 2

                # 좌우
                if hand_x < center_x - 0.05:
                    side = "왼쪽"
                elif hand_x > center_x + 0.05:
                    side = "오른쪽"
                else:
                    side = "중앙"

                # 상하
                if hand_y < shoulder_y + (hip_y - shoulder_y) * 0.35:
                    part = "가슴"
                elif hand_y < shoulder_y + (hip_y - shoulder_y) * 0.7:
                    part = "배"
                else:
                    part = "아래배"

                self.scratching_history[hand_key].clear()
                return f"{side} {part}"

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

    def _detect_wrist_bending(self, left_hand_landmarks, right_hand_landmarks):
        """
        손목 꺾기 개선 버전

        로직:
        1. 손목(0) - 중지기저(9) 벡터 (팔뚝 방향)
        2. 중지기저(9) - 중지끝(12) 벡터 (손가락 방향)
        3. 두 벡터의 각도가 140도 미만이면 꺾임
        4. 손바닥 평면도 함께 고려 (옆으로 꺾기 vs 앞으로 꺾기)
        """

        def calculate_wrist_bend(hand_landmarks, hand_name):
            if not hand_landmarks:
                return None

            # 주요 포인트
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]  # 검지 기저
            middle_mcp = hand_landmarks.landmark[9]  # 중지 기저
            ring_mcp = hand_landmarks.landmark[13]  # 약지 기저
            pinky_mcp = hand_landmarks.landmark[17]  # 새끼 기저

            # 손바닥 중심 (4개 손가락 기저의 평균)
            palm_center_x = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
            palm_center_y = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4
            palm_center_z = (index_mcp.z + middle_mcp.z + ring_mcp.z + pinky_mcp.z) / 4

            # 벡터 1: 손목 → 손바닥 중심 (팔뚝 연장선)
            vec1_x = palm_center_x - wrist.x
            vec1_y = palm_center_y - wrist.y
            vec1_z = palm_center_z - wrist.z

            # 벡터 2: 손바닥 중심 → 중지 끝 (손가락 방향)
            middle_tip = hand_landmarks.landmark[12]
            vec2_x = middle_tip.x - palm_center_x
            vec2_y = middle_tip.y - palm_center_y
            vec2_z = middle_tip.z - palm_center_z

            # 3D 각도 계산
            dot_product = vec1_x * vec2_x + vec1_y * vec2_y + vec1_z * vec2_z
            mag1 = math.sqrt(vec1_x ** 2 + vec1_y ** 2 + vec1_z ** 2)
            mag2 = math.sqrt(vec2_x ** 2 + vec2_y ** 2 + vec2_z ** 2)

            if mag1 < 0.01 or mag2 < 0.01:  # 너무 가까우면 무시
                return None

            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))

            angle = math.degrees(math.acos(cos_angle))

            # 꺾임 방향 판단 (2D 평면에서)
            # y축 기준 (위아래)
            bend_direction = None

            # 앞으로 꺾기 (손등이 위로)
            if vec2_y < vec1_y * 0.5:  # 손가락이 상대적으로 위로
                bend_direction = "앞으로"
            # 뒤로 꺾기 (손바닥이 위로)
            elif vec2_y > vec1_y * 1.5:  # 손가락이 상대적으로 아래로
                bend_direction = "뒤로"
            # 옆으로 꺾기 (x축)
            else:
                x_diff = abs(vec2_x - vec1_x)
                if x_diff > abs(vec1_x) * 0.5:
                    bend_direction = "옆으로"

            return {
                'angle': angle,
                'direction': bend_direction,
                'hand': hand_name
            }

        # 양손 체크
        left_result = calculate_wrist_bend(left_hand_landmarks, '왼손')
        right_result = calculate_wrist_bend(right_hand_landmarks, '오른손')

        # 결과 저장
        if left_result:
            self.wrist_angles['left'].append(left_result)

        if right_result:
            self.wrist_angles['right'].append(right_result)

        # 감지 (더 엄격한 조건)
        for hand_key in ['left', 'right']:
            if len(self.wrist_angles[hand_key]) >= 12:  # 12프레임 필요
                recent_data = list(self.wrist_angles[hand_key])[-12:]

                # 평균 각도
                avg_angle = sum(d['angle'] for d in recent_data) / len(recent_data)

                # 각도 변화가 적어야 함 (안정적으로 유지)
                angles = [d['angle'] for d in recent_data]
                angle_variance = sum((a - avg_angle) ** 2 for a in angles) / len(angles)
                angle_std = math.sqrt(angle_variance)

                # 방향이 일정해야 함
                directions = [d['direction'] for d in recent_data if d['direction']]
                if len(directions) < 8:  # 방향이 감지된 경우가 8개 이상
                    continue

                most_common_direction = max(set(directions), key=directions.count)
                direction_consistency = directions.count(most_common_direction) / len(directions)

                # 손목 꺾임 조건:
                # 1. 평균 각도가 135도 미만
                # 2. 각도 변화가 작음 (표준편차 < 10도)
                # 3. 방향이 일정함 (70% 이상)
                # 4. 방향이 감지됨

                is_bent = (
                        avg_angle < 135 and
                        angle_std < 10 and
                        direction_consistency > 0.7 and
                        most_common_direction is not None
                )

                if is_bent:
                    hand_name = '왼손' if hand_key == 'left' else '오른손'

                    # 심각도 판단
                    if avg_angle < 110:
                        severity = "매우"
                    elif avg_angle < 120:
                        severity = "많이"
                    else:
                        severity = "약간"

                    # 히스토리 일부만 초기화 (계속 감지 방지)
                    # 전체 초기화 대신 절반만 제거
                    for _ in range(len(self.wrist_angles[hand_key]) // 2):
                        if len(self.wrist_angles[hand_key]) > 0:
                            self.wrist_angles[hand_key].popleft()

                    return f'{hand_name}/{severity}/{most_common_direction}'

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