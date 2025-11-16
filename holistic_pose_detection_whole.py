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

        self.rubbing_history = deque(maxlen=30)
        self.finger_cracking_history = deque(maxlen=15)

        # 귀 위치 추적 추가
        self.left_ear_positions = deque(maxlen=15)
        self.right_ear_positions = deque(maxlen=15)

        # 팔꿈치 위치 추적 추가
        self.left_elbow_positions = deque(maxlen=15)
        self.right_elbow_positions = deque(maxlen=15)

        self.prev_finger_states = None

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
        if self._detect_head_shake(landmarks):
            self.detected_gestures.append('머리동작/흔들기')

        # 고개 숙이기
        if self._detect_head_bow(landmarks):
            self.detected_gestures.append('머리동작/숙이기')

    def _detect_head_shake(self, landmarks):
        """귀의 움직임으로 고개 흔들기 감지"""
        # 랜드마크 가져오기
        left_ear = landmarks[self.mp_holistic.PoseLandmark.LEFT_EAR.value]
        right_ear = landmarks[self.mp_holistic.PoseLandmark.RIGHT_EAR.value]

        # 현재 귀 위치 저장 (x 좌표)
        self.left_ear_positions.append(left_ear.x)
        self.right_ear_positions.append(right_ear.x)

        # 최소 프레임 수 확보 (10프레임)
        if len(self.left_ear_positions) < 10:
            return False

        # 최근 프레임들의 움직임 범위 계산
        left_range = max(self.left_ear_positions) - min(self.left_ear_positions)
        right_range = max(self.right_ear_positions) - min(self.right_ear_positions)

        # 임계값: 0.08 = 화면 너비의 8% 움직임
        threshold = 0.08

        # 왼쪽 귀나 오른쪽 귀 중 하나라도 많이 움직였으면 감지
        if left_range > threshold or right_range > threshold:
            # 감지 후 히스토리 클리어
            self.left_ear_positions.clear()
            self.right_ear_positions.clear()
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

        # 팔짱끼기
        if self._detect_arms_crossed(landmarks):
            self.detected_gestures.append("팔짱 끼기")

            # 무의미 반동
        if self._detect_meaningless_arm_movement(landmarks):
            self.detected_gestures.append('팔동작/무의미반동')

        # 뒷짐
        if self._detect_hands_behind_back(landmarks):
            self.detected_gestures.append('팔동작/뒷짐')

    def _detect_arms_crossed(self, landmarks):
        """팔짱끼기 감지 - 손목이 반대쪽 어깨와 가까워지면"""
        # 랜드마크 가져오기
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

        # 왼쪽 손목이 오른쪽 어깨 x좌표와 유사한지 체크
        left_wrist_near_right_shoulder = abs(left_wrist.x - right_shoulder.x) < 0.1

        # 오른쪽 손목이 왼쪽 어깨 x좌표와 유사한지 체크
        right_wrist_near_left_shoulder = abs(right_wrist.x - left_shoulder.x) < 0.1

        # 둘 다 만족하면 팔짱끼기
        if left_wrist_near_right_shoulder and right_wrist_near_left_shoulder:
            return True

        return False

    def _detect_meaningless_arm_movement(self, landmarks):
        """팔꿈치의 빠른 움직임으로 무의미 반동 감지"""
        # 팔꿈치 랜드마크
        left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]

        # 팔꿈치가 보이지 않으면 히스토리 클리어하고 리턴
        # visibility: 0.0~1.0 (0.5 이상이면 보인다고 판단)
        if left_elbow.visibility < 0.5 and right_elbow.visibility < 0.5:
            self.left_elbow_positions.clear()
            self.right_elbow_positions.clear()
            return False

        # 현재 팔꿈치 위치 저장 (x, y 좌표 튜플)
        self.left_elbow_positions.append((left_elbow.x, left_elbow.y))
        self.right_elbow_positions.append((right_elbow.x, right_elbow.y))

        # 최소 프레임 수 확보
        if len(self.left_elbow_positions) < 10:
            return False

        # 왼쪽 팔꿈치 움직임 범위 계산 (보일 때만)
        left_movement = 0
        if left_elbow.visibility >= 0.5:
            left_x_range = max(pos[0] for pos in self.left_elbow_positions) - \
                           min(pos[0] for pos in self.left_elbow_positions)
            left_y_range = max(pos[1] for pos in self.left_elbow_positions) - \
                           min(pos[1] for pos in self.left_elbow_positions)
            left_movement = (left_x_range ** 2 + left_y_range ** 2) ** 0.5

        # 오른쪽 팔꿈치 움직임 범위 계산 (보일 때만)
        right_movement = 0
        if right_elbow.visibility >= 0.5:
            right_x_range = max(pos[0] for pos in self.right_elbow_positions) - \
                            min(pos[0] for pos in self.right_elbow_positions)
            right_y_range = max(pos[1] for pos in self.right_elbow_positions) - \
                            min(pos[1] for pos in self.right_elbow_positions)
            right_movement = (right_x_range ** 2 + right_y_range ** 2) ** 0.5

        # 임계값: 0.12 이상 움직임
        threshold = 0.12

        # 양쪽 중 하나라도 많이 움직였으면 감지
        if left_movement > threshold or right_movement > threshold:
            # 감지 후 히스토리 클리어
            self.left_elbow_positions.clear()
            self.right_elbow_positions.clear()
            return True

        return False

    def _detect_hands_behind_back(self, landmarks):
        """뒷짐 - 손목이 보이지 않고 팔꿈치 위치가 특정 조건 만족"""
        # 손목과 팔꿈치 랜드마크
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]

        # 어깨 랜드마크 (참고용)
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

        # 1. 양쪽 손목이 안 보임 (visibility 낮음)
        left_wrist_hidden = left_wrist.visibility < 0.3
        right_wrist_hidden = right_wrist.visibility < 0.3

        # 2. 팔꿈치는 보임 (팔이 완전히 사라진 건 아님)
        left_elbow_visible = left_elbow.visibility > 0.5
        right_elbow_visible = right_elbow.visibility > 0.5

        # 3. 팔꿈치가 몸통 뒤쪽에 위치 (z 좌표가 어깨보다 뒤)
        left_elbow_behind = left_elbow.z > left_shoulder.z + 0.03
        right_elbow_behind = right_elbow.z > right_shoulder.z + 0.03

        # 모든 조건 만족하면 뒷짐
        if (left_wrist_hidden and right_wrist_hidden and
                left_elbow_visible and right_elbow_visible and
                left_elbow_behind and right_elbow_behind):
            return True

        return False

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

    def _detect_hand_to_face_region(self, pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks):
        """손-얼굴 접촉 (부위별 분류) - 거리 기반 + 입술"""

        if not pose_landmarks or not face_landmarks:
            return None

        # 얼굴 주요 랜드마크
        nose_tip = face_landmarks.landmark[1]
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]

        # 귀는 face_landmarks 사용
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]

        # 입술 랜드마크
        upper_lip = face_landmarks.landmark[13]  # 윗입술 중앙
        lower_lip = face_landmarks.landmark[14]  # 아랫입술 중앙
        mouth_center_x = (upper_lip.x + lower_lip.x) / 2
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2

        # 얼굴 중심 및 크기 계산
        face_center_x = nose_tip.x
        face_center_y = nose_tip.y

        face_width = abs(right_ear.x - left_ear.x)
        face_height_top = abs(nose_tip.y - forehead.y)
        face_height_bottom = abs(chin.y - nose_tip.y)

        # 각 부위의 대표 좌표 정의
        head_center = (nose_tip.x, forehead.y - face_height_top * 0.5)  # 이마 위쪽
        forehead_center = (nose_tip.x, forehead.y + face_height_top * 0.1)  # 이마 중간
        nose_center = (nose_tip.x, nose_tip.y)  # 코
        mouth_center = (mouth_center_x, mouth_center_y)  # 입술
        chin_center = (nose_tip.x, chin.y)  # 턱
        left_ear_center = (left_ear.x, left_ear.y)  # 왼쪽 귀
        right_ear_center = (right_ear.x, right_ear.y)  # 오른쪽 귀

        def check_hand_region(hand_landmarks, hand_name):
            """손이 닿은 얼굴 부위 확인 - 거리 기반"""
            if not hand_landmarks:
                return None

            # 손가락 끝들 (엄지, 검지, 중지, 약지, 새끼)
            fingertips = [4, 8, 12, 16, 20]

            min_distance_overall = float('inf')
            closest_region = None

            for idx in fingertips:
                point = hand_landmarks.landmark[idx]

                # 얼굴 전체 범위 체크 (넓게)
                distance_x = abs(point.x - face_center_x)

                # 얼굴 영역 밖이면 스킵
                if distance_x > face_width * 1.0 or point.y > chin.y + face_height_bottom * 0.3:
                    continue

                # 각 부위까지의 거리 계산
                distances = {}

                # 머리 (이마보다 위일 때만)
                if point.y < forehead.y:
                    distances['손동작/머리'] = math.sqrt(
                        (point.x - head_center[0]) ** 2 + (point.y - head_center[1]) ** 2
                    )

                # 이마
                if forehead.y - face_height_top * 0.3 < point.y < nose_tip.y:
                    distances['손동작/이마'] = math.sqrt(
                        (point.x - forehead_center[0]) ** 2 + (point.y - forehead_center[1]) ** 2
                    )

                # 코
                if nose_tip.y - face_height_top * 0.3 < point.y < nose_tip.y + face_height_bottom * 0.3:
                    distances['손동작/코'] = math.sqrt(
                        (point.x - nose_center[0]) ** 2 + (point.y - nose_center[1]) ** 2
                    )

                # 입술 (코와 턱 사이)
                if nose_tip.y + face_height_bottom * 0.1 < point.y < chin.y - face_height_bottom * 0.2:
                    distances['손동작/입술'] = math.sqrt(
                        (point.x - mouth_center[0]) ** 2 + (point.y - mouth_center[1]) ** 2
                    )

                # 턱
                if point.y > nose_tip.y + face_height_bottom * 0.2:
                    distances['손동작/턱'] = math.sqrt(
                        (point.x - chin_center[0]) ** 2 + (point.y - chin_center[1]) ** 2
                    )

                # 왼쪽 귀
                left_ear_dist = math.sqrt(
                    (point.x - left_ear_center[0]) ** 2 + (point.y - left_ear_center[1]) ** 2
                )
                if left_ear_dist < 0.15:  # 귀 범위 내
                    distances['손동작/왼쪽 귀'] = left_ear_dist

                # 오른쪽 귀
                right_ear_dist = math.sqrt(
                    (point.x - right_ear_center[0]) ** 2 + (point.y - right_ear_center[1]) ** 2
                )
                if right_ear_dist < 0.15:  # 귀 범위 내
                    distances['손동작/오른쪽 귀'] = right_ear_dist

                # 이번 손가락에서 가장 가까운 부위 찾기
                if distances:
                    min_region = min(distances, key=distances.get)
                    min_distance = distances[min_region]

                    # 전체에서 가장 가까운 거리 업데이트
                    if min_distance < min_distance_overall:
                        min_distance_overall = min_distance
                        closest_region = min_region

            return closest_region

        # 양손 체크 (왼손 우선)
        detected_region = check_hand_region(left_hand_landmarks, "왼손")

        if not detected_region:
            detected_region = check_hand_region(right_hand_landmarks, "오른손")

        # 히스토리 관리
        if detected_region:
            self.hand_face_distances.append(detected_region)
        else:
            self.hand_face_distances.append(None)

        # 최근 프레임에서 안정적으로 감지되었는지 확인
        if len(self.hand_face_distances) >= 8:
            recent_detections = list(self.hand_face_distances)[-8:]

            # None이 아닌 값들만 추출
            valid_regions = [r for r in recent_detections if r is not None]

            # 최근 8프레임 중 5프레임 이상 감지되었고
            if len(valid_regions) >= 5:
                # 가장 많이 감지된 부위를 반환
                from collections import Counter
                most_common = Counter(valid_regions).most_common(1)[0][0]

                self.hand_face_distances.clear()
                return most_common

        return None

    def _detect_hand_rubbing(self, left_hand_landmarks, right_hand_landmarks):
        """양손 비비기 - 지속적인 문지름 감지 (개선)"""

        if not left_hand_landmarks or not right_hand_landmarks:
            return False

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

        recent_distances = [f['distance'] for f in list(self.rubbing_history)[-20:]]
        avg_distance = sum(recent_distances) / len(recent_distances)

        if avg_distance > 0.15:
            return False

        # 반대 방향 + 유의미한 움직임만 카운트
        opposite_with_movement = 0
        total_movement = 0
        history_list = list(self.rubbing_history)

        for i in range(1, len(history_list)):
            prev = history_list[i - 1]
            curr = history_list[i]

            left_dx = curr['left'][0] - prev['left'][0]
            right_dx = curr['right'][0] - prev['right'][0]
            left_dy = curr['left'][1] - prev['left'][1]
            right_dy = curr['right'][1] - prev['right'][1]

            # 전체 움직임 크기
            combined_movement = math.sqrt(left_dx ** 2 + left_dy ** 2 + right_dx ** 2 + right_dy ** 2)

            # 임계값 증가: 0.008 → 0.015 (더 큰 움직임만)
            if combined_movement > 0.015:
                total_movement += 1

                # 반대 방향 체크
                if (left_dx * right_dx < 0) or (left_dy * right_dy < 0):
                    opposite_with_movement += 1

        # 조건 강화:
        # 1. 최소 15번 이상 실제로 움직였고 (12 → 15)
        # 2. 그 중 10번 이상 반대 방향으로 움직임 (8 → 10)
        return total_movement >= 15 and opposite_with_movement >= 10


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