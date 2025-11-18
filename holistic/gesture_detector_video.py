import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from datetime import datetime


class GestureDetector:
    def __init__(self, fps=30):
        """
        실시간 제스처 감지기

        Args:
            fps: 카메라 FPS (프레임 기반 임계값 계산에 사용)
        """
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Holistic 모델 (포즈, 손 감지)
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        # FaceMesh 모델 (얼굴, 눈 깜빡임)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.fps = fps
        self._init_parameters()
        self._init_state_variables()
        self._init_counters()

    def _init_parameters(self):
        """파라미터 설정"""
        # 공통
        self.VISIBILITY_THRESHOLD = 0.3
        self.FACE_VISIBILITY_THRESHOLD = 0.5  # 얼굴 가시성 임계값

        # === Head Pose ===
        self.NPR_CEILING_THRESHOLD = 0.20
        self.NPR_FLOOR_THRESHOLD = 0.44
        self.POSE_FRAME_COUNT = int(self.fps * 0.8)

        # === Head Bow (고개 숙이기) ===
        self.HEAD_BOW_THRESHOLD = -0.2
        self.HEAD_BOW_FRAME_COUNT = 10

        # === Lip Bite ===
        self.LIP_BITE_RATIO_THRESHOLD = 0.25
        self.LIP_BITE_FRAME_COUNT = int(self.fps * 0.5)

        # === Hand Near Face ===
        self.HAND_NEAR_FACE_THRESHOLD = 0.15
        self.HAND_GESTURE_FRAME_COUNT = int(self.fps * 0.5)

        # === Slant ===
        self.SLANT_RATIO_THRESHOLD = 0.03
        self.SLANT_FRAME_COUNT = int(self.fps * 1.5)

        # === Rigid Attention ===
        self.RIGID_FRAME_COUNT = max(self.fps * 2, 60)
        self.ATTENTION_DIST_RATIO = 0.5  # 손목이 엉덩이로부터 어깨너비의 50% 이내
        self.RIGID_MOVEMENT_RATIO = 0.02  # 움직임 표준편차가 어깨너비의 2% 이하

        # === Blink ===
        self.EAR_THRESHOLD = 0.20
        self.BLINK_FRAME_COUNT = max(int(self.fps * 0.05), 2)

        # === Head Shake ===
        self.HEAD_SHAKE_THRESHOLD = 0.08
        self.HEAD_SHAKE_FRAME_COUNT = 10

        # === Arms Crossed ===
        self.ARMS_CROSSED_THRESHOLD = 0.1

        # === Hands Behind Back ===
        self.HANDS_BEHIND_BACK_THRESHOLD = 0.03

        # === Hand Rubbing ===
        self.HAND_RUBBING_DISTANCE = 0.15
        self.HAND_RUBBING_MOVEMENT = 0.015
        self.HAND_RUBBING_FRAME_COUNT = 20

        # === Fig Leaf (무화과 잎 자세) ===
        self.FIG_LEAF_HANDS_CLOSE_THRESHOLD = 100  # 픽셀 거리
        self.FIG_LEAF_FRAME_COUNT = int(self.fps * 1.0)  # 1초 유지

    def _init_state_variables(self):
        """상태 변수 초기화"""
        # Head Pose
        self.pose_status = "Forward"
        self.pose_counter = 0
        self.last_pose_event = ""

        # Head Bow
        self.head_angles = deque(maxlen=self.HEAD_BOW_FRAME_COUNT)
        self.last_head_bow_event = False

        # Lip Bite
        self.lip_bite_status = False
        self.lip_bite_counter = 0
        self.last_lip_bite_event = False

        # Hand Near Face
        self.hand_near_face_status = False
        self.hand_near_face_counter = 0
        self.last_hand_near_face_event = False

        # Slant
        self.slant_status = False
        self.slant_counter = 0
        self.last_slant_event = False

        # Rigid
        self.rigid_status = False
        self.rigid_counter = 0
        self.last_rigid_event = False

        # Blink
        self.blink_status_closed = False
        self.blink_counter = 0

        # Head Shake
        self.last_head_shake_event = False

        # Arms Crossed
        self.arms_crossed_status = False

        # Hands Behind Back
        self.hands_behind_back_status = False

        # Hand Rubbing
        self.last_hand_rubbing_event = False

        # Hand to Face Region
        self.last_hand_to_face_event = None

        # Fig Leaf
        self.fig_leaf_status = False
        self.fig_leaf_counter = 0
        self.last_fig_leaf_event = False

        # 히스토리
        self.left_wrist_history = deque(maxlen=self.RIGID_FRAME_COUNT)
        self.right_wrist_history = deque(maxlen=self.RIGID_FRAME_COUNT)
        self.left_ear_positions = deque(maxlen=self.HEAD_SHAKE_FRAME_COUNT)
        self.right_ear_positions = deque(maxlen=self.HEAD_SHAKE_FRAME_COUNT)
        self.rubbing_history = deque(maxlen=30)
        self.hand_face_distances = deque(maxlen=15)

    def _init_counters(self):
        """이벤트 카운터 초기화"""
        self.total_floor_events = 0
        self.total_ceiling_events = 0
        self.total_head_bow_events = 0
        self.total_lip_bite_events = 0
        self.total_hand_near_face_events = 0
        self.total_slant_events = 0
        self.total_rigid_events = 0
        self.total_blink_events = 0
        self.total_head_shake_events = 0
        self.total_arms_crossed_events = 0
        self.total_hands_behind_back_events = 0
        self.total_hand_rubbing_events = 0
        self.total_hand_to_face_events = {}
        self.total_fig_leaf_events = 0

        # 타임라인 추적
        self.timeline = {
            '바닥보기': [],
            '천장보기': [],
            '고개숙이기': [],
            '입술깨물기': [],
            '눈깜빡임': [],
            '고개흔들기': [],
            '비스듬한자세': [],
            '경직된차려': [],
            '팔짱끼기': [],
            '뒷짐': [],
            '손비비기': [],
            '무화과잎자세': [],
            '머리터치': [],
            '이마터치': [],
            '코터치': [],
            '입술터치': [],
            '턱터치': [],
            '왼쪽귀터치': [],
            '오른쪽귀터치': []
        }

        self.gesture_log = []

    def get_distance_2d(self, lm1, lm2, w, h):
        """2D 픽셀 거리 계산"""
        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)
        return math.hypot(x1 - x2, y1 - y2)

    def _check_face_visible(self, face_landmarks):
        """얼굴이 충분히 보이는지 확인"""
        if not face_landmarks:
            return False

        # 주요 얼굴 랜드마크 체크 (코, 이마, 턱)
        try:
            nose = face_landmarks.landmark[1]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]

            # 세 점의 가시성이 모두 임계값 이상이어야 함
            if (nose.visibility > self.FACE_VISIBILITY_THRESHOLD and
                    forehead.visibility > self.FACE_VISIBILITY_THRESHOLD and
                    chin.visibility > self.FACE_VISIBILITY_THRESHOLD):
                return True
        except:
            pass

        return False

    def process_frame(self, image, current_time_str):
        """한 프레임 처리"""
        height, width, _ = image.shape
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe 실행
        results_holistic = self.holistic.process(imgRGB)
        results_facemesh = self.face_mesh.process(imgRGB)

        # 현재 프레임에서 새로 발생한 이벤트
        new_events = []

        # 얼굴 가시성 체크
        face_visible = False
        if results_facemesh.multi_face_landmarks:
            face_visible = self._check_face_visible(results_facemesh.multi_face_landmarks[0])

        # === A. 얼굴 분석 (얼굴이 보일 때만) ===
        if face_visible:
            face_events = self._process_face(
                results_facemesh.multi_face_landmarks[0],
                width, height, current_time_str
            )
            new_events.extend(face_events)
        else:
            # 얼굴이 안 보이면 관련 상태 리셋
            self._reset_face_states()

        # === B. 포즈 분석 ===
        if results_holistic.pose_landmarks:
            pose_events = self._process_pose(
                results_holistic.pose_landmarks,
                width, height, current_time_str,
                face_visible  # 얼굴 가시성 정보 전달
            )
            new_events.extend(pose_events)

        # === C. 손 분석 (얼굴이 보일 때만) ===
        if face_visible and (results_holistic.left_hand_landmarks or results_holistic.right_hand_landmarks):
            hand_events = self._process_hands(
                results_holistic,
                results_facemesh.multi_face_landmarks[0],
                width, height, current_time_str
            )
            new_events.extend(hand_events)

        # === D. 시각화 (랜드마크만) ===
        processed_image = self._visualize(
            image, results_holistic, results_facemesh
        )

        return processed_image, new_events

    def _reset_face_states(self):
        """얼굴 관련 상태 리셋"""
        # Head Pose
        self.pose_counter = 0
        self.pose_status = "Forward"
        self.last_pose_event = ""

        # Head Bow
        self.head_angles.clear()
        self.last_head_bow_event = False

        # Lip Bite
        self.lip_bite_status = False
        self.lip_bite_counter = 0
        self.last_lip_bite_event = False

        # Blink
        self.blink_status_closed = False
        self.blink_counter = 0

        # Head Shake
        self.left_ear_positions.clear()
        self.right_ear_positions.clear()
        self.last_head_shake_event = False

        # Hand to Face
        self.hand_face_distances.clear()
        self.last_hand_to_face_event = None

    def _process_face(self, face_landmarks, width, height, current_time_str):
        """얼굴 분석"""
        events = []
        landmarks = face_landmarks.landmark

        try:
            # === 1. Head Pose (천장/바닥만) ===
            current_pose_direction = "Forward"

            eye_center_y = landmarks[6].y
            nose_tip_y = landmarks[1].y
            chin_tip_y = landmarks[152].y
            face_vertical_height = chin_tip_y - eye_center_y

            if face_vertical_height > 0.01:
                avg_npr = (nose_tip_y - eye_center_y) / face_vertical_height

                if avg_npr > self.NPR_FLOOR_THRESHOLD:
                    current_pose_direction = "Floor"
                elif avg_npr < self.NPR_CEILING_THRESHOLD:
                    current_pose_direction = "Ceiling"

            # Head Pose 이벤트 처리
            if current_pose_direction != "Forward":
                if current_pose_direction == self.pose_status:
                    self.pose_counter += 1
                else:
                    self.pose_status = current_pose_direction
                    self.pose_counter = 1
                    self.last_pose_event = ""

                if self.pose_counter >= self.POSE_FRAME_COUNT and self.last_pose_event != self.pose_status:
                    self.last_pose_event = self.pose_status

                    if self.pose_status == "Floor":
                        self.total_floor_events += 1
                        self.timeline['바닥보기'].append(current_time_str)
                        log_msg = f"[{current_time_str}] 바닥보기"
                        self.gesture_log.append(log_msg)
                        print(log_msg)
                        events.append("바닥보기")
                    elif self.pose_status == "Ceiling":
                        self.total_ceiling_events += 1
                        self.timeline['천장보기'].append(current_time_str)
                        log_msg = f"[{current_time_str}] 천장보기"
                        self.gesture_log.append(log_msg)
                        print(log_msg)
                        events.append("천장보기")
            else:
                self.pose_counter = 0
                self.pose_status = "Forward"
                self.last_pose_event = ""

            # === 2. Lip Bite ===
            upper_outer_lip_y = landmarks[12].y
            upper_inner_lip_y = landmarks[13].y
            lower_inner_lip_y = landmarks[14].y
            lower_outer_lip_y = landmarks[17].y
            total_lip_height = abs(lower_outer_lip_y - upper_outer_lip_y)
            inner_lip_height = lower_inner_lip_y - upper_inner_lip_y

            current_lip_bite = False
            if total_lip_height > 0.005:
                avg_lip_ratio = inner_lip_height / total_lip_height

                if avg_lip_ratio < self.LIP_BITE_RATIO_THRESHOLD:
                    current_lip_bite = True

            if current_lip_bite:
                self.lip_bite_status = True
                self.lip_bite_counter += 1

                if self.lip_bite_counter >= self.LIP_BITE_FRAME_COUNT and not self.last_lip_bite_event:
                    self.last_lip_bite_event = True
                    self.total_lip_bite_events += 1
                    self.timeline['입술깨물기'].append(current_time_str)
                    log_msg = f"[{current_time_str}] 입술깨물기"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append("입술깨물기")
            else:
                self.lip_bite_status = False
                self.lip_bite_counter = 0
                self.last_lip_bite_event = False

            # === 3. Blink ===
            left_v_top = landmarks[386]
            left_v_bottom = landmarks[374]
            left_h_left = landmarks[362]
            left_h_right = landmarks[263]
            right_v_top = landmarks[159]
            right_v_bottom = landmarks[145]
            right_h_left = landmarks[133]
            right_h_right = landmarks[33]

            left_vertical_dist = self.get_distance_2d(left_v_top, left_v_bottom, width, height)
            left_horizontal_dist = self.get_distance_2d(left_h_left, left_h_right, width, height)
            right_vertical_dist = self.get_distance_2d(right_v_top, right_v_bottom, width, height)
            right_horizontal_dist = self.get_distance_2d(right_h_left, right_h_right, width, height)

            current_blink = False
            if left_horizontal_dist > 0 and right_horizontal_dist > 0:
                left_ear = left_vertical_dist / left_horizontal_dist
                right_ear = right_vertical_dist / right_horizontal_dist
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < self.EAR_THRESHOLD:
                    current_blink = True

            if current_blink:
                self.blink_counter += 1
                self.blink_status_closed = True
            else:
                if self.blink_counter >= self.BLINK_FRAME_COUNT:
                    self.total_blink_events += 1
                    self.timeline['눈깜빡임'].append(current_time_str)
                    log_msg = f"[{current_time_str}] 눈깜빡임"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append("눈깜빡임")

                self.blink_counter = 0
                self.blink_status_closed = False

            # === 4. Head Shake ===
            left_ear = landmarks[234]
            right_ear = landmarks[454]

            self.left_ear_positions.append(left_ear.x)
            self.right_ear_positions.append(right_ear.x)

            if len(self.left_ear_positions) >= self.HEAD_SHAKE_FRAME_COUNT:
                left_range = max(self.left_ear_positions) - min(self.left_ear_positions)
                right_range = max(self.right_ear_positions) - min(self.right_ear_positions)

                if left_range > self.HEAD_SHAKE_THRESHOLD or right_range > self.HEAD_SHAKE_THRESHOLD:
                    if not self.last_head_shake_event:
                        self.last_head_shake_event = True
                        self.total_head_shake_events += 1
                        self.timeline['고개흔들기'].append(current_time_str)
                        log_msg = f"[{current_time_str}] 고개흔들기"
                        self.gesture_log.append(log_msg)
                        print(log_msg)
                        events.append("고개흔들기")

                    self.left_ear_positions.clear()
                    self.right_ear_positions.clear()
                else:
                    self.last_head_shake_event = False

        except Exception as e:
            pass

        return events

    def _process_pose(self, pose_landmarks, width, height, current_time_str, face_visible):
        """포즈 분석"""
        events = []
        landmarks = pose_landmarks.landmark

        try:
            # === 1. Head Bow (고개 숙이기) - 얼굴이 보일 때만 ===
            if face_visible:
                nose = landmarks[self.mp_holistic.PoseLandmark.NOSE.value]
                left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

                if (nose.visibility > self.VISIBILITY_THRESHOLD and
                        left_shoulder.visibility > self.VISIBILITY_THRESHOLD and
                        right_shoulder.visibility > self.VISIBILITY_THRESHOLD):

                    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
                    head_position = nose.y - shoulder_mid_y

                    self.head_angles.append(head_position)

                    if len(self.head_angles) >= self.HEAD_BOW_FRAME_COUNT:
                        current_angle = self.head_angles[-1]

                        if current_angle > self.HEAD_BOW_THRESHOLD:
                            recent_bowing = sum(1 for angle in list(self.head_angles)[-5:]
                                                if angle > self.HEAD_BOW_THRESHOLD)
                            if recent_bowing >= 3:
                                if not self.last_head_bow_event:
                                    self.last_head_bow_event = True
                                    self.total_head_bow_events += 1
                                    self.timeline['고개숙이기'].append(current_time_str)
                                    log_msg = f"[{current_time_str}] 고개숙이기"
                                    self.gesture_log.append(log_msg)
                                    print(log_msg)
                                    events.append("고개숙이기")
                                self.head_angles.clear()
                        else:
                            self.last_head_bow_event = False

            # === 2. Slant & Rigid ===
            left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

            if (left_shoulder.visibility > self.VISIBILITY_THRESHOLD and
                    right_shoulder.visibility > self.VISIBILITY_THRESHOLD):

                shoulder_width_px = self.get_distance_2d(left_shoulder, right_shoulder, width, height)
                shoulder_y_diff_px = abs(int(left_shoulder.y * height) - int(right_shoulder.y * height))

                # Slant
                slant_ratio = 0.0
                if shoulder_width_px > 0:
                    slant_ratio = shoulder_y_diff_px / shoulder_width_px

                current_slant_pose = slant_ratio > self.SLANT_RATIO_THRESHOLD

                if current_slant_pose:
                    self.slant_status = True
                    self.slant_counter += 1

                    if self.slant_counter >= self.SLANT_FRAME_COUNT and not self.last_slant_event:
                        self.last_slant_event = True
                        self.total_slant_events += 1
                        self.timeline['비스듬한자세'].append(current_time_str)
                        log_msg = f"[{current_time_str}] 비스듬한자세"
                        self.gesture_log.append(log_msg)
                        print(log_msg)
                        events.append("비스듬한자세")
                else:
                    self.slant_status = False
                    self.slant_counter = 0
                    self.last_slant_event = False

                # === Rigid (경직된 차려자세) ===
                # 어깨 너비가 충분히 보일 때만 분석
                if shoulder_width_px > 10:
                    left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
                    left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
                    right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]

                    if (left_wrist.visibility > self.VISIBILITY_THRESHOLD and
                            right_wrist.visibility > self.VISIBILITY_THRESHOLD and
                            left_hip.visibility > self.VISIBILITY_THRESHOLD and
                            right_hip.visibility > self.VISIBILITY_THRESHOLD):

                        # 동적 임계값 계산 (어깨 너비 기준)
                        dynamic_attention_thresh = shoulder_width_px * self.ATTENTION_DIST_RATIO  # 50%
                        dynamic_rigid_thresh = shoulder_width_px * self.RIGID_MOVEMENT_RATIO  # 2%

                        # 1단계: 차렷자세 감지 (손목이 엉덩이 근처)
                        dist_l_wrist_hip = self.get_distance_2d(left_wrist, left_hip, width, height)
                        dist_r_wrist_hip = self.get_distance_2d(right_wrist, right_hip, width, height)

                        is_attention_pose = (dist_l_wrist_hip < dynamic_attention_thresh and
                                             dist_r_wrist_hip < dynamic_attention_thresh)

                        # 2단계: 손목 위치 히스토리 수집 (2초간)
                        lw_x_px, lw_y_px = int(left_wrist.x * width), int(left_wrist.y * height)
                        rw_x_px, rw_y_px = int(right_wrist.x * width), int(right_wrist.y * height)
                        self.left_wrist_history.append((lw_x_px, lw_y_px))
                        self.right_wrist_history.append((rw_x_px, rw_y_px))

                        # 3단계: 경직 감지 (움직임 표준편차 계산)
                        is_rigid = False
                        if len(self.left_wrist_history) == self.RIGID_FRAME_COUNT:
                            # 각 손목의 X, Y 좌표 표준편차 계산
                            l_std_x = np.std([pos[0] for pos in self.left_wrist_history])
                            l_std_y = np.std([pos[1] for pos in self.left_wrist_history])
                            r_std_x = np.std([pos[0] for pos in self.right_wrist_history])
                            r_std_y = np.std([pos[1] for pos in self.right_wrist_history])

                            # 평균 움직임 표준편차
                            movement_std = (l_std_x + l_std_y + r_std_x + r_std_y) / 4.0

                            # 움직임이 임계값보다 작으면 경직
                            if movement_std < dynamic_rigid_thresh:
                                is_rigid = True

                        # 4단계: 최종 판정 (차렷자세 + 경직)
                        current_rigid_pose = is_attention_pose and is_rigid

                        if current_rigid_pose:
                            self.rigid_status = True
                            self.rigid_counter += 1

                            if not self.last_rigid_event:
                                self.last_rigid_event = True
                                self.total_rigid_events += 1
                                self.timeline['경직된차려'].append(current_time_str)
                                log_msg = f"[{current_time_str}] 경직된차려 (움직임 표준편차: {movement_std:.1f}px)"
                                self.gesture_log.append(log_msg)
                                print(log_msg)
                                events.append("경직된차려")
                        else:
                            self.rigid_status = False
                            self.rigid_counter = 0
                            self.last_rigid_event = False

            # === 3. Arms Crossed ===
            left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
            left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

            left_wrist_near_right_shoulder = abs(left_wrist.x - right_shoulder.x) < self.ARMS_CROSSED_THRESHOLD
            right_wrist_near_left_shoulder = abs(right_wrist.x - left_shoulder.x) < self.ARMS_CROSSED_THRESHOLD

            if left_wrist_near_right_shoulder and right_wrist_near_left_shoulder:
                if not self.arms_crossed_status:
                    self.arms_crossed_status = True
                    self.total_arms_crossed_events += 1
                    self.timeline['팔짱끼기'].append(current_time_str)
                    log_msg = f"[{current_time_str}] 팔짱끼기"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append("팔짱끼기")
            else:
                self.arms_crossed_status = False

            # === 4. Hands Behind Back ===
            left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
            left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

            left_wrist_hidden = left_wrist.visibility < 0.3
            right_wrist_hidden = right_wrist.visibility < 0.3
            left_elbow_visible = left_elbow.visibility > 0.5
            right_elbow_visible = right_elbow.visibility > 0.5
            left_elbow_behind = left_elbow.z > left_shoulder.z + self.HANDS_BEHIND_BACK_THRESHOLD
            right_elbow_behind = right_elbow.z > right_shoulder.z + self.HANDS_BEHIND_BACK_THRESHOLD

            if (left_wrist_hidden and right_wrist_hidden and
                    left_elbow_visible and right_elbow_visible and
                    left_elbow_behind and right_elbow_behind):
                if not self.hands_behind_back_status:
                    self.hands_behind_back_status = True
                    self.total_hands_behind_back_events += 1
                    self.timeline['뒷짐'].append(current_time_str)
                    log_msg = f"[{current_time_str}] 뒷짐"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append("뒷짐")
            else:
                self.hands_behind_back_status = False

            # === 5. Fig Leaf (무화과 잎 자세) ===
            left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP.value]

            if (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
                    left_hip.visibility > 0.5 and right_hip.visibility > 0.5):

                dist_hands = self.get_distance_2d(left_wrist, right_wrist, width, height)
                hip_center_y = (left_hip.y + right_hip.y) / 2.0
                hands_center_y = (left_wrist.y + right_wrist.y) / 2.0

                current_fig_leaf_pose = False
                if dist_hands < self.FIG_LEAF_HANDS_CLOSE_THRESHOLD:
                    if hands_center_y > hip_center_y:
                        current_fig_leaf_pose = True

                if current_fig_leaf_pose:
                    self.fig_leaf_status = True
                    self.fig_leaf_counter += 1

                    if self.fig_leaf_counter >= self.FIG_LEAF_FRAME_COUNT and not self.last_fig_leaf_event:
                        self.last_fig_leaf_event = True
                        self.total_fig_leaf_events += 1
                        self.timeline['무화과잎자세'].append(current_time_str)
                        log_msg = f"[{current_time_str}] 무화과잎자세"
                        self.gesture_log.append(log_msg)
                        print(log_msg)
                        events.append("무화과잎자세")
                else:
                    self.fig_leaf_status = False
                    self.fig_leaf_counter = 0
                    self.last_fig_leaf_event = False

        except Exception as e:
            pass

        return events

    def _process_hands(self, results_holistic, face_landmarks, width, height, current_time_str):
        """손 분석"""
        events = []

        try:
            # === 1. Hand to Face Region ===
            face_region = self._detect_hand_to_face_region(
                results_holistic.pose_landmarks,
                results_holistic.left_hand_landmarks,
                results_holistic.right_hand_landmarks,
                face_landmarks,
                width, height
            )

            if face_region:
                if face_region != self.last_hand_to_face_event:
                    self.last_hand_to_face_event = face_region

                    if face_region not in self.total_hand_to_face_events:
                        self.total_hand_to_face_events[face_region] = 0

                    self.total_hand_to_face_events[face_region] += 1
                    self.timeline[face_region].append(current_time_str)
                    log_msg = f"[{current_time_str}] {face_region}"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append(face_region)
            else:
                self.last_hand_to_face_event = None

            # === 2. Hand Rubbing ===
            if self._detect_hand_rubbing(
                    results_holistic.left_hand_landmarks,
                    results_holistic.right_hand_landmarks):
                if not self.last_hand_rubbing_event:
                    self.last_hand_rubbing_event = True
                    self.total_hand_rubbing_events += 1
                    self.timeline['손비비기'].append(current_time_str)
                    log_msg = f"[{current_time_str}] 손비비기"
                    self.gesture_log.append(log_msg)
                    print(log_msg)
                    events.append("손비비기")
            else:
                self.last_hand_rubbing_event = False

        except Exception as e:
            pass

        return events

    def _detect_hand_to_face_region(self, pose_landmarks, left_hand_landmarks,
                                    right_hand_landmarks, face_landmarks, width, height):
        """손-얼굴 접촉 부위 감지"""
        if not pose_landmarks or not face_landmarks:
            return None

        nose_tip = face_landmarks.landmark[1]
        forehead = face_landmarks.landmark[10]
        chin = face_landmarks.landmark[152]
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]

        mouth_center_x = (upper_lip.x + lower_lip.x) / 2
        mouth_center_y = (upper_lip.y + lower_lip.y) / 2

        face_center_x = nose_tip.x
        face_center_y = nose_tip.y
        face_width = abs(right_ear.x - left_ear.x)
        face_height_top = abs(nose_tip.y - forehead.y)
        face_height_bottom = abs(chin.y - nose_tip.y)

        head_center = (nose_tip.x, forehead.y - face_height_top * 0.5)
        forehead_center = (nose_tip.x, forehead.y + face_height_top * 0.1)
        nose_center = (nose_tip.x, nose_tip.y)
        mouth_center = (mouth_center_x, mouth_center_y)
        chin_center = (nose_tip.x, chin.y)
        left_ear_center = (left_ear.x, left_ear.y)
        right_ear_center = (right_ear.x, right_ear.y)

        def check_hand_region(hand_landmarks):
            if not hand_landmarks:
                return None

            fingertips = [4, 8, 12, 16, 20]
            min_distance_overall = float('inf')
            closest_region = None

            for idx in fingertips:
                point = hand_landmarks.landmark[idx]
                distance_x = abs(point.x - face_center_x)

                if distance_x > face_width * 1.0 or point.y > chin.y + face_height_bottom * 0.3:
                    continue

                distances = {}

                if point.y < forehead.y:
                    distances['머리터치'] = math.sqrt(
                        (point.x - head_center[0]) ** 2 + (point.y - head_center[1]) ** 2
                    )

                if forehead.y - face_height_top * 0.3 < point.y < nose_tip.y:
                    distances['이마터치'] = math.sqrt(
                        (point.x - forehead_center[0]) ** 2 + (point.y - forehead_center[1]) ** 2
                    )

                if nose_tip.y - face_height_top * 0.3 < point.y < nose_tip.y + face_height_bottom * 0.3:
                    distances['코터치'] = math.sqrt(
                        (point.x - nose_center[0]) ** 2 + (point.y - nose_center[1]) ** 2
                    )

                if nose_tip.y + face_height_bottom * 0.1 < point.y < chin.y - face_height_bottom * 0.2:
                    distances['입술터치'] = math.sqrt(
                        (point.x - mouth_center[0]) ** 2 + (point.y - mouth_center[1]) ** 2
                    )

                if point.y > nose_tip.y + face_height_bottom * 0.2:
                    distances['턱터치'] = math.sqrt(
                        (point.x - chin_center[0]) ** 2 + (point.y - chin_center[1]) ** 2
                    )

                left_ear_dist = math.sqrt(
                    (point.x - left_ear_center[0]) ** 2 + (point.y - left_ear_center[1]) ** 2
                )
                if left_ear_dist < 0.15:
                    distances['왼쪽귀터치'] = left_ear_dist

                right_ear_dist = math.sqrt(
                    (point.x - right_ear_center[0]) ** 2 + (point.y - right_ear_center[1]) ** 2
                )
                if right_ear_dist < 0.15:
                    distances['오른쪽귀터치'] = right_ear_dist

                if distances:
                    min_region = min(distances, key=distances.get)
                    min_distance = distances[min_region]

                    if min_distance < min_distance_overall:
                        min_distance_overall = min_distance
                        closest_region = min_region

            return closest_region

        detected_region = check_hand_region(left_hand_landmarks)
        if not detected_region:
            detected_region = check_hand_region(right_hand_landmarks)

        if detected_region:
            self.hand_face_distances.append(detected_region)
        else:
            self.hand_face_distances.append(None)

        if len(self.hand_face_distances) >= 8:
            recent_detections = list(self.hand_face_distances)[-8:]
            valid_regions = [r for r in recent_detections if r is not None]

            if len(valid_regions) >= 5:
                from collections import Counter
                most_common = Counter(valid_regions).most_common(1)[0][0]
                self.hand_face_distances.clear()
                return most_common

        return None

    def _detect_hand_rubbing(self, left_hand_landmarks, right_hand_landmarks):
        """손 비비기 감지"""
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

        if len(self.rubbing_history) < self.HAND_RUBBING_FRAME_COUNT:
            return False

        recent_distances = [f['distance'] for f in list(self.rubbing_history)[-20:]]
        avg_distance = sum(recent_distances) / len(recent_distances)

        if avg_distance > self.HAND_RUBBING_DISTANCE:
            return False

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

            combined_movement = math.sqrt(left_dx ** 2 + left_dy ** 2 +
                                          right_dx ** 2 + right_dy ** 2)

            if combined_movement > self.HAND_RUBBING_MOVEMENT:
                total_movement += 1

                if (left_dx * right_dx < 0) or (left_dy * right_dy < 0):
                    opposite_with_movement += 1

        return total_movement >= 15 and opposite_with_movement >= 10

    def _visualize(self, image, results_holistic, results_facemesh):
        """시각화 - 랜드마크만 그리기"""
        # 손 랜드마크
        if results_holistic.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results_holistic.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        if results_holistic.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results_holistic.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        # 포즈 랜드마크
        if results_holistic.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results_holistic.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )

        # 얼굴 랜드마크 (눈만)
        if results_facemesh.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results_facemesh.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1
                )
            )
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results_facemesh.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1
                )
            )

        return image

    def get_statistics(self):
        """통계 정보 반환"""
        stats = {
            '바닥보기': self.total_floor_events,
            '천장보기': self.total_ceiling_events,
            '고개숙이기': self.total_head_bow_events,
            '입술깨물기': self.total_lip_bite_events,
            '눈깜빡임': self.total_blink_events,
            '고개흔들기': self.total_head_shake_events,
            '비스듬한자세': self.total_slant_events,
            '경직된차려': self.total_rigid_events,
            '팔짱끼기': self.total_arms_crossed_events,
            '뒷짐': self.total_hands_behind_back_events,
            '손비비기': self.total_hand_rubbing_events,
            '무화과잎자세': self.total_fig_leaf_events
        }

        # 손얼굴터치 세부 항목 추가
        for region, count in self.total_hand_to_face_events.items():
            stats[region] = count

        return stats

    def save_log(self, output_path, video_path, processing_time, total_frames):
        """로그 파일 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                     비언어 행동 분석 결과\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"동영상: {video_path}\n")
            f.write(f"처리 시간: {processing_time:.2f}초\n")
            f.write(f"총 프레임: {total_frames}\n")
            f.write(f"FPS: {self.fps}\n\n")

            f.write("=" * 80 + "\n")
            f.write("                          통계 요약\n")
            f.write("=" * 80 + "\n\n")

            # 통계를 카테고리별로 정리
            categories = {
                '머리 움직임': ['바닥보기', '천장보기', '고개숙이기', '고개흔들기'],
                '얼굴 표정': ['입술깨물기', '눈깜빡임'],
                '자세': ['비스듬한자세', '경직된차려', '무화과잎자세'],
                '팔 동작': ['팔짱끼기', '뒷짐', '손비비기'],
                '손-얼굴 터치': ['머리터치', '이마터치', '코터치', '입술터치', '턱터치', '왼쪽귀터치', '오른쪽귀터치']
            }

            for category, gestures in categories.items():
                f.write(f"[{category}]\n")
                for gesture in gestures:
                    count = 0
                    if gesture in ['바닥보기', '천장보기', '고개숙이기', '입술깨물기', '눈깜빡임',
                                   '고개흔들기', '비스듬한자세', '경직된차려', '팔짱끼기', '뒷짐',
                                   '손비비기', '무화과잎자세']:
                        count = getattr(self, f'total_{self._get_counter_name(gesture)}_events', 0)
                    else:  # 손-얼굴 터치
                        count = self.total_hand_to_face_events.get(gesture, 0)

                    timeline = self.timeline.get(gesture, [])

                    if count > 0:
                        timeline_str = ', '.join(timeline) if timeline else '-'
                        f.write(f"  • {gesture}: {count}회\n")
                        f.write(f"    타임라인: {timeline_str}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("                        상세 로그\n")
            f.write("=" * 80 + "\n\n")

            for log in self.gesture_log:
                f.write(log + "\n")

    def _get_counter_name(self, gesture):
        """제스처 이름을 카운터 변수명으로 변환"""
        mapping = {
            '바닥보기': 'floor',
            '천장보기': 'ceiling',
            '고개숙이기': 'head_bow',
            '입술깨물기': 'lip_bite',
            '눈깜빡임': 'blink',
            '고개흔들기': 'head_shake',
            '비스듬한자세': 'slant',
            '경직된차려': 'rigid',
            '팔짱끼기': 'arms_crossed',
            '뒷짐': 'hands_behind_back',
            '손비비기': 'hand_rubbing',
            '무화과잎자세': 'fig_leaf'
        }
        return mapping.get(gesture, '')

    def close(self):
        """리소스 정리"""
        self.holistic.close()
        self.face_mesh.close()


# ========================================
# MP4 동영상 분석
# ========================================

def analyze_video(video_path, output_video_path=None, output_log_path=None):
    """MP4 동영상 제스처 분석"""

    # 기본 출력 경로 설정
    if output_video_path is None:
        output_video_path = video_path.replace('.mp4', '_analyzed.mp4')
    if output_log_path is None:
        output_log_path = video_path.replace('.mp4', '_log.txt')

    # 동영상 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ 동영상을 열 수 없습니다: {video_path}")
        return

    # 동영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 60)
    print("📹 MP4 동영상 제스처 분석 시작!")
    print("=" * 60)
    print(f"📂 입력 파일: {video_path}")
    print(f"📊 동영상 정보: {width}x{height} @ {fps}fps")
    print(f"🎬 총 프레임: {total_frames}")
    print(f"⏱️  예상 시간: {total_frames / fps:.1f}초")
    print("=" * 60)
    print("🎥 분석 중...\n")

    # 제스처 감지기 초기화
    detector = GestureDetector(fps=fps)

    # 결과 동영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    start_time = time.time()
    frame_count = 0

    while True:
        success, image = cap.read()
        if not success:
            break

        frame_count += 1

        # 현재 시간 계산
        current_time_sec = frame_count / fps
        current_time_str = time.strftime('%H:%M:%S', time.gmtime(current_time_sec))

        # 진행률 표시 (2초마다)
        if frame_count % (fps * 2) == 0 or frame_count == 1:
            progress = (frame_count / total_frames) * 100
            print(f"처리 중... {progress:.1f}% ({frame_count}/{total_frames})")

        # 프레임 처리
        processed_image, new_events = detector.process_frame(image, current_time_str)

        # 결과 동영상에 저장
        out.write(processed_image)

    # 정리
    cap.release()
    out.release()

    end_time = time.time()
    processing_time = end_time - start_time

    # 로그 저장
    detector.save_log(output_log_path, video_path, processing_time, total_frames)

    # 최종 통계 출력
    print("\n" + "=" * 60)
    print("📊 분석 완료!")
    print("=" * 60)
    print(f"⏱️  처리 시간: {processing_time:.2f}초")
    print(f"📹 결과 동영상: {output_video_path}")
    print(f"📝 로그 파일: {output_log_path}")

    print("\n=== 비언어 행동 통계 ===")
    stats = detector.get_statistics()
    for gesture, count in stats.items():
        if count > 0:
            timeline = detector.timeline.get(gesture, [])
            timeline_str = ', '.join(timeline[:5])  # 처음 5개만 표시
            if len(timeline) > 5:
                timeline_str += f", ... (총 {len(timeline)}회)"
            print(f"{gesture}: {count}회 | {timeline_str}")

    print("=" * 60)

    detector.close()

    print("\n✅ 모든 결과가 저장되었습니다!\n")


# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\n사용법:")
        print("  python script.py <입력_동영상.mp4> [출력_동영상.mp4] [로그파일.txt]")
        print("\n예시:")
        print("  python script.py input.mp4")
        print("  python script.py input.mp4 output.mp4")
        print("  python script.py input.mp4 output.mp4 log.txt\n")
    else:
        video_path = sys.argv[1]
        output_video = sys.argv[2] if len(sys.argv) > 2 else None
        output_log = sys.argv[3] if len(sys.argv) > 3 else None

        analyze_video(video_path, output_video, output_log)