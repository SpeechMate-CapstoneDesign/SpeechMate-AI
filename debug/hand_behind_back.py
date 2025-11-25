import cv2
import mediapipe as mp
import time


class HandsBehindBackDetector:
    def __init__(self, fps=30):
        """뒷짐 전용 감지기"""
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        self.fps = fps

        # 파라미터
        self.HANDS_BEHIND_BACK_FRAME_COUNT = int(self.fps * 0.5)  # 0.5초

        # 상태 변수
        self.hands_behind_back_counter = 0
        self.last_hands_behind_back_event = False
        self.total_hands_behind_back_events = 0

        # 디버그 정보
        self.debug_info = {}
        self.start_time = time.time()

        # 프레임 카운터
        self.frame_count = 0

    def process_frame(self, image):
        """한 프레임 처리"""
        height, width, _ = image.shape
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # MediaPipe 실행
        results = self.holistic.process(imgRGB)

        # 이벤트
        event_detected = False

        # 디버그 정보 초기화
        self.debug_info = {
            'frame_valid': False,
            'counter': self.hands_behind_back_counter,
            'threshold': self.HANDS_BEHIND_BACK_FRAME_COUNT
        }

        # 뒷짐 체크
        if results.pose_landmarks:
            event_detected = self._detect_hands_behind_back(
                results.pose_landmarks, width, height
            )

        # 시각화
        processed_image = self._visualize(image, results)

        self.frame_count += 1

        return processed_image, event_detected, self.debug_info

    def _detect_hands_behind_back(self, pose_landmarks, width, height):
        """뒷짐 감지"""
        landmarks = pose_landmarks.landmark
        event_detected = False

        try:
            # 랜드마크 추출
            left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
            left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
            left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

            current_hands_behind = False

            # === 좌표 정보 출력 ===
            print("\n" + "=" * 70)
            print("📍 좌표 정보")
            print("=" * 70)
            print(f"어깨:")
            print(f"  왼쪽  X: {left_shoulder.x:.3f}  Y: {left_shoulder.y:.3f}  Z: {left_shoulder.z:.3f}")
            print(f"  오른쪽 X: {right_shoulder.x:.3f}  Y: {right_shoulder.y:.3f}  Z: {right_shoulder.z:.3f}")
            print(f"\n팔꿈치:")
            print(
                f"  왼쪽  X: {left_elbow.x:.3f}  Y: {left_elbow.y:.3f}  Z: {left_elbow.z:.3f}  가시성: {left_elbow.visibility:.2f}")
            print(
                f"  오른쪽 X: {right_elbow.x:.3f}  Y: {right_elbow.y:.3f}  Z: {right_elbow.z:.3f}  가시성: {right_elbow.visibility:.2f}")
            print(f"\n손목:")
            print(
                f"  왼쪽  X: {left_wrist.x:.3f}  Y: {left_wrist.y:.3f}  Z: {left_wrist.z:.3f}  가시성: {left_wrist.visibility:.2f}")
            print(
                f"  오른쪽 X: {right_wrist.x:.3f}  Y: {right_wrist.y:.3f}  Z: {right_wrist.z:.3f}  가시성: {right_wrist.visibility:.2f}")

            # 디버그 정보 저장
            self.debug_info.update({
                'left_shoulder_x': left_shoulder.x,
                'left_shoulder_z': left_shoulder.z,
                'right_shoulder_x': right_shoulder.x,
                'right_shoulder_z': right_shoulder.z,
                'left_elbow_x': left_elbow.x,
                'left_elbow_z': left_elbow.z,
                'right_elbow_x': right_elbow.x,
                'right_elbow_z': right_elbow.z,
                'left_wrist_x': left_wrist.x,
                'left_wrist_z': left_wrist.z,
                'right_wrist_x': right_wrist.x,
                'right_wrist_z': right_wrist.z,
                'left_wrist_vis': left_wrist.visibility,
                'right_wrist_vis': right_wrist.visibility,
            })

            # === 뒷짐 조건 체크 (한쪽만 있어도 OK) ===
            print("\n" + "🔍 조건 체크 (한쪽 팔만 뒤로 가도 감지)")
            print("-" * 70)

            # 각 팔 개별 체크
            left_hand_behind = False
            right_hand_behind = False

            # 왼쪽 팔 체크
            print(f"\n왼팔 분석:")
            left_wrist_hidden = left_wrist.visibility < 0.4
            left_elbow_behind = left_elbow.z > left_shoulder.z + 0.05

            print(f"  [{'✓' if left_wrist_hidden else '✗'}] 손목 가시성 낮음: {left_wrist_hidden}")
            print(f"      (손목 가시성: {left_wrist.visibility:.2f} < 0.4)")
            print(f"  [{'✓' if left_elbow_behind else '✗'}] 팔꿈치가 어깨보다 뒤: {left_elbow_behind}")
            print(
                f"      (팔꿈치 Z: {left_elbow.z:.3f} > 어깨 Z: {left_shoulder.z:.3f} + 0.05 = {left_shoulder.z + 0.05:.3f})")

            # 왼팔 뒷짐 판정: 손목이 안 보이고 AND 팔꿈치가 뒤에
            left_hand_behind = left_wrist_hidden and left_elbow_behind
            print(f"  ➜ 왼팔 뒷짐: {'✅ YES' if left_hand_behind else '❌ NO'} (손목 안 보임 AND 팔꿈치 뒤)")

            # 오른쪽 팔 체크
            print(f"\n오른팔 분석:")
            right_wrist_hidden = right_wrist.visibility < 0.4
            right_elbow_behind = right_elbow.z > right_shoulder.z + 0.05

            print(f"  [{'✓' if right_wrist_hidden else '✗'}] 손목 가시성 낮음: {right_wrist_hidden}")
            print(f"      (손목 가시성: {right_wrist.visibility:.2f} < 0.4)")
            print(f"  [{'✓' if right_elbow_behind else '✗'}] 팔꿈치가 어깨보다 뒤: {right_elbow_behind}")
            print(
                f"      (팔꿈치 Z: {right_elbow.z:.3f} > 어깨 Z: {right_shoulder.z:.3f} + 0.05 = {right_shoulder.z + 0.05:.3f})")

            # 오른팔 뒷짐 판정: 손목이 안 보이고 AND 팔꿈치가 뒤에
            right_hand_behind = right_wrist_hidden and right_elbow_behind
            print(f"  ➜ 오른팔 뒷짐: {'✅ YES' if right_hand_behind else '❌ NO'} (손목 안 보임 AND 팔꿈치 뒤)")

            # 최종 판정: 한쪽만 뒷짐져도 OK
            current_hands_behind = left_hand_behind or right_hand_behind

            # 어느 팔이 뒷짐인지 표시
            behind_hands = []
            if left_hand_behind:
                behind_hands.append("왼손")
            if right_hand_behind:
                behind_hands.append("오른손")

            print("\n" + "-" * 70)
            if current_hands_behind:
                print(f"➡️  최종 판정: ✅ 뒷짐 자세 ({', '.join(behind_hands)} 뒤로)")
            else:
                print(f"➡️  최종 판정: ❌ 뒷짐 아님")
            print("=" * 70)

            # 뒷짐 이벤트 처리
            if current_hands_behind:
                self.hands_behind_back_counter += 1
                self.debug_info['counter'] = self.hands_behind_back_counter
                self.debug_info['behind_hands'] = behind_hands

                print(f"\n⏱️  카운터: {self.hands_behind_back_counter}/{self.HANDS_BEHIND_BACK_FRAME_COUNT}")

                if (self.hands_behind_back_counter >= self.HANDS_BEHIND_BACK_FRAME_COUNT and
                        not self.last_hands_behind_back_event):
                    self.last_hands_behind_back_event = True
                    self.total_hands_behind_back_events += 1
                    event_detected = True

                    elapsed = time.time() - self.start_time
                    print("\n" + "=" * 70)
                    print("✅ 뒷짐 감지!")
                    print("=" * 70)
                    print(f"감지 시간: {elapsed:.1f}초")
                    print(f"뒤로 간 손: {', '.join(behind_hands)}")
                    print(f"총 감지 횟수: {self.total_hands_behind_back_events}회")
                    print("=" * 70 + "\n")
            else:
                if self.hands_behind_back_counter > 0:
                    print(f"\n❌ 카운터 리셋 (조건 미충족)\n")
                self.hands_behind_back_counter = 0
                self.last_hands_behind_back_event = False
                self.debug_info['counter'] = 0

        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

        return event_detected

    def _visualize(self, image, results):
        """시각화 - Z축 깊이 정보"""
        h, w, _ = image.shape

        # 포즈 랜드마크 그리기
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark

            # 텍스트 정보 표시
            y_offset = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # 왼쪽 팔 정보
            if 'left_wrist_vis' in self.debug_info:
                left_vis = self.debug_info['left_wrist_vis']
                left_elbow_z = self.debug_info['left_elbow_z']
                left_shoulder_z = self.debug_info['left_shoulder_z']
                left_z_diff = left_elbow_z - left_shoulder_z

                left_hidden = left_vis < 0.4
                left_behind = left_z_diff > 0.05
                left_ok = left_hidden and left_behind

                color_left = (0, 255, 0) if left_ok else (0, 0, 255)

                cv2.putText(image, f"Left Hand:",
                            (10, y_offset), font, font_scale, (255, 255, 255), thickness)
                y_offset += 25
                cv2.putText(image, f"  Vis: {left_vis:.2f} {'<' if left_hidden else '>='} 0.4",
                            (10, y_offset), font, 0.5, color_left if left_hidden else (0, 0, 255), 1)
                y_offset += 20
                cv2.putText(image, f"  Z: {left_z_diff:+.3f} {'>' if left_behind else '<='} 0.05",
                            (10, y_offset), font, 0.5, color_left if left_behind else (0, 0, 255), 1)
                y_offset += 25
                cv2.putText(image, f"  Behind: {'YES' if left_ok else 'NO'}",
                            (10, y_offset), font, 0.5, color_left, 2)
                y_offset += 35

            # 오른쪽 팔 정보
            if 'right_wrist_vis' in self.debug_info:
                right_vis = self.debug_info['right_wrist_vis']
                right_elbow_z = self.debug_info['right_elbow_z']
                right_shoulder_z = self.debug_info['right_shoulder_z']
                right_z_diff = right_elbow_z - right_shoulder_z

                right_hidden = right_vis < 0.4
                right_behind = right_z_diff > 0.05
                right_ok = right_hidden and right_behind

                color_right = (0, 255, 0) if right_ok else (0, 0, 255)

                cv2.putText(image, f"Right Hand:",
                            (10, y_offset), font, font_scale, (255, 255, 255), thickness)
                y_offset += 25
                cv2.putText(image, f"  Vis: {right_vis:.2f} {'<' if right_hidden else '>='} 0.4",
                            (10, y_offset), font, 0.5, color_right if right_hidden else (0, 0, 255), 1)
                y_offset += 20
                cv2.putText(image, f"  Z: {right_z_diff:+.3f} {'>' if right_behind else '<='} 0.05",
                            (10, y_offset), font, 0.5, color_right if right_behind else (0, 0, 255), 1)
                y_offset += 25
                cv2.putText(image, f"  Behind: {'YES' if right_ok else 'NO'}",
                            (10, y_offset), font, 0.5, color_right, 2)

            # 카운터 및 감지된 손 표시
            if self.hands_behind_back_counter > 0:
                counter_text = f"Counter: {self.hands_behind_back_counter}/{self.HANDS_BEHIND_BACK_FRAME_COUNT}"
                cv2.putText(image, counter_text, (10, h - 50),
                            font, 0.8, (0, 255, 255), 2)

            # 뒷짐 감지 시 표시
            if 'behind_hands' in self.debug_info and self.debug_info['behind_hands']:
                behind_text = f"Behind: {', '.join(self.debug_info['behind_hands'])}"
                cv2.putText(image, behind_text, (10, h - 20),
                            font, 0.8, (0, 255, 0), 2)

        return image

    def close(self):
        """리소스 정리"""
        self.holistic.close()


def main():
    """메인 함수"""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    print("\n" + "=" * 70)
    print("🙌 뒷짐 분석 테스트")
    print("=" * 70)
    print("⌨️  조작법:")
    print("   - ESC: 종료")
    print("   - 'r': 카운터 초기화")
    print("=" * 70)
    print("\n📋 화면 표시:")
    print("   - 초록색: 조건 충족")
    print("   - 빨간색: 조건 미충족")
    print("   - L/R: 왼쪽/오른쪽 손목-어깨 Z축 차이")
    print("=" * 70)
    print("\n💡 감지 조건 (한쪽만 만족해도 됨):")
    print("   - 왼손: 손목 가시성 < 0.4 AND 팔꿈치 Z > 어깨 Z + 0.05")
    print("   - 오른손: 손목 가시성 < 0.4 AND 팔꿈치 Z > 어깨 Z + 0.05")
    print("=" * 70 + "\n")

    detector = HandsBehindBackDetector(fps=fps)
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1

        # 프레임 처리
        processed_image, event_detected, debug_info = detector.process_frame(image)

        # 화면 표시 (좌우 반전)
        cv2.imshow('Hands Behind Back Test', cv2.flip(processed_image, 1))

        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # 리셋
            detector.hands_behind_back_counter = 0
            detector.last_hands_behind_back_event = False
            print("\n🔄 카운터 초기화\n")

    # 최종 통계
    elapsed_time = time.time() - detector.start_time
    print("\n" + "=" * 70)
    print("📊 최종 통계")
    print("=" * 70)
    print(f"총 뒷짐 감지: {detector.total_hands_behind_back_events}회")
    print(f"총 실행 시간: {elapsed_time:.1f}초")
    print(f"처리된 프레임: {frame_count}개")
    if elapsed_time > 0:
        print(f"평균 FPS: {frame_count / elapsed_time:.1f}")
    print("=" * 70)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    print("\n👋 테스트 종료\n")


if __name__ == "__main__":
    main()