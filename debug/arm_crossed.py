import cv2
import mediapipe as mp
import time


class ArmsCrossedDetector:
    def __init__(self, fps=30):
        """팔짱끼기 전용 감지기 - 한쪽 팔만 교차해도 감지"""
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
        self.ARMS_CROSSED_FRAME_COUNT = int(self.fps * 0.3)  # 0.3초

        # 상태 변수
        self.arms_crossed_counter = 0
        self.last_arms_crossed_event = False
        self.total_arms_crossed_events = 0

        # 디버그 정보
        self.debug_info = {}
        self.start_time = time.time()

        # 프레임 카운터
        self.frame_count = 0

        # 이전 상태 저장
        self.last_conditions = None
        self.last_visibility = None

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
            'counter': self.arms_crossed_counter,
            'threshold': self.ARMS_CROSSED_FRAME_COUNT
        }

        # 팔짱끼기 체크
        if results.pose_landmarks:
            event_detected = self._detect_arms_crossed(
                results.pose_landmarks, width, height
            )

        # 시각화
        processed_image = self._visualize(image, results)

        self.frame_count += 1

        return processed_image, event_detected, self.debug_info

    def _detect_arms_crossed(self, pose_landmarks, width, height):
        """팔짱끼기 감지 - 한쪽 팔만 교차해도 감지"""
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

            current_arms_crossed = False

            # 몸통 중심 계산
            body_center_x = (left_shoulder.x + right_shoulder.x) / 2

            # === 좌표 정보 출력 ===
            print("\n" + "=" * 70)
            print("📍 좌표 정보")
            print("=" * 70)
            print(f"어깨:")
            print(f"  왼쪽  X: {left_shoulder.x:.3f}  Y: {left_shoulder.y:.3f}")
            print(f"  오른쪽 X: {right_shoulder.x:.3f}  Y: {right_shoulder.y:.3f}")
            print(f"  중심   X: {body_center_x:.3f}")
            print(f"\n팔꿈치:")
            print(f"  왼쪽  X: {left_elbow.x:.3f}  Y: {left_elbow.y:.3f}  가시성: {left_elbow.visibility:.2f}")
            print(f"  오른쪽 X: {right_elbow.x:.3f}  Y: {right_elbow.y:.3f}  가시성: {right_elbow.visibility:.2f}")
            print(f"\n손목:")
            print(f"  왼쪽  X: {left_wrist.x:.3f}  Y: {left_wrist.y:.3f}  가시성: {left_wrist.visibility:.2f}")
            print(f"  오른쪽 X: {right_wrist.x:.3f}  Y: {right_wrist.y:.3f}  가시성: {right_wrist.visibility:.2f}")

            # 디버그 정보 저장
            self.debug_info.update({
                'left_shoulder_x': left_shoulder.x,
                'right_shoulder_x': right_shoulder.x,
                'body_center_x': body_center_x,
                'left_elbow_x': left_elbow.x,
                'right_elbow_x': right_elbow.x,
                'left_wrist_x': left_wrist.x,
                'right_wrist_x': right_wrist.x,
                'left_elbow_vis': left_elbow.visibility,
                'right_elbow_vis': right_elbow.visibility,
                'left_wrist_vis': left_wrist.visibility,
                'right_wrist_vis': right_wrist.visibility,
            })

            # === 팔짱끼기 조건 체크 (한쪽만 있어도 OK) ===
            print("\n" + "🔍 조건 체크 (한쪽 팔만 교차해도 감지)")
            print("-" * 70)

            # 체크 0: 가시성 (한쪽만 있어도 됨)
            left_visible = left_elbow.visibility > 0.5
            right_visible = right_elbow.visibility > 0.5

            print(f"[{'✓' if left_visible else '✗'}] 왼쪽 팔 가시성: {left_visible} ({left_elbow.visibility:.2f})")
            print(f"[{'✓' if right_visible else '✗'}] 오른쪽 팔 가시성: {right_visible} ({right_elbow.visibility:.2f})")

            if not (left_visible or right_visible):
                print(f"    ❌ 양쪽 팔꿈치 모두 잘 보이지 않음")
                print("=" * 70)
                self.arms_crossed_counter = 0
                self.last_arms_crossed_event = False
                return False

            # 체크 1: 각 팔의 교차 여부 개별 확인
            # 왼팔 체크
            left_arm_crossed = False
            if left_visible:
                left_elbow_crossed = left_elbow.x > body_center_x  # 왼팔꿈치가 중심보다 오른쪽
                left_range_min = body_center_x
                left_range_max = left_shoulder.x + 0.15
                left_elbow_inside = left_range_min < left_elbow.x < left_range_max

                # 왼손목 체크 (보이면)
                left_wrist_ok = True
                if left_wrist.visibility > 0.5:
                    left_wrist_ok = left_wrist.x < body_center_x  # 왼손목은 중심보다 왼쪽

                left_arm_crossed = left_elbow_crossed and left_elbow_inside and left_wrist_ok

                print(f"\n왼팔 분석:")
                print(f"  [{'✓' if left_elbow_crossed else '✗'}] 팔꿈치 교차: {left_elbow_crossed}")
                print(f"      (팔꿈치 {left_elbow.x:.3f} > 중심 {body_center_x:.3f})")
                print(f"  [{'✓' if left_elbow_inside else '✗'}] 팔꿈치 범위: {left_elbow_inside}")
                print(f"      (범위 [{left_range_min:.3f}, {left_range_max:.3f}])")
                print(f"  [{'✓' if left_wrist_ok else '✗'}] 손목 위치: {left_wrist_ok}")
                if left_wrist.visibility > 0.5:
                    print(f"      (손목 {left_wrist.x:.3f} < 중심 {body_center_x:.3f})")
                else:
                    print(f"      (손목 가시성 부족 - 스킵)")
                print(f"  ➜ 왼팔 교차: {'✅ YES' if left_arm_crossed else '❌ NO'}")

            # 오른팔 체크
            right_arm_crossed = False
            if right_visible:
                right_elbow_crossed = right_elbow.x < body_center_x  # 오른팔꿈치가 중심보다 왼쪽
                right_range_min = right_shoulder.x - 0.15
                right_range_max = body_center_x
                right_elbow_inside = right_range_min < right_elbow.x < right_range_max

                # 오른손목 체크 (보이면)
                right_wrist_ok = True
                if right_wrist.visibility > 0.5:
                    right_wrist_ok = right_wrist.x > body_center_x  # 오른손목은 중심보다 오른쪽

                right_arm_crossed = right_elbow_crossed and right_elbow_inside and right_wrist_ok

                print(f"\n오른팔 분석:")
                print(f"  [{'✓' if right_elbow_crossed else '✗'}] 팔꿈치 교차: {right_elbow_crossed}")
                print(f"      (팔꿈치 {right_elbow.x:.3f} < 중심 {body_center_x:.3f})")
                print(f"  [{'✓' if right_elbow_inside else '✗'}] 팔꿈치 범위: {right_elbow_inside}")
                print(f"      (범위 [{right_range_min:.3f}, {right_range_max:.3f}])")
                print(f"  [{'✓' if right_wrist_ok else '✗'}] 손목 위치: {right_wrist_ok}")
                if right_wrist.visibility > 0.5:
                    print(f"      (손목 {right_wrist.x:.3f} > 중심 {body_center_x:.3f})")
                else:
                    print(f"      (손목 가시성 부족 - 스킵)")
                print(f"  ➜ 오른팔 교차: {'✅ YES' if right_arm_crossed else '❌ NO'}")

            # 최종 판정: 한쪽만 교차해도 OK
            current_arms_crossed = left_arm_crossed or right_arm_crossed

            # 어느 팔이 교차했는지 표시
            crossed_arms = []
            if left_arm_crossed:
                crossed_arms.append("왼팔")
            if right_arm_crossed:
                crossed_arms.append("오른팔")

            print("-" * 70)
            if current_arms_crossed:
                print(f"➡️  최종 판정: ✅ 팔짱끼기 자세 ({', '.join(crossed_arms)} 교차)")
            else:
                print(f"➡️  최종 판정: ❌ 팔짱끼기 아님")
            print("=" * 70)

            # 팔짱 이벤트 처리
            if current_arms_crossed:
                self.arms_crossed_counter += 1
                self.debug_info['counter'] = self.arms_crossed_counter
                self.debug_info['crossed_arms'] = crossed_arms

                print(f"\n⏱️  카운터: {self.arms_crossed_counter}/{self.ARMS_CROSSED_FRAME_COUNT}")

                if (self.arms_crossed_counter >= self.ARMS_CROSSED_FRAME_COUNT and
                        not self.last_arms_crossed_event):
                    self.last_arms_crossed_event = True
                    self.total_arms_crossed_events += 1
                    event_detected = True

                    elapsed = time.time() - self.start_time
                    print("\n" + "=" * 70)
                    print("✅ 팔짱끼기 감지!")
                    print("=" * 70)
                    print(f"감지 시간: {elapsed:.1f}초")
                    print(f"교차된 팔: {', '.join(crossed_arms)}")
                    print(f"총 감지 횟수: {self.total_arms_crossed_events}회")
                    print("=" * 70 + "\n")
            else:
                if self.arms_crossed_counter > 0:
                    print(f"\n❌ 카운터 리셋 (조건 미충족)\n")
                self.arms_crossed_counter = 0
                self.last_arms_crossed_event = False
                self.debug_info['counter'] = 0

        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

        return event_detected

    def _visualize(self, image, results):
        """시각화 - X좌표 라인"""
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

            # === X좌표 라인 시각화 ===
            if 'body_center_x' in self.debug_info:
                # 몸통 중심선 (노란색 - 두껍게)
                center_x_px = int(self.debug_info['body_center_x'] * w)
                cv2.line(image, (center_x_px, 0), (center_x_px, h), (0, 255, 255), 3)

                # 왼쪽 어깨 X좌표 (빨간색)
                left_shoulder_px = int(self.debug_info['left_shoulder_x'] * w)
                cv2.line(image, (left_shoulder_px, 0), (left_shoulder_px, h), (0, 0, 255), 2)

                # 오른쪽 어깨 X좌표 (빨간색)
                right_shoulder_px = int(self.debug_info['right_shoulder_x'] * w)
                cv2.line(image, (right_shoulder_px, 0), (right_shoulder_px, h), (0, 0, 255), 2)

                # 왼쪽 팔꿈치 X좌표 (파란색) + 강조 원
                left_elbow = landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW.value]
                left_elbow_px = int(left_elbow.x * w)
                left_elbow_py = int(left_elbow.y * h)
                cv2.line(image, (left_elbow_px, 0), (left_elbow_px, h), (255, 0, 0), 2)
                cv2.circle(image, (left_elbow_px, left_elbow_py), 12, (255, 0, 0), -1)

                # 오른쪽 팔꿈치 X좌표 (파란색) + 강조 원
                right_elbow = landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW.value]
                right_elbow_px = int(right_elbow.x * w)
                right_elbow_py = int(right_elbow.y * h)
                cv2.line(image, (right_elbow_px, 0), (right_elbow_px, h), (255, 0, 0), 2)
                cv2.circle(image, (right_elbow_px, right_elbow_py), 12, (255, 0, 0), -1)

                # 손목 X좌표 (초록색) - 보이면
                if self.debug_info.get('left_wrist_vis', 0) > 0.5:
                    left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
                    left_wrist_px = int(left_wrist.x * w)
                    left_wrist_py = int(left_wrist.y * h)
                    cv2.line(image, (left_wrist_px, 0), (left_wrist_px, h), (0, 255, 0), 1)
                    cv2.circle(image, (left_wrist_px, left_wrist_py), 8, (0, 255, 0), -1)

                if self.debug_info.get('right_wrist_vis', 0) > 0.5:
                    right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
                    right_wrist_px = int(right_wrist.x * w)
                    right_wrist_py = int(right_wrist.y * h)
                    cv2.line(image, (right_wrist_px, 0), (right_wrist_px, h), (0, 255, 0), 1)
                    cv2.circle(image, (right_wrist_px, right_wrist_py), 8, (0, 255, 0), -1)

                # 상태 텍스트 표시
                if 'crossed_arms' in self.debug_info and self.debug_info['crossed_arms']:
                    text = f"Crossed: {', '.join(self.debug_info['crossed_arms'])}"
                    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

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
    print("🤞 팔짱끼기 분석 테스트 (한쪽 팔만 교차해도 감지)")
    print("=" * 70)
    print("⌨️  조작법:")
    print("   - ESC: 종료")
    print("   - 'r': 카운터 초기화")
    print("=" * 70)
    print("\n📋 화면 표시:")
    print("   - 빨간 선: 어깨 X좌표")
    print("   - 노란 선: 몸통 중심 X좌표")
    print("   - 파란 선: 팔꿈치 X좌표 (파란 원으로 강조)")
    print("   - 초록 선: 손목 X좌표 (보일 때)")
    print("=" * 70)
    print("\n💡 감지 조건 (한쪽만 만족해도 됨):")
    print("   - 왼팔: 팔꿈치가 중심보다 오른쪽 + 어깨 범위 안 + 손목이 중심보다 왼쪽")
    print("   - 오른팔: 팔꿈치가 중심보다 왼쪽 + 어깨 범위 안 + 손목이 중심보다 오른쪽")
    print("=" * 70 + "\n")

    detector = ArmsCrossedDetector(fps=fps)
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1

        # 프레임 처리
        processed_image, event_detected, debug_info = detector.process_frame(image)

        # 화면 표시 (좌우 반전)
        cv2.imshow('Arms Crossed Test (Single Arm)', cv2.flip(processed_image, 1))

        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # 리셋
            detector.arms_crossed_counter = 0
            detector.last_arms_crossed_event = False
            print("\n🔄 카운터 초기화\n")

    # 최종 통계
    elapsed_time = time.time() - detector.start_time
    print("\n" + "=" * 70)
    print("📊 최종 통계")
    print("=" * 70)
    print(f"총 팔짱끼기 감지: {detector.total_arms_crossed_events}회")
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