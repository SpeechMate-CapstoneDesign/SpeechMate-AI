import cv2
import mediapipe as mp
import time


class HeadPoseDetector:
    def __init__(self, fps=30):
        """천장보기/고개숙이기 전용 감지기"""
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.fps = fps

        # 파라미터
        self.NPR_CEILING_THRESHOLD = 0.25  # 천장보기 임계값 (0.20 → 0.25로 증가)
        self.NPR_FLOOR_THRESHOLD = 0.44  # 고개숙이기 임계값
        self.POSE_FRAME_COUNT = int(self.fps * 0.8)  # 0.8초

        # 상태 변수
        self.pose_status = "Forward"
        self.pose_counter = 0
        self.last_pose_event = ""

        # 카운터
        self.total_head_bow_events = 0
        self.total_ceiling_events = 0

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
        results = self.face_mesh.process(imgRGB)

        # 이벤트
        event_detected = None

        # 디버그 정보 초기화
        self.debug_info = {
            'face_detected': False,
            'counter': self.pose_counter,
            'threshold': self.POSE_FRAME_COUNT,
            'status': self.pose_status
        }

        # 얼굴 감지 체크
        if results.multi_face_landmarks:
            event_detected = self._detect_head_pose(
                results.multi_face_landmarks[0],
                width, height
            )

        # 시각화
        processed_image = self._visualize(image, results)

        self.frame_count += 1

        return processed_image, event_detected, self.debug_info

    def _detect_head_pose(self, face_landmarks, width, height):
        """천장보기/고개숙이기 감지"""
        landmarks = face_landmarks.landmark
        event_detected = None

        try:
            # 랜드마크 추출
            eye_center = landmarks[6]  # 눈 중심
            nose_tip = landmarks[1]  # 코끝
            chin_tip = landmarks[152]  # 턱

            # === 좌표 정보 출력 ===
            print("\n" + "=" * 70)
            print("📍 얼굴 랜드마크 좌표")
            print("=" * 70)
            print(f"눈 중심(6):  Y: {eye_center.y:.3f}")
            print(f"코끝(1):    Y: {nose_tip.y:.3f}")
            print(f"턱(152):    Y: {chin_tip.y:.3f}")

            # 얼굴 세로 높이
            face_vertical_height = chin_tip.y - eye_center.y
            print(f"\n얼굴 세로 높이: {face_vertical_height:.3f}")

            # NPR (Nose Position Ratio) 계산
            current_pose_direction = "Forward"

            if face_vertical_height > 0.01:
                avg_npr = (nose_tip.y - eye_center.y) / face_vertical_height

                print(f"NPR 계산: (코끝 - 눈중심) / 얼굴높이")
                print(f"NPR = ({nose_tip.y:.3f} - {eye_center.y:.3f}) / {face_vertical_height:.3f}")
                print(f"NPR = {avg_npr:.3f}")

                # 디버그 정보 저장
                self.debug_info.update({
                    'face_detected': True,
                    'eye_center_y': eye_center.y,
                    'nose_tip_y': nose_tip.y,
                    'chin_tip_y': chin_tip.y,
                    'face_height': face_vertical_height,
                    'npr': avg_npr
                })

                # === NPR 기반 방향 판정 ===
                print("\n" + "🔍 방향 판정")
                print("-" * 70)

                if avg_npr > self.NPR_FLOOR_THRESHOLD:
                    current_pose_direction = "HeadBow"  # 고개숙이기
                    print(f"[✓] NPR {avg_npr:.3f} > {self.NPR_FLOOR_THRESHOLD} → 고개숙이기")
                elif avg_npr < self.NPR_CEILING_THRESHOLD:
                    current_pose_direction = "Ceiling"  # 천장보기
                    print(f"[✓] NPR {avg_npr:.3f} < {self.NPR_CEILING_THRESHOLD} → 천장보기")
                else:
                    current_pose_direction = "Forward"  # 정면
                    print(f"[○] NPR {avg_npr:.3f} (정면 범위: {self.NPR_CEILING_THRESHOLD} ~ {self.NPR_FLOOR_THRESHOLD})")

                print("-" * 70)
            else:
                print("\n❌ 얼굴 높이가 너무 작음 (감지 실패)")
                print("=" * 70)
                return None

            # === 이벤트 처리 ===
            print(f"\n현재 상태: {current_pose_direction}")
            print(f"이전 상태: {self.pose_status}")
            print(f"카운터: {self.pose_counter}/{self.POSE_FRAME_COUNT}")

            if current_pose_direction != "Forward":
                # 정면이 아닌 자세
                if current_pose_direction == self.pose_status:
                    # 같은 자세 유지 중
                    self.pose_counter += 1
                    print(f"→ 같은 자세 유지 중, 카운터 증가: {self.pose_counter}")
                else:
                    # 새로운 자세로 변경
                    print(f"→ 자세 변경: {self.pose_status} → {current_pose_direction}")
                    self.pose_status = current_pose_direction
                    self.pose_counter = 1
                    self.last_pose_event = ""

                # 충분히 유지했는지 체크
                if (self.pose_counter >= self.POSE_FRAME_COUNT and
                        self.last_pose_event != self.pose_status):
                    self.last_pose_event = self.pose_status
                    elapsed = time.time() - self.start_time

                    print("\n" + "=" * 70)
                    if self.pose_status == "HeadBow":
                        self.total_head_bow_events += 1
                        event_detected = "고개숙이기"
                        print("✅ 고개숙이기 감지!")
                    elif self.pose_status == "Ceiling":
                        self.total_ceiling_events += 1
                        event_detected = "천장보기"
                        print("✅ 천장보기 감지!")

                    print("=" * 70)
                    print(f"감지 시간: {elapsed:.1f}초")
                    print(f"유지 프레임: {self.pose_counter}")
                    print(f"고개숙이기 총: {self.total_head_bow_events}회")
                    print(f"천장보기 총: {self.total_ceiling_events}회")
                    print("=" * 70 + "\n")
                else:
                    remaining = self.POSE_FRAME_COUNT - self.pose_counter
                    print(f"→ {remaining}프레임 더 유지 필요")
            else:
                # 정면으로 돌아옴
                if self.pose_counter > 0:
                    print(f"→ 정면으로 복귀, 카운터 리셋")
                self.pose_counter = 0
                self.pose_status = "Forward"
                self.last_pose_event = ""

            print("=" * 70)

            self.debug_info['counter'] = self.pose_counter
            self.debug_info['status'] = self.pose_status

        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

        return event_detected

    def _visualize(self, image, results):
        """시각화"""
        h, w, _ = image.shape

        # 얼굴 랜드마크 그리기 (눈만)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # 눈 그리기
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1
                )
            )
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1
                )
            )

            # 주요 포인트 표시
            if 'eye_center_y' in self.debug_info:
                landmarks = face_landmarks.landmark

                # 눈 중심 (초록)
                eye_center = landmarks[6]
                eye_px = (int(eye_center.x * w), int(eye_center.y * h))
                cv2.circle(image, eye_px, 8, (0, 255, 0), -1)
                cv2.circle(image, eye_px, 10, (255, 255, 255), 2)

                # 코끝 (파랑)
                nose_tip = landmarks[1]
                nose_px = (int(nose_tip.x * w), int(nose_tip.y * h))
                cv2.circle(image, nose_px, 8, (255, 0, 0), -1)
                cv2.circle(image, nose_px, 10, (255, 255, 255), 2)

                # 턱 (빨강)
                chin_tip = landmarks[152]
                chin_px = (int(chin_tip.x * w), int(chin_tip.y * h))
                cv2.circle(image, chin_px, 8, (0, 0, 255), -1)
                cv2.circle(image, chin_px, 10, (255, 255, 255), 2)

                # 연결선
                cv2.line(image, eye_px, nose_px, (255, 255, 0), 2)
                cv2.line(image, nose_px, chin_px, (255, 255, 0), 2)

        # 텍스트 정보 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        if 'face_detected' in self.debug_info and self.debug_info['face_detected']:
            # NPR 정보
            npr = self.debug_info.get('npr', 0)

            # NPR 값 색상 결정
            if npr > self.NPR_FLOOR_THRESHOLD:
                npr_color = (0, 165, 255)  # 주황 (고개숙이기)
                status_text = "HEAD BOW"
            elif npr < self.NPR_CEILING_THRESHOLD:
                npr_color = (255, 0, 255)  # 보라 (천장보기)
                status_text = "CEILING"
            else:
                npr_color = (0, 255, 0)  # 초록 (정면)
                status_text = "FORWARD"

            cv2.putText(image, f"NPR: {npr:.3f}",
                        (10, y_offset), font, 0.7, npr_color, 2)
            y_offset += 35

            cv2.putText(image, status_text,
                        (10, y_offset), font, 0.8, npr_color, 2)
            y_offset += 40

            # NPR 범위 표시
            cv2.putText(image, f"Ceiling < {self.NPR_CEILING_THRESHOLD:.2f}",
                        (10, y_offset), font, 0.5, (255, 0, 255), 1)
            y_offset += 25
            cv2.putText(image, f"Forward: {self.NPR_CEILING_THRESHOLD:.2f} ~ {self.NPR_FLOOR_THRESHOLD:.2f}",
                        (10, y_offset), font, 0.5, (0, 255, 0), 1)
            y_offset += 25
            cv2.putText(image, f"HeadBow > {self.NPR_FLOOR_THRESHOLD:.2f}",
                        (10, y_offset), font, 0.5, (0, 165, 255), 1)
            y_offset += 40

            # 카운터 정보
            counter = self.debug_info.get('counter', 0)
            threshold = self.debug_info.get('threshold', 0)

            counter_color = (0, 255, 0) if counter >= threshold else (0, 165, 255)
            cv2.putText(image, f"Counter: {counter}/{threshold}",
                        (10, y_offset), font, 0.7, counter_color, 2)
            y_offset += 35

            # 진행 바
            bar_width = 300
            bar_height = 20
            bar_x = 10
            bar_y = y_offset

            # 배경 (회색)
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (100, 100, 100), -1)

            # 진행 상황 (초록)
            if threshold > 0:
                progress = min(counter / threshold, 1.0)
                progress_width = int(bar_width * progress)
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                              (0, 255, 0), -1)

            # 테두리
            cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (255, 255, 255), 2)

            y_offset += 40

            # 통계
            cv2.putText(image, f"HeadBow: {self.total_head_bow_events}",
                        (10, h - 60), font, 0.6, (0, 165, 255), 2)
            cv2.putText(image, f"Ceiling: {self.total_ceiling_events}",
                        (10, h - 30), font, 0.6, (255, 0, 255), 2)

        else:
            cv2.putText(image, "Waiting for face...",
                        (10, y_offset), font, 0.7, (0, 165, 255), 2)

        return image

    def close(self):
        """리소스 정리"""
        self.face_mesh.close()


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
    print("👀 천장보기/고개숙이기 분석 테스트")
    print("=" * 70)
    print("⌨️  조작법:")
    print("   - ESC: 종료")
    print("   - 'r': 카운터 초기화")
    print("=" * 70)
    print("\n📋 화면 표시:")
    print("   - 초록 원: 눈 중심")
    print("   - 파란 원: 코끝")
    print("   - 빨간 원: 턱")
    print("   - 노란 선: 얼굴 라인")
    print("=" * 70)
    print("\n💡 NPR (Nose Position Ratio):")
    print("   - NPR < 0.25: 천장보기 (보라색) ← 더 민감하게")
    print("   - 0.25 ~ 0.44: 정면 (초록색)")
    print("   - NPR > 0.44: 고개숙이기 (주황색)")
    print("   - 0.8초 유지 시 감지")
    print("=" * 70 + "\n")

    detector = HeadPoseDetector(fps=fps)
    frame_count = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1

        # 프레임 처리
        processed_image, event_detected, debug_info = detector.process_frame(image)

        # 화면 표시 (좌우 반전)
        cv2.imshow('Head Pose Test', cv2.flip(processed_image, 1))

        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # 리셋
            detector.pose_counter = 0
            detector.pose_status = "Forward"
            detector.last_pose_event = ""
            print("\n🔄 카운터 초기화\n")

    # 최종 통계
    elapsed_time = time.time() - detector.start_time
    print("\n" + "=" * 70)
    print("📊 최종 통계")
    print("=" * 70)
    print(f"고개숙이기 감지: {detector.total_head_bow_events}회")
    print(f"천장보기 감지: {detector.total_ceiling_events}회")
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