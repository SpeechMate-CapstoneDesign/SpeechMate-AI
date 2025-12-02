import cv2
import mediapipe as mp
import time
from gesture_detector import GestureDetector

# 한글 -> 영어 번역 맵
GESTURE_TRANSLATION = {
    '고개숙이기': 'Head Down',
    '천장보기': 'Looking Up',
    '입술깨물기': 'Lip Bite',
    '눈깜빡임': 'Frequent Blink',
    '잦은 눈깜빡임': 'Frequent Blink',
    '고개흔들기': 'Head Shake',
    '비스듬한자세': 'Slanted Posture',
    '경직된차려': 'Rigid Attention',
    '팔짱끼기': 'Arms Crossed',
    '뒷짐': 'Hands Behind Back',
    '손비비기': 'Hand Rubbing',
    '무화과잎자세': 'Fig Leaf Pose',
    '머리터치': 'Head Touch',
    '이마터치': 'Forehead Touch',
    '코터치': 'Nose Touch',
    '입술터치': 'Lip Touch',
    '턱터치': 'Chin Touch',
    '왼쪽귀터치': 'Left Ear Touch',
    '오른쪽귀터치': 'Right Ear Touch'
}


def translate_gesture(korean_text):
    """한글 제스처명을 영어로 변환"""
    # 횟수 정보가 포함된 경우 처리
    for korean, english in GESTURE_TRANSLATION.items():
        if korean in korean_text:
            return korean_text.replace(korean, english)
    return korean_text


def run_webcam():
    """웹캠 실시간 제스처 감지 (3초간 텍스트 표시)"""
    # MediaPipe 시각화 도구
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    print("\n" + "=" * 60)
    print("📹 웹캠 제스처 감지 시작!")
    print("=" * 60)
    print("⌨️  조작법:")
    print("  - ESC: 종료")
    print("  - 's': 현재 통계 출력")
    print("  - 'r': 통계 초기화")
    print("=" * 60)
    print("💡 참고: 카메라가 좌우반전되어 보이는 것은")
    print("   웹캠의 기본 동작입니다 (거울 모드)")
    print("=" * 60)
    print("🎥 감지 중...\n")

    # 제스처 감지기 초기화
    detector = GestureDetector(fps=fps)

    # MediaPipe 시각화용 초기화
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0

    # 제스처별 표시 시작 시간 추적 (3초 후 자동 제거)
    gesture_timestamps = {}
    TEXT_DISPLAY_DURATION = 3.0  # 3초

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        frame_count += 1
        image_height, image_width, _ = image.shape
        current_time = time.time()

        # 제스처 감지
        _, new_events = detector.process_frame(image)

        # 새로운 이벤트 발생 시 콘솔 출력 및 타임스탬프 기록
        if new_events:
            elapsed = current_time - detector.start_time
            time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            for event in new_events:
                print(f"[{time_str}] ✅ {event}")

                # "해제" 이벤트는 무시 (3초 후 자동으로 사라짐)
                if "해제" not in event:
                    # 눈깜빡임은 특별 처리 (횟수 정보 포함)
                    if "눈깜빡임" in event:
                        # 기존 눈깜빡임 관련 항목 제거
                        keys_to_remove = [k for k in gesture_timestamps.keys() if "눈깜빡임" in k]
                        for k in keys_to_remove:
                            del gesture_timestamps[k]

                    # 새 제스처 타임스탬프 기록
                    gesture_timestamps[event] = current_time

        # 3초 지난 제스처 제거
        expired_gestures = [
            gesture for gesture, timestamp in gesture_timestamps.items()
            if current_time - timestamp > TEXT_DISPLAY_DURATION
        ]
        for gesture in expired_gestures:
            del gesture_timestamps[gesture]

        # 시각화 (랜드마크 그리기)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_holistic = holistic.process(imgRGB)
        results_facemesh = face_mesh.process(imgRGB)

        # 손 랜드마크
        if results_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results_holistic.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results_holistic.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        # 포즈 랜드마크
        if results_holistic.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results_holistic.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # FaceMesh 그리기 (Contours + Irises만)
        if results_facemesh.multi_face_landmarks:
            for face_landmarks in results_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # === 화면에 활성 제스처 텍스트 표시 (왼쪽 상단, 3초간) ===
        y_offset = 50
        for gesture in sorted(gesture_timestamps.keys()):
            # 영어로 변환
            english_gesture = translate_gesture(gesture)

            # 텍스트 크기 측정
            text_size = cv2.getTextSize(english_gesture, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_width = text_size[0]

            # 왼쪽 상단 정렬
            x_pos = 20

            # 배경 박스 그리기 (가독성 향상)
            cv2.rectangle(
                image,
                (x_pos - 10, y_offset - 35),
                (x_pos + text_width + 10, y_offset + 5),
                (0, 0, 0),  # 검은색 배경
                -1
            )

            # 텍스트 그리기
            cv2.putText(
                image,
                english_gesture,
                (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,  # 폰트 크기
                (255, 140, 0),  # 진한 하늘색 (BGR)
                2,  # 두께
                cv2.LINE_AA
            )
            y_offset += 50

        # 화면 표시 (원본 그대로 - 카메라 자체가 거울모드)
        cv2.imshow('Gesture Detection', image)

        # 키 입력 처리
        key = cv2.waitKey(5) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('s'):  # 통계 출력
            print("\n" + "=" * 60)
            print("📊 현재 통계")
            print("=" * 60)

            # 모든 제스처 목록
            all_gestures = [
                '고개숙이기', '천장보기', '입술깨물기', '눈깜빡임',
                '고개흔들기', '비스듬한자세', '경직된차려', '팔짱끼기', '뒷짐',
                '손비비기', '무화과잎자세', '머리터치', '이마터치', '코터치',
                '입술터치', '턱터치', '왼쪽귀터치', '오른쪽귀터치'
            ]

            stats = detector.get_statistics()

            for gesture in all_gestures:
                count = stats.get(gesture, 0)
                timeline = detector.timeline.get(gesture, [])

                if count > 0:
                    timeline_str = ', '.join(timeline[:5])
                    if len(timeline) > 5:
                        timeline_str += f", ... (총 {len(timeline)}회)"
                    print(f"{gesture}: {count}회")
                    print(f"  타임라인: {timeline_str}")
                else:
                    print(f"{gesture}: 0회")

            print("=" * 60 + "\n")

        elif key == ord('r'):  # 통계 초기화
            detector._init_counters()
            gesture_timestamps.clear()
            print("\n🔄 통계가 초기화되었습니다.\n")

    # 종료 전 최종 통계 출력
    print("\n" + "=" * 60)
    print("📊 최종 통계")
    print("=" * 60)

    all_gestures = [
        '고개숙이기', '천장보기', '입술깨물기', '눈깜빡임',
        '고개흔들기', '비스듬한자세', '경직된차려', '팔짱끼기', '뒷짐',
        '손비비기', '무화과잎자세', '머리터치', '이마터치', '코터치',
        '입술터치', '턱터치', '왼쪽귀터치', '오른쪽귀터치'
    ]

    stats = detector.get_statistics()

    for gesture in all_gestures:
        count = stats.get(gesture, 0)
        timeline = detector.timeline.get(gesture, [])

        if count > 0:
            timeline_str = ', '.join(timeline[:10])
            if len(timeline) > 10:
                timeline_str += f", ... (총 {len(timeline)}회)"
            print(f"{gesture}: {count}회")
            print(f"  타임라인: {timeline_str}")
        else:
            print(f"{gesture}: 0회")

    elapsed_time = time.time() - detector.start_time
    print(f"\n총 실행 시간: {elapsed_time:.1f}초")
    print(f"처리된 프레임: {frame_count}개")
    if elapsed_time > 0:
        print(f"평균 FPS: {frame_count / elapsed_time:.1f}")
    print("=" * 60)

    # 리소스 정리
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    face_mesh.close()
    detector.close()

    print("\n👋 프로그램 종료\n")


if __name__ == "__main__":
    run_webcam()