#!/usr/bin/env python3
"""
비디오 랜드마크 시각화 → MP4 생성 (제스처 분석 없음)
"""
import cv2
import mediapipe as mp
import sys


def visualize_landmarks_to_mp4(video_path, output_video_path):
    """비디오에 랜드마크를 그려서 새 MP4 생성"""

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return False

    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Input: {video_path}")
    print(f"   {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"🎬 Output: {output_video_path}")
    print("-" * 60)

    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # MediaPipe 초기화
    mp_holistic = mp.solutions.holistic
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 프레임 처리
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 랜드마크 감지
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_holistic = holistic.process(imgRGB)
        results_facemesh = face_mesh.process(imgRGB)

        # === 포즈 랜드마크 그리기 ===
        if results_holistic.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        # === 손 랜드마크 그리기 ===
        if results_holistic.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        if results_holistic.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results_holistic.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # === FaceMesh 랜드마크 그리기 ===
        if results_facemesh.multi_face_landmarks:
            for face_landmarks in results_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

        # 프레임 저장
        out.write(frame)

        frame_idx += 1

        # 진행률 표시
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            progress = (frame_idx / total_frames) * 100
            print(f"\rProgress: {progress:.1f}% ({frame_idx}/{total_frames})", end='', flush=True)

    print()  # 줄바꿈

    # 리소스 정리
    cap.release()
    out.release()
    holistic.close()
    face_mesh.close()

    print("-" * 60)
    print(f"✅ Video saved: {output_video_path}")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_to_mp4.py <video_path> [output_video]")
        print("Example: python visualize_to_mp4.py input.mp4 output_landmarks.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else video_path.replace('.mp4', '_landmarks.mp4')

    visualize_landmarks_to_mp4(video_path, output_path)