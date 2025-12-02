#!/usr/bin/env python3
"""
비디오 제스처 분석 → JSON 생성 (호환성 개선 버전)
"""
import cv2
import json
import sys
import time


class VideoGestureDetector:
    """비디오 전용 제스처 감지기 (gesture_detector.py 래퍼)"""

    def __init__(self, fps=30):
        # 원본 GestureDetector import
        from gesture_detector import GestureDetector

        self.detector = GestureDetector(fps=fps)
        self.start_time = time.time()

        # timeline을 ms 기반으로 재구성
        self.ms_timeline = {
            '고개숙이기': [], '천장보기': [], '입술깨물기': [], '눈깜빡임': [],
            '고개흔들기': [], '비스듬한자세': [], '경직된차려': [], '팔짱끼기': [],
            '뒷짐': [], '손비비기': [], '무화과잎자세': [], '머리터치': [],
            '이마터치': [], '코터치': [], '입술터치': [], '턱터치': [],
            '왼쪽귀터치': [], '오른쪽귀터치': []
        }

        # 마지막 이벤트 추적
        self.last_events = set()

    def process_frame(self, image, timestamp_ms):
        """프레임 처리 (비디오 타임스탬프 사용)"""
        # 원본 detector 실행 (timestamp_ms 없이)
        _, events = self.detector.process_frame(image)

        # 새로운 이벤트만 필터링 (해제 이벤트 제외)
        new_gestures = []
        for event in events:
            if "해제" not in event:
                # 제스처 이름 추출
                gesture_name = event.split('(')[0].strip()  # "잦은 눈깜빡임 (15회/10초)" -> "잦은 눈깜빡임"

                # 눈깜빡임 특별 처리
                if "눈깜빡임" in gesture_name:
                    gesture_name = "눈깜빡임"

                # 새로운 이벤트인 경우만 기록
                if gesture_name in self.ms_timeline:
                    if gesture_name not in self.last_events:
                        # ms를 HH:MM:SS 형식으로 변환
                        total_seconds = timestamp_ms // 1000
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        seconds = total_seconds % 60
                        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                        self.ms_timeline[gesture_name].append(time_str)
                        new_gestures.append(gesture_name)
                        self.last_events.add(gesture_name)
            else:
                # 해제 이벤트 처리
                gesture_name = event.replace(" 해제", "").strip()
                self.last_events.discard(gesture_name)

        return new_gestures

    def get_timeline(self):
        """밀리초 기반 타임라인 반환"""
        return self.ms_timeline

    def close(self):
        """리소스 정리"""
        self.detector.close()


def analyze_video_to_json(video_path, json_output_path):
    """비디오 분석하여 JSON 생성"""

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return None

    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # 제스처 감지기 초기화
    detector = VideoGestureDetector(fps=fps)

    # 제스처 그룹
    behavior_groups = {
        "HEAD": ["고개숙이기", "천장보기", "고개흔들기"],
        "ARMS": ["팔짱끼기", "뒷짐"],
        "HANDS": ["손비비기", "무화과잎자세"],
        "POSTURE": ["비스듬한자세", "경직된차려"],
        "FACE": ["입술깨물기", "눈깜빡임", "머리터치", "이마터치",
                 "코터치", "입술터치", "턱터치", "왼쪽귀터치", "오른쪽귀터치"]
    }

    # 프레임 처리
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === 프레임 번호 기반 타임스탬프 계산 (더 정확) ===
        timestamp_ms = int((frame_idx / fps) * 1000)

        # 제스처 감지
        new_gestures = detector.process_frame(frame, timestamp_ms)

        frame_idx += 1

    # JSON 생성
    results = {}
    total_count = 0  # 제스처 종류 수
    timeline = detector.get_timeline()

    for group_key, behaviors_list in behavior_groups.items():
        group_behaviors = []

        for behavior_name in behaviors_list:
            timestamps = timeline.get(behavior_name, [])

            if timestamps:
                group_behaviors.append({
                    "name": behavior_name,
                    "count": len(timestamps),
                    "timestamps": timestamps
                })
                total_count += 1  # 종류 수 증가 (횟수가 아님)

        if group_behaviors:
            results[group_key] = group_behaviors

    json_data = {
        "totalCount": total_count,
        "results": results
    }

    # JSON 저장
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))

    # 리소스 정리
    cap.release()
    detector.close()

    return json_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_to_json.py <video_path> [output_json]")
        print("Example: python video_to_json.py input.mp4 result.json")
        sys.exit(1)

    video_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else video_path.replace('.mp4', '_result.json')

    analyze_video_to_json(video_path, json_path)