#!/usr/bin/env python3
"""
비디오 제스처 분석 - 균형 최적화 버전
정확도를 유지하면서 속도 향상
"""
import cv2
import json
import sys
import time


class BalancedVideoGestureDetector:
    """균형잡힌 비디오 제스처 감지기"""

    def __init__(self, fps=30):
        from gesture_detector import GestureDetector

        self.detector = GestureDetector(fps=fps)

        # 균형 설정 (고정)
        self.target_width = 960  # 중간 해상도
        self.frame_skip = 1  # 모든 프레임 처리
        self.use_downscale = True

        self.ms_timeline = {
            '고개숙이기': [], '천장보기': [], '입술깨물기': [], '눈깜빡임': [],
            '고개흔들기': [], '비스듬한자세': [], '경직된차려': [], '팔짱끼기': [],
            '뒷짐': [], '손비비기': [], '무화과잎자세': [], '머리터치': [],
            '이마터치': [], '코터치': [], '입술터치': [], '턱터치': [],
            '왼쪽귀터치': [], '오른쪽귀터치': []
        }
        self.last_events = set()
        self.frame_count = 0

    def process_frame(self, image, timestamp_ms):
        """프레임 처리"""
        self.frame_count += 1

        # 프레임 스킵
        if self.frame_count % self.frame_skip != 0:
            return []

        # 해상도 다운스케일
        if self.use_downscale and self.target_width:
            h, w = image.shape[:2]
            if w > self.target_width:
                scale = self.target_width / w
                new_w = self.target_width
                new_h = int(h * scale)
                image = cv2.resize(image, (new_w, new_h))

        # 제스처 감지
        _, events = self.detector.process_frame(image)

        # 이벤트 처리
        new_gestures = []
        for event in events:
            if "해제" not in event:
                gesture_name = event.split('(')[0].strip()
                if "눈깜빡임" in gesture_name:
                    gesture_name = "눈깜빡임"

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
                gesture_name = event.replace(" 해제", "").strip()
                self.last_events.discard(gesture_name)

        return new_gestures

    def get_timeline(self):
        return self.ms_timeline

    def close(self):
        self.detector.close()


def analyze_video_balanced(video_path, json_output_path):
    """균형잡힌 비디오 분석"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = BalancedVideoGestureDetector(fps=fps)

    behavior_groups = {
        "HEAD": ["고개숙이기", "천장보기", "고개흔들기"],
        "ARMS": ["팔짱끼기", "뒷짐"],
        "HANDS": ["손비비기", "무화과잎자세"],
        "POSTURE": ["비스듬한자세", "경직된차려"],
        "FACE": ["입술깨물기", "눈깜빡임", "머리터치", "이마터치",
                 "코터치", "입술터치", "턱터치", "왼쪽귀터치", "오른쪽귀터치"]
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 번호 기반 타임스탬프 계산 (더 정확)
        timestamp_ms = int((frame_idx / fps) * 1000)
        detector.process_frame(frame, timestamp_ms)

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

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))

    cap.release()
    detector.close()

    return json_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python balanced_video_to_json.py <video_path> [output_json]")
        print("Example: python balanced_video_to_json.py input.mp4 result.json")
        sys.exit(1)

    video_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else video_path.replace('.mp4', '_result.json')

    analyze_video_balanced(video_path, json_path)