import cv2
import json
from gesture_detector import GestureDetector


class VideoGestureAnalyzer:
    def __init__(self, video_path):
        """비디오 파일 분석기 (서버용 최적화)"""
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # 비디오 정보
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30

        # GestureDetector 초기화
        self.detector = GestureDetector(fps=self.fps)

        # 제스처 그룹 매핑
        self.behavior_groups = {
            "HEAD": ["고개숙이기", "천장보기", "고개흔들기"],
            "ARMS": ["팔짱끼기", "뒷짐"],
            "HANDS": ["손비비기", "무화과잎자세"],
            "POSTURE": ["비스듬한자세", "경직된차려"],
            "FACE": ["입술깨물기", "눈깜빡임", "머리터치", "이마터치",
                     "코터치", "입술터치", "턱터치", "왼쪽귀터치", "오른쪽귀터치"]
        }

    def analyze(self):
        """비디오 전체 분석 (최적화)"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 프레임 처리 (시각화 없음)
            self.detector.process_frame(frame)

        self.cap.release()

    def _time_to_ms(self, time_str):
        """HH:MM:SS를 ms로 변환"""
        parts = time_str.split(':')
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return (hours * 3600 + minutes * 60 + seconds) * 1000

    def generate_json(self):
        """JSON 생성 (최적화)"""
        results = {}
        total_count = 0

        for group_key, behaviors_list in self.behavior_groups.items():
            group_behaviors = []

            for behavior_name in behaviors_list:
                timestamps = self.detector.timeline.get(behavior_name, [])

                if timestamps:
                    group_behaviors.append({
                        "name": behavior_name,
                        "count": len(timestamps),
                        "timestamps": [self._time_to_ms(ts) for ts in timestamps]
                    })
                    total_count += 1

            if group_behaviors:
                results[group_key] = group_behaviors

        return {
            "totalCount": total_count,
            "results": results
        }

    def save_json(self, output_path):
        """JSON 파일 저장"""
        json_data = self.generate_json()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, separators=(',', ':'))

        return json_data


def analyze_video(video_path, output_path=None):
    """
    비디오 분석 및 JSON 생성 (서버용 함수)

    Args:
        video_path: 비디오 파일 경로
        output_path: JSON 저장 경로 (None이면 자동 생성)

    Returns:
        dict: 분석 결과 JSON
    """
    analyzer = VideoGestureAnalyzer(video_path)
    analyzer.analyze()

    if output_path is None:
        import os
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_analysis.json"

    return analyzer.save_json(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = analyze_video(video_path, output_path)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)