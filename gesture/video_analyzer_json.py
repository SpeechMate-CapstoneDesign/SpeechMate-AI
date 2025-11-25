import cv2
import time
import sys
import os
import json
from gesture_detector import GestureDetector


class VideoGestureAnalyzer:
    def __init__(self, video_path):
        """비디오 파일 분석기"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        # 비디오 정보
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            self.fps = 30

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # GestureDetector 초기화
        self.detector = GestureDetector(fps=self.fps)

        # 제스처 그룹 매핑
        self.behavior_groups = {
            "HEAD": {
                "label": "머리",
                "behaviors": ["고개숙이기", "천장보기", "고개흔들기"]
            },
            "ARMS": {
                "label": "팔",
                "behaviors": ["팔짱끼기", "뒷짐"]
            },
            "HANDS": {
                "label": "손",
                "behaviors": ["손비비기", "무화과잎자세"]
            },
            "POSTURE": {
                "label": "자세",
                "behaviors": ["비스듬한자세", "경직된차려"]
            },
            "FACE": {
                "label": "얼굴",
                "behaviors": [
                    "입술깨물기", "눈깜빡임",
                    "머리터치", "이마터치", "코터치",
                    "입술터치", "턱터치", "왼쪽귀터치", "오른쪽귀터치"
                ]
            }
        }

    def analyze(self, show_progress=True):
        """비디오 전체 분석"""
        print("\n" + "=" * 70)
        print("🎬 비디오 분석 시작")
        print("=" * 70)
        print(f"파일: {self.video_path}")
        print(f"해상도: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"총 프레임: {self.total_frames}")
        print(f"길이: {self.format_time(self.duration)}")
        print("=" * 70 + "\n")

        frame_count = 0
        start_time = time.time()
        last_progress_time = start_time

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # 프레임 처리
            processed_frame, events = self.detector.process_frame(frame)

            # 진행률 표시 (1초마다)
            current_time = time.time()
            if show_progress and (current_time - last_progress_time >= 1.0):
                progress = (frame_count / self.total_frames) * 100
                elapsed = current_time - start_time

                if frame_count > 0:
                    eta = (elapsed / frame_count) * (self.total_frames - frame_count)
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    print(f"진행률: {progress:.1f}% ({frame_count}/{self.total_frames} 프레임) - "
                          f"처리속도: {fps_processing:.1f} FPS - ETA: {self.format_time(eta)}", end='\r')
                    last_progress_time = current_time

        self.cap.release()

        elapsed_total = time.time() - start_time
        print(f"\n\n{'=' * 70}")
        print(f"✅ 분석 완료!")
        print(f"처리 시간: {self.format_time(elapsed_total)}")
        if elapsed_total > 0:
            print(f"처리 속도: {frame_count / elapsed_total:.1f} FPS")
        print("=" * 70 + "\n")

    def time_to_seconds(self, time_str):
        """HH:MM:SS를 초로 변환"""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0

    def generate_kotlin_json(self):
        """Kotlin 데이터 클래스 구조에 맞춘 JSON 생성"""

        # 각 그룹별로 동작 정리
        results = {}
        total_behavior_types = 0  # 발생한 동작 종류의 개수

        for group_key, group_info in self.behavior_groups.items():
            behaviors = []

            for behavior_name in group_info["behaviors"]:
                timestamps_list = self.detector.timeline.get(behavior_name, [])

                if timestamps_list:
                    # ms 단위 숫자 배열로 변환
                    time_ranges = []
                    for ts in timestamps_list:
                        seconds = self.time_to_seconds(ts)
                        milliseconds = seconds * 1000  # ms로 변환
                        time_ranges.append(milliseconds)

                    behaviors.append({
                        "name": behavior_name,
                        "count": len(timestamps_list),
                        "timestamps": time_ranges  # [11000, 23000, 32000] 형식
                    })

                    total_behavior_types += 1  # 동작 종류 1개 추가

            # 해당 그룹에 동작이 있으면 추가
            if behaviors:
                results[group_key] = behaviors

        # 최종 JSON 구조
        analysis = {
            "totalCount": total_behavior_types,  # 발생한 동작 종류의 개수
            "results": results
        }

        return analysis

    def save_kotlin_json(self, output_path):
        """Kotlin 구조의 JSON을 파일로 저장"""
        try:
            json_data = self.generate_kotlin_json()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            print(f"📄 JSON 리포트가 저장되었습니다: {output_path}")

            # JSON 내용 미리보기
            print("\n" + "=" * 70)
            print("📊 분석 결과 (Kotlin JSON 형식)")
            print("=" * 70)
            print(json.dumps(json_data, ensure_ascii=False, indent=2))
            print("=" * 70)

            # 통계 출력
            print(f"\n📈 발생한 동작 종류: {json_data['totalCount']}개")
            print(f"📊 그룹별 동작 종류:")
            for group_key, behaviors in json_data['results'].items():
                group_label = self.behavior_groups[group_key]['label']
                behavior_types = len(behaviors)
                total_occurrences = sum(b['count'] for b in behaviors)
                print(f"  - {group_label} ({group_key}): {behavior_types}종류 (총 {total_occurrences}회 발생)")

        except Exception as e:
            print(f"❌ JSON 리포트 저장 실패: {e}")

    @staticmethod
    def format_time(seconds):
        """초를 HH:MM:SS 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main():
    """메인 함수"""
    print("\n" + "=" * 70)
    print("🎥 비디오 제스처 분석기 (Kotlin JSON)")
    print("=" * 70)

    # 비디오 파일 경로 입력
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("\n사용법: python video_analyzer_json.py <video_path>")
        print("또는 비디오 파일 경로를 입력하세요:")
        video_path = input("비디오 경로: ").strip().strip('"').strip("'")

    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"\n❌ 파일을 찾을 수 없습니다: {video_path}")
        return

    try:
        # 분석기 생성
        analyzer = VideoGestureAnalyzer(video_path)

        # 분석 실행
        analyzer.analyze(show_progress=True)

        # JSON 리포트 저장
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = f"{base_name}_analysis.json"

        analyzer.save_kotlin_json(json_path)

        print("\n✅ 분석이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()