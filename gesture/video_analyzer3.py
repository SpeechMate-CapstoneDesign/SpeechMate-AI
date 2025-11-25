import cv2
import time
import sys
import os
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

    def analyze(self, show_progress=True, save_video=False, output_video_path=None):
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

        # 비디오 저장 설정
        video_writer = None
        if save_video and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            print(f"📹 처리된 비디오 저장 중: {output_video_path}\n")

        frame_count = 0
        start_time = time.time()
        last_progress_time = start_time

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_timestamp = frame_count / self.fps

            # 프레임 처리
            processed_frame, events = self.detector.process_frame(frame)

            # 비디오 저장
            if video_writer is not None:
                video_writer.write(processed_frame)

            # 이벤트 기록 및 출력
            for event in events:
                time_str = self.format_time(current_timestamp)
                print(f"[{time_str}] {event}")

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

        # 완료
        if video_writer is not None:
            video_writer.release()

        self.cap.release()

        elapsed_total = time.time() - start_time
        print(f"\n\n{'=' * 70}")
        print(f"✅ 분석 완료!")
        print(f"처리 시간: {self.format_time(elapsed_total)}")
        if elapsed_total > 0:
            print(f"처리 속도: {frame_count / elapsed_total:.1f} FPS")
        print("=" * 70 + "\n")

    def generate_report(self):
        """분석 결과 리포트 생성"""
        print("\n" + "=" * 70)
        print("📊 분석 결과 리포트")
        print("=" * 70 + "\n")

        # 통계
        stats = self.detector.get_statistics()

        print("📈 동작별 감지 횟수")
        print("-" * 70)

        # 얼굴터치 통계
        face_touch_stats = {
            '머리터치': stats.get('머리터치', 0),
            '이마터치': stats.get('이마터치', 0),
            '코터치': stats.get('코터치', 0),
            '입술터치': stats.get('입술터치', 0),
            '턱터치': stats.get('턱터치', 0),
            '왼쪽귀터치': stats.get('왼쪽귀터치', 0),
            '오른쪽귀터치': stats.get('오른쪽귀터치', 0)
        }

        face_touch_total = sum(face_touch_stats.values())

        # 얼굴터치 상세 출력
        if face_touch_total > 0:
            print(f"\n얼굴터치 (전체): {face_touch_total}회")
            print("  📍 얼굴터치 상세:")
            for name, count in face_touch_stats.items():
                if count > 0:
                    print(f"    - {name}: {count}회")
            print()
        else:
            print(f"얼굴터치 (전체): 0회\n")

        # 나머지 제스처
        other_gestures = [
            ('팔짱끼기', stats.get('팔짱끼기', 0)),
            ('뒷짐', stats.get('뒷짐', 0)),
            ('무화과잎자세', stats.get('무화과잎자세', 0)),
            ('비스듬한자세', stats.get('비스듬한자세', 0)),
            ('경직된차려', stats.get('경직된차려', 0)),
            ('천장보기', stats.get('천장보기', 0)),
            ('고개숙이기', stats.get('고개숙이기', 0)),
            ('고개흔들기', stats.get('고개흔들기', 0)),
            ('입술깨물기', stats.get('입술깨물기', 0)),
            ('잦은 눈깜빡임', stats.get('눈깜빡임', 0)),
            ('손비비기', stats.get('손비비기', 0)),
        ]

        for name, count in other_gestures:
            print(f"{name}: {count}회")

        print("\n" + "-" * 70)

        # 타임라인
        has_timeline = any(len(timestamps) > 0 for timestamps in self.detector.timeline.values())

        if has_timeline:
            print("\n📅 동작 타임라인")
            print("-" * 70)

            for gesture_name, timestamps in sorted(self.detector.timeline.items()):
                if timestamps:
                    print(f"\n{gesture_name}:")

                    # 연속된 시간대를 구간으로 묶기
                    intervals = self._group_timestamps(timestamps)

                    for start, end in intervals:
                        if end:
                            print(f"  {start} ~ {end}")
                        else:
                            print(f"  {start}")

        print("\n" + "=" * 70)

    def _group_timestamps(self, timestamps):
        """타임스탬프를 구간으로 그룹화 (3초 간격)"""
        if not timestamps:
            return []

        intervals = []
        start = timestamps[0]
        prev = timestamps[0]

        for ts in timestamps[1:]:
            # 3초 이상 차이나면 새로운 구간
            prev_seconds = self._parse_time(prev)
            ts_seconds = self._parse_time(ts)

            if ts_seconds - prev_seconds > 3:
                intervals.append((start, prev))
                start = ts

            prev = ts

        # 마지막 구간
        if start == prev:
            intervals.append((start, None))
        else:
            intervals.append((start, prev))

        return intervals

    def _parse_time(self, time_str):
        """시간 문자열을 초로 변환 (HH:MM:SS)"""
        try:
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0

    def save_report(self, output_path):
        """리포트를 파일로 저장"""
        original_stdout = sys.stdout

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                sys.stdout = f
                self.generate_report()
                sys.stdout = original_stdout

            print(f"📄 리포트가 저장되었습니다: {output_path}")

        except Exception as e:
            sys.stdout = original_stdout
            print(f"❌ 리포트 저장 실패: {e}")

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
    print("🎥 비디오 제스처 분석기")
    print("=" * 70)

    # 비디오 파일 경로 입력
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("\n사용법: python video_analyzer.py <video_path>")
        print("또는 비디오 파일 경로를 입력하세요:")
        video_path = input("비디오 경로: ").strip().strip('"').strip("'")

    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"\n❌ 파일을 찾을 수 없습니다: {video_path}")
        return

    try:
        # 분석기 생성
        analyzer = VideoGestureAnalyzer(video_path)

        # 처리된 비디오 저장 여부
        print("\n처리된 비디오를 저장하시겠습니까? (y/n): ", end='')
        save_choice = input().strip().lower()

        save_video = (save_choice == 'y')
        output_video_path = None

        if save_video:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = f"{base_name}_processed.mp4"

        # 분석 실행
        analyzer.analyze(show_progress=True, save_video=save_video, output_video_path=output_video_path)

        # 리포트 출력
        analyzer.generate_report()

        # 리포트 저장 여부 확인
        print("\n리포트를 파일로 저장하시겠습니까? (y/n): ", end='')
        save_report_choice = input().strip().lower()

        if save_report_choice == 'y':
            # 출력 파일명 생성
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            report_path = f"{base_name}_analysis_report.txt"

            analyzer.save_report(report_path)

        print("\n✅ 분석이 완료되었습니다!")

    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()