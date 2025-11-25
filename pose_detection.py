import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
my_pose = mpPose.Pose()  # Pose 객체 생성
mpDraw = mp.solutions.drawing_utils

while cv2.waitKey(33) < 0:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 'my_hands.process' 대신 'my_pose.process' 호출
    results = my_pose.process(imgRGB)

    # 3. 'multi_hand_landmarks' 대신 'pose_landmarks' 확인
    if results.pose_landmarks:

        # 4. 'handLms' 대신 'results.pose_landmarks' 사용
        poseLms = results.pose_landmarks

        # 좌표 변환 로직 (이전 답변에서 수정된 픽셀 변환 로직 사용)
        h, w, c = img.shape

        # 5. 모든 랜드마크 반복
        for id, lm in enumerate(poseLms.landmark):
            cx = int(lm.x * w)
            cy = int(lm.y * h)

            # (예시: 코 끝(ID=0)에 원을 그림)
            if id == 0:
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

                # 6. Pose 연결선 그리기
        mpDraw.draw_landmarks(img, poseLms, mpPose.POSE_CONNECTIONS)

    cv2.imshow("Mediapipe Project", img)
cap.release()
cv2.destroyAllWindows()