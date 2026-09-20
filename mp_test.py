# mp_test.py 测试MediaPipe人体骨骼识别
import cv2
import mediapipe as mp

# 初始化mediapipe姿态识别
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 打开电脑摄像头
cap = cv2.VideoCapture(0)

print("摄像头启动，对着人就能画出骨骼，按 q 关闭窗口")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 图像镜像翻转，方便看
    frame = cv2.flip(frame,1)
    h,w,c = frame.shape
    # BGR转RGB，mediapipe要求
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    # 如果检测到人体，绘制骨骼关键点
    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("MediaPipe人体姿态测试", frame)
    # 按q退出
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("测试结束，MediaPipe正常可用！")