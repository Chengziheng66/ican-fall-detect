# fall_detect.py 人体摔倒检测核心程序
import cv2
import mediapipe as mp
import time
import os

# 初始化mediapipe姿态识别
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 截图保存文件夹
save_dir = "alert_snapshot"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# 摔倒状态标记
is_fall = False
fall_cooldown = 0  # 冷却，防止短时间重复多次告警

cap = cv2.VideoCapture(0)
print("摔倒检测程序启动！模拟摔倒测试，按 q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    # 默认状态
    status_text = "No Person"
    color = (255,255,255)

    if result.pose_landmarks:
        # 绘制骨骼
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        # 获取头部、髋部、脚踝关键点
        nose = landmarks[0]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]

        # 计算髋部平均坐标
        hip_y = (left_hip.y + right_hip.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        nose_y = nose.y

        # 计算人体高度、宽度（归一化坐标）
        body_height = ankle_y - nose_y
        # 摔倒判定规则：人体高度很小，说明人横向躺倒
        if body_height < 0.3:
            status_text = "FALL DETECTED！摔倒告警"
            color = (0,0,255) #红色
            # 冷却判断，避免重复截图
            if fall_cooldown <= 0:
                is_fall = True
                # 保存截图
                save_name = os.path.join(save_dir, f"fall_{time.time():.0f}.jpg")
                cv2.imwrite(save_name, frame)
                print(f"【摔倒事件】已保存截图: {save_name}")
                fall_cooldown = 50
        else:
            status_text = "Standing 正常站立"
            color = (0,255,0) #绿色

    # 冷却倒计时
    if fall_cooldown > 0:
        fall_cooldown -= 1

    # 在画面上打印状态文字
    cv2.putText(frame, status_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("摔倒检测Demo", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("程序退出")