# app.py V1.1 增加音频告警
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import os
import json
import winsound

app = Flask(__name__, static_folder="alert_snapshot", static_url_path="/alert_snapshot")

# 初始化mediapipe姿态识别
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

save_dir = "alert_snapshot"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

log_file = "record_log.json"
if not os.path.exists(log_file):
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump([], f)

fall_cooldown = 0
cap = cv2.VideoCapture(0)

# 播放告警提示音 Windows专属
def play_alert_sound():
    # 频率800Hz，持续500ms
    winsound.Beep(800, 500)
    time.sleep(0.2)
    winsound.Beep(800, 500)

# 模拟发送告警信息
def send_alert_msg(time_str):
    msg = f"【社区AI无障碍守护系统告警】\n检测到老人摔倒！\n告警时间：{time_str}"
    print("\n========= 模拟推送告警消息 =========")
    print(msg)
    print("====================================\n")

# 写入告警日志
def write_log(time_str, img_name):
    data = {
        "time": time_str,
        "snapshot": img_name,
        "event": "老人摔倒告警"
    }
    with open(log_file, "r", encoding="utf-8") as f:
        log_list = json.load(f)
    log_list.append(data)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_list, f, ensure_ascii=False, indent=2)

# 视频流生成
def gen_frame():
    global fall_cooldown
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame,1)
        h,w,c = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)
        status_text = "No Person"
        color = (255,255,255)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            nose = landmarks[0]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            hip_y = (left_hip.y + right_hip.y)/2
            ankle_y = (left_ankle.y + right_ankle.y)/2
            nose_y = nose.y
            shoulder_y = (left_shoulder.y + right_shoulder.y)/2

            body_height = ankle_y - nose_y
            # 摔倒判定：身高缩小 + 肩膀低于髋部
            if body_height < 0.3 and shoulder_y > hip_y:
                status_text = "FALL DETECTED！摔倒告警"
                color = (0,0,255)
                if fall_cooldown <= 0:
                    timestamp = f"{time.time():.0f}"
                    img_name = f"fall_{timestamp}.jpg"
                    img_path = os.path.join(save_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    write_log(time_str, img_name)
                    send_alert_msg(time_str)
                    play_alert_sound() # 播放提示音
                    print(f"【摔倒告警】{time_str} 截图已保存 {img_path}")
                    fall_cooldown = 80
            else:
                status_text = "Standing 正常站立"
                color = (0,255,0)
        
        if fall_cooldown>0:
            fall_cooldown -=1
        
        cv2.putText(frame, status_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_log')
def get_log():
    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    return jsonify(log_data)

# 新增接口：清空日志
@app.route('/clear_log')
def clear_log():
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump([], f)
    return "ok"

if __name__ == '__main__':
    print("服务器启动，浏览器打开 http://127.0.0.1:5000 访问系统")
    app.run(debug=True)