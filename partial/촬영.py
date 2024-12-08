import cv2
import keyboard

# カメラの初期化
cap = cv2.VideoCapture(0)

# 動画の書き出し設定
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームの表示
    cv2.imshow('frame', frame)

    # 'r'キーで録画の開始・停止を切り替え
    if keyboard.is_pressed('r'):
        if recording:
            # 録画停止
            recording = False
            out.release()
            print("Recording stopped")
        else:
            # 録画開始
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print("Recording started")

    # 録画中ならフレームを保存
    if recording:
        out.write(frame)

    # 'q'キーで終了
    if keyboard.is_pressed('q'):
        break

# リソースの解放
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()

