import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os


# # CUDA가 가능하면 FP16 적용, CPU 모드에서는 사용하지 않음
# if torch.cuda.is_available():
#     model.model.half()  # GPU에서만 FP16 적용
# else:
#     model.model.float()  # CPU에서는 FP32 사용

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 


# 특정 구역
frame_count = 0
save_frame_interval = 10  # 10 프레임마다 저장

image_num = 1
start_image = image_num
max_images = 15

# 저장 경로
save_dir = r"C:\\Users\\USER\\Track\\training"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Tkinter 윈도우 생성
root = tk.Tk()
label = tk.Label(root)
label.pack()


def show_frame():
    global frame_count
    global image_num
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 480))  # 해상도 조정
    
    # 구역 표시
    frame_count += 1

    for result in frame:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()

            if conf > 0.5:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 10 프레임마다 이미지 저장
                if frame_count % save_frame_interval == 0:
                    image_number = str(image_num).zfill(5)
                    cropped_frame = frame[y1:y2, x1:x2]
                    
                    blurred = cv2.GaussianBlur(cropped_frame, (9, 9), 0)
                    edges = cv2.Canny(blurred, 30, 150)
                    
                    image_path = os.path.join(save_dir, f"{image_number}.png")
                    cv2.imwrite(image_path, edges)
                    image_num += 1
                    
                    # 35장 저장 후 프로그램 종료
                    if image_num - start_image >= max_images:
                        print(f"{max_images}장의 이미지를 저장했으므로 프로그램을 종료합니다.")
                        cap.release()
                        cv2.destroyAllWindows()
                        root.quit()  # Tkinter 루프 종료
                        return

    # OpenCV 이미지를 Pillow 이미지로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    # Tkinter Label에 이미지 표시
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # 1ms마다 업데이트
    root.after(1, show_frame)

# 실시간 영상 표시 시작
show_frame()

# Tkinter 이벤트 루프 시작
root.mainloop()

# 비디오 캡처 및 윈도우 해제
cap.release()
cv2.destroyAllWindows()
