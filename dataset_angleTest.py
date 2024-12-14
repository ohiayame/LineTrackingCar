import os
import cv2
import numpy as np
import shutil

# 화살표 그리기 함수
def draw_arrow(frame, angle, index):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    length = 50  # 화살표 길이

    # 왼쪽 0도, 위쪽 90도, 오른쪽 180도를 반영하기 위한 각도 변환
    angle_rad = np.radians(180 - angle)  # OpenCV의 기준에서 변환

    end_x = int(center_x + length * np.cos(angle_rad))
    end_y = int(center_y - length * np.sin(angle_rad))  # OpenCV의 y축은 아래로 증가하므로 음수로 변환
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3, tipLength=0.3)
    text = f"Image: {index + 1}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# 이미지 검증 함수
def validate_images(folder_path):
    files = sorted(os.listdir(folder_path))  # 폴더 내 파일 정렬
    deleat_folder = os.path.join(folder_path, "deleat")
    os.makedirs(deleat_folder, exist_ok=True)

    index = 0
    total_index = 0
    while 0 <= index < len(files):
        file_name = files[index]
        file_path = os.path.join(folder_path, file_name)

        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            index += 1
            continue

        # 이미지 읽기
        image = cv2.imread(file_path)
        if image is None:
            print(f"Cannot read file: {file_name}")
            index += 1
            continue

        # 폴더 이름에서 각도 추출
        try:
            angle = int(os.path.basename(folder_path))  # 폴더 이름이 각도라고 가정
        except ValueError:
            print(f"Invalid folder name for angle: {folder_path}")
            break

        # 화살표 추가
        annotated_image = draw_arrow(image.copy(), angle, total_index)

        # 이미지 표시
        cv2.imshow("Image Validation", annotated_image)
        key = cv2.waitKey(1)

        if key == ord('d'):  # 삭제 키
            print(f"Deleting {file_name}...")
            shutil.move(file_path, os.path.join(deleat_folder, file_name))
            files.pop(index)  # 파일 리스트에서 제거
        elif key == ord('l'):  # 오른쪽 화살표 (다음 사진)
            index += 1
            total_index += 1
        elif key == ord('j'):  # 왼쪽 화살표 (이전 사진)
            index = max(index - 1, 0)
            total_index -= 1
        elif key == 27:  # ESC 키로 종료
            break

    cv2.destroyAllWindows()

# 실행 예시
folder_path = r"C:/Users/USER/LineTrackingCar/Dataset/ResizeData/validation/65"  # 이미지 폴더 경로
validate_images(folder_path)
# "C:\Users\USER\LineTrackingCar\Dataset\ResizeData\training\class65"