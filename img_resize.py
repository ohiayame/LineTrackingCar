import cv2
import os
import numpy as np

def crop_image(image):
    # 이미지 크기 확인
    height, width, _ = image.shape

    # ROI 설정 (도로 영역만 남기기)
    roi_top = int(height * 0.3)  # 상단 30% 잘라내기 (도로만 남기기)
    roi = image[roi_top:height, :]  # 아래쪽 70%만 사용
    return roi

def resize_image(image, target_width=224, target_height=224):
    # 자른 이미지 크기 확인
    new_height, new_width, _ = image.shape

    # 비율 유지하면서 크기 조정
    ratio = min(target_width / new_width, target_height / new_height)
    target_width_resized = int(new_width * ratio)
    target_height_resized = int(new_height * ratio)

    # OpenCV로 리사이즈 (색상 변경을 피하기 위해 PIL을 사용하지 않음)
    resized_image = cv2.resize(image, (target_width_resized, target_height_resized), interpolation=cv2.INTER_LANCZOS4)

    # 새로운 이미지를 224x224 크기로 설정 (검은색 배경)
    new_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 리사이즈된 이미지를 중앙에 배치
    new_image[(target_height - target_height_resized) // 2:(target_height - target_height_resized) // 2 + target_height_resized,
              (target_width - target_width_resized) // 2:(target_width - target_width_resized) // 2 + target_width_resized] = resized_image

    return new_image

def process_images(input_folder, output_folder):
    # 저장할 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내 모든 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"): 
            # 원본 이미지 경로
            image_path = os.path.join(input_folder, filename)
            
            # 이미지 불러오기 (BGR 형식)
            image = cv2.imread(image_path)
            if image is None:
                print(f"이미지를 불러오지 못했습니다: {image_path}")
                continue

            # 이미지 자르기
            cropped_image = crop_image(image)
            
            # 이미지 크기 조정
            resized_image = resize_image(cropped_image)
            
            # 결과 이미지 저장 경로
            output_path = os.path.join(output_folder, f"{filename}")

            # 결과 저장 (BGR 형식)
            cv2.imwrite(output_path, resized_image)
            print(f"전처리된 이미지를 저장했습니다: {output_path}")

# 예시: 폴더 경로와 원하는 크기 설정
# dataset = ["training", "validation"]
lavel = ["60", "75", "86", "105", "120"]

# for i in range(len(dataset)):
for j in range(len(lavel)):
    input_folder = f'Data/{lavel[j]}'  # 원본 이미지가 있는 폴더 경로
    output_folder = f'Data/ResizeData/{lavel[j]}'  # 리사이즈된 이미지를 저장할 폴더 경로

    process_images(input_folder, output_folder)

print("모든 이미지의 전처리가 완료되었습니다.")
