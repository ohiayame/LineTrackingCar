import os
import matplotlib.pyplot as plt

def count_images_in_folders(parent_folder):
    folder_counts = {}
    total_count = 0
    
    # 각 하위 폴더 탐색
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        
        # 디렉토리인지 확인
        if os.path.isdir(folder_path):
            # 이미지 파일 필터링
            image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg'))]
            folder_counts[folder_name] = len(image_files)
            total_count += len(image_files)
    return folder_counts, total_count

def plot_folder_counts(folder_counts, total_count):
    # 데이터 분리
    folders = list(folder_counts.keys())
    counts = list(folder_counts.values())
    
    # 그래프 생성
    plt.bar(folders, counts, color='skyblue')
    plt.xlabel('Folders')
    plt.ylabel('Number of Images')
    plt.title('Image Counts in Each Folder')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print(f"Total number of images across all folders: {total_count}")

# 메인 실행 코드
if __name__ == "__main__":
    parent_folder_path = "Data"  # 여기에 폴더 경로 입력
    folder_counts, total_count  = count_images_in_folders(parent_folder_path)
    
    plot_folder_counts(folder_counts, total_count)
