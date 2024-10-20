import cv2

# 이미지 읽기
img = cv2.imread(r"C:\Users\USER\Pictures\line2.jpg")

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 다양한 커널 크기로 블러 적용
kernel_sizes = [(3, 3), (5, 5), (9, 9)]  # 테스트할 커널 크기

for kernel_size in kernel_sizes:
    blurred = cv2.GaussianBlur(gray, kernel_size, 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 결과 이미지 표시
    cv2.imshow(f"Blurred Image with kernel {kernel_size}", blurred)
    cv2.imshow(f"Edges with kernel {kernel_size}", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()