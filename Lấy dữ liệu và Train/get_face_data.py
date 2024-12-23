import cv2
import os
import dlib
import numpy as np
from PIL import Image
import random

# Đường dẫn đến các mô hình
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Danh sách các tệp hình ảnh (có thể thay đổi đường dẫn đến các bức ảnh của bạn)
image_files = [
    'E:/IOT/SinhVien/1.6251020072.jpg',
    'E:/IOT/SinhVien/2.6251020073.jpg',
    'E:/IOT/SinhVien/3.6251020074.jpg',
    'E:/IOT/SinhVien/4.6251020075.jpg'
]

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists("Datasheet"):
    os.makedirs("Datasheet")

# Số lượng hình ảnh cần lưu cho mỗi người
desired_count = 50  # Thay đổi số lượng hình ảnh mong muốn thành 50

# Tạo đối tượng phát hiện khuôn mặt Dlib
detector = dlib.get_frontal_face_detector()
# Tạo đối tượng dự đoán các điểm đặc trưng trên khuôn mặt
predictor = dlib.shape_predictor(shape_predictor_path)
# Tạo đối tượng nhận diện khuôn mặt
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Hàm tăng cường dữ liệu
def augment_image(face):
    # Lật ảnh
    flipped_face = cv2.flip(face, 1)
    return flipped_face

# Lặp qua từng tệp hình ảnh
for img_path in image_files:
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Không thể đọc hình ảnh từ {img_path}.")
        continue

    # Lấy ID từ tên tệp (giả sử tên tệp là 'user.ID.jpg')
    face_id = os.path.splitext(os.path.basename(img_path))[0] # Lấy tên tệp mà không có phần mở rộng

    # Chuyển đổi hình ảnh sang màu xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = detector(gray)

    # Biến đếm số lượng khuôn mặt đã lưu
    count = 0

    # Vẽ hình chữ nhật quanh khuôn mặt và lưu ảnh
    for face in faces:
        while count < desired_count:
            # Lưu khuôn mặt đã phát hiện
            cv2.imwrite(f"Datasheet/User.{face_id}.{count + 1}.JPG", gray[face.top():face.bottom(), face.left():face.right()])
            count += 1

            # Tăng cường dữ liệu: Lật ảnh
            augmented_face = augment_image(gray[face.top():face.bottom(), face.left():face.right()])
            cv2.imwrite(f"Datasheet/User.{face_id}.{count + 1}.JPG", augmented_face)
            count += 1

            # Hiển thị hình ảnh
            cv2.imshow('Image', img)
            cv2.waitKey(100)  # Hiển thị mỗi hình ảnh trong 100ms

    # Nếu không phát hiện khuôn mặt, in ra thông báo
    if count == 0:
        print(f"Không phát hiện khuôn mặt trong hình ảnh {img_path}.")

print("\n [INFO] Đã thu thập khuôn mặt. Thoát")

cv2.destroyAllWindows()