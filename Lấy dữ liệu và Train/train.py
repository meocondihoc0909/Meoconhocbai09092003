import os
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from mtcnn import MTCNN
import dlib

# Đường dẫn đến các mô hình
shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# Đường dẫn đến thư mục chứa ảnh
path = 'Datasheet'

# Tạo đối tượng phát hiện khuôn mặt MTCNN
detector = MTCNN()
# Tạo đối tượng dự đoán các điểm đặc trưng trên khuôn mặt
predictor = dlib.shape_predictor(shape_predictor_path)
# Tạo đối tượng nhận diện khuôn mặt
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        try:
            # Mở ảnh và chuyển đổi sang ảnh RGB
            PIL_img = Image.open(imagePath).convert('RGB')
            img_numpy = np.array(PIL_img)

            # Lấy ID từ tên tệp (giả sử tên tệp có định dạng User.1.6251020072.JPG)
            id = os.path.split(imagePath)[-1].split(".")[2]  # Lấy phần sau dấu chấm thứ hai
            
            # Phát hiện khuôn mặt bằng MTCNN
            faces = detector.detect_faces(img_numpy)
            if not faces:  # Nếu không có khuôn mặt nào được phát hiện
                print(f"No faces found in image: {imagePath}. Marking as Unknown.")
                continue  # Bỏ qua hình ảnh này

            for face in faces:
                x, y, width, height = face['box']
                # Dự đoán các điểm đặc trưng trên khuôn mặt
                shape = predictor(img_numpy, dlib.rectangle(x, y, x + width, y + height))
                # Tính toán các đặc trưng khuôn mặt
                face_descriptor = face_rec_model.compute_face_descriptor(img_numpy, shape)
                faceSamples.append(np.array(face_descriptor))  # Lưu đặc trưng khuôn mặt
                ids.append(id)  # Lưu ID

                # Vẽ hình chữ nhật quanh khuôn mặt
                cv2.rectangle(img_numpy, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # Hiển thị ID
                cv2.putText(img_numpy, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Hiển thị hình ảnh với khuôn mặt đã phát hiện
            cv2.imshow("Image", img_numpy)
            cv2.waitKey(100)  # Hiển thị mỗi hình ảnh trong 100ms

        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")

    return faceSamples, ids

def save_to_xml(faces, ids, filename):
    root = ET.Element("trainer_data")
    
    for face, id in zip(faces, ids):
        face_element = ET.SubElement(root, "face")
        id_element = ET.SubElement(face_element, "id")
        id_element.text = str(id)  # Lưu ID
        descriptor_element = ET.SubElement(face_element, "descriptor")
        descriptor_element.text = ','.join(map(str, face))  # Chuyển đổi mảng thành chuỗi
        
    tree = ET.ElementTree(root)
    tree.write(filename)

print("\n [INFO] Đang huấn luyện dữ liệu....")
faces, ids = getImagesAndLabels(path)

if len(faces) > 0:
    # Chuyển đổi danh sách đặc trưng khuôn mặt thành mảng NumPy
    faces = np.array(faces)
    ids = np.array(ids)
    
    # Lưu dữ liệu vào tệp trainer.xml
    save_to_xml(faces, ids, 'trainer/trainer.xml')
    
    print("\n [INFO] {0} khuôn mặt được huấn luyện. Dữ liệu đã được lưu vào trainer.xml .".format(len(np.unique(ids))))
else:
    print("\n [INFO] Không có khuôn mặt nào được phát hiện để huấn luyện.")

# Đóng tất cả các cửa sổ hiển thị
cv2.destroyAllWindows()