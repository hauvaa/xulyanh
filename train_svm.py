import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder


# Hàm đọc dữ liệu từ folder
def load_data(data_dir):
    images = []
    labels = []

    # Lặp qua từng folder trong thư mục chính
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        # Nếu là folder, đọc các hình ảnh bên trong
        if os.path.isdir(folder_path):
            label = folder_name  # Sử dụng tên folder làm label
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở dạng xám
                img = cv2.resize(img, (28, 28))  # Resize ảnh về kích thước 28x28
                images.append(img.flatten())  # Làm phẳng ảnh
                labels.append(label)

    return np.array(images), np.array(labels)


# Đọc dữ liệu
data_dir = 'data_char'
X, y = load_data(data_dir)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mã hóa nhãn
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Huấn luyện mô hình SVM
model = svm.SVC(kernel='linear', probability=True)  # Sử dụng kernel tuyến tính
model.fit(X_train, y_train_encoded)

# Dự đoán trên tập huấn luyện và kiểm tra
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Tính accuracy
train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)

# Tính loss
y_train_prob = model.predict_proba(X_train)
y_test_prob = model.predict_proba(X_test)

# Lấy danh sách các lớp
labels = np.unique(y_train_encoded)

# Tính log loss với tham số labels
train_loss = log_loss(y_train_encoded, y_train_prob, labels=labels)
test_loss = log_loss(y_test_encoded, y_test_prob, labels=labels)

# In kết quả
print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')
