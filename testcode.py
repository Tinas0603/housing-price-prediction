import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tkinter import ttk, messagebox
import tkinter as tk

def load_and_train_model():
    # Load data
    df = pd.read_csv("housing_price_dataset.csv")
    df['Price'] = df['Price'].apply(lambda x: max(0, x))
    # Preprocess data
    df_encoded = pd.get_dummies(df, columns=['Neighborhood'])
    df_encoded = df_encoded.astype(int)
    scaler = StandardScaler()
    df_encoded = scaler.fit_transform(df_encoded)
    target = df_encoded[:, 4]
    features = np.delete(df_encoded, 4, axis=1)
    # Train the model
    model = LinearRegression()
    model.fit(features, target)
    return model

class HousePricePredictionApp:
    def __init__(self, master, model):
        self.master = master
        self.model = model
        self.master.title("CHƯƠNG TRÌNH DỰ ĐOÁN GIÁ NHÀ")
        self.master.configure(bg='#EEDD82')  # Màu nền cho cửa sổ chính

        # Lấy chiều rộng và chiều cao của màn hình
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # Lấy chiều rộng và chiều cao của cửa sổ
        window_width = 450
        window_height = 300

        # Tính toán để cửa sổ được đặt giữa màn hình
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        # Đặt cửa sổ ở giữa màn hình
        self.master.geometry(f'{window_width}x{window_height}+{x_position}+{y_position}')

        label_title = ttk.Label(self.master, text="DỰ ĐOÁN GIÁ NHÀ DÙNG HỒI QUY TUYẾN TÍNH!", background='#EEDD82',foreground='black')
        label_title.grid(row=0, column=0, columnspan=2,padx=30 ,pady=10, sticky="n")
        self.master.columnconfigure(0, weight=1)  # Cấu hình cột 0 để co giãn theo chiều ngang

        # Tạo các biến kiểm soát Entry
        self.entry_vars = {
            "squarefeet": tk.IntVar(),
            "bedrooms": tk.IntVar(),
            "bathrooms": tk.IntVar(),
            "yearbuilt": tk.IntVar(),
            "neighborhood": tk.StringVar(),
        }

        # Tạo và đặt giá trị cho Entry
        entries = [
            ("Số m²:", "squarefeet", 1),
            ("Phòng ngủ:", "bedrooms", 2),
            ("Phòng tắm:", "bathrooms", 3),
            ("Năm xây dựng:", "yearbuilt", 4),
            ("Vùng (1:nông thôn, 2:ngoại thành, 3:đô thị):", "neighborhood", 5),
        ]
        for label_text, var_name, row in entries:
            label = ttk.Label(self.master, text=label_text, background='#EEDD82',foreground='black')
            label.grid(row=row, column=0, padx=10, pady=10,sticky="w")
            entry = ttk.Entry(self.master, textvariable=self.entry_vars[var_name])
            entry.grid(row=row, column=1, padx=30, pady=10)
            entry.configure(background='#EEDD82',foreground='black')  # Màu nền cho Entry

        # Tạo Button để dự đoán giá nhà
        ttk.Button(self.master, text="Dự đoán giá nhà", command=self.predict_price,).grid(row=6, column=0, columnspan=2, pady=10)

    def validate_neighborhood(self, value):
        try:
            neighborhood = int(value)
            return 1 <= neighborhood <= 3
        except ValueError:
            return False

    def predict_price(self):
        try:
            # Lấy giá trị đầu vào từ Entry
            inputs = {key: value.get() for key, value in self.entry_vars.items()}
            inputs["neighborhood"] = int(inputs["neighborhood"])  # Chuyển đổi giá trị vùng thành số nguyên
            # Kiểm tra giá trị nhập vào cho Neighborhood
            if not self.validate_neighborhood(inputs["neighborhood"]):
                messagebox.showerror("Lỗi", "Vùng chỉ được nhập 1, 2 hoặc 3.")
                return
            # Chuẩn bị dữ liệu để dự đoán
            new_example = np.array([float(inputs["squarefeet"]), int(inputs["bedrooms"]), int(inputs["bathrooms"]), int(inputs["yearbuilt"]), 0, 0, 0])
            new_example[inputs["neighborhood"]] = 4  # Đặt giá trị 1 cho neighborhood tương ứng
            # Dự đoán giá nhà
            predicted_price = self.model.predict(new_example.reshape(1, -1))
            predicted_price_in_currency = predicted_price[0] * 100
            # Hiển thị kết quả
            messagebox.showinfo("Kết quả dự đoán", f"Giá được dự đoán: {predicted_price_in_currency:.2f}$")
            # Xóa nội dung của tất cả các Entry
            for var in self.entry_vars.values():
                var.set("")
        except ValueError:
            messagebox.showerror("Lỗi", "Giá trị nhập vào không hợp lệ.")

def main():
    root = tk.Tk()
    model = load_and_train_model()
    app = HousePricePredictionApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
