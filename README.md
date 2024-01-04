# housing-price-prediction

Numpy(import numpy as np):
•	Mục đích: Numpy là một thư viện tính toán số cho Python. Nó cung cấp hỗ trợ cho các mảng đa chiều và ma trận lớn, cùng với các hàm toán học để thực hiện các phép toán trên các mảng này.
•	Chức năng: Numpy được sử dụng cho các phép toán số, thay đối mảng, đại số tuyến tính, tạo số ngẫu nhiên và các phép tính toán toán học khác.
Pandas (import pandas as pd):
•	Mục đích: Pandas là một thư viện xử lý và phân tích dữ liệu cho Python. Nó cung cấp các cấu trúc dữ liệu như Series và DataFrame để thực hiện các phép xử lý và phân tích dữ liệu một cách hiệu quả.
•	Chức năng: Pandas thường được sử dụng để đọc và ghi dữ liệu ở các định dạng khác nhau (CSV,Excel, SQL), làm sạch dữ liệu, lọc dữ liệu, nhóm dữ liệu, kết hợp và phân tích thống kê cơ bản.

Seaborn (import seaborn as sns):
Mục đích: Seaborn là một thư viện trực quan hóa dữ liệu dựa trên Matplotlib. Nó cung cấp một giao diện cấp cao để tạo đổ thị thống kê hấp dẫn và thông tin.
Chức năng: Seaborn đơn giản hóa quá trình tạo đồ thị thống kê, chẳng hạn như đồ thị phân tán, đồ thị thanh, biểu đồ nhiệt, và nhiều hơn nữa. Thường được sử dụng để làm đồ họa trực quan hấp dẫn hơn và đơn giản hóa mà để tạo các biểu đồ phức tạp.

Scikit-learn (from sklearn...):
•	Mục đích: Scikit-learn là một thư viện máy học cho Python. Nó cung cấp các công cụ đơn giản và hiệu quả cho khai thác dữ liệu và phân tích dữ liệu, dựa trên NumPy, SciPy và Matplotlib.
•	Chức năng: Scikit-learn bao gồm các mô-đun cho các nhiệm vụ như phân loại, hồi quy, gom nhóm, giảm chiều, và nhiều hơn nữa. 
•	train test_split: Chia một tập dữ liệu thành tập huấn luyện và tập kiểm tra.
•	LinearRegression: Thực hiện hồi quy tuyến tính, một thuật toán hồi quy phổ biến.
•	LabelEncoder. Mà hóa nhân phân loại thành giá trị số.
•	OneHotEncoder: Chuyển đổi các đặc trưng phân loại số nguyên thành biểu diễn mã hóa một-hot.
•	StandardScaler: Chuẩn hóa các đặc trưng bằng cách loại bỏ trung bình và tỷ lệ thành phương sai đơn vị.
•	ColumnTransformer: Áp dụng các bộ biến đổi cho các cột của một mảng hoặc DataFrame.
•	mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score: Các độ đo để đánh giá mô hình hồi quy. 
Matplotlib(from matplotlib import pylot as plt):
Mục đích: Matplotlib là một thư viện vẽ đồ thị 2D cho Python. Nó tạo ra đồ hoạ tĩnh, động và tương tác chất lượng cao trong Python.
Chức năng: Matplotlib được sử dụng rộng rãi để tạo ra nhiều loại đồ thị, bao gồm đồ thị đường, đồ thị phân tán, đồ thị thanh, biểu đồ tần suất và nhiều loại khác.

Chương 3: Xây dựng ứng dụng bằng Python
3.1 Xây dựng ứng dụng và giải thích
Sau khi import các thư viện cần dùng vào chương trình, tiếp đến cần lấy dữ liệu từ file “housing_price_dataset.csv”
		 
Hình 3.1.a: Tập dữ liệu được load vào chương trình bằng thư viện Pandas
Tiếp theo ta dùng hàm describe() để mô tả dữ liệu:
 
	Hình 3.1.b: Thống kê dữ liệu bằng hàm describe().

Từ bảng mô tả trên ta có thể phân tích dữ liệu như sau: 
- Dữ liệu trên có 5 thuộc tính với 50,000 quan sát
- Các thông số khác như: trung bình (mean), độ lệch chuẩn(std),...
- Xuất hiện ngoại lai(outlier): giá min "price" = -36588.165397.
- Lịch sử dữ liệu: 75% dữ liệu giá nhà được xây dưới năm 2003.

Ta thấy được giá trị Price nhỏ nhất là một giá trị âm, điều này có thể làm quá trình huấn luyện mô hình gặp khó khăn khi đưa ra dự đoán. Ta có thể xem như ngôi nhà không được định giá rõ và ta phải xoá đi các quan sát trên.
 
Hình 3.1.c: Hình ảnh dữ liệu sau khi xoá đi các quan sát không rõ ràng ở giá trị ‘Price’.
 
Hình 3.1.d: Biểu đồ giá theo năm trước khi xử lý ngoại lai.
 
Hình 3.1.e Biểu đồ giá theo năm sau khi xử lý ngoại lai.
Tiếp đến ta kiểm tra phân phối của dữ liệu, việc kiểm tra phân phối sẽ giúp chúng ta xem được giá trị đầu ra kỳ vọng dựa trên tập dữ liệu ta đã có.
 
sns.hisplot(df[‘Price’],kde = True, bins=30, color=’skyblue’):
•	sns.histplot: Đây là hàm trong thư viện Seaborn được sử dụng để vẽ biểu đồ histogram.
•	df[‘Price’]: Lấy cột “Price” từ DataFrame df.
•	kde-True: Đặt giá trị này là True để hiển thị cả đường phân phôi xác suất (Kernel Density Estimate- KDE) trên biểu đồ histogram. KDE là một cách để ước lượng hàm mật độ xác suất của một biến ngẫu nhiên.
•	bins-30: Số lượng bins (ngăn) trong histogram, tức là số lượng cột dọc trên trục x.
•	color='skyblue': Đặt màu săc của histogram là 'skyblue'.


plt.title("Phân phối giá"):
•	plt.title: Đặt tiêu đề cho biểu đồ. Trong trường hợp này, tiêu đề là 'Phân phối giá.
plt.xlabel('Giá'):
•	plt.xlabel: Đặt nhãn cho trục x của biểu đồ. Trong trường hợp này, nhãn là ‘Giá’.
pit.ylabel(Tần suất):
•	plt.ylabel: Đặt nhãn cho trục y của biểu đồ. Trong trường hợp này, nhãn là ‘Tần suất’.
plt.show():
•	plt.show: Hiển thị biểu đồ đã vẽ.
Nếu ta chọn giá làm đầu ra dự đoán, kết quả từ mô hình được huấn luyện có khả năng cao rơi vào 200,000$ đến 300,000$ xoay quanh giá trị trung bình.
Ngoài ra, còn có các phân phối khác như:      
Cách thuộc tính còn lại điều cho biểu đồ phân phối đều, điều này giúp cho mô hình có thể học các thuộc tính một cách công bằng, không thiên vị trọng số.
Riêng biểu đồ “Phân phối YearBuilt” ta có thể thấy phân phối của dữ liệu bị ngắt khoảng giữa các năm,dẫn đến không đều trong dữ liệu. Điều này có thế ảnh hưởng đến mô hình vì mô hình sẽ quan tâm nhiềuhơn các dữ liệu có nhiều quan sát hơn so với các năm bị ít quan sát.
Tiếp theo, ta cần biến đổi dữ liệu thổ sang dữ liệu huấn luyện (dữ liệu số). Trước tiên, ta cần biến đổi thuộctính “Neighborhood" từ kiểu phân loại gồm 3 loại như: Rural, Suburb, Urban sang kiểu dữ liệu số bằng -phương pháp mã hóa One-Hot được mô tả bên dưới:
 
pd.get dummies(df,columns=[Neighborhood’]);
•	pd.get dummies: Đây là một hàm của thư viện Pandas được sử dụng để thực hiện quá trình One-Hot Encoding Kết quả của pd.get_dummies thường là một DataFrame mới với các cột nhị phân (True/False) cho các giá trị khác nhau của biến phân loại.
•	df: DataFrame cần được chuyển đổi.
•	columns=[Neighborhooď]: Xác định cột hoặc danh sách các cột cần thực hiện One-Hot Encoding.Trong trường hợp này, cột "Neighborhood" được chon để chuyển đổi.
df_encoded.astype(int): Chuyển đổi True/False thành 0/1.

Tiếp đến ta chuẩn hóa dữ liệu, đưa dữ liệu về cùng một khoảng giá trị có thể giúp mô hình học tập tốt hơn.Trong bài toán này, ta chọn chuẩn hóa theo theo phân phối chuẩn.
 
Scaler = StandardScaler():
•	Tạo một đối tượng StandardScaler, là một trình chuẩn hóa từ thư viện scikit-learn. StandardScaler chuẩn hóa dữ liệu bằng cách loại bỏ trung bình và chia cho độ lệch chuẩn, biến đổi dữ liệu sao cho nó có phân phối chuẩn với trung bình 0 và độ lệch chuẩn 1.
df_encoded=scaler.fit_transform(df_encoded):
•	scaler.fit_transform(df_encoded):Gọi phương thức fit_transform của StandardScaler để thực hiện quá trình chuẩn hóa trên DataFrame 
•	df_encoded. Trong quá trình này, fit được sử dụng để tính toán trung bình và độ lệch chuẩn của từng cột, và sau đó, transform được sử dụng để chuẩn hóa dữ liệu dựa trên các giá trị này.
•	Kết quả của quá trình này là một mảng NumPy chứa dữ liệu đã được chuẩn hóa.


pd.DataFrame(df_encoded):
•	Tạo một DataFrame mới từ màng NumPy chứa dữ liệu dã được chuẩn hóa.
•	Nhằm mục đích hiển thị dữ liệu màng một cách dễ nhìn
Ta cần tách dữ liệu đầu vào X và đầu ra y từ bộ dữ liệu xử lý trên:
 
y = df_encoded[:,4]:
•	Tạo biến y từ cột thứ 4 (cột 5 trong biểu đồ số) của DataFrame df_encoded.
•	df_encoded[:,4]: Lấy tất cả các dòng từ cột thứ 4.

X = np.delete(df_encoded, 4 ,axis=1)
•	Tạo biến X từ DataFrame df_encoded bằng cách loại bỏ cột thứ 4.
•	np.delete(df_encoded, 4 ,axis=1): Sử dụng hàm np.delete từ thư viện NumPy để loại bỏ cột thứ 4( cột thứ 5 trong biểu đồ số) từ df_encoded, axis=1 chỉ định rằng  chúng ta dang thực hiện thao tác trên trục cột.
pd.DataFrame(X), pd.DataFrame(y):
•	Tạo DataFrame mới từ mảng NumPy X và y để hiển thị dữ liệu dưới dạng DataFrame. 
•	Nhằm mục đích hiển thị dữ liệu mảng một cách dễ nhìn.

Tiếp theo cần chia tập huấn luyện (X_train,y_train), tập test(X_test, y_test) từ tập X, y:
 
Train_test_split(X,y,test_size=0.2, random_state=42):
•	X: Ma trận chứa dữ liệu đặc trưng.
•	y: Mảng chứa dữ liệu mục tiêu (biến phụ thuộc).
•	test_size=0.2: Xác định tỷ lệ dữ liệu được chia thành bộ kiểm thử. Trong trường hợp này, 20% của dữ liệu sẽ được sử dụng làm dữ liệu kiểm thử.
•	random_state=42: Seed cho quá trình tạo số ngẫu nhiên, đảm bảo rằng việc chia dữ liệu sẽ được thực hiện 1 cách ngẫu nhiên nhưng có thể tái tạo được.
•	X_train,X_test,y_train,y_test: Các biến chứa dữ liệu của bộ huấn luyện và kiểm thử, X_train và y_train là dữ liệu được sử dụng để huấn luyện mô hình, trong khi X_test và y_test là dữ liệu được sử dụng để kiểm thử mô hình.
Sau khi ta có được tập huấn luyện và tập test, ta tiến hành tạo mô hình và huẩn luyện:
 
model = LinearRegression(): Tạo một đối tượng mô hình hồi quy tuyến tính. Trong trường hợp này, đang sử dụng mô hình hồi quy tuyến tính từ thư viện scikit-learn.
model.fit(X_train, y_train): Sử dụng phương thức fit để huấn luyện mô hình dựa trên dữ liệu đào tạo(X_train,y_train).
•	X_train: Dữ liệu đặc trưng dùng để huấn luyện mô hình.
•	y_train: Dữ liệu mục tiêu(biến phụ thuộc) dùng để huấn luyện mô hình.
Sau khi huấn luyện mô hình hoàn tất, ta có thể đánh giá mô hình qua tập test:
 
Chỉ số R bình phương là 0.57 tức chỉ giải thích được 57% sự biến thiên của y_test. So với ứng dụng thực tế (>0,7).

Ta tiến hành plot dữ liệu để xem khả năng dự đoán so với thực tế : 
 
•	plt.plot(y_test,label= 'Giá thật'): Vẽ đồ thị cho giá trị thực (y_test). Label='Giá thật' là nhãn được sử dụng khi tạo hình thức mô tả(legend) sau này.
•	plt.plot(y_pred, label= 'Giá dự đoán'): Vẽ đồ thị cho giá trị dự đoán (y_pred). label='Giá dự đoán' cũng là nhãn được sử dụng trong hình thức mô tả.
•	plt.xlabel('Số mẫu test'): Đặt nhãn cho trục x, trong trường hợp nãy là số mẫu trong tập kiểm thử.
•	plt.ylabel('Giá nhà'): Đặt nhãn cho trục y, là giá nhà.
•	plt.title('Dự đoán giá nhà'): Đặt tiêu đề cho biểu đồ.
•	plt.legend(): Hiển thị hình thức mô tả cho các dòng đồ thị, dựa trên nhãn được đặt trước đó.
•	plt.show(): Hiển thị biểu đồ.
Chúng ta sẽ quan sát biểu đồ gần hơn với 70 quan sát.
 

•	plt.plot(y_test[-70:],label= 'Giá thật'): Vẽ đồ thị cho giá trị thực (y_test) chỉ trên 70 mẫu cuối cùng. Label='Giá thật' là nhãn được sử dụng khi tạo hình thức mô tả sau này.
•	plt.plot(y_pred[-70:], label= 'Giá dự đoán'): Vẽ đồ thị cho giá trị dự đoán (y_pred) chỉ trên 70 mẫu cuối cùng. label='Giá dự đoán' cũng là nhãn được sử dụng trong hình thức mô tả.
•	plt.xlabel('Số mẫu test'): Đặt nhãn cho trục x, trong trường hợp nãy là số mẫu trong tập kiểm thử.
•	plt.ylabel('Giá nhà'): Đặt nhãn cho trục y, là giá nhà.
•	plt.title('Dự đoán giá nhà (Zoom Mode)'): Đặt tiêu đề cho biểu đồ, chỉ rõ đây là chế độ Zoom.
•	plt.legend(): Hiển thị hình thức mô tả cho các dòng đồ thị, dựa trên nhãn được đặt trước đó.
•	plt.show(): Hiển thị biểu đồ.
Chúng ta sẽ xem xét trọng số đã học của mô hình để đánh giá mô hình:
 
Dựa vào hệ số phương trình: 
•	Biến x1 (SquaredFeet) có tác động dương theo giá (biến x1 tăng thì Price tăng ) và tác động lớn nhất trong các hệ số còn lại(0.751)
•	Các biến còn lại cũng có tác động nhưng không đáng kể
