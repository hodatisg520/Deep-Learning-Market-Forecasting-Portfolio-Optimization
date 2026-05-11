# DL4AI-240112-project

## Giới thiệu dự án (Project Overview)

Dự án này là bài tập lớn kết thúc học phần **CS313 - Deep Learning for Artificial Intelligence (Spring 2026)**. Mục tiêu chính là làm quen với dữ liệu chuỗi thời gian (time-series data) và ứng dụng các kỹ thuật học sâu (Deep Learning), cụ thể là mô hình **LSTM (Long Short-Term Memory)**, để phân tích và dự báo thị trường chứng khoán Nasdaq và Việt Nam .

Dự án bao gồm một quy trình toàn diện từ tiền xử lý dữ liệu, kỹ thuật đặc trưng (feature engineering), xây dựng mô hình dự báo giá, nhận diện tín hiệu giao dịch, cho đến quản trị rủi ro và tối ưu hóa danh mục đầu tư .

## Công nghệ sử dụng (Technology Stack)

**Ngôn ngữ lập trình:** Python.

**Thư viện học sâu:** TensorFlow / Keras.

**Phân tích dữ liệu:** Pandas, Numpy, Scikit-learn.

**Trực quan hóa:** Matplotlib, Seaborn.

**Tối ưu hóa danh mục:** PyPortfolioOpt .
 
**Môi trường:** Google Colab.


## Cấu trúc các tác vụ (Project Tasks)

Task 1: Dự báo giá cổ phiếu Nasdaq 

**1.1 Mở rộng đa đặc trưng:** Sử dụng nhiều đặc trưng đầu vào (High, Low, Open, Close, Adj Close, Volume) thay vì chỉ giá Open để dự báo.

**1.2 Dự báo ngày thứ k:** Dự báo giá cổ phiếu tại ngày thứ 3 hoặc ngày thứ 7 trong tương lai.

**1.3 Dự báo chuỗi k ngày:** Dự báo liên tiếp giá cổ phiếu cho k ngày tiếp theo.

Task 2: Dự báo giá cổ phiếu Việt Nam 

* Tương tự Task 1 nhưng áp dụng trên dữ liệu thị trường Việt Nam (mã SAM-VNINDEX), kết hợp xử lý dữ liệu đặc thù như sắp xếp theo trình tự thời gian và chuẩn hóa dữ liệu .



Task 3: Nhận diện tín hiệu giao dịch 

**3.1 Tín hiệu Mua (Buying Signal):** Xây dựng mô hình phân loại nhị phân để xác định thời điểm mua tiềm năng dựa trên xác suất sinh lời .

**3.2 Tín hiệu Bán (Selling Signal):** Xác định các điểm thoát lệnh dựa trên các chỉ báo kỹ thuật như Bollinger Bands và lợi nhuận mục tiêu .

Task 4: Quản trị danh mục và rủi ro 

**4.1 Lựa chọn cổ phiếu:** Đánh giá và chọn lọc các mã cổ phiếu tiềm năng dựa trên xác suất dự báo của mô hình LSTM .

**4.2 Quản trị rủi ro:** Tính toán điểm rủi ro tổng hợp (Composite Risk Score) kết hợp từ xác suất bán của AI, độ biến động và các chỉ báo kỹ thuật .

**4.3 Tối ưu hóa danh mục:** Sử dụng mô hình Markowitz (Efficient Frontier) để phân bổ vốn tối ưu cho các cổ phiếu vượt qua bộ lọc rủi ro .


Task 5: Triển khai mô hình (Extra Credit) 

* Xuất mô hình dưới dạng SavedModel thân thiện với CPU để triển khai qua API và chuẩn bị dữ liệu cho Dashboard SaaS .



## Cấu trúc thư mục dữ liệu (Data Structure)

Mã nguồn yêu cầu cấu trúc thư mục trên Google Drive như sau :

```
/content/drive/MyDrive/data-vn-20230228/
├── stock-historical-data/   # Chứa file CSV lịch sử giá cổ phiếu
├── financial-ratio/         # Chỉ số tài chính (P/E, ROE, ROA)
├── dividend-history/        # Lịch sử chi trả cổ tức
├── companies.csv            # Danh sách công ty
└── ticker-overview.csv      # Tổng quan về các mã chứng khoán

```

## Hướng dẫn cài đặt và chạy (Installation & Usage)

1. **Clone repository:**
```bash
git clone https://github.com/hodatisg520/DL4AI-240112-project.git

```


2. 
**Môi trường chạy:** Mở file `Final_project_DL4AI.ipynb` bằng **Google Colab**.


3. **Kết nối dữ liệu:**
* Tải bộ dữ liệu lên Google Drive cá nhân.
* Thay đổi biến `base_path` trong notebook để trỏ đúng vào thư mục chứa dữ liệu của bạn.




4. 
**Cài đặt thư viện bổ sung:** Chạy lệnh sau trong notebook để cài đặt thư viện tối ưu hóa danh mục:


```python
!pip install PyPortfolioOpt

```


5. **Chạy Notebook:** Chọn `Runtime` -> `Run all` để thực thi toàn bộ quy trình từ đầu đến cuối.

## Kết quả và Quan sát (Findings)

* Mô hình LSTM cho kết quả dự báo giá khá bám sát thực tế với các chỉ số MAE và MAPE ở mức thấp cho thị trường Việt Nam .


* Việc kết hợp Phân tích Kỹ thuật (Technical Analysis) và Phân tích Cơ bản (Fundamental Ratios) giúp cải thiện đáng kể khả năng nhận diện tín hiệu giao dịch .



## Tác giả (Author)

* **Họ và tên:** [Nguyen Hong Dang]
* **Mã số sinh viên:** 240112 
* 
**Lớp:** CS313 Spring 2026
