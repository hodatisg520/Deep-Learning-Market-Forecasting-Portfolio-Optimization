# Task 5.3: AI Engineering Workflow (Phương án 1)

## 1. Mục tiêu tổng thể (Overall Goal)
Mục tiêu của Task 5.3 là thiết kế một **luồng kỹ thuật tự động hóa (Automated Engineering Workflow)**, biến một mô hình Deep Learning tĩnh trên Jupyter Notebook thành một hệ thống thực tế (production-ready). 
Hệ thống này bao phủ toàn bộ vòng đời của dữ liệu: từ thu thập (Ingestion), biến đổi (Transformation), dự đoán (Prediction) cho đến lưu trữ (Storage) một cách tự động, liên tục và nhất quán.

## 2. Sơ đồ hệ thống (System Diagram)
Luồng công việc (Data Pipeline) được triển khai theo cấu trúc **ELT (Extract, Load, Transform)** kết hợp với dự đoán học sâu. 

```mermaid
graph TD
    %% Định nghĩa các node
    Ext[Nguồn dữ liệu CSV/API]
    DB_Raw[(Database: Raw Tables)]
    DB_Feat[(Database: Feature Tables)]
    DB_Pred[(Database: Predictions)]
    
    subgraph "1. Airbyte (Data Ingestion)"
        AB[Airbyte]
    end
    
    subgraph "2. dbt (Data Transformation)"
        DBT[dbt]
    end
    
    subgraph "3. Python (Prediction)"
        PY[TensorFlow Script]
    end
    
    subgraph "4. Airflow (Orchestration)"
        AF((Apache Airflow))
    end

    %% Luồng dữ liệu
    Ext -->|Đọc file/API| AB
    AB -->|Nạp dữ liệu thô (Extract & Load)| DB_Raw
    
    DB_Raw -.->|Truy vấn bằng SQL| DBT
    DBT -->|Xử lý, tính SMA/RSI (Transform)| DB_Feat
    
    DB_Feat -.->|Kéo dữ liệu đã xử lý| PY
    PY -->|Chạy mô hình & Lưu dự đoán| DB_Pred

    %% Luồng điều khiển của Airflow
    AF ===>|Task 1: Kích hoạt đồng bộ| AB
    AF ===>|Task 2: Kích hoạt biến đổi| DBT
    AF ===>|Task 3: Kích hoạt chạy Model| PY

    %% Style
    classDef orchestrator fill:#f9f0ff,stroke:#d8b4e2,stroke-width:2px;
    class AF orchestrator;
```

## 3. Giải thích từng bước trong Pipeline và tương tác công cụ

Hệ thống được thiết kế dựa trên 4 công cụ chuyên nghiệp trong ngành kỹ thuật dữ liệu (Modern Data Stack), tương tác với nhau theo thứ tự sau:

### Bước 1: Airbyte (Data Ingestion)
- **Vai trò:** Chịu trách nhiệm Extract (Trích xuất) và Load (Nạp) dữ liệu.
- **Cách hoạt động:** Airbyte kết nối với các nguồn dữ liệu bên ngoài (như các file CSV lịch sử của Nasdaq/Việt Nam hoặc API trực tuyến). Định kỳ, Airbyte sẽ tự động đồng bộ (sync) và nạp toàn bộ dữ liệu thô này vào một Cơ sở dữ liệu tập trung (ví dụ: PostgreSQL hoặc MongoDB) thành các bảng thô (`raw_daily_prices`). Bước này đảm bảo database luôn có dữ liệu mới nhất mà không cần tải tay.

### Bước 2: dbt - Data Build Tool (Data Transformation)
- **Vai trò:** Đảm nhiệm bước Transform (Biến đổi) trực tiếp bên trong Database.
- **Cách hoạt động:** Sau khi dữ liệu thô đã nằm trong Database, dbt sẽ thực thi các truy vấn SQL (`stock_features.sql`) để làm sạch dữ liệu. Nó tự động xử lý các giá trị rỗng và tính toán các đặc trưng kỹ thuật phức tạp (như Simple Moving Average - SMA, Relative Strength Index - RSI, độ biến động - Volatility) bằng Window Functions. Dữ liệu sau biến đổi được lưu ngay lại thành một bảng mới (`stock_features`), sạch sẽ và sẵn sàng cho máy học. Việc dùng dbt thay vì pandas giúp tận dụng tối đa sức mạnh tính toán song song của Database.

### Bước 3: Python Script (Prediction & Storage)
- **Vai trò:** Thực hiện suy luận (Inference) dựa trên mô hình Deep Learning (TensorFlow).
- **Cách hoạt động:** Một file Python script (`predict_and_store.py`) được thiết lập để kéo 20 ngày dữ liệu mới nhất từ bảng `stock_features` (đã được dbt chuẩn bị sẵn). Nó định dạng lại dữ liệu thành các tensor 3 chiều, đẩy vào mô hình LSTM (đã được huấn luyện ở Task 1 & 2) để tính ra xác suất tăng/giảm. Cuối cùng, script này sẽ lưu kết quả dự đoán (Ví dụ: `BUY` với độ tin cậy `0.85`) trở lại một bảng mới trong Database (`stock_predictions`). Từ bảng này, các công cụ Dashboard (như Superset/Tableau) có thể truy vấn để hiển thị biểu đồ trực quan.

### Bước 4: Apache Airflow (Orchestration)
- **Vai trò:** Làm "nhạc trưởng" (Orchestrator) điều phối toàn bộ quy trình trên.
- **Cách hoạt động:** Các công cụ ở Bước 1, 2, 3 không tự biết khi nào nên chạy. Airflow sẽ định nghĩa một **DAG (Directed Acyclic Graph)** để hẹn giờ (ví dụ: chạy vào lúc 16h30 mỗi ngày sau khi thị trường đóng cửa). 
  - Đầu tiên, Airflow gọi Airbyte lấy dữ liệu mới.
  - Chờ Airbyte báo thành công, Airflow tiếp tục gọi lệnh `dbt run`.
  - Chờ dbt biến đổi xong, Airflow kích hoạt lệnh `python predict_and_store.py`.
  - Sự phụ thuộc này (`Airbyte >> dbt >> Python`) đảm bảo dữ liệu luôn liền mạch. Nếu một bước thất bại, Airflow sẽ gửi cảnh báo và dừng tiến trình, không để dữ liệu sai lọt vào mô hình dự đoán.

## 4. Kết luận về luồng công việc
Việc sử dụng Phương án 1 (Airbyte + dbt + Airflow + TensorFlow) không chỉ giúp tự động hóa quá trình dự đoán chứng khoán mà còn đáp ứng tiêu chuẩn công nghiệp về MLOps (Machine Learning Operations). Dữ liệu được quản lý minh bạch, tách biệt rõ ràng giữa khâu xử lý data (Data Engineering) và khâu chạy mô hình (Data Science), giúp hệ thống dễ bảo trì và mở rộng trong tương lai.
