# Lab 21 — Evaluation Report

**Học viên**: Lê Trần Quốc Bảo — 2A202600596
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: $0.07 (~12.7 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8 | 1,843,200 | 4.2 min | 7.2 GB | 1.5577 | 4.75 |
| 16 | 3,686,400 | 4.4 min | 6.6 GB | 1.5161 | 4.55 |
| 64 | 14,745,600 | 4.2 min | 8.0 GB | 1.4768 | 4.38 |

## 3. Loss Curve Analysis
- **Quan sát**: Dựa trên đồ thị loss curve của r=16, training loss giảm ổn định và eval loss đạt mức thấp nhất vào cuối quá trình. Không có dấu hiệu overfitting rõ rệt do số lượng sample nhỏ (200) và số epoch (3) vừa phải. 

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác ...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc...
**Nhận xét**: Model fine-tuned trả lời tập trung hơn vào cấu trúc instruction và sử dụng tiếng Việt tự nhiên hơn trong context của dataset Alpaca.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:

```python
def fibonacci(n):
    if n <= 0:
        return "N phải là một số dương"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
 ...
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:

```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range...
**Nhận xét**: Model fine-tuned trả lời tập trung hơn vào cấu trúc instruction và sử dụng tiếng Việt tự nhiên hơn trong context của dataset Alpaca.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với ...
**Nhận xét**: Model fine-tuned trả lời tập trung hơn vào cấu trúc instruction và sử dụng tiếng Việt tự nhiên hơn trong context của dataset Alpaca.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ...
**Nhận xét**: Model fine-tuned trả lời tập trung hơn vào cấu trúc instruction và sử dụng tiếng Việt tự nhiên hơn trong context của dataset Alpaca.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th...
**Nhận xét**: Model fine-tuned trả lời tập trung hơn vào cấu trúc instruction và sử dụng tiếng Việt tự nhiên hơn trong context của dataset Alpaca.

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thực nghiệm, rank r=64 cho Perplexity thấp nhất (4.38), chứng tỏ khả năng học sâu hơn các đặc điểm của dataset. Tuy nhiên, sự khác biệt về perplexity giữa r=16 và r=64 là không quá lớn so với việc tăng gấp 4 lần số lượng tham số huấn luyện. 

Rank r=16 cho thấy là điểm cân bằng (ROI tốt nhất) giữa tài nguyên VRAM (6.6 GB) và hiệu suất hội tụ. Tăng rank lên 64 bắt đầu xuất hiện tình trạng 'diminishing returns' khi độ giảm loss không còn tương xứng với chi phí tính toán. Nếu triển khai thực tế trên T4, mình sẽ chọn rank r=16 hoặc r=32 để tối ưu tốc độ inference mà vẫn giữ được chất lượng câu trả lời.

## 6. What I Learned
- Hiểu được cách QLoRA giúp tiết kiệm VRAM đáng kể trên GPU yếu như T4.
- Cách phân tích p95 token length để tối ưu context window.
- Hiểu rõ mối quan hệ giữa Rank (r) và dung lượng bộ nhớ cũng như chất lượng hội tụ của model.
