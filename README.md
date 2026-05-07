# 🔮 Vietnamese Fortune Teller GPT (AI Thầy Bói)

Dự án này là một Mô hình Ngôn ngữ (Language Model) dựa trên kiến trúc Transformer, được **xây dựng hoàn toàn từ con số 0 (From Scratch) bằng PyTorch**. AI được huấn luyện trên hàng ngàn dữ liệu tử vi để học cách sinh văn bản theo văn phong tâm linh của một "Thầy Đồ" Việt Nam.

## 🚀 Trải nghiệm trực tiếp (Live Demo)
Bạn không cần cài đặt code, có thể bấm vào link dưới đây để xin quẻ ngay lập tức:
👉 **[Chơi thử AI Thầy Bói trên Hugging Face](https://huggingface.co/spaces/tieukiet/Vietnamese_fortune_teller)**

## 🛠️ Công nghệ sử dụng (Tech Stack)
* **Framework:** PyTorch
* **Kiến trúc lõi:** Transformer (Self-Attention, Causal Mask, Feed-Forward) - Code from scratch!
* **Tokenizer:** Tiktoken (OpenAI cl100k_base)
* **Giao diện Web:** Streamlit
* **Triển khai (Deployment):** Hugging Face Spaces

## 🧠 Quá trình huấn luyện
* Mô hình tự dự đoán từ tiếp theo (Auto-regressive).
* Sử dụng kỹ thuật **Logit Biasing** để ép AI tập trung vào đúng con giáp của người dùng.
* Sử dụng **Temperature** và **Top-K sampling** để tạo sự bay bổng, sáng tạo nhưng không bị lạc đề.
