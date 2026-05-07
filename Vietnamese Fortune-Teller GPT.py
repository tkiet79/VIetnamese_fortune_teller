import torch
import torch.nn as nn
import tiktoken
import os

enc = tiktoken.get_encoding("cl100k_base")
file_path = "/tu_vi_data.txt"

if not os.path.exists(file_path):
    print(f"LỖI: Không tìm thấy file tại {file_path}.")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    print(f"Đã nạp thành công dữ liệu. Độ dài: {len(text_data):,} ký tự.")
    data_ids = enc.encode(text_data)
    data_tensor = torch.tensor(data_ids, dtype=torch.long)
    print(f"Tổng cộng: {len(data_tensor):,} tokens.")


def get_batch(data, seq_len, batch_size):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i+seq_len] for i in ix])
    y = torch.stack([data[i+1 : i+seq_len+1] for i in ix])
    return x, y


class Vietnamese_Fortune_Teller_GPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, max_len=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4, batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B,T = x.shape
        device = x.device
        x = self.token_emb(x)
        positions = torch.arange(0,T,device=device)
        x += self.pos_emb(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        x = self.blocks(x, mask=mask, is_causal=True)
        logits = self.head(x)
        return logits


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Đang chạy bằng sức mạnh của: {device.upper()}")

model_path = "project.pth"
model = Vietnamese_Fortune_Teller_GPT(vocab_size=enc.n_vocab).to(device)


so_vong_lap = 5000
if so_vong_lap > 0:
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
   criterion = nn.CrossEntropyLoss()
   data_tensor = data_tensor.to(device)
   model.train()

for step in range(so_vong_lap):
    X_batch, Y_batch = get_batch(data_tensor, seq_len = 64, batch_size = 32)
    optimizer.zero_grad()
    logits = model(X_batch)
    B,T,C = logits.shape
    loss = criterion(logits.view(B*T, C), Y_batch.view(B*T))
    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
          print(f"   ⏳ Vòng {step+1:4d} | Loss: {loss.item():.4f}")

print(f"đã train xong và được lưu ở {model_path}")
torch.save(model.state_dict(), model_path)

if os.path.exists(model_path):
    print("\n✨ AI BẮT ĐẦU SÁNG TÁC:")
    model.eval()
    user_input = input("nhập năm sinh hoặc con giáp của bạn: ")
    idx = torch.tensor([enc.encode(user_input)], dtype=torch.long).to(device)

    for _ in range(100):
        idx_cond = idx[:, -64:] # Cắt ngữ cảnh để tránh quá tải
        with torch.no_grad():
            logits = model(idx_cond)

        diem_tu_cuoi = logits[:, -1, :] / 0.8
        xac_suat = torch.softmax(diem_tu_cuoi, dim=-1)
        tu_moi = torch.multinomial(xac_suat, num_samples=1)
        idx = torch.cat((idx, tu_moi), dim=1)

    print(f"\n{'-'*40}\n{enc.decode(idx[0].tolist())}\n{'-'*40}")