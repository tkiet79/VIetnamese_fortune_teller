import streamlit as st
import torch
import torch.nn as nn
import tiktoken
import os

# Danh sách 12 con giáp dùng chung cho toàn hệ thống
DANH_SACH_CON_GIAP = ["Thân", "Dậu", "Tuất", "Hợi", "Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi"]

def tinh_con_giap(nam_sinh):
    return DANH_SACH_CON_GIAP[nam_sinh % 12]

def tim_con_giap_trong_chu(text):
    """Hàm phụ trợ: Nếu người dùng nhập chữ 'tuổi hợi' thay vì số, ta bóc tách ra chữ 'Hợi'"""
    for cg in DANH_SACH_CON_GIAP:
        if cg.lower() in text.lower():
            return cg
    return None

# ==========================================
# 1. CẤU TRÚC NÃO BỘ
# ==========================================
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

# ==========================================
# 2. TẢI BỘ NHỚ VÀO CACHE 
# ==========================================
@st.cache_resource
def load_ai_model():
    enc = tiktoken.get_encoding("cl100k_base")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = Vietnamese_Fortune_Teller_GPT(vocab_size=enc.n_vocab).to(device)
    model_path = "project.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model, enc, device
    else:
        return None, None, None

model, enc, device = load_ai_model()

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB
# ==========================================
st.set_page_config(page_title="AI Thầy Bói", page_icon="🔮")

st.title("🔮 AI Thầy Đồ - Phán Tử Vi")
st.markdown("Cỗ máy AI này đã được huấn luyện trên **hàng ngàn văn bản tử vi**. Hãy nhập một năm sinh hoặc tên con giáp để xem AI bói nhé!")

if model is None:
    st.error("❌ Không tìm thấy file `project.pth`. Trò hãy đảm bảo file này nằm cùng thư mục với file code nhé!")
else:
    # Bố cục Web
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_input("✍️ Nhập năm sinh hoặc con giáp của bạn:", value="2003")
    
    with col2:
        nhiet_do = st.slider("🔥 Độ phiêu (Temperature)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)
        so_tu = st.slider("📏 Độ dài lời phán (Tokens)", min_value=50, max_value=2000, value=2000, step=50)

    # Nút bấm Xin Quẻ
    if st.button("✨ Xin Quẻ Ngay", type="primary"):
        with st.spinner("Thầy đang gieo quẻ bấm độn..."):
            
            con_giap_muc_tieu = None
            
            # Xử lý input của người dùng để trích xuất ra con giáp
            try:
                # Nếu người dùng nhập số (VD: 2003)
                nam_sinh_int = int(user_input.strip())
                con_giap_muc_tieu = tinh_con_giap(nam_sinh_int)
                cau_moi = f"Tử vi tuổi {con_giap_muc_tieu} năm nay"
                st.info(f"⚙️ Tâm linh tương thông: Bạn sinh năm {nam_sinh_int}, cầm tinh con {con_giap_muc_tieu}.")
            except ValueError:
                # Nếu người dùng nhập chữ (VD: tuổi hợi)
                con_giap_muc_tieu = tim_con_giap_trong_chu(user_input)
                cau_moi = user_input
                if con_giap_muc_tieu:
                    st.info(f"⚙️ Hệ thống nhận diện bạn đang xem cho tuổi: {con_giap_muc_tieu}")
                else:
                    st.warning("⚠️ Không nhận diện được con giáp cụ thể. AI sẽ tự do phán.")
            
            # Đưa Câu Mồi vào cho AI
            idx = torch.tensor([enc.encode(cau_moi)], dtype=torch.long).to(device)
            
            for _ in range(so_tu):
                idx_cond = idx[:, -64:]
                with torch.no_grad():
                    logits = model(idx_cond)
                
                # Lấy điểm số của từ cuối cùng
                diem_tu_cuoi = logits[:, -1, :] / nhiet_do
                
                # ==========================================
                # NGHỆ THUẬT ÉP BUỘC AI (LOGIT BIASING)
                # Tăng điểm con giáp hiện tại, trừ điểm con giáp khác
                # ==========================================
                if con_giap_muc_tieu:
                    for cg in DANH_SACH_CON_GIAP:
                        if cg != con_giap_muc_tieu: # Chỉ xét các con giáp KHÁC
                            cg_tokens = enc.encode(cg) + enc.encode(" " + cg) + enc.encode(cg.lower())
                            for token_id in cg_tokens:
                                diem_tu_cuoi[0, token_id] -= 5.0  # Phạt nặng điểm số (Nerf)
                # ==========================================

                xac_suat = torch.softmax(diem_tu_cuoi, dim=-1)
                tu_moi = torch.multinomial(xac_suat, num_samples=1)
                idx = torch.cat((idx, tu_moi), dim=1)
            
            ket_qua = enc.decode(idx[0].tolist())
            
            # Hiển thị kết quả
            st.success("Tác phẩm hoàn thành!")
            st.markdown(f"### Lời phán:\n> *{ket_qua}*")