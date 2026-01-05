import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="Potato Forecast Pro", page_icon="🥔", layout="wide")

DEFAULT_DATA_PATH = "Dataset_Optimized_Full_Features.csv"
TARGET_COL = 'Yield (Production-Hector M. Ton)'

MODEL_PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}
MAX_PENALTY_THRESHOLD = 0.7 

# --- 2. XỬ LÝ DỮ LIỆU & HUẤN LUYỆN ---
@st.cache_resource(show_spinner="Đang huấn luyện mô hình từ dữ liệu...")
def train_model_from_source(data_source, source_name="Default"):
    try:
        # Đọc file CSV với các định dạng khác nhau
        try:
            if hasattr(data_source, 'seek'): data_source.seek(0)
            df = pd.read_csv(data_source, sep=';', decimal=',')
            if len(df.columns) < 5:
                if hasattr(data_source, 'seek'): data_source.seek(0)
                df = pd.read_csv(data_source)
        except Exception as e:
            return None, None, None, f"Lỗi đọc file CSV: {str(e)}"

        # --- TÌM ĐOẠN NÀY TRONG HÀM train_model_from_source ---

        # Loại bỏ các cột không dùng để huấn luyện
        cols_to_drop = [TARGET_COL, 'Date', 'Season_ID', 'Cumulative_Yield_Ton']
        drop_actual = [c for c in cols_to_drop if c in df.columns]
        
        # ĐÂY LÀ CHỖ THAY ĐỔI:
        X_raw = df.drop(columns=drop_actual)
        
        # Chỉ giữ lại các cột dữ liệu dạng số (Loại bỏ Silt Loam và các cột chữ khác)
        X = X_raw.select_dtypes(include=[np.number])
        
        # Kiểm tra xem có dữ liệu sau khi lọc không
        if X.empty:
            return None, None, None, "❌ File CSV không chứa các cột dữ liệu dạng số hợp lệ."

        y = df[TARGET_COL]

        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(X, y)
        
        avg_stats = {
            'yield_max': float(y.max()),
            'yield_mean': float(y.mean()),
            'input_means': X.mean().to_dict(),
            'source': source_name,
            'rows': len(df)
        }
        return model, X.columns.tolist(), avg_stats, None

    except Exception as e:
        return None, None, None, f"Lỗi hệ thống: {str(e)}"

# --- 3. SIDEBAR: CẤU HÌNH DỮ LIỆU ---
st.sidebar.title("Cấu Hình Dữ Liệu")
uploaded_file = st.sidebar.file_uploader("📂 Upload Dữ Liệu Mới (.csv)", type=['csv'])

if uploaded_file is not None:
    current_source = uploaded_file
    source_label = uploaded_file.name
else:
    current_source = DEFAULT_DATA_PATH
    source_label = "Mặc định (Local)"

model, feature_names, avg_stats, error_msg = train_model_from_source(current_source, source_label)

# Chốt chặn lỗi để không bị lỗi 'NoneType' ở các dòng sau
if error_msg:
    st.error(error_msg)
    st.stop()

if avg_stats is None:
    st.warning("⚠️ Không thể khởi tạo dữ liệu. Vui lòng kiểm tra file đầu vào.")
    st.stop()

# --- 4. QUY TẮC CHUYÊN GIA (EXPERT RULES) ---
def apply_expert_rules(raw_pred, inputs):
    current_yield = raw_pred
    logs = []
    
    # 4.1 Thời gian
    day = inputs['Day_In_Season']
    if day > 130:
        decay = 0.1 * ((day - 130) / 10)
        current_yield *= (1 - decay)
        logs.append(f"🍂 Quá hạn ({day} ngày): Giảm {decay*100:.1f}%")

    # 4.2 Nhiệt độ
    temp = inputs['avg_temp_C']
    if temp > 30:
        pen = 0.2 if temp <= 35 else 0.8 
        current_yield *= (1 - pen)
        logs.append(f"🔥 Nhiệt cao ({temp}°C): Giảm {pen*100:.0f}%")
    
    # 4.3 Mây che phủ
    cloud = inputs.get('cloudcover%', 0)
    cloud_penalty = cloud * 0.0005 
    current_yield *= (1 - cloud_penalty)
    if cloud > 80:
        logs.append(f"☁️ Mây mù dày đặc ({cloud}%): Giảm quang hợp")

    # 4.4 Stress môi trường
    heat_stress = inputs['Heat_Stress_Days (HSD)']
    temp_shock = inputs['Temp_Shock (TS)']
    frost = inputs.get('Frost_Days (FD)', 0)
    
    stress_pen = 0
    if heat_stress > 5: stress_pen += min(heat_stress * 0.03, 0.3)
    if temp_shock > 2:  stress_pen += 0.1
    if frost > 0:       stress_pen += 0.3 * frost
    
    if stress_pen > 0:
        current_yield *= (1 - stress_pen)
        logs.append(f"⚠️ Stress môi trường: Giảm {stress_pen*100:.1f}%")

    # 4.5 Quản lý nước
    total_water = inputs['precipitation_mm'] + inputs['Irrigation_Applied_mm']
    if total_water < 150: 
        current_yield *= 0.5
        logs.append(f"🌵 Thiếu nước nặng: Giảm 50%")
    elif total_water > 800:
        current_yield *= 0.7
        logs.append(f"🌊 Dư nước: Giảm 30%")

    # 4.6 Đất & Dinh dưỡng
    if inputs['Sand_%'] < 60 or inputs['Clay_%'] > 35:
        current_yield *= 0.8
        logs.append("🧱 Đất bí: Giảm 20%")

    ph = inputs['pH']
    if ph < 4.8 or ph > 7.5:
        current_yield *= 0.85
        logs.append(f"🧪 pH {ph} không tối ưu: Giảm 15%")

    # Giới hạn mức phạt
    loss_ratio = 1 - (current_yield / raw_pred) if raw_pred > 0 else 0
    if loss_ratio > MAX_PENALTY_THRESHOLD and frost == 0:
        current_yield = raw_pred * (1 - MAX_PENALTY_THRESHOLD)
        logs.append("🛡️ Đã áp dụng giới hạn mức giảm tối đa (Penalty Cap)")

    return max(0, current_yield), logs

# --- 5. GIAO DIỆN NGƯỜI DÙNG ---
st.title("🥔 Potato Yield Forecast Pro")
st.caption(f"Nguồn dữ liệu: **{avg_stats['source']}** | Quy mô: {avg_stats['rows']} mẫu")

col_input, col_dashboard = st.columns([1, 1.8])

with col_input:
    st.subheader("🛠️ Thiết lập thông số")
    inputs = {}
    
    with st.container(border=True):
        st.markdown("##### 💧 Quản lý nước")
        c1, c2 = st.columns(2)
        inputs['precipitation_mm'] = c1.number_input('Mưa (mm)', 0.0, 3000.0, 100.0)
        inputs['Irrigation_Applied_mm'] = c2.number_input('Tưới (mm)', 0.0, 3000.0, 200.0)
    
    with st.expander("1️⃣ Thời gian & Khí hậu", expanded=True):
        inputs['Day_In_Season'] = st.slider('Ngày trong mùa (DAS)', 1, 150, 90)
        inputs['avg_temp_C'] = st.slider('Nhiệt độ TB (°C)', 5.0, 40.0, 22.0)
        inputs['cloudcover%'] = st.slider('Độ phủ mây (%)', 0, 100, 50)
        inputs['Day_of_Year'] = (inputs['Day_In_Season'] + 270) % 365 or 365
        inputs['solarradiation_W/m2'] = 180.0 * (1 - (inputs['cloudcover%'] / 100) * 0.8)

    with st.expander("2️⃣ Chỉ số Stress & Đất"):
        inputs['Heat_Stress_Days (HSD)'] = st.slider('Ngày Sốc nhiệt', 0, 30, 0)
        inputs['Temp_Shock (TS)'] = st.slider('Lần biến động nhiệt', 0, 15, 0)
        inputs['Frost_Days (FD)'] = st.slider('Ngày Sương giá', 0, 10, 0)
        inputs['Sand_%'] = st.slider('Cát (%)', 0, 100, 80)
        inputs['Clay_%'] = st.slider('Sét (%)', 0, 100, 15)
        inputs['pH'] = st.slider('pH', 4.0, 9.0, 6.0)
        inputs['Organic_Matter_%'] = st.slider('Hữu cơ (%)', 0.0, 10.0, 3.5)

    with st.expander("3️⃣ Dinh dưỡng & Vật lý"):
        inputs['Nitrogen_%'] = st.number_input('Đạm (N%)', 0.0, 2.0, 0.25)
        inputs['Phosphorus_mg_kg'] = st.number_input('Lân (P)', 0.0, 500.0, 45.0)
        inputs['Potassium_mg_kg'] = st.number_input('Kali (K)', 0.0, 500.0, 200.0)
        inputs['Moisture_Content_%'] = st.slider('Độ ẩm đất (%)', 0, 100, 70)
        inputs['Bulk_Density_g_cm3'] = st.slider('Dung trọng', 0.5, 2.0, 1.1)
        inputs['Soil_Compaction_kPa'] = st.slider('Độ nén', 0, 500, 100)
        inputs['Tuber_Yield_Potential_kg_ha'] = 80000.0

    # Chuẩn bị dữ liệu cho Model
    defaults = avg_stats['input_means']
    final_features = {col: inputs.get(col, defaults.get(col, 0)) for col in feature_names}
    input_df = pd.DataFrame([final_features])[feature_names]

# --- 6. DASHBOARD KẾT QUẢ ---
with col_dashboard:
    ai_raw_pred = model.predict(input_df)[0]
    final_pred, rule_logs = apply_expert_rules(ai_raw_pred, inputs)
    
    st.header("📊 Kết Quả Phân Tích")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("AI Model (Gốc)", f"{ai_raw_pred:,.2f}", "Tấn/ha")
    m2.metric("Dự báo cuối cùng", f"{final_pred:,.2f}", f"{final_pred - ai_raw_pred:,.2f}")
    m3.metric("Hiệu suất", f"{(final_pred / avg_stats['yield_max']) * 100:.1f}%", "so với Max")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_pred,
        number = {'valueformat': ",.2f"},
        title = {'text': "Năng Suất Thực Tế (Tấn/ha)"},
        gauge = {
            'axis': {'range': [None, avg_stats['yield_max'] * 1.1]},
            'bar': {'color': "#27AE60" if not rule_logs else "#E67E22"},
            'steps': [
                {'range': [0, avg_stats['yield_mean']], 'color': "#F2F3F4"},
                {'range': [avg_stats['yield_mean'], avg_stats['yield_max']], 'color': "#D5F5E3"}]
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Nhật ký hệ thống")
    if rule_logs:
        for log in rule_logs: st.info(f"🔻 {log}")
    else:
        st.success("✅ Điều kiện sinh trưởng lý tưởng.")

    with st.expander("🔍 Xem mảng dữ liệu đầu vào"):
        st.dataframe(input_df)