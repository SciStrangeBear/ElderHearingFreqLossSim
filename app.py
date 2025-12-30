import streamlit as st
import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import io

# --- 1. 页面配置与样式优化 ---
st.set_page_config(
    page_title="老人听觉模拟器", 
    page_icon="👂", 
    layout="wide",
    initial_sidebar_state="collapsed" # 收起侧边栏，使用主界面布局
)

# 自定义 CSS 优化视觉体验
st.markdown("""
<style>
    /* 全局字体与间距优化 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 按钮样式优化 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 3rem;
    }
    
    /* 标题样式 */
    h1 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* 卡片式布局背景 (可选，视主题而定) */
    div[data-testid="stExpander"] {
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心逻辑与性能优化 ---

@st.cache_data(show_spinner=False)
def load_audio_data(file_content, sr=None):
    """
    缓存音频加载结果，减少重复IO
    注意：这里传入 file_content (bytes) 而不是 file object 以利用缓存哈希
    """
    # 将 bytes 转为 IO 供 librosa 读取
    return librosa.load(io.BytesIO(file_content), sr=sr)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """低通滤波器实现"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # 避免截止频率超过奈奎斯特频率
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

@st.cache_data(show_spinner=False)
def process_audio(data, fs, cutoff_freq):
    """
    执行音频处理 (带缓存)
    """
    # 1. 滤波
    filtered_data = butter_lowpass_filter(data, cutoff_freq, fs)
    
    # 2. 自动增益补偿
    max_val = np.max(np.abs(data))
    if max_val > 0:
        filtered_data = librosa.util.normalize(filtered_data) * max_val
    
    return filtered_data

def get_default_sample_path():
    """获取默认样本路径"""
    possible_paths = ["Sample1.mp3", "ElderHearingFreqLoss/Sample1.mp3"]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

# --- 3. 主界面布局 ---
def main():
    # 顶部导航区
    col_title, col_info = st.columns([4, 1])
    with col_title:
        st.title("👵 老人听觉模拟器")
        st.caption("Hearing Loss Simulator: 体验不同年龄段的听觉世界")
    
    with col_info:
        with st.expander("ℹ️ 使用说明"):
            st.markdown("""
            1. **上传/选择音频**
            2. **设定年龄/频率**
            3. **点击生成**
            4. **对比与分析**
            """)

    st.divider()

    # 主内容区：双栏布局
    col_input, col_output = st.columns([1, 1], gap="large")

    # === 左侧：设置与输入 ===
    with col_input:
        st.subheader("🎛️ 参数设置")
        
        # 1. 音频源选择
        uploaded_file = st.file_uploader("上传音频 (mp3, wav)", type=["mp3", "wav", "m4a"], label_visibility="collapsed")
        
        file_source = None
        is_sample = False
        file_bytes = None # 用于缓存键值
        
        if uploaded_file:
            file_source = uploaded_file
            file_bytes = uploaded_file.getvalue()
            st.success(f"已加载: {uploaded_file.name}")
        else:
            sample_path = get_default_sample_path()
            if sample_path:
                file_source = sample_path
                with open(sample_path, "rb") as f:
                    file_bytes = f.read()
                is_sample = True
                st.info(f"使用默认样本: {os.path.basename(sample_path)}")
            else:
                st.warning("请上传音频文件")

        # 2. 模拟参数
        age_map = {
            "20岁 (正常听力)": 15000,
            "50岁 (轻度衰退 - 6kHz)": 6000,
            "65岁 (中度衰退 - 3kHz)": 3000,
            "80岁 (重度衰退 - 1.5kHz)": 1500,
            "自定义频率": 0
        }
        
        selected_age = st.selectbox("选择模拟年龄段", list(age_map.keys()))
        
        if selected_age == "自定义频率":
            cutoff_freq = st.slider("截止频率 (Hz)", 500, 10000, 2000, step=100)
        else:
            cutoff_freq = age_map[selected_age]
            st.metric("当前截止频率", f"{cutoff_freq} Hz")

        # 3. 动作按钮
        process_btn = st.button("🚀 生成模拟音频", type="primary", disabled=(file_source is None))

    # === 右侧：结果展示 ===
    with col_output:
        st.subheader("� 试听对比")
        
        if file_source and file_bytes:
            # 加载原始音频
            try:
                data, fs = load_audio_data(file_bytes)
            except Exception as e:
                st.error(f"音频加载失败: {e}")
                st.stop()

            # 显示原始音频
            st.markdown("**原始音频**")
            st.audio(file_bytes, format='audio/wav') # 直接播放原始 bytes

            # 处理逻辑
            if process_btn:
                with st.spinner("正在处理音频频谱..."):
                    # 核心处理
                    filtered_data = process_audio(data, fs, cutoff_freq)
                    
                    # 导出为 WAV
                    out_io = io.BytesIO()
                    sf.write(out_io, filtered_data, fs, format='WAV')
                    out_io.seek(0)
                    
                    # 存入 Session State 以持久化显示
                    st.session_state['result_audio'] = out_io
                    st.session_state['result_data'] = filtered_data
                    st.session_state['result_params'] = (selected_age, cutoff_freq)
            
            # 显示处理结果 (如果有)
            if 'result_audio' in st.session_state:
                st.markdown(f"**模拟结果** ({st.session_state.get('result_params', ('', ''))[0]})")
                st.audio(st.session_state['result_audio'])
                
                # 下载按钮
                st.download_button(
                    label="⬇️ 下载模拟音频",
                    data=st.session_state['result_audio'],
                    file_name="simulated_hearing.wav",
                    mime="audio/wav"
                )
        else:
            st.info("👈 请先在左侧配置并生成音频")

    # === 底部：可视化分析 (仅当有结果时) ===
    if 'result_data' in st.session_state and file_source:
        st.divider()
        with st.expander("📊 频谱可视化分析", expanded=True):
            col_viz1, col_viz2 = st.columns(2)
            
            # 统一绘图参数
            vmin, vmax = -80, 0
            
            with col_viz1:
                st.caption("原始音频频谱")
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
                librosa.display.specshow(D_orig, sr=fs, x_axis='time', y_axis='hz', ax=ax1, vmin=vmin, vmax=vmax)
                ax1.set_ylim(0, 12000)
                ax1.label_outer()
                st.pyplot(fig1)

            with col_viz2:
                st.caption(f"模拟音频频谱 ({st.session_state['result_params'][1]}Hz Cutoff)")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                D_filt = librosa.amplitude_to_db(np.abs(librosa.stft(st.session_state['result_data'])), ref=np.max)
                librosa.display.specshow(D_filt, sr=fs, x_axis='time', y_axis='hz', ax=ax2, vmin=vmin, vmax=vmax)
                ax2.set_ylim(0, 12000)
                ax2.label_outer()
                st.pyplot(fig2)

if __name__ == "__main__":
    main()
