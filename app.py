import sys
import asyncio
import os

# Настройки для MacOS
if sys.platform == "darwin":
    if sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

# Настройки Torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from pathlib import Path
import numpy as np
import streamlit as st

# Инициализация Torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.device(device)

# Импорт пользовательских модулей
from src.pipeline import pipeline

def main():
    st.title("Download your data")
    st.header("You can input video or audio files\nUse one of the following formats:")
    
    st.subheader("For audio:")
    st.text("mp3, wav, ogg")
    
    st.subheader("For video:")
    st.text("mp4, avi, mov, mkv")

    # Создаем папку для сохранения медиафайлов
    UPLOAD_DIR = "uploaded_media"
    Path(UPLOAD_DIR).mkdir(exist_ok=True)

    # Загрузка медиафайлов
    uploaded_file = st.file_uploader(
        "Download mediafile (video or audio)",
        type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "ogg"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        file_details = {
            "File name": uploaded_file.name,
            "Content type": "Video" if file_type == "video" else "Audio",
            "MIME-type": uploaded_file.type,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB"
        }
        st.write(file_details)

        if file_type == "video":
            st.video(uploaded_file)
            result = pipeline(uploaded_file)
            st.write("Analysis Result:", result)
        elif file_type == "audio":
            st.audio(uploaded_file)
        else:
            st.error("Unsupported file type")

if __name__ == "__main__":
    # Запускаем синхронно, без asyncio.run()
    main()