import pyaudio
import numpy as np
import torch
import os
import json
from stream_vits_zh import VITSModel  # 替换为封装后的类

# **配置参数**
CONFIG_PATH = "configs/bert_vits.json"  # VITS 配置文件路径
MODEL_PATH = "vits_bert_model.pth"      # VITS 训练模型路径
TEXT_INPUT = """
在一个宁静的村庄里，住着一位名叫小明的少年。他聪明好学，深受村民喜爱。一天，小明在村口的老树下发现了一本古老的书籍，书页泛黄，封面上写着“智慧之书”。

小明带着好奇心翻开书页，发现里面记载了许多关于勇气、智慧和善良的故事。每当他读完一个故事，书中的文字就会消失，取而代之的是一段新的文字，仿佛在引导他不断探索。

有一天，村庄附近的河流突然泛滥，威胁到村民的安全。小明想起书中提到的一个关于建造堤坝的故事，决定带领村民们一起筑堤防洪。他运用从书中学到的知识，指导大家齐心协力，终于成功地保护了村庄。

洪水退去后，村民们对小明充满感激，称他为“智慧少年”。然而，当小明再次翻开那本书时，发现书中的文字已全部消失，取而代之的是一面镜子，映照出他自己的影像。小明这才明白，真正的智慧源自于自身的学习和实践，而那本书只是引导他发现内心潜能的工具。

从此，小明更加勤奋地学习和帮助他人，村庄也因此变得更加和谐美好。
"""  # 需要合成的文本

# **自动检测 VITS 语音参数**
def get_vits_audio_params(model):
    """ 自动查询 VITS 采样率和格式 """
    RATE = model.hps.data.sampling_rate  # 读取采样率
    CHUNK_SIZE = 1024  # 音频块大小
    CHANNELS = 1  # VITS 输出通常是单声道

    # 运行一次 inference_stream 以检测数据格式
    sample_audio = next(model.inference_stream("测试"))
    
    if sample_audio.dtype == np.int16:
        FORMAT = pyaudio.paInt16
    elif sample_audio.dtype == np.float32:
        FORMAT = pyaudio.paFloat32
    else:
        raise ValueError("未知的音频格式")

    print(f"自动检测参数: RATE={RATE}, CHUNK_SIZE={CHUNK_SIZE}, FORMAT={FORMAT}, CHANNELS={CHANNELS}")
    return RATE, CHUNK_SIZE, FORMAT, CHANNELS

# **初始化 VITSModel**
vits = VITSModel(CONFIG_PATH, MODEL_PATH)
RATE, CHUNK_SIZE, FORMAT, CHANNELS = get_vits_audio_params(vits)

# **初始化 PyAudio**
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK_SIZE)


try:
    for audio_chunk in vits.inference_stream(TEXT_INPUT):
        if audio_chunk is not None and len(audio_chunk) > 0:
            # 确保数据是 float32
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            # 播放音频
            stream.write(audio_chunk.tobytes())
except KeyboardInterrupt:
    print("\n音频播放被中断，程序正在退出...")

# **关闭 PyAudio 流**
stream.stop_stream()
stream.close()
p.terminate()
