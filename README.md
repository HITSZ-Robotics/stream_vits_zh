# Stream_Vits_Zh

Stream_Vits_Zh 是一个基于 **VITS** 的流式 TTS 方案，适用于需要**低延迟**的语音合成应用。

本项目基于 [vits_chinese](https://github.com/PlayVoice/vits_chinese)，提供了一个支持**流式输出**的中文 VITS 模型接口，专为 **中文 TTS** 任务优化。

## **✨ 特性**

✅ **基于 VITS_chinese**：专为中文语音合成优化，提供高质量的 TTS 输出。

✅ **支持流式输出**：低延迟、逐步生成音频，提高实时性。

✅ **兼容 Coqui-TTS**：可作为 **Coqui-TTS** 的中文流式模型补充，提供更好的流式 TTS 性能。

✅ **易于部署**：简洁的 API 设计，方便集成，适用于各种应用场景。

---

## **🚀 在线 Demo**

https://github.com/user-attachments/assets/788da5c5-ff62-4e09-bdfd-e33c240e1cda

---

## **📦 安装依赖**

### **1️⃣ 克隆项目**
```bash
git clone https://github.com/zzl410/stream_vits_zh.git
cd stream_vits_zh
```

### **2️⃣ 安装依赖**
```bash
pip install -r requirements.txt
```

⚠️ **推荐 Python 3.8+ 版本**

### **3️⃣ 安装 `monotonic_align` 依赖**
```bash
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

---

## **📥 使用预训练模型进行推理**

### **1️⃣ 下载预训练模型**

🔗 [vits_chinese/releases](https://github.com/zzl410/stream_vits_zh/releases/tag/v1.0)

将 **`prosody_model.pt`** 放入 `./bert/prosody_model.pt`

将 **`vits_bert_model.pth`** 放入 `./vits_bert_model.pth`

### **2️⃣ 运行推理**
```bash
python stream_example.py
```

生成的音频块将**逐步输出**，支持实时播放。

---

## **📌 贡献：提供流式 VITS 中文 TTS 接口**

本项目提供了一个能 **流式输出中文** 的 VITS 模型接口，适用于**低延迟 TTS** 应用，并可作为 **Coqui-TTS** 的补充。

### **🔧 核心接口：`stream_vits_zh.py`**

```python
def inference_stream(config: str, model: str, txt: str):
    """
    进行 VITS 推理，流式返回音频块。
    
    Args:
        config (str): 配置文件路径。
        model (str): 训练模型路径。
        txt (str): 输入文本。
    
    Yields:
        np.ndarray: 单个流式音频块。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_path = os.path.join(os.path.dirname(__file__), 'bert')
    tts_front = VITS_PinYin(bert_path, device)
    hps = utils.get_hparams_from_file(config)
    
    net_g = utils.load_class(hps.train.eval_class)(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    utils.load_model(model, net_g)
    net_g.eval()
    net_g.to(device)
    
    phonemes, char_embeds = tts_front.chinese_to_phonemes(txt)
    input_ids = cleaned_text_to_sequence(phonemes)
    buffer_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    
    def producer():
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
            
            for chunk in net_g.inference_stream(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5, length_scale=1):
                buffer_queue.put(chunk)
        stop_event.set()
    
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    
    while not (stop_event.is_set() and buffer_queue.empty()):
        chunk = buffer_queue.get()
        yield chunk
```

---

## **🌟 欢迎 Star & 贡献**

💖 如果你觉得这个项目有用，请给个 ⭐ **Star** 支持！

📢 期待你的 PR 和 Issue 反馈，共同优化 Stream_Vits_Zh 🚀

📬 **联系作者：** [GitHub Issues](https://github.com/zzl410/stream_vits_zh/issues)

---


