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
cythonize -i core.pyx
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


### **VITSModel 使用示例**

#### 1️⃣ 初始化模型
您可以通过以下方式初始化 `VITSModel`，并加载必要的配置与预训练模型。

```python
from stream_vits_zh import VITSModel

model = VITSModel(config_path='path/to/config.json', model_path='path/to/vits_bert_model.pth')
```

#### 2️⃣ 进行推理
使用 `inference_stream` 方法输入文本并生成音频。

```python
txt = "这是一段中文语音合成的测试文本。"
for audio_chunk in model.inference_stream(txt):
    # 使用 PyAudio 播放音频块
    stream.write(audio_chunk.tobytes())
```

在上述代码中，`audio_chunk` 是一个逐步生成的音频块，可以实时播放。

### **完整的 VITSModel 类定义**:

```python
class VITSModel:
    def __init__(self, config_path: str, model_path: str):
        """
        初始化 VITS 模型，加载配置文件和预训练模型。
        
        Args:
            config_path (str): 配置文件路径。
            model_path (str): 训练好的模型路径。
        """
        # 加载配置文件和模型
        self.config = utils.get_hparams_from_file(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_g = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """
        加载 VITS 模型。
        
        Args:
            model_path (str): 模型文件路径。
        
        Returns:
            torch.nn.Module: 加载的模型。
        """
        # 加载模型
        model_class = utils.load_class(self.config.train.eval_class)
        model = model_class(len(symbols), self.config.data.filter_length // 2 + 1, 
                            self.config.train.segment_size // self.config.data.hop_length, 
                            **self.config.model)
        utils.load_model(model_path, model)
        model.eval()
        model.to(self.device)
        return model

    def inference_stream(self, txt: str):
        """
        流式返回音频块。
        
        Args:
            txt (str): 输入的文本。
        
        Yields:
            np.ndarray: 逐步生成的音频块。
        """
        # 生成音频块的逻辑（略）
        phonemes, char_embeds = self._text_to_phonemes(txt)
        input_ids = cleaned_text_to_sequence(phonemes)
        buffer_queue = queue.Queue(maxsize=5)
        stop_event = threading.Event()

        def producer():
            with torch.no_grad():
                x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
                x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
                x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)

                for chunk in self.net_g.inference_stream(x_tst, x_tst_lengths, x_tst_prosody):
                    buffer_queue.put(chunk)
            stop_event.set()

        # 开启线程进行推理
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()

        # 持续输出音频块
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


