import os
import queue
import threading
import torch
import utils
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin

class VITSModel:
    def __init__(self, config_path: str, model_path: str, device: str = None):
        """
        初始化 VITS 语音合成模型。
        
        Args:
            config_path (str): 配置文件路径。
            model_path (str): 训练模型路径。
            device (str, optional): 设备 ('cuda' 或 'cpu')。
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.model_path = model_path
        
        # 初始化前端
        bert_path = os.path.join(os.path.dirname(__file__), 'bert')
        self.tts_front = VITS_PinYin(bert_path, self.device)
        
        # 加载配置
        self.hps = utils.get_hparams_from_file(config_path)
        
        # 加载模型
        self.net_g = utils.load_class(self.hps.train.eval_class)(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        )
        utils.load_model(model_path, self.net_g)
        self.net_g.eval()
        self.net_g.to(self.device)

    def inference_stream(self, text: str):
        """
        进行 VITS 推理，流式返回音频块。

        Args:
            text (str): 输入文本。

        Yields:
            np.ndarray: 单个流式音频块。
        """
        # 解析文本
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(text)
        input_ids = cleaned_text_to_sequence(phonemes)

        # 缓冲区队列
        buffer_queue = queue.Queue(maxsize=8)
        stop_event = threading.Event()

        def producer():
            try:
                with torch.no_grad():
                    x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
                    x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
                    x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)

                    for chunk in self.net_g.inference_stream(x_tst, x_tst_lengths, x_tst_prosody, noise_scale=0.5, length_scale=1):
                        buffer_queue.put(chunk)
            except Exception as e:
                print(f"Producer thread error: {e}")
            finally:
                stop_event.set()  # 线程结束时标记

        # 设置守护线程，避免阻塞退出
        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while not (stop_event.is_set() and buffer_queue.empty()):
            try:
                yield buffer_queue.get(timeout=1)  # 设置超时，避免阻塞
            except queue.Empty:
                continue  # 如果队列为空，继续检查 stop_event

        print("Inference stream finished.")