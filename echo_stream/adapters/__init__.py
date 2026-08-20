"""真模組的 adapter——Phase 1+ 才會有內容。

**這是唯一允許 import 子系統的地方。** contracts 與 core 一律不得 import
echo_stt / echo_tts / echo_memory / session control，那條線一旦破了，
整合層就退化成耦合點。

規劃：

* ``speak_indextts.py``（Phase 1）— 包 ``IndexTTS2Backend``。
  重點是 push→pull 橋接：後端是同步 callback
  ``on_segment_audio(tensor, idx, total)``，契約是 async pull，
  用 :class:`~echo_stream.core.channel.StreamChannel` 轉接
* ``think_llm.py``（Phase 2）— LLM 串流 + SentenceSplitter
* ``input_stt.py``（Phase 3）— 包 ``MultiChannelSTTEngine``
* ``think_memory.py``（Phase 4）— EchoMemory + SessionControl
"""
