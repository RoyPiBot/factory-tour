/**
 * voice.js - 語音互動系統（Web Speech API）
 */

(function () {
  'use strict';

  class VoiceSystem {
    constructor() {
      this.recognition = null;
      this.synthesis = window.speechSynthesis || null;
      this.isRecording = false;
      this.ttsEnabled = localStorage.getItem('ttsEnabled') !== 'false';

      // 回呼
      this._onResult = null;
      this._onInterim = null;
      this._onError = null;
    }

    /**
     * 初始化語音辨識
     */
    init() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        console.warn('[Voice] 瀏覽器不支援語音辨識');
        return false;
      }

      this.recognition = new SpeechRecognition();
      this.recognition.lang = 'zh-TW';
      this.recognition.continuous = false;
      this.recognition.interimResults = true;

      this.recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) {
            finalTranscript += result[0].transcript;
          } else {
            interimTranscript += result[0].transcript;
          }
        }

        if (interimTranscript && this._onInterim) {
          this._onInterim(interimTranscript);
        }
        if (finalTranscript && this._onResult) {
          this._onResult(finalTranscript);
        }
      };

      this.recognition.onerror = (event) => {
        console.warn('[Voice] 辨識錯誤:', event.error);
        this.isRecording = false;
        this._updateButton(false);
        if (this._onError) {
          this._onError(event.error);
        }
      };

      this.recognition.onend = () => {
        this.isRecording = false;
        this._updateButton(false);
      };

      return true;
    }

    /**
     * 檢查是否支援語音功能
     */
    isSupported() {
      const hasStt = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
      const hasTts = !!window.speechSynthesis;
      return hasStt || hasTts;
    }

    /**
     * 切換錄音狀態
     */
    toggleRecording() {
      if (this.isRecording) {
        this.stopListening();
      } else {
        this.startListening();
      }
    }

    /**
     * 開始錄音
     */
    startListening() {
      if (!this.recognition) {
        console.warn('[Voice] 語音辨識未初始化');
        return;
      }
      if (this.isRecording) return;

      try {
        this.recognition.start();
        this.isRecording = true;
        this._updateButton(true);
      } catch (e) {
        console.warn('[Voice] 無法啟動錄音:', e.message);
      }
    }

    /**
     * 停止錄音
     */
    stopListening() {
      if (!this.recognition || !this.isRecording) return;

      try {
        this.recognition.stop();
      } catch (e) {
        // 忽略
      }
      this.isRecording = false;
      this._updateButton(false);
    }

    /**
     * 語音合成（TTS）
     */
    speak(text) {
      if (!this.synthesis || !this.ttsEnabled) return;

      // 取消目前的語音
      this.synthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'zh-TW';
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      // 嘗試選擇中文語音
      const voices = this.synthesis.getVoices();
      const zhVoice = voices.find((v) => v.lang.startsWith('zh'));
      if (zhVoice) {
        utterance.voice = zhVoice;
      }

      this.synthesis.speak(utterance);
    }

    /**
     * 切換 TTS 開關
     */
    toggleTTS() {
      this.ttsEnabled = !this.ttsEnabled;
      localStorage.setItem('ttsEnabled', this.ttsEnabled.toString());

      if (!this.ttsEnabled && this.synthesis) {
        this.synthesis.cancel();
      }

      return this.ttsEnabled;
    }

    /**
     * 註冊回呼
     */
    onResult(callback) {
      this._onResult = callback;
    }

    onInterim(callback) {
      this._onInterim = callback;
    }

    onError(callback) {
      this._onError = callback;
    }

    /**
     * 更新錄音按鈕狀態
     */
    _updateButton(recording) {
      const btn = document.getElementById('voice-btn');
      if (!btn) return;

      if (recording) {
        btn.classList.add('recording');
        btn.title = '錄音中...點擊停止';
      } else {
        btn.classList.remove('recording');
        btn.title = '點擊開始語音輸入';
      }
    }
  }

  window.VoiceSystem = VoiceSystem;
})();
