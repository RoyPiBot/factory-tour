/**
 * dialog.js - RPG 風格對話框系統 v2.0
 *
 * v2.0 新增：
 *   - TTS 回調 (onResponseComplete)
 *   - 持久化 sessionId (localStorage)
 */

export class DialogSystem {
  constructor() {
    this.overlay = document.getElementById('dialog-overlay');
    this.portraitEl = document.getElementById('dialog-portrait');
    this.speakerEl = document.getElementById('dialog-speaker');
    this.textEl = document.getElementById('dialog-text');
    this.typingEl = document.getElementById('dialog-typing');
    this.hintEl = document.getElementById('dialog-hint');

    this.isOpen = false;
    this.isTyping = false;
    this.isWaitingAPI = false;
    this.currentNpc = null;

    // 持久化 session ID（跨分頁保留）
    this.sessionId = localStorage.getItem('factory-tour-session') ||
      ('rpg-' + Math.random().toString(36).substring(2, 10));
    localStorage.setItem('factory-tour-session', this.sessionId);

    // 打字機效果
    this.fullText = '';
    this.displayedChars = 0;
    this.typeSpeed = 30; // ms per char
    this.typeTimer = null;

    // 對話歷史（追蹤已打過招呼的 NPC）
    this.greeted = new Set();

    // v2.0 — TTS 回調
    this.onResponseComplete = null; // function(text)
  }

  /**
   * 開啟對話框 — 與 NPC 互動
   */
  async interact(npc) {
    if (this.isOpen && !this.isWaitingAPI) {
      this.close();
      return;
    }
    if (this.isWaitingAPI) return;

    this.currentNpc = npc;
    this.isOpen = true;
    this.overlay.classList.remove('hidden');

    this.portraitEl.textContent = npc.portrait;
    this.speakerEl.textContent = npc.name;

    if (!this.greeted.has(npc.id)) {
      this.greeted.add(npc.id);
      this.typeText(npc.greeting);
      this.hintEl.textContent = '按 E 繼續 / T 提問';
    } else {
      await this.askAI(`我又來找你了，有什麼新的資訊可以告訴我嗎？關於「${npc.name.split('—')[1]?.trim() || '這個區域'}」`);
    }
  }

  /**
   * 自由提問（按 T 觸發）
   */
  async askFreeQuestion(question) {
    if (!question.trim()) return;

    if (this.currentNpc) {
      this.portraitEl.textContent = this.currentNpc.portrait;
      this.speakerEl.textContent = this.currentNpc.name;
    } else {
      this.portraitEl.textContent = '🤖';
      this.speakerEl.textContent = 'AI 導覽助手';
    }

    this.isOpen = true;
    this.overlay.classList.remove('hidden');
    await this.askAI(question);
  }

  /**
   * 進入區域時自動觸發（首次進入）
   */
  async triggerAreaEntry(room, npc) {
    if (this.greeted.has('area_' + room.id)) return;
    this.greeted.add('area_' + room.id);

    this.currentNpc = npc;
    this.isOpen = true;
    this.overlay.classList.remove('hidden');

    this.portraitEl.textContent = npc ? npc.portrait : '🏭';
    this.speakerEl.textContent = npc ? npc.name : '系統';

    await this.askAI(
      `我剛走進了「${room.name}」，請用 2-3 句話簡單介紹這個區域的功能和重點，包含安全注意事項。`
    );
  }

  /**
   * 呼叫後端 AI API
   */
  async askAI(message) {
    this.isWaitingAPI = true;
    this.hintEl.textContent = '思考中...';

    this.textEl.textContent = '';
    this.typingEl.style.display = 'inline';

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          session_id: this.sessionId,
        }),
      });

      this.typingEl.style.display = 'none';

      if (res.ok) {
        const data = await res.json();
        this.typeText(data.reply);
      } else {
        const err = await res.json().catch(() => ({ detail: '伺服器錯誤' }));
        this.typeText(`⚠️ ${err.detail || '連線錯誤，請稍後再試'}`);
      }
    } catch (e) {
      this.typingEl.style.display = 'none';
      this.typeText(`⚠️ 無法連接伺服器：${e.message}`);
    }

    this.isWaitingAPI = false;
    this.hintEl.textContent = '按 E 關閉 / T 提問';
  }

  /**
   * 打字機效果
   */
  typeText(text) {
    if (this.typeTimer) clearInterval(this.typeTimer);

    this.fullText = text;
    this.displayedChars = 0;
    this.isTyping = true;
    this.textEl.textContent = '';

    this.typeTimer = setInterval(() => {
      this.displayedChars++;
      this.textEl.textContent = this.fullText.substring(0, this.displayedChars);

      if (this.displayedChars >= this.fullText.length) {
        clearInterval(this.typeTimer);
        this.typeTimer = null;
        this.isTyping = false;

        // v2.0 — 通知 TTS 回調
        if (this.onResponseComplete) {
          try {
            this.onResponseComplete(this.fullText);
          } catch (e) {
            console.warn('TTS callback error:', e);
          }
        }
      }
    }, this.typeSpeed);
  }

  /**
   * 跳過打字動畫（立即顯示全文）
   */
  skipTyping() {
    if (this.isTyping && this.typeTimer) {
      clearInterval(this.typeTimer);
      this.typeTimer = null;
      this.textEl.textContent = this.fullText;
      this.isTyping = false;

      // 跳過也要觸發 TTS
      if (this.onResponseComplete) {
        try {
          this.onResponseComplete(this.fullText);
        } catch (e) {
          console.warn('TTS callback error:', e);
        }
      }
    }
  }

  /**
   * 關閉對話框
   */
  close() {
    if (this.isWaitingAPI) return;
    this.skipTyping();
    this.isOpen = false;
    this.overlay.classList.add('hidden');
  }

  /**
   * 處理 E 鍵按下
   */
  handleInteractKey(npc) {
    if (this.isOpen) {
      if (this.isTyping) {
        this.skipTyping();
      } else if (!this.isWaitingAPI) {
        this.close();
      }
    } else if (npc) {
      this.interact(npc);
    }
  }

  getIsOpen() {
    return this.isOpen;
  }
}
