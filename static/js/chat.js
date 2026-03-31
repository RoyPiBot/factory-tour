/**
 * chat.js - 聊天面板邏輯
 */

export class ChatPanel {
  constructor() {
    this.messagesEl = document.getElementById('chat-messages');
    this.inputEl = document.getElementById('chat-input');
    this.sendBtn = document.getElementById('chat-send');
    this.sessionId = 'game-' + Math.random().toString(36).substring(2, 10);
    this.isWaiting = false;

    // 綁定事件
    this.sendBtn.addEventListener('click', () => this.sendUserMessage());
    this.inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendUserMessage();
      }
    });

    // 歡迎訊息
    this.addMessage('歡迎來到工廠導覽！用 WASD 或方向鍵移動，走進不同區域我會為您介紹。💡', 'system');
  }

  addMessage(text, sender = 'bot') {
    const div = document.createElement('div');
    div.className = `chat-msg chat-msg-${sender}`;

    if (sender === 'system') {
      div.innerHTML = `<span class="msg-icon">🏭</span> ${this.escapeHtml(text)}`;
    } else if (sender === 'user') {
      div.innerHTML = this.escapeHtml(text);
    } else if (sender === 'guide') {
      div.innerHTML = `<span class="msg-icon">🧑‍🏭</span> ${this.formatMessage(text)}`;
    } else {
      div.innerHTML = `<span class="msg-icon">🤖</span> ${this.formatMessage(text)}`;
    }

    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  showTyping() {
    const div = document.createElement('div');
    div.className = 'chat-msg chat-msg-bot typing-indicator';
    div.id = 'typing';
    div.innerHTML = '<span class="msg-icon">🧑‍🏭</span> <span class="dots"><span>.</span><span>.</span><span>.</span></span>';
    this.messagesEl.appendChild(div);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  hideTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
  }

  async sendUserMessage() {
    const msg = this.inputEl.value.trim();
    if (!msg || this.isWaiting) return;

    this.addMessage(msg, 'user');
    this.inputEl.value = '';
    await this.sendToAgent(msg);
  }

  async sendToAgent(message, isAutomatic = false) {
    this.isWaiting = true;
    this.sendBtn.disabled = true;
    this.showTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: this.sessionId }),
      });

      this.hideTyping();

      if (res.ok) {
        const data = await res.json();
        this.addMessage(data.reply, isAutomatic ? 'guide' : 'bot');
      } else {
        const err = await res.json().catch(() => ({ detail: '伺服器錯誤' }));
        this.addMessage(`⚠️ ${err.detail || '連線錯誤'}`, 'system');
      }
    } catch (e) {
      this.hideTyping();
      this.addMessage(`⚠️ 無法連接伺服器：${e.message}`, 'system');
    }

    this.isWaiting = false;
    this.sendBtn.disabled = false;
  }

  /**
   * 自動觸發 — 進入區域時
   */
  async triggerAreaIntro(room) {
    this.addMessage(`📍 您已進入：${room.emoji} ${room.name}`, 'system');
    await this.sendToAgent(
      `我剛走進了「${room.name}」，請用 2-3 句話簡單介紹這個區域，包含重要的安全注意事項。`,
      true
    );
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  formatMessage(text) {
    // 簡單格式化：粗體、換行
    return this.escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br>');
  }
}
