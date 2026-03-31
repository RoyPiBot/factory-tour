/**
 * feedback.js - 評分回饋系統（RPG 風格）
 */

(function () {
  'use strict';

  class FeedbackSystem {
    constructor(sessionId) {
      this.modal = null;
      this.sessionId = sessionId || 'default';
      this.areasVisited = [];
      this.rating = 0;
      this.submitted = false;
      this._quizSystem = null;
    }

    /**
     * 設定測驗系統參照（用於顯示分數）
     */
    setQuizSystem(qs) {
      this._quizSystem = qs;
    }

    /**
     * 是否已提交過
     */
    hasSubmitted() {
      return this.submitted || !!localStorage.getItem('feedback_submitted_' + this.sessionId);
    }

    /**
     * 測驗/回饋是否正在顯示
     */
    isActive() {
      return this.modal && !this.modal.classList.contains('hidden');
    }

    /**
     * 顯示回饋表單
     * @param {string|string[]} sessionIdOrAreas - sessionId 或 areasVisited 陣列
     * @param {string[]|undefined} areasVisited
     */
    show(sessionIdOrAreas, areasVisited) {
      // 支援兩種呼叫方式: show(areasVisited) 或 show(sessionId, areasVisited)
      if (Array.isArray(sessionIdOrAreas)) {
        areasVisited = sessionIdOrAreas;
      } else if (typeof sessionIdOrAreas === 'string') {
        this.sessionId = sessionIdOrAreas;
      }

      // 已提交過就不再顯示
      if (localStorage.getItem('feedback_submitted_' + this.sessionId)) {
        return;
      }

      this.areasVisited = areasVisited || [];
      this.rating = 0;
      this.submitted = false;

      this._createModal();
      this.modal.classList.remove('hidden');
    }

    /**
     * 隱藏回饋表單
     */
    hide() {
      if (this.modal) {
        this.modal.classList.add('hidden');
      }
    }

    /**
     * 提交回饋
     */
    async submit() {
      if (this.submitted || this.rating === 0) return;

      const commentEl = this.modal.querySelector('.feedback-comment');
      const comment = commentEl ? commentEl.value.trim() : '';

      const submitBtn = this.modal.querySelector('.feedback-submit');
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = '送出中...';
      }

      try {
        const res = await fetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: this.sessionId,
            rating: this.rating,
            comment: comment,
            areas_visited: this.areasVisited,
          }),
        });

        if (res.ok) {
          this.submitted = true;
          localStorage.setItem('feedback_submitted_' + this.sessionId, 'true');
          this._showThankYou();
        } else {
          if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = '送出';
          }
          console.warn('[Feedback] 送出失敗:', res.status);
        }
      } catch (e) {
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.textContent = '送出';
        }
        console.warn('[Feedback] 送出錯誤:', e.message);
      }
    }

    /**
     * 建立 RPG 風格的回饋 Modal
     */
    _createModal() {
      // 移除舊的
      if (this.modal) {
        this.modal.remove();
      }

      this.modal = document.createElement('div');
      this.modal.className = 'feedback-overlay hidden';
      this.modal.innerHTML = `
        <div class="feedback-modal">
          <div class="feedback-header">
            <span class="feedback-icon">📜</span>
            <h2>冒險評價</h2>
          </div>
          <div class="feedback-body">
            <p class="feedback-prompt">勇者，這趟工廠導覽如何？</p>
            <div class="feedback-areas">
              已探索 <strong>${this.areasVisited.length}</strong> 個區域
            </div>
            <div class="feedback-stars">
              ${[1, 2, 3, 4, 5].map((n) =>
                `<span class="feedback-star" data-rating="${n}">⭐</span>`
              ).join('')}
            </div>
            <div class="feedback-rating-text"></div>
            <textarea class="feedback-comment" placeholder="留下你的冒險心得...（選填）" rows="3"></textarea>
            <button class="feedback-submit" disabled>送出評價</button>
          </div>
          <button class="feedback-close">&times;</button>
        </div>
      `;

      // 套用樣式
      this._applyStyles();

      document.body.appendChild(this.modal);

      // 綁定事件
      const stars = this.modal.querySelectorAll('.feedback-star');
      const ratingText = this.modal.querySelector('.feedback-rating-text');
      const submitBtn = this.modal.querySelector('.feedback-submit');
      const closeBtn = this.modal.querySelector('.feedback-close');

      const ratingLabels = ['', '不太好', '普通', '不錯', '很棒', '超讚！'];

      stars.forEach((star) => {
        star.addEventListener('click', () => {
          this.rating = parseInt(star.dataset.rating, 10);
          submitBtn.disabled = false;

          // 更新星星顯示
          stars.forEach((s) => {
            const val = parseInt(s.dataset.rating, 10);
            s.style.opacity = val <= this.rating ? '1' : '0.3';
            s.style.transform = val <= this.rating ? 'scale(1.2)' : 'scale(1)';
          });

          ratingText.textContent = ratingLabels[this.rating] || '';
        });

        star.addEventListener('mouseenter', () => {
          const hoverVal = parseInt(star.dataset.rating, 10);
          stars.forEach((s) => {
            const val = parseInt(s.dataset.rating, 10);
            s.style.opacity = val <= hoverVal ? '1' : '0.3';
          });
        });

        star.addEventListener('mouseleave', () => {
          stars.forEach((s) => {
            const val = parseInt(s.dataset.rating, 10);
            s.style.opacity = val <= this.rating ? '1' : '0.3';
            s.style.transform = val <= this.rating ? 'scale(1.2)' : 'scale(1)';
          });
        });
      });

      submitBtn.addEventListener('click', () => this.submit());
      closeBtn.addEventListener('click', () => this.hide());
    }

    /**
     * 顯示感謝訊息
     */
    _showThankYou() {
      const body = this.modal.querySelector('.feedback-body');
      if (!body) return;

      body.innerHTML = `
        <div class="feedback-thankyou">
          <span class="feedback-thankyou-icon">🎉</span>
          <h3>感謝你的評價！</h3>
          <p>你的回饋將幫助我們改善導覽體驗。</p>
          <p>勇者，期待你的下次冒險！</p>
        </div>
      `;

      setTimeout(() => this.hide(), 4000);
    }

    /**
     * 動態注入樣式
     */
    _applyStyles() {
      if (document.getElementById('feedback-styles')) return;

      const style = document.createElement('style');
      style.id = 'feedback-styles';
      style.textContent = `
        .feedback-overlay {
          position: fixed;
          top: 0; left: 0;
          width: 100%; height: 100%;
          background: rgba(0, 0, 0, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 9000;
        }
        .feedback-overlay.hidden { display: none; }
        .feedback-modal {
          background: linear-gradient(135deg, #1a1a2e, #16213e);
          border: 2px solid #e2b857;
          border-radius: 16px;
          padding: 28px 32px;
          max-width: 420px;
          width: 90%;
          position: relative;
          box-shadow: 0 0 40px rgba(226, 184, 87, 0.2);
          font-family: "Microsoft JhengHei", sans-serif;
          color: #eee;
        }
        .feedback-header {
          text-align: center;
          margin-bottom: 16px;
        }
        .feedback-header h2 {
          margin: 4px 0 0;
          font-size: 22px;
          color: #e2b857;
        }
        .feedback-icon { font-size: 36px; }
        .feedback-prompt {
          text-align: center;
          font-size: 15px;
          color: #ccc;
          margin-bottom: 8px;
        }
        .feedback-areas {
          text-align: center;
          font-size: 13px;
          color: #999;
          margin-bottom: 16px;
        }
        .feedback-stars {
          display: flex;
          justify-content: center;
          gap: 12px;
          margin-bottom: 8px;
        }
        .feedback-star {
          font-size: 32px;
          cursor: pointer;
          opacity: 0.3;
          transition: all 0.15s;
          user-select: none;
        }
        .feedback-star:hover { transform: scale(1.3); }
        .feedback-rating-text {
          text-align: center;
          font-size: 14px;
          color: #e2b857;
          height: 20px;
          margin-bottom: 12px;
        }
        .feedback-comment {
          width: 100%;
          box-sizing: border-box;
          background: rgba(255,255,255,0.08);
          border: 1px solid rgba(255,255,255,0.15);
          border-radius: 8px;
          color: #eee;
          padding: 10px;
          font-size: 14px;
          font-family: inherit;
          resize: vertical;
          margin-bottom: 16px;
        }
        .feedback-comment::placeholder { color: #777; }
        .feedback-submit {
          display: block;
          width: 100%;
          padding: 12px;
          background: linear-gradient(135deg, #e2b857, #c9952e);
          color: #1a1a2e;
          font-weight: bold;
          font-size: 16px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-family: inherit;
          transition: opacity 0.2s;
        }
        .feedback-submit:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }
        .feedback-submit:hover:not(:disabled) { opacity: 0.9; }
        .feedback-close {
          position: absolute;
          top: 8px; right: 12px;
          background: none;
          border: none;
          color: #888;
          font-size: 24px;
          cursor: pointer;
          line-height: 1;
        }
        .feedback-close:hover { color: #fff; }
        .feedback-thankyou {
          text-align: center;
          padding: 20px 0;
        }
        .feedback-thankyou-icon { font-size: 48px; }
        .feedback-thankyou h3 {
          color: #e2b857;
          margin: 12px 0 8px;
        }
        .feedback-thankyou p {
          color: #ccc;
          font-size: 14px;
          margin: 4px 0;
        }
      `;
      document.head.appendChild(style);
    }
  }

  window.FeedbackSystem = FeedbackSystem;
})();
