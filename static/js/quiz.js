/**
 * quiz.js - 區域問答系統（RPG 風格）
 */

(function () {
  'use strict';

  class QuizSystem {
    constructor(sessionId) {
      this.sessionId = sessionId || 'default';
      this.overlay = null;
      this.totalCorrect = 0;
      this.totalAnswered = 0;
      this.quizzedAreas = new Set();

      // 當前測驗狀態
      this._questions = [];
      this._currentIndex = 0;
      this._currentAreaId = null;
      this._active = false;

      // 回調
      this.onScoreUpdate = null; // function({ totalCorrect, totalAnswered })

      this._applyStyles();
    }

    /**
     * 測驗是否正在進行
     */
    isActive() {
      return this._active;
    }

    /**
     * 該區域是否已測驗過
     */
    isAreaQuizzed(areaId) {
      return this.quizzedAreas.has(areaId);
    }

    /**
     * 取得目前總分
     */
    getScore() {
      return {
        correct: this.totalCorrect,
        answered: this.totalAnswered,
        areas: this.quizzedAreas.size,
      };
    }

    /**
     * 開始區域測驗
     */
    async startQuiz(areaId) {
      this._currentAreaId = areaId;
      this._currentIndex = 0;
      this._questions = [];

      this._active = true;
      this._createOverlay();
      this._showLoading();
      this.overlay.classList.remove('hidden');

      try {
        const res = await fetch(`/quiz/${encodeURIComponent(areaId)}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        this._questions = data.questions || [];

        if (this._questions.length === 0) {
          this._showMessage('目前此區域沒有題目', true);
          return;
        }

        this.quizzedAreas.add(areaId);
        this._renderQuestion();
      } catch (e) {
        console.warn('[Quiz] 取得題目失敗:', e.message);
        this._showMessage('無法載入題目，請稍後再試', true);
      }
    }

    /**
     * 提交答案
     */
    async submitAnswer(questionId, answer) {
      this.totalAnswered++;

      // 停用選項按鈕
      const btns = this.overlay.querySelectorAll('.quiz-option');
      btns.forEach((btn) => {
        btn.disabled = true;
      });

      try {
        const res = await fetch('/quiz/answer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: this.sessionId,
            question_id: questionId,
            answer: answer,
            area_id: this._currentAreaId,
          }),
        });

        if (res.ok) {
          const data = await res.json();
          const isCorrect = data.correct;
          if (isCorrect) this.totalCorrect++;
          this._showResult(isCorrect, data.explanation || '', answer, data.correct_answer);
          // 通知分數更新
          if (this.onScoreUpdate) {
            this.onScoreUpdate({ totalCorrect: this.totalCorrect, totalAnswered: this.totalAnswered });
          }
        } else {
          this._showResult(false, '伺服器回應錯誤', answer, null);
        }
      } catch (e) {
        console.warn('[Quiz] 提交答案失敗:', e.message);
        this._showResult(false, '無法連接伺服器', answer, null);
      }
    }

    /**
     * 建立覆蓋層
     */
    _createOverlay() {
      if (this.overlay) {
        this.overlay.remove();
      }

      this.overlay = document.createElement('div');
      this.overlay.className = 'quiz-overlay hidden';
      this.overlay.innerHTML = '<div class="quiz-modal"><div class="quiz-content"></div></div>';
      document.body.appendChild(this.overlay);
    }

    /**
     * 顯示載入中
     */
    _showLoading() {
      const content = this.overlay.querySelector('.quiz-content');
      content.innerHTML = `
        <div class="quiz-header">
          <span class="quiz-icon">📝</span>
          <h2>知識挑戰</h2>
        </div>
        <p class="quiz-loading">載入題目中...</p>
      `;
    }

    /**
     * 渲染當前題目
     */
    _renderQuestion() {
      const q = this._questions[this._currentIndex];
      if (!q) return;

      const content = this.overlay.querySelector('.quiz-content');
      const labels = ['A', 'B', 'C', 'D'];
      const progress = `${this._currentIndex + 1} / ${this._questions.length}`;

      content.innerHTML = `
        <div class="quiz-header">
          <span class="quiz-icon">📝</span>
          <h2>知識挑戰</h2>
          <span class="quiz-progress">${progress}</span>
        </div>
        <p class="quiz-question">${this._escapeHtml(q.question)}</p>
        <div class="quiz-options">
          ${(q.options || []).map((opt, i) => `
            <button class="quiz-option" data-answer="${labels[i]}">
              <span class="quiz-option-label">${labels[i]}</span>
              ${this._escapeHtml(opt)}
            </button>
          `).join('')}
        </div>
        <div class="quiz-result-area"></div>
      `;

      // 綁定選項點擊
      const btns = content.querySelectorAll('.quiz-option');
      btns.forEach((btn) => {
        btn.addEventListener('click', () => {
          const answer = btn.dataset.answer;
          // 標記選中
          btns.forEach((b) => b.classList.remove('selected'));
          btn.classList.add('selected');
          this.submitAnswer(q.id, answer);
        });
      });
    }

    /**
     * 顯示答題結果
     */
    _showResult(isCorrect, explanation, selectedAnswer, correctAnswer) {
      const resultArea = this.overlay.querySelector('.quiz-result-area');
      if (!resultArea) return;

      // 標記正確/錯誤的選項
      const btns = this.overlay.querySelectorAll('.quiz-option');
      btns.forEach((btn) => {
        if (correctAnswer && btn.dataset.answer === correctAnswer) {
          btn.classList.add('correct');
        }
        if (!isCorrect && btn.dataset.answer === selectedAnswer) {
          btn.classList.add('wrong');
        }
      });

      const isLast = this._currentIndex >= this._questions.length - 1;
      const nextLabel = isLast ? '查看結果' : '下一題 ▶';

      resultArea.innerHTML = `
        <div class="quiz-feedback ${isCorrect ? 'quiz-correct' : 'quiz-wrong'}">
          <span>${isCorrect ? '✅ 答對了！' : '❌ 答錯了'}</span>
          ${explanation ? `<p class="quiz-explanation">${this._escapeHtml(explanation)}</p>` : ''}
        </div>
        <button class="quiz-next">${nextLabel}</button>
      `;

      const nextBtn = resultArea.querySelector('.quiz-next');
      nextBtn.addEventListener('click', () => {
        if (isLast) {
          this._showSummary();
        } else {
          this._currentIndex++;
          this._renderQuestion();
        }
      });
    }

    /**
     * 顯示結果摘要
     */
    _showSummary() {
      const content = this.overlay.querySelector('.quiz-content');
      const total = this._questions.length;
      const correct = this.totalCorrect; // 累積的，但這裡只看本輪
      const pct = total > 0 ? Math.round((this.totalCorrect / this.totalAnswered) * 100) : 0;

      let rank = '見習生';
      let rankEmoji = '🌱';
      if (pct >= 90) { rank = '工廠大師'; rankEmoji = '👑'; }
      else if (pct >= 70) { rank = '資深技師'; rankEmoji = '🏆'; }
      else if (pct >= 50) { rank = '合格員工'; rankEmoji = '🎖️'; }

      content.innerHTML = `
        <div class="quiz-header">
          <span class="quiz-icon">${rankEmoji}</span>
          <h2>挑戰完成！</h2>
        </div>
        <div class="quiz-summary">
          <div class="quiz-score-big">${this.totalCorrect} / ${this.totalAnswered}</div>
          <p class="quiz-rank">稱號：<strong>${rank}</strong></p>
          <p class="quiz-stat">正確率：${pct}%</p>
          <p class="quiz-stat">已挑戰區域：${this.quizzedAreas.size}</p>
        </div>
        <button class="quiz-close-btn">關閉</button>
      `;

      const closeBtn = content.querySelector('.quiz-close-btn');
      closeBtn.addEventListener('click', () => {
        this._active = false;
        this.overlay.classList.add('hidden');
      });
    }

    /**
     * 顯示訊息（錯誤等）
     */
    _showMessage(msg, showClose) {
      const content = this.overlay.querySelector('.quiz-content');
      content.innerHTML = `
        <div class="quiz-header">
          <span class="quiz-icon">📝</span>
          <h2>知識挑戰</h2>
        </div>
        <p class="quiz-loading">${this._escapeHtml(msg)}</p>
        ${showClose ? '<button class="quiz-close-btn">關閉</button>' : ''}
      `;

      if (showClose) {
        const closeBtn = content.querySelector('.quiz-close-btn');
        closeBtn.addEventListener('click', () => {
          this._active = false;
          this.overlay.classList.add('hidden');
        });
      }
    }

    /**
     * HTML 跳脫
     */
    _escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    /**
     * 動態注入樣式
     */
    _applyStyles() {
      if (document.getElementById('quiz-styles')) return;

      const style = document.createElement('style');
      style.id = 'quiz-styles';
      style.textContent = `
        .quiz-overlay {
          position: fixed;
          top: 0; left: 0;
          width: 100%; height: 100%;
          background: rgba(0, 0, 0, 0.75);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 8500;
        }
        .quiz-overlay.hidden { display: none; }
        .quiz-modal {
          background: linear-gradient(135deg, #1a1a2e, #16213e);
          border: 2px solid #29b6f6;
          border-radius: 16px;
          padding: 24px 28px;
          max-width: 480px;
          width: 90%;
          box-shadow: 0 0 40px rgba(41, 182, 246, 0.15);
          font-family: "Microsoft JhengHei", sans-serif;
          color: #eee;
        }
        .quiz-header {
          text-align: center;
          margin-bottom: 16px;
          position: relative;
        }
        .quiz-header h2 {
          margin: 4px 0 0;
          font-size: 20px;
          color: #29b6f6;
        }
        .quiz-icon { font-size: 32px; }
        .quiz-progress {
          position: absolute;
          top: 0; right: 0;
          font-size: 13px;
          color: #888;
        }
        .quiz-question {
          font-size: 16px;
          line-height: 1.6;
          margin-bottom: 16px;
          color: #ddd;
        }
        .quiz-loading {
          text-align: center;
          color: #999;
          padding: 20px 0;
        }
        .quiz-options {
          display: flex;
          flex-direction: column;
          gap: 10px;
          margin-bottom: 12px;
        }
        .quiz-option {
          display: flex;
          align-items: center;
          gap: 12px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 10px;
          padding: 12px 16px;
          color: #ddd;
          font-size: 14px;
          font-family: inherit;
          cursor: pointer;
          text-align: left;
          transition: all 0.15s;
        }
        .quiz-option:hover:not(:disabled) {
          background: rgba(41, 182, 246, 0.12);
          border-color: rgba(41, 182, 246, 0.4);
        }
        .quiz-option.selected {
          border-color: #29b6f6;
          background: rgba(41, 182, 246, 0.15);
        }
        .quiz-option.correct {
          border-color: #4CAF50;
          background: rgba(76, 175, 80, 0.15);
        }
        .quiz-option.wrong {
          border-color: #ef5350;
          background: rgba(239, 83, 80, 0.15);
        }
        .quiz-option:disabled { cursor: default; }
        .quiz-option-label {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          border-radius: 50%;
          background: rgba(255,255,255,0.1);
          font-weight: bold;
          font-size: 13px;
          flex-shrink: 0;
        }
        .quiz-result-area { margin-top: 12px; }
        .quiz-feedback {
          padding: 12px 16px;
          border-radius: 10px;
          margin-bottom: 12px;
          font-size: 15px;
        }
        .quiz-correct {
          background: rgba(76, 175, 80, 0.12);
          border: 1px solid rgba(76, 175, 80, 0.3);
        }
        .quiz-wrong {
          background: rgba(239, 83, 80, 0.12);
          border: 1px solid rgba(239, 83, 80, 0.3);
        }
        .quiz-explanation {
          font-size: 13px;
          color: #bbb;
          margin: 8px 0 0;
          line-height: 1.5;
        }
        .quiz-next, .quiz-close-btn {
          display: block;
          width: 100%;
          padding: 12px;
          background: linear-gradient(135deg, #29b6f6, #0288d1);
          color: #fff;
          font-weight: bold;
          font-size: 15px;
          border: none;
          border-radius: 8px;
          cursor: pointer;
          font-family: inherit;
          transition: opacity 0.2s;
        }
        .quiz-next:hover, .quiz-close-btn:hover { opacity: 0.85; }
        .quiz-summary { text-align: center; padding: 12px 0 20px; }
        .quiz-score-big {
          font-size: 48px;
          font-weight: bold;
          color: #29b6f6;
          margin-bottom: 8px;
        }
        .quiz-rank {
          font-size: 18px;
          color: #e2b857;
          margin: 8px 0;
        }
        .quiz-stat {
          font-size: 14px;
          color: #999;
          margin: 4px 0;
        }
      `;
      document.head.appendChild(style);
    }
  }

  window.QuizSystem = QuizSystem;
})();
