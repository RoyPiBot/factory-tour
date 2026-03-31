/**
 * triggers.js - 區域進入偵測與觸發（RPG 版）v2.0
 *
 * v2.0 新增：
 *   - 測驗系統觸發（進入新區域後觸發測驗）
 *   - 回饋系統觸發（導覽完成後觸發評分）
 */
import { getRoomAt } from './map.js';

export class TriggerSystem {
  constructor(dialog, npcSystem) {
    this.dialog = dialog;
    this.npcSystem = npcSystem;
    this.currentAreaId = null;
    this.visitedAreas = new Set();
    this.triggeredAreas = new Set(); // 已自動觸發過的區域

    // v2.0 — 測驗與回饋
    this.quizSystem = null;
    this.feedbackSystem = null;
    this.quizPendingArea = null; // 等待對話結束後觸發測驗
    this.feedbackTriggered = false;
  }

  setQuizSystem(qs) {
    this.quizSystem = qs;
  }

  setFeedbackSystem(fs) {
    this.feedbackSystem = fs;
  }

  update(playerX, playerY) {
    const room = getRoomAt(playerX, playerY);
    const newAreaId = room ? room.id : null;

    if (newAreaId !== this.currentAreaId) {
      this.currentAreaId = newAreaId;

      if (room) {
        const isFirstVisit = !this.visitedAreas.has(room.id);
        this.visitedAreas.add(room.id);

        // 首次進入 → 自動觸發區域介紹
        if (!this.triggeredAreas.has(room.id)) {
          this.triggeredAreas.add(room.id);
          const npc = this.npcSystem.getNpcs().find(n => n.roomId === room.id);
          this.dialog.triggerAreaEntry(room, npc);

          // 記住此區域，等對話結束後觸發測驗
          if (this.quizSystem && !this.quizSystem.isAreaQuizzed(room.id)) {
            this.quizPendingArea = room.id;
            this._waitForDialogCloseThenQuiz();
          }
        }

        // 檢查是否導覽完成 → 觸發回饋
        this._checkTourComplete();
      }
    }

    // 持續檢查是否有 pending quiz
    if (this.quizPendingArea && this.quizSystem && !this.dialog.getIsOpen() && !this.quizSystem.isActive()) {
      const areaId = this.quizPendingArea;
      this.quizPendingArea = null;
      // 延遲 1 秒再彈出測驗
      setTimeout(() => {
        if (this.quizSystem && !this.quizSystem.isAreaQuizzed(areaId)) {
          this.quizSystem.startQuiz(areaId);
        }
      }, 1500);
    }
  }

  _waitForDialogCloseThenQuiz() {
    // 透過輪詢等待對話框關閉
    const check = () => {
      if (!this.dialog.getIsOpen() && this.quizPendingArea) {
        // 對話已關閉，update loop 會處理
        return;
      }
      if (this.quizPendingArea) {
        setTimeout(check, 500);
      }
    };
    setTimeout(check, 1000);
  }

  _checkTourComplete() {
    const progress = this.getProgress();
    if (progress.visited >= progress.total && !this.feedbackTriggered) {
      this.feedbackTriggered = true;

      // 延遲 3 秒彈出回饋
      setTimeout(() => {
        if (this.feedbackSystem && !this.feedbackSystem.hasSubmitted()) {
          const visitedList = Array.from(this.visitedAreas);
          this.feedbackSystem.show(visitedList);
        }
      }, 3000);
    }
  }

  getVisitedAreas() {
    return this.visitedAreas;
  }

  getCurrentArea() {
    return this.currentAreaId;
  }

  getProgress() {
    return { visited: this.visitedAreas.size, total: 5 };
  }
}
