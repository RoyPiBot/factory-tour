/**
 * triggers.js - 區域進入偵測與觸發
 */
import { getRoomAt } from './map.js';

export class TriggerSystem {
  constructor(chatPanel) {
    this.chat = chatPanel;
    this.currentAreaId = null;
    this.visitedAreas = new Set();
    this.lastTriggerTime = {};
    this.cooldown = 30000; // 30 秒冷卻
    this.areaCache = {};   // 快取區域資料
  }

  update(playerX, playerY) {
    const room = getRoomAt(playerX, playerY);
    const newAreaId = room ? room.id : null;

    if (newAreaId !== this.currentAreaId) {
      const previousArea = this.currentAreaId;
      this.currentAreaId = newAreaId;

      if (room && this.shouldTrigger(room.id)) {
        this.visitedAreas.add(room.id);
        this.lastTriggerTime[room.id] = Date.now();
        this.onEnterArea(room);
      }
    }
  }

  shouldTrigger(areaId) {
    const lastTime = this.lastTriggerTime[areaId] || 0;
    return (Date.now() - lastTime) > this.cooldown;
  }

  async onEnterArea(room) {
    // 先載入快取的區域資料
    if (!this.areaCache[room.id]) {
      try {
        const res = await fetch(`/areas/${room.id}`);
        if (res.ok) {
          this.areaCache[room.id] = await res.json();
        }
      } catch (e) {
        // 靜默失敗
      }
    }

    // 觸發 AI 導覽
    await this.chat.triggerAreaIntro(room);
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
