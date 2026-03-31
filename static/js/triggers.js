/**
 * triggers.js - 區域進入偵測與觸發（RPG 版）
 * 進入區域時自動觸發 AI 對話框介紹
 */
import { getRoomAt } from './map.js';

export class TriggerSystem {
  constructor(dialog, npcSystem) {
    this.dialog = dialog;
    this.npcSystem = npcSystem;
    this.currentAreaId = null;
    this.visitedAreas = new Set();
    this.triggeredAreas = new Set(); // 已自動觸發過的區域
  }

  update(playerX, playerY) {
    const room = getRoomAt(playerX, playerY);
    const newAreaId = room ? room.id : null;

    if (newAreaId !== this.currentAreaId) {
      this.currentAreaId = newAreaId;

      if (room) {
        this.visitedAreas.add(room.id);

        // 首次進入 → 自動觸發區域介紹
        if (!this.triggeredAreas.has(room.id)) {
          this.triggeredAreas.add(room.id);
          // 找到該區域的 NPC
          const npc = this.npcSystem.getNpcs().find(n => n.roomId === room.id);
          this.dialog.triggerAreaEntry(room, npc);
        }
      }
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
