/**
 * npc.js - NPC 系統（RPG 風格導覽員）
 */
import { TILE_SIZE, ROOMS } from './config.js';

// NPC 定義 — 每個房間一個導覽員
const NPC_DEFS = [
  {
    id: 'npc_lobby',
    roomId: 'lobby',
    name: '小美 — 接待員',
    emoji: '👩‍💼',
    portrait: '👩‍💼',
    color: '#66bb6a',
    offsetX: 4, offsetY: 2,  // 在房間內的 tile 偏移
    greeting: '歡迎來到智慧工廠！我是接待員小美，需要我帶您參觀嗎？',
    idleFrames: ['👩‍💼', '👩‍💼', '🙋‍♀️', '👩‍💼'],
  },
  {
    id: 'npc_assembly',
    roomId: 'assembly_a',
    name: '阿強 — 產線工程師',
    emoji: '👷',
    portrait: '👷',
    color: '#ffa726',
    offsetX: 4, offsetY: 4,
    greeting: '嘿！歡迎來到 SMT 產線！這裡每小時可以處理 8 萬個零件呢！',
    idleFrames: ['👷', '👷', '🔧', '👷'],
  },
  {
    id: 'npc_qc',
    roomId: 'qc_room',
    name: '小琳 — 品管主管',
    emoji: '👩‍🔬',
    portrait: '👩‍🔬',
    color: '#29b6f6',
    offsetX: 5, offsetY: 2,
    greeting: '您好！我是品管室的小琳。我們的不良率控制在 0.1% 以下！',
    idleFrames: ['👩‍🔬', '👩‍🔬', '🔍', '👩‍🔬'],
  },
  {
    id: 'npc_warehouse',
    roomId: 'warehouse',
    name: '大叔 — 倉管主任',
    emoji: '👨‍🔧',
    portrait: '👨‍🔧',
    color: '#ef5350',
    offsetX: 5, offsetY: 3,
    greeting: '歡迎到倉儲區！我們使用自動化倉儲系統，可存放一萬個棧板。',
    idleFrames: ['👨‍🔧', '👨‍🔧', '📦', '👨‍🔧'],
  },
  {
    id: 'npc_conference',
    roomId: 'conference',
    name: '陳經理 — 廠長',
    emoji: '👨‍💼',
    portrait: '👨‍💼',
    color: '#ab47bc',
    offsetX: 2, offsetY: 2,
    greeting: '辛苦了！導覽即將結束，有任何問題都可以問我。',
    idleFrames: ['👨‍💼', '👨‍💼', '☕', '👨‍💼'],
  },
];

const INTERACT_RANGE = TILE_SIZE * 2.2; // 互動距離（像素）

export class NPCSystem {
  constructor() {
    this.npcs = NPC_DEFS.map(def => {
      const room = ROOMS.find(r => r.id === def.roomId);
      if (!room) return null;
      return {
        ...def,
        x: (room.x + def.offsetX) * TILE_SIZE + TILE_SIZE / 2,
        y: (room.y + def.offsetY) * TILE_SIZE + TILE_SIZE / 2,
        animTimer: Math.random() * 4, // 隨機起始相位
        animFrame: 0,
        bobOffset: 0,
        talking: false,
      };
    }).filter(Boolean);

    this.nearestNpc = null;
    this.interactDistance = 0;
  }

  update(playerX, playerY, dt) {
    // 更新 NPC 動畫
    for (const npc of this.npcs) {
      npc.animTimer += dt;
      // 呼吸/漂浮動畫
      npc.bobOffset = Math.sin(npc.animTimer * 2) * 2;
      // 切換 idle 表情
      if (npc.animTimer > 1.5) {
        npc.animTimer = 0;
        npc.animFrame = (npc.animFrame + 1) % npc.idleFrames.length;
      }
    }

    // 尋找最近的 NPC
    let nearest = null;
    let minDist = Infinity;
    for (const npc of this.npcs) {
      const dx = playerX - npc.x;
      const dy = playerY - npc.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) {
        minDist = dist;
        nearest = npc;
      }
    }

    this.nearestNpc = (minDist <= INTERACT_RANGE) ? nearest : null;
    this.interactDistance = minDist;
  }

  /**
   * 取得可互動的 NPC（在互動範圍內）
   */
  getInteractableNpc() {
    return this.nearestNpc;
  }

  /**
   * 繪製所有 NPC
   */
  draw(ctx, camera) {
    for (const npc of this.npcs) {
      const sx = npc.x - camera.x;
      const sy = npc.y - camera.y + npc.bobOffset;

      // 跳過畫面外的 NPC
      if (sx < -40 || sx > camera.w + 40 || sy < -60 || sy > camera.h + 40) continue;

      // NPC 陰影
      ctx.fillStyle = 'rgba(0,0,0,0.25)';
      ctx.beginPath();
      ctx.ellipse(sx, sy + 18, 14, 6, 0, 0, Math.PI * 2);
      ctx.fill();

      // NPC 身體底座（圓形 + 顏色）
      ctx.fillStyle = npc.color;
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(sx, sy, 20, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // NPC 邊框
      ctx.strokeStyle = npc.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 18, 0, Math.PI * 2);
      ctx.stroke();

      // NPC emoji
      const frame = npc.idleFrames[npc.animFrame];
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(frame, sx, sy);

      // NPC 名牌
      ctx.font = 'bold 10px "Microsoft JhengHei", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const nameLabel = npc.name.split('—')[0].trim();
      const nameW = ctx.measureText(nameLabel).width + 10;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      roundRect(ctx, sx - nameW / 2, sy - 32, nameW, 16, 4);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.fillText(nameLabel, sx, sy - 30);

      // 互動提示 — 最近的 NPC 顯示 "按 E 對話"
      if (npc === this.nearestNpc) {
        const hintY = sy - 50;
        const pulse = Math.sin(Date.now() / 300) * 0.3 + 0.7;
        ctx.globalAlpha = pulse;

        // 提示氣泡
        ctx.fillStyle = 'rgba(79, 195, 247, 0.9)';
        const hint = '按 E 對話';
        const hintW = ctx.measureText(hint).width + 16;
        roundRect(ctx, sx - hintW / 2, hintY - 6, hintW, 18, 6);
        ctx.fill();

        // 小三角
        ctx.beginPath();
        ctx.moveTo(sx - 5, hintY + 12);
        ctx.lineTo(sx + 5, hintY + 12);
        ctx.lineTo(sx, hintY + 18);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#000';
        ctx.font = 'bold 10px "Microsoft JhengHei", sans-serif';
        ctx.fillText(hint, sx, hintY - 2);

        ctx.globalAlpha = 1;
      }
    }
  }

  getNpcs() {
    return this.npcs;
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
