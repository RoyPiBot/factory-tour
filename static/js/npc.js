/**
 * npc.js - NPC 系統（RPG 風格導覽員）v2.0
 *
 * v2.0 新增：
 *   - 巡邏行為（在房間內走動）
 *   - 語音氣泡（主動打招呼）
 *   - 面向玩家
 *   - 狀態機 (idle / walking / talking / greeting)
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
    offsetX: 5, offsetY: 3,
    greeting: '歡迎來到智慧工廠！我是接待員小美，需要我帶您參觀嗎？',
    idleFrames: ['👩‍💼', '👩‍💼', '🙋‍♀️', '👩‍💼'],
    // 巡邏路點（房間內 tile 偏移）
    patrolPoints: [
      { x: 5, y: 3 }, { x: 3, y: 5 }, { x: 8, y: 6 }, { x: 5, y: 3 },
    ],
    // 隨機招呼語
    greetings: [
      '需要幫忙嗎？😊', '歡迎回來！', '這邊請～', '有什麼問題嗎？',
      '今天參觀愉快嗎？', '記得戴好安全帽喔！',
    ],
  },
  {
    id: 'npc_assembly',
    roomId: 'assembly_a',
    name: '阿強 — 產線工程師',
    emoji: '👷',
    portrait: '👷',
    color: '#ffa726',
    offsetX: 6, offsetY: 5,
    greeting: '嘿！歡迎來到 SMT 產線！這裡每小時可以處理 8 萬個零件呢！',
    idleFrames: ['👷', '👷', '🔧', '👷'],
    patrolPoints: [
      { x: 6, y: 5 }, { x: 10, y: 4 }, { x: 4, y: 7 }, { x: 12, y: 5 },
    ],
    greetings: [
      '小心燙！回焊爐很熱', '要不要看看 SMT 流程？', '今日良率 99.87%！',
      '機台運轉正常 👍', '這台印刷機很厲害喔',
    ],
  },
  {
    id: 'npc_qc',
    roomId: 'qc_room',
    name: '小琳 — 品管主管',
    emoji: '👩‍🔬',
    portrait: '👩‍🔬',
    color: '#29b6f6',
    offsetX: 7, offsetY: 3,
    greeting: '您好！我是品管室的小琳。我們的不良率控制在 0.1% 以下！',
    idleFrames: ['👩‍🔬', '👩‍🔬', '🔍', '👩‍🔬'],
    patrolPoints: [
      { x: 7, y: 3 }, { x: 4, y: 4 }, { x: 10, y: 3 }, { x: 7, y: 5 },
    ],
    greetings: [
      '品質是我們的命脈', 'AOI 正在掃描中...', '來看看 X-ray 檢測？',
      '今天的批次全數通過！', '不良品都在這裡被攔截 🔍',
    ],
  },
  {
    id: 'npc_warehouse',
    roomId: 'warehouse',
    name: '大叔 — 倉管主任',
    emoji: '👨‍🔧',
    portrait: '👨‍🔧',
    color: '#ef5350',
    offsetX: 7, offsetY: 4,
    greeting: '歡迎到倉儲區！我們使用自動化倉儲系統，可存放一萬個棧板。',
    idleFrames: ['👨‍🔧', '👨‍🔧', '📦', '👨‍🔧'],
    patrolPoints: [
      { x: 7, y: 4 }, { x: 4, y: 3 }, { x: 10, y: 5 }, { x: 7, y: 3 },
    ],
    greetings: [
      '堆高機通道請注意！', '今天出了 200 箱貨', '倉位 A3 剛進新料',
      '自動倉儲好方便啊', '注意腳下安全！📦',
    ],
  },
  {
    id: 'npc_conference',
    roomId: 'conference',
    name: '陳經理 — 廠長',
    emoji: '👨‍💼',
    portrait: '👨‍💼',
    color: '#ab47bc',
    offsetX: 4, offsetY: 3,
    greeting: '辛苦了！導覽即將結束，有任何問題都可以問我。',
    idleFrames: ['👨‍💼', '👨‍💼', '☕', '👨‍💼'],
    patrolPoints: [
      { x: 4, y: 3 }, { x: 6, y: 4 }, { x: 3, y: 5 }, { x: 7, y: 3 },
    ],
    greetings: [
      '導覽怎麼樣？', '有什麼建議嗎？', '我們的目標是零災害',
      '歡迎提出問題 ☕', '感謝您來參觀！',
    ],
  },
];

const INTERACT_RANGE = TILE_SIZE * 2.2;
const GREETING_RANGE = TILE_SIZE * 3.5; // 主動招呼距離
const GREETING_COOLDOWN = 25; // 招呼冷卻秒數
const PATROL_SPEED = 30; // 巡邏移動速度 (px/s)
const PATROL_PAUSE_MIN = 2; // 巡邏停頓最短秒數
const PATROL_PAUSE_MAX = 5; // 巡邏停頓最長秒數

export class NPCSystem {
  constructor() {
    this.npcs = NPC_DEFS.map(def => {
      const room = ROOMS.find(r => r.id === def.roomId);
      if (!room) return null;

      const homeX = (room.x + def.offsetX) * TILE_SIZE + TILE_SIZE / 2;
      const homeY = (room.y + def.offsetY) * TILE_SIZE + TILE_SIZE / 2;

      return {
        ...def,
        room,
        x: homeX,
        y: homeY,
        homeX, homeY,
        // 動畫
        animTimer: Math.random() * 4,
        animFrame: 0,
        bobOffset: 0,
        talking: false,
        // 巡邏狀態
        state: 'idle', // idle | walking | talking
        patrolIndex: 0,
        patrolPauseTimer: Math.random() * PATROL_PAUSE_MAX,
        targetX: homeX,
        targetY: homeY,
        direction: 'down', // up/down/left/right
        walkAnimPhase: 0,
        // 語音氣泡
        speechBubble: null,
        speechTimer: 0,
        greetCooldown: 0,
        hasProactiveGreeted: false,
      };
    }).filter(Boolean);

    this.nearestNpc = null;
    this.interactDistance = 0;
  }

  update(playerX, playerY, dt) {
    for (const npc of this.npcs) {
      // 更新動畫計時器
      npc.animTimer += dt;
      npc.bobOffset = Math.sin(npc.animTimer * 2) * 2;

      // 切換 idle 表情
      if (npc.state !== 'walking') {
        if (npc.animTimer > 1.5) {
          npc.animTimer = 0;
          npc.animFrame = (npc.animFrame + 1) % npc.idleFrames.length;
        }
      } else {
        // 走路時更快切換
        if (npc.animTimer > 0.5) {
          npc.animTimer = 0;
          npc.animFrame = (npc.animFrame + 1) % npc.idleFrames.length;
        }
      }

      // 更新招呼冷卻
      if (npc.greetCooldown > 0) {
        npc.greetCooldown -= dt;
      }

      // 更新語音氣泡
      if (npc.speechBubble) {
        npc.speechTimer -= dt;
        if (npc.speechTimer <= 0) {
          npc.speechBubble = null;
        }
      }

      // 巡邏邏輯
      this._updatePatrol(npc, playerX, playerY, dt);

      // 主動打招呼
      this._updateProactiveGreeting(npc, playerX, playerY);
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

  _updatePatrol(npc, playerX, playerY, dt) {
    // 靠近玩家時停止巡邏，面向玩家
    const dx = playerX - npc.x;
    const dy = playerY - npc.y;
    const distToPlayer = Math.sqrt(dx * dx + dy * dy);

    if (distToPlayer < INTERACT_RANGE) {
      npc.state = 'idle';
      // 面向玩家
      if (Math.abs(dx) > Math.abs(dy)) {
        npc.direction = dx > 0 ? 'right' : 'left';
      } else {
        npc.direction = dy > 0 ? 'down' : 'up';
      }
      return;
    }

    if (npc.state === 'idle') {
      npc.patrolPauseTimer -= dt;
      if (npc.patrolPauseTimer <= 0) {
        // 移動到下一個巡邏點
        npc.patrolIndex = (npc.patrolIndex + 1) % npc.patrolPoints.length;
        const point = npc.patrolPoints[npc.patrolIndex];
        npc.targetX = (npc.room.x + point.x) * TILE_SIZE + TILE_SIZE / 2;
        npc.targetY = (npc.room.y + point.y) * TILE_SIZE + TILE_SIZE / 2;
        npc.state = 'walking';
      }
    } else if (npc.state === 'walking') {
      const tdx = npc.targetX - npc.x;
      const tdy = npc.targetY - npc.y;
      const dist = Math.sqrt(tdx * tdx + tdy * tdy);

      if (dist < 3) {
        // 到達目標
        npc.x = npc.targetX;
        npc.y = npc.targetY;
        npc.state = 'idle';
        npc.patrolPauseTimer = PATROL_PAUSE_MIN + Math.random() * (PATROL_PAUSE_MAX - PATROL_PAUSE_MIN);
      } else {
        // 移動
        const speed = PATROL_SPEED * dt;
        const nx = (tdx / dist) * speed;
        const ny = (tdy / dist) * speed;
        npc.x += nx;
        npc.y += ny;

        // 更新方向
        if (Math.abs(tdx) > Math.abs(tdy)) {
          npc.direction = tdx > 0 ? 'right' : 'left';
        } else {
          npc.direction = tdy > 0 ? 'down' : 'up';
        }

        // 走路動畫
        npc.walkAnimPhase += dt * 6;
      }
    }
  }

  _updateProactiveGreeting(npc, playerX, playerY) {
    if (npc.greetCooldown > 0) return;
    if (npc.speechBubble) return;

    const dx = playerX - npc.x;
    const dy = playerY - npc.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    // 在招呼範圍內但不在互動範圍內時，顯示氣泡
    if (dist < GREETING_RANGE && dist > INTERACT_RANGE * 0.8) {
      const greetings = npc.greetings || ['你好！'];
      const text = greetings[Math.floor(Math.random() * greetings.length)];
      npc.speechBubble = text;
      npc.speechTimer = 3; // 顯示 3 秒
      npc.greetCooldown = GREETING_COOLDOWN;
    }
  }

  getInteractableNpc() {
    return this.nearestNpc;
  }

  draw(ctx, camera) {
    for (const npc of this.npcs) {
      const sx = npc.x - camera.x;
      const sy = npc.y - camera.y + npc.bobOffset;

      // 跳過畫面外的 NPC
      if (sx < -60 || sx > camera.w + 60 || sy < -80 || sy > camera.h + 60) continue;

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

      // NPC 邊框（走路時稍微抖動）
      const wobble = npc.state === 'walking' ? Math.sin(npc.walkAnimPhase) * 1.5 : 0;
      ctx.strokeStyle = npc.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx + wobble, sy, 18, 0, Math.PI * 2);
      ctx.stroke();

      // NPC emoji
      const frame = npc.idleFrames[npc.animFrame];
      ctx.font = '24px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(frame, sx + wobble, sy);

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

      // 語音氣泡
      if (npc.speechBubble) {
        this._drawSpeechBubble(ctx, sx, sy - 55, npc.speechBubble, npc.speechTimer);
      }

      // 互動提示 — 最近的 NPC 顯示 "按 E 對話"
      if (npc === this.nearestNpc && !npc.speechBubble) {
        const hintY = sy - 50;
        const pulse = Math.sin(Date.now() / 300) * 0.3 + 0.7;
        ctx.globalAlpha = pulse;

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

  _drawSpeechBubble(ctx, x, y, text, timer) {
    ctx.save();

    // 漸隱效果
    const alpha = Math.min(timer, 1);
    ctx.globalAlpha = alpha;

    ctx.font = '11px "Microsoft JhengHei", sans-serif';
    const textW = ctx.measureText(text).width + 16;
    const bx = x - textW / 2;
    const by = y - 10;

    // 氣泡背景
    ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
    roundRect(ctx, bx, by, textW, 22, 8);
    ctx.fill();

    // 氣泡邊框
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.lineWidth = 1;
    roundRect(ctx, bx, by, textW, 22, 8);
    ctx.stroke();

    // 小三角
    ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
    ctx.beginPath();
    ctx.moveTo(x - 4, by + 22);
    ctx.lineTo(x + 4, by + 22);
    ctx.lineTo(x, by + 28);
    ctx.closePath();
    ctx.fill();

    // 文字
    ctx.fillStyle = '#333';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x, by + 11);

    ctx.restore();
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
