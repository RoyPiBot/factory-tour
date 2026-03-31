/**
 * player.js - RPG 風格玩家角色
 */
import { TILE_SIZE, PLAYER_SPEED } from './config.js';
import { isWalkable } from './map.js';

export class Player {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.radius = 14;
    this.direction = 'down'; // up, down, left, right
    this.animFrame = 0;
    this.animTimer = 0;
    this.moving = false;
    this.stepCount = 0;
  }

  update(keys, dt) {
    let dx = 0, dy = 0;
    if (keys['ArrowUp'] || keys['KeyW']) { dy = -1; this.direction = 'up'; }
    if (keys['ArrowDown'] || keys['KeyS']) { dy = 1; this.direction = 'down'; }
    if (keys['ArrowLeft'] || keys['KeyA']) { dx = -1; this.direction = 'left'; }
    if (keys['ArrowRight'] || keys['KeyD']) { dx = 1; this.direction = 'right'; }

    // 正規化對角移動
    if (dx !== 0 && dy !== 0) {
      dx *= 0.707;
      dy *= 0.707;
    }

    this.moving = dx !== 0 || dy !== 0;

    if (this.moving) {
      const speed = PLAYER_SPEED * dt;
      const newX = this.x + dx * speed;
      const newY = this.y + dy * speed;
      const r = this.radius - 2;

      if (this.canMoveTo(newX, this.y, r)) this.x = newX;
      if (this.canMoveTo(this.x, newY, r)) this.y = newY;

      // 行走動畫
      this.animTimer += dt;
      if (this.animTimer > 0.12) {
        this.animTimer = 0;
        this.animFrame = (this.animFrame + 1) % 4;
        this.stepCount++;
      }
    } else {
      this.animFrame = 0;
    }
  }

  canMoveTo(x, y, r) {
    const corners = [
      { x: x - r, y: y - r },
      { x: x + r, y: y - r },
      { x: x - r, y: y + r },
      { x: x + r, y: y + r },
    ];
    return corners.every(c => {
      const tx = Math.floor(c.x / TILE_SIZE);
      const ty = Math.floor(c.y / TILE_SIZE);
      return isWalkable(tx, ty);
    });
  }

  draw(ctx, camera) {
    const sx = this.x - camera.x;
    const sy = this.y - camera.y;

    // 行走搖擺
    const wobble = this.moving ? Math.sin(this.stepCount * Math.PI / 2) * 2 : 0;

    // ── 陰影 ──
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.beginPath();
    ctx.ellipse(sx, sy + 16, 12, 5, 0, 0, Math.PI * 2);
    ctx.fill();

    // ── 身體（RPG 角色） ──
    const bodyY = sy + wobble * 0.3;

    // 安全帽
    ctx.fillStyle = '#FDD835'; // 黃色安全帽
    ctx.beginPath();
    ctx.ellipse(sx, bodyY - 14, 11, 8, 0, Math.PI, Math.PI * 2);
    ctx.fill();
    ctx.fillRect(sx - 13, bodyY - 14, 26, 3);

    // 頭
    ctx.fillStyle = '#FFCC80'; // 膚色
    ctx.beginPath();
    ctx.arc(sx, bodyY - 6, 9, 0, Math.PI * 2);
    ctx.fill();

    // 眼睛（根據方向）
    ctx.fillStyle = '#333';
    if (this.direction === 'left') {
      ctx.fillRect(sx - 5, bodyY - 8, 3, 3);
    } else if (this.direction === 'right') {
      ctx.fillRect(sx + 2, bodyY - 8, 3, 3);
    } else if (this.direction === 'up') {
      // 背面，不畫眼睛
    } else {
      // 正面
      ctx.fillRect(sx - 5, bodyY - 8, 3, 3);
      ctx.fillRect(sx + 2, bodyY - 8, 3, 3);
      // 微笑
      ctx.beginPath();
      ctx.arc(sx, bodyY - 3, 3, 0.1, Math.PI - 0.1);
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // 身體（工作服）
    ctx.fillStyle = '#42A5F5'; // 藍色工作服
    ctx.fillRect(sx - 8, bodyY + 2, 16, 12);

    // 工作服領口
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(sx - 3, bodyY + 2);
    ctx.lineTo(sx, bodyY + 6);
    ctx.lineTo(sx + 3, bodyY + 2);
    ctx.closePath();
    ctx.fill();

    // 訪客證
    ctx.fillStyle = '#fff';
    ctx.fillRect(sx + 4, bodyY + 5, 4, 6);
    ctx.fillStyle = '#F44336';
    ctx.fillRect(sx + 4, bodyY + 5, 4, 2);

    // 腿（行走動畫）
    ctx.fillStyle = '#37474F';
    if (this.moving) {
      const legOffset = Math.sin(this.stepCount * Math.PI / 2) * 3;
      ctx.fillRect(sx - 5, bodyY + 14, 4, 6 + legOffset);
      ctx.fillRect(sx + 1, bodyY + 14, 4, 6 - legOffset);
    } else {
      ctx.fillRect(sx - 5, bodyY + 14, 4, 6);
      ctx.fillRect(sx + 1, bodyY + 14, 4, 6);
    }

    // 鞋子
    ctx.fillStyle = '#212121';
    const shoeY = this.moving ? bodyY + 19 + Math.abs(wobble) * 0.3 : bodyY + 19;
    ctx.fillRect(sx - 6, shoeY, 5, 3);
    ctx.fillRect(sx + 1, shoeY, 5, 3);

    // ── 方向指示光暈 ──
    if (this.moving) {
      ctx.strokeStyle = 'rgba(66, 165, 245, 0.3)';
      ctx.lineWidth = 1.5;
      const pulseR = 20 + Math.sin(Date.now() / 200) * 3;
      ctx.beginPath();
      ctx.arc(sx, bodyY + 4, pulseR, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}
