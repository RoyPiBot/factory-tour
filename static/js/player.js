/**
 * player.js - 玩家角色
 */
import { TILE_SIZE, PLAYER_SPEED } from './config.js';
import { isWalkable } from './map.js';

export class Player {
  constructor(x, y) {
    this.x = x;
    this.y = y;
    this.radius = 12;
    this.direction = 'down'; // up, down, left, right
    this.animFrame = 0;
    this.animTimer = 0;
    this.moving = false;
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

      // 分軸碰撞偵測
      const r = this.radius - 2; // 稍微縮小碰撞半徑

      // 嘗試 X 軸移動
      if (this.canMoveTo(newX, this.y, r)) {
        this.x = newX;
      }
      // 嘗試 Y 軸移動
      if (this.canMoveTo(this.x, newY, r)) {
        this.y = newY;
      }

      // 動畫
      this.animTimer += dt;
      if (this.animTimer > 0.15) {
        this.animTimer = 0;
        this.animFrame = (this.animFrame + 1) % 4;
      }
    }
  }

  canMoveTo(x, y, r) {
    // 檢查四個角落
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

    // 陰影
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.beginPath();
    ctx.ellipse(sx, sy + 4, this.radius, this.radius * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();

    // 身體
    ctx.fillStyle = '#4FC3F7';
    ctx.beginPath();
    ctx.arc(sx, sy, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0288D1';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 方向指示 (小三角形)
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    const arrowSize = 6;
    switch (this.direction) {
      case 'up':
        ctx.moveTo(sx, sy - arrowSize - 2);
        ctx.lineTo(sx - arrowSize / 2, sy - 1);
        ctx.lineTo(sx + arrowSize / 2, sy - 1);
        break;
      case 'down':
        ctx.moveTo(sx, sy + arrowSize + 2);
        ctx.lineTo(sx - arrowSize / 2, sy + 1);
        ctx.lineTo(sx + arrowSize / 2, sy + 1);
        break;
      case 'left':
        ctx.moveTo(sx - arrowSize - 2, sy);
        ctx.lineTo(sx - 1, sy - arrowSize / 2);
        ctx.lineTo(sx - 1, sy + arrowSize / 2);
        break;
      case 'right':
        ctx.moveTo(sx + arrowSize + 2, sy);
        ctx.lineTo(sx + 1, sy - arrowSize / 2);
        ctx.lineTo(sx + 1, sy + arrowSize / 2);
        break;
    }
    ctx.closePath();
    ctx.fill();

    // 走路動畫 - 腳步波紋
    if (this.moving) {
      ctx.strokeStyle = 'rgba(79, 195, 247, 0.3)';
      ctx.lineWidth = 1;
      const pulseR = this.radius + 4 + Math.sin(this.animFrame * Math.PI / 2) * 3;
      ctx.beginPath();
      ctx.arc(sx, sy, pulseR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // 「訪客」標籤
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px "Microsoft JhengHei", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('👤', sx, sy + 4);
  }
}
