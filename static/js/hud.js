/**
 * hud.js - 遊戲 HUD (抬頭顯示器)
 */
import { ROOMS } from './config.js';

export class HUD {
  constructor() {
    this.banner = null;
    this.bannerTimer = 0;
    this.bannerDuration = 3; // 秒
  }

  showBanner(text, emoji) {
    this.banner = { text, emoji };
    this.bannerTimer = this.bannerDuration;
  }

  update(dt) {
    if (this.bannerTimer > 0) {
      this.bannerTimer -= dt;
    }
  }

  draw(ctx, canvasW, canvasH, triggerSystem) {
    const progress = triggerSystem.getProgress();
    const currentArea = triggerSystem.getCurrentArea();

    // 頂部：當前區域名稱
    if (currentArea) {
      const room = ROOMS.find(r => r.id === currentArea);
      if (room) {
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        const textW = ctx.measureText(room.emoji + ' ' + room.name).width + 40;
        const bx = (canvasW - textW) / 2;
        roundRect(ctx, bx, 8, textW, 32, 16);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px "Microsoft JhengHei", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(room.emoji + ' ' + room.name, canvasW / 2, 30);
      }
    }

    // 左上：導覽進度
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    roundRect(ctx, 8, 8, 160, 36, 8);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = '13px "Microsoft JhengHei", sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`📍 導覽進度：${progress.visited} / ${progress.total}`, 16, 30);

    // 進度條
    const barX = 16, barY = 36, barW = 140, barH = 4;
    ctx.fillStyle = 'rgba(255,255,255,0.2)';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = '#4CAF50';
    ctx.fillRect(barX, barY, barW * (progress.visited / progress.total), barH);

    // 右上：操作提示
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    roundRect(ctx, canvasW - 188, 8, 180, 30, 8);
    ctx.fill();
    ctx.fillStyle = '#ccc';
    ctx.font = '12px "Microsoft JhengHei", sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('⌨️ WASD / 方向鍵移動', canvasW - 16, 28);

    // 底部：已參觀區域圖示
    const visited = triggerSystem.getVisitedAreas();
    const iconSize = 24;
    const totalW = ROOMS.length * (iconSize + 8);
    const startX = (canvasW - totalW) / 2;

    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    roundRect(ctx, startX - 8, canvasH - 42, totalW + 16, 36, 8);
    ctx.fill();

    ROOMS.forEach((room, i) => {
      const ix = startX + i * (iconSize + 8);
      const iy = canvasH - 34;

      if (visited.has(room.id)) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = room.color;
      } else {
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = '#666';
      }

      roundRect(ctx, ix, iy, iconSize, iconSize, 4);
      ctx.fill();

      ctx.globalAlpha = 1;
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(room.emoji, ix + iconSize / 2, iy + iconSize - 5);
    });
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
