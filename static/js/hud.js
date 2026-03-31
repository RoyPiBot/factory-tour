/**
 * hud.js - 全螢幕 RPG HUD
 */
import { ROOMS } from './config.js';

export class HUD {
  constructor() {
    this.banner = null;
    this.bannerTimer = 0;
    this.bannerDuration = 3;
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

    // ── 頂部中央：當前區域名稱橫幅 ──
    if (currentArea) {
      const room = ROOMS.find(r => r.id === currentArea);
      if (room) {
        const label = room.emoji + ' ' + room.name;
        ctx.font = 'bold 18px "Microsoft JhengHei", sans-serif';
        const textW = ctx.measureText(label).width + 40;
        const bx = (canvasW - textW) / 2;

        // 半透明背景
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        roundRect(ctx, bx, 12, textW, 36, 18);
        ctx.fill();

        // 區域顏色邊框
        ctx.strokeStyle = room.color;
        ctx.lineWidth = 2;
        roundRect(ctx, bx, 12, textW, 36, 18);
        ctx.stroke();

        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, canvasW / 2, 30);
      }
    }

    // ── 左上：導覽進度面板 ──
    const panelX = 16, panelY = 16;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    roundRect(ctx, panelX, panelY, 200, 56, 10);
    ctx.fill();

    ctx.fillStyle = '#fff';
    ctx.font = 'bold 14px "Microsoft JhengHei", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`🏭 導覽進度`, panelX + 12, panelY + 8);

    ctx.font = '12px "Microsoft JhengHei", sans-serif';
    ctx.fillStyle = '#aaa';
    ctx.fillText(`${progress.visited} / ${progress.total} 區域已探索`, panelX + 12, panelY + 28);

    // 進度條
    const barX = panelX + 12, barY = panelY + 44, barW = 176, barH = 6;
    ctx.fillStyle = 'rgba(255,255,255,0.15)';
    roundRect(ctx, barX, barY, barW, barH, 3);
    ctx.fill();

    const pct = progress.visited / progress.total;
    if (pct > 0) {
      const gradient = ctx.createLinearGradient(barX, 0, barX + barW * pct, 0);
      gradient.addColorStop(0, '#4CAF50');
      gradient.addColorStop(1, '#81C784');
      ctx.fillStyle = gradient;
      roundRect(ctx, barX, barY, barW * pct, barH, 3);
      ctx.fill();
    }

    // ── 右上：操作提示 ──
    const hints = [
      '⌨️ WASD 移動',
      '🗣️ E 對話',
      '💬 T 提問',
    ];
    const hintX = canvasW - 150, hintY = 16;

    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    roundRect(ctx, hintX, hintY, 134, 70, 10);
    ctx.fill();

    ctx.fillStyle = '#999';
    ctx.font = '11px "Microsoft JhengHei", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    hints.forEach((hint, i) => {
      ctx.fillText(hint, hintX + 10, hintY + 10 + i * 20);
    });

    // ── 底部中央：已參觀區域圖示列 ──
    const visited = triggerSystem.getVisitedAreas();
    const iconSize = 32;
    const iconGap = 12;
    const totalW = ROOMS.length * (iconSize + iconGap) - iconGap;
    const startX = (canvasW - totalW) / 2;
    const startY = canvasH - 56;

    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    roundRect(ctx, startX - 12, startY - 6, totalW + 24, iconSize + 20, 12);
    ctx.fill();

    ROOMS.forEach((room, i) => {
      const ix = startX + i * (iconSize + iconGap);
      const iy = startY;

      // 背景方塊
      if (visited.has(room.id)) {
        ctx.fillStyle = room.color;
        ctx.globalAlpha = 0.8;
      } else {
        ctx.fillStyle = '#333';
        ctx.globalAlpha = 0.4;
      }
      roundRect(ctx, ix, iy, iconSize, iconSize, 6);
      ctx.fill();
      ctx.globalAlpha = 1;

      // 邊框（當前區域高亮）
      if (room.id === currentArea) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        roundRect(ctx, ix, iy, iconSize, iconSize, 6);
        ctx.stroke();
      }

      // Emoji
      ctx.font = '18px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(room.emoji, ix + iconSize / 2, iy + iconSize / 2);
    });

    // 完成提示
    if (progress.visited >= progress.total) {
      ctx.fillStyle = 'rgba(76, 175, 80, 0.9)';
      ctx.font = 'bold 13px "Microsoft JhengHei", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('🎉 導覽完成！', canvasW / 2, startY - 14);
    }
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
