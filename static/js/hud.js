/**
 * hud.js - 全螢幕 RPG HUD v2.0
 *
 * v2.0 新增：
 *   - 即時感測器數據面板
 *   - 測驗分數顯示
 */
import { ROOMS } from './config.js';

export class HUD {
  constructor() {
    this.banner = null;
    this.bannerTimer = 0;
    this.bannerDuration = 3;

    // 感測器數據
    this.sensorData = null;

    // 測驗分數
    this.quizScore = null; // { totalCorrect, totalAnswered }
  }

  updateSensorData(data) {
    this.sensorData = data;
  }

  updateQuizScore(score) {
    this.quizScore = score;
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

        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        roundRect(ctx, bx, 12, textW, 36, 18);
        ctx.fill();

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
    const panelH = this.quizScore ? 76 : 56;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    roundRect(ctx, panelX, panelY, 200, panelH, 10);
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

    // 測驗分數
    if (this.quizScore && this.quizScore.totalAnswered > 0) {
      ctx.font = 'bold 11px "Microsoft JhengHei", sans-serif';
      ctx.fillStyle = '#FFD54F';
      ctx.fillText(
        `📝 測驗 ${this.quizScore.totalCorrect}/${this.quizScore.totalAnswered} 正確`,
        panelX + 12, panelY + 56
      );
    }

    // ── 左側：感測器數據面板 ──
    if (this.sensorData && currentArea) {
      this._drawSensorPanel(ctx, panelX, panelY + panelH + 12, currentArea);
    }

    // ── 右上：操作提示 ──
    const hints = [
      '⌨️ WASD 移動',
      '🗣️ E 對話',
      '💬 T 提問',
      '🎤 V 語音',
    ];
    const hintX = canvasW - 150, hintY = 16;

    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    roundRect(ctx, hintX, hintY, 134, 88, 10);
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

      if (room.id === currentArea) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        roundRect(ctx, ix, iy, iconSize, iconSize, 6);
        ctx.stroke();
      }

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

  /**
   * 繪製感測器數據面板
   */
  _drawSensorPanel(ctx, x, y, currentArea) {
    const areas = this.sensorData.areas || {};
    const areaData = areas[currentArea];
    if (!areaData) return;

    const panelW = 200;
    const lines = [];

    // 溫度
    if (areaData.temperature !== undefined) {
      const temp = areaData.temperature.toFixed(1);
      const tempColor = areaData.temperature > 35 ? '#ef5350' :
                         areaData.temperature > 30 ? '#ffa726' : '#66bb6a';
      lines.push({ icon: '🌡️', label: '溫度', value: `${temp}°C`, color: tempColor });
    }

    // 濕度
    if (areaData.humidity !== undefined) {
      lines.push({ icon: '💧', label: '濕度', value: `${areaData.humidity.toFixed(1)}%`, color: '#29b6f6' });
    }

    // 產線速度
    if (areaData.line_speed !== undefined) {
      lines.push({ icon: '⚡', label: '產速', value: `${Math.round(areaData.line_speed)}/h`, color: '#ffa726' });
    }

    // 良率
    if (areaData.yield_rate !== undefined) {
      const yr = areaData.yield_rate.toFixed(2);
      const yrColor = areaData.yield_rate < 99 ? '#ef5350' : '#66bb6a';
      lines.push({ icon: '✅', label: '良率', value: `${yr}%`, color: yrColor });
    }

    if (lines.length === 0) return;

    const panelH = 28 + lines.length * 22;
    ctx.fillStyle = 'rgba(0,0,0,0.65)';
    roundRect(ctx, x, y, panelW, panelH, 10);
    ctx.fill();

    // 標題
    ctx.font = 'bold 12px "Microsoft JhengHei", sans-serif';
    ctx.fillStyle = '#4FC3F7';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('📊 即時數據', x + 12, y + 6);

    // 數據行
    ctx.font = '11px "Microsoft JhengHei", sans-serif';
    lines.forEach((line, i) => {
      const ly = y + 24 + i * 22;

      ctx.fillStyle = '#999';
      ctx.fillText(`${line.icon} ${line.label}`, x + 12, ly);

      ctx.fillStyle = line.color;
      ctx.textAlign = 'right';
      ctx.font = 'bold 12px "Microsoft JhengHei", sans-serif';
      ctx.fillText(line.value, x + panelW - 12, ly);

      ctx.textAlign = 'left';
      ctx.font = '11px "Microsoft JhengHei", sans-serif';
    });

    // 告警
    const alerts = this.sensorData.alerts || [];
    const areaAlerts = alerts.filter(a => a.area === currentArea);
    if (areaAlerts.length > 0) {
      const alertY = y + panelH + 4;
      ctx.fillStyle = 'rgba(239, 83, 80, 0.2)';
      roundRect(ctx, x, alertY, panelW, 22, 6);
      ctx.fill();

      ctx.fillStyle = '#ef5350';
      ctx.font = 'bold 11px "Microsoft JhengHei", sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`⚠️ ${areaAlerts[0].type} 異常`, x + 8, alertY + 5);
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
