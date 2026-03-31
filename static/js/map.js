/**
 * map.js - 地圖建構與渲染 (v2 — 升級版視覺效果)
 */
import { TILE_SIZE, MAP_COLS, MAP_ROWS, TILE, ROOMS, CORRIDORS, DOORS } from './config.js';

let mapGrid = [];
let animTime = 0; // 全域動畫計時器

/**
 * 建構地圖 tile 陣列
 */
export function buildMap() {
  mapGrid = Array.from({ length: MAP_ROWS }, () =>
    Array.from({ length: MAP_COLS }, () => TILE.WALL)
  );

  for (const room of ROOMS) {
    for (let y = room.y; y < room.y + room.h; y++) {
      for (let x = room.x; x < room.x + room.w; x++) {
        if (y >= 0 && y < MAP_ROWS && x >= 0 && x < MAP_COLS) {
          mapGrid[y][x] = TILE.FLOOR;
        }
      }
    }
  }

  for (const cor of CORRIDORS) {
    for (let y = cor.y; y < cor.y + cor.h; y++) {
      for (let x = cor.x; x < cor.x + cor.w; x++) {
        if (y >= 0 && y < MAP_ROWS && x >= 0 && x < MAP_COLS) {
          mapGrid[y][x] = TILE.FLOOR;
        }
      }
    }
  }

  for (const door of DOORS) {
    if (door.y >= 0 && door.y < MAP_ROWS && door.x >= 0 && door.x < MAP_COLS) {
      mapGrid[door.y][door.x] = TILE.DOOR;
    }
  }

  return mapGrid;
}

export function isWalkable(tileX, tileY) {
  if (tileX < 0 || tileX >= MAP_COLS || tileY < 0 || tileY >= MAP_ROWS) return false;
  return mapGrid[tileY][tileX] !== TILE.WALL;
}

export function getRoomAt(pixelX, pixelY) {
  const tx = Math.floor(pixelX / TILE_SIZE);
  const ty = Math.floor(pixelY / TILE_SIZE);
  for (const room of ROOMS) {
    if (tx >= room.x && tx < room.x + room.w &&
        ty >= room.y && ty < room.y + room.h) {
      return room;
    }
  }
  return null;
}

/* ══════════════════════════════════════════
   繪製地圖 — 主函式
   ══════════════════════════════════════════ */
export function drawMap(ctx, camera, dt) {
  animTime += (dt || 0.016);

  const startCol = Math.max(0, Math.floor(camera.x / TILE_SIZE));
  const endCol   = Math.min(MAP_COLS, Math.ceil((camera.x + camera.w) / TILE_SIZE) + 1);
  const startRow = Math.max(0, Math.floor(camera.y / TILE_SIZE));
  const endRow   = Math.min(MAP_ROWS, Math.ceil((camera.y + camera.h) / TILE_SIZE) + 1);

  // ── 1. 底層 tiles ──
  for (let y = startRow; y < endRow; y++) {
    for (let x = startCol; x < endCol; x++) {
      const tile = mapGrid[y][x];
      const sx = x * TILE_SIZE - camera.x;
      const sy = y * TILE_SIZE - camera.y;

      if (tile === TILE.WALL) {
        drawWallTile(ctx, sx, sy, x, y);
      } else if (tile === TILE.DOOR) {
        drawDoorTile(ctx, sx, sy);
      } else {
        drawCorridorFloor(ctx, sx, sy);
      }
    }
  }

  // ── 2. 房間（覆蓋樓層）──
  for (const room of ROOMS) {
    drawRoom(ctx, room, camera);
  }

  // ── 3. 設備 ──
  drawEquipment(ctx, camera);

  // ── 4. 牆壁頂部陰影（深度感）──
  drawWallShadows(ctx, camera, startCol, endCol, startRow, endRow);
}

/* ══════════════════════════════════════════
   牆壁 — 3D 立體感
   ══════════════════════════════════════════ */
function drawWallTile(ctx, sx, sy, tileX, tileY) {
  // 牆壁主色
  const base = 42 + ((tileX * 7 + tileY * 13) % 8);  // 微妙色差
  ctx.fillStyle = `rgb(${base}, ${base}, ${base + 12})`;
  ctx.fillRect(sx, sy, TILE_SIZE, TILE_SIZE);

  // 磚塊紋理
  ctx.strokeStyle = `rgba(20,20,30,0.4)`;
  ctx.lineWidth = 0.5;
  // 橫線
  ctx.beginPath();
  ctx.moveTo(sx, sy + TILE_SIZE / 2);
  ctx.lineTo(sx + TILE_SIZE, sy + TILE_SIZE / 2);
  ctx.stroke();
  // 交錯豎線
  const offset = (tileY % 2) * (TILE_SIZE / 2);
  ctx.beginPath();
  ctx.moveTo(sx + offset, sy);
  ctx.lineTo(sx + offset, sy + TILE_SIZE / 2);
  ctx.moveTo(sx + offset + TILE_SIZE / 2, sy + TILE_SIZE / 2);
  ctx.lineTo(sx + offset + TILE_SIZE / 2, sy + TILE_SIZE);
  ctx.stroke();

  // 頂部高光
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(sx, sy, TILE_SIZE, 2);
}

/* ══════════════════════════════════════════
   門 — 帶光暈
   ══════════════════════════════════════════ */
function drawDoorTile(ctx, sx, sy) {
  // 門口地板
  ctx.fillStyle = '#c49a6c';
  ctx.fillRect(sx, sy, TILE_SIZE, TILE_SIZE);

  // 門框
  ctx.strokeStyle = '#8B6914';
  ctx.lineWidth = 2;
  ctx.strokeRect(sx + 3, sy + 3, TILE_SIZE - 6, TILE_SIZE - 6);

  // 門把
  ctx.fillStyle = '#DAA520';
  ctx.beginPath();
  ctx.arc(sx + TILE_SIZE - 10, sy + TILE_SIZE / 2, 3, 0, Math.PI * 2);
  ctx.fill();

  // 門口光暈
  const glow = ctx.createRadialGradient(
    sx + TILE_SIZE / 2, sy + TILE_SIZE / 2, 0,
    sx + TILE_SIZE / 2, sy + TILE_SIZE / 2, TILE_SIZE
  );
  glow.addColorStop(0, 'rgba(255,220,100,0.15)');
  glow.addColorStop(1, 'rgba(255,220,100,0)');
  ctx.fillStyle = glow;
  ctx.fillRect(sx - TILE_SIZE / 2, sy - TILE_SIZE / 2, TILE_SIZE * 2, TILE_SIZE * 2);
}

/* ══════════════════════════════════════════
   走廊地板
   ══════════════════════════════════════════ */
function drawCorridorFloor(ctx, sx, sy) {
  ctx.fillStyle = '#d0d0c8';
  ctx.fillRect(sx, sy, TILE_SIZE, TILE_SIZE);
  // 格線
  ctx.strokeStyle = '#bbb';
  ctx.lineWidth = 0.3;
  ctx.strokeRect(sx + 0.5, sy + 0.5, TILE_SIZE - 1, TILE_SIZE - 1);
}

/* ══════════════════════════════════════════
   房間繪製 — 精緻地板 + 邊框 + 名稱
   ══════════════════════════════════════════ */
function drawRoom(ctx, room, camera) {
  const rx = room.x * TILE_SIZE - camera.x;
  const ry = room.y * TILE_SIZE - camera.y;
  const rw = room.w * TILE_SIZE;
  const rh = room.h * TILE_SIZE;

  // ── 地板底色 ──
  ctx.fillStyle = room.floorColor;
  ctx.fillRect(rx, ry, rw, rh);

  // ── 地板紋路 ──
  drawFloorPattern(ctx, rx, ry, rw, rh, room.floorPattern);

  // ── 環境光 ──
  if (room.ambientColor) {
    const ambient = ctx.createRadialGradient(
      rx + rw / 2, ry + rh / 2, 0,
      rx + rw / 2, ry + rh / 2, Math.max(rw, rh) * 0.7
    );
    ambient.addColorStop(0, room.ambientColor.replace('0.06', '0.12'));
    ambient.addColorStop(1, 'transparent');
    ctx.fillStyle = ambient;
    ctx.fillRect(rx, ry, rw, rh);
  }

  // ── 房間邊框 ──
  ctx.save();
  // 外框陰影
  ctx.shadowColor = room.color;
  ctx.shadowBlur = 6;
  ctx.strokeStyle = room.color;
  ctx.lineWidth = 3;
  roundRect(ctx, rx + 2, ry + 2, rw - 4, rh - 4, 4);
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.restore();

  // ── 房間名稱標籤 ──
  const labelW = ctx.measureText ? 160 : 160;
  const labelX = rx + rw / 2;
  const labelY = ry + 18;

  // 標籤背景
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  roundRect(ctx, labelX - 80, labelY - 12, 160, 18, 9);
  ctx.fill();

  ctx.fillStyle = '#fff';
  ctx.font = 'bold 11px "Microsoft JhengHei", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(room.emoji + ' ' + room.name, labelX, labelY);

  // ── 危險區域標示 ──
  if (room.hazard) {
    drawHazardBorder(ctx, rx, ry, rw, rh);
  }
}

/* ── 地板紋路 ── */
function drawFloorPattern(ctx, x, y, w, h, pattern) {
  ctx.save();
  ctx.globalAlpha = 0.15;

  switch (pattern) {
    case 'marble': {
      // 大理石 — 細微紋路
      for (let dy = 0; dy < h; dy += TILE_SIZE) {
        for (let dx = 0; dx < w; dx += TILE_SIZE) {
          ctx.strokeStyle = '#888';
          ctx.lineWidth = 0.3;
          ctx.strokeRect(x + dx + 0.5, y + dy + 0.5, TILE_SIZE - 1, TILE_SIZE - 1);
          // 對角裝飾線
          if ((dx / TILE_SIZE + dy / TILE_SIZE) % 2 === 0) {
            ctx.beginPath();
            ctx.moveTo(x + dx + 4, y + dy + 4);
            ctx.lineTo(x + dx + TILE_SIZE - 4, y + dy + TILE_SIZE - 4);
            ctx.stroke();
          }
        }
      }
      break;
    }
    case 'industrial': {
      // 工業地板 — 防滑紋
      ctx.strokeStyle = '#999';
      ctx.lineWidth = 0.5;
      for (let dy = 0; dy < h; dy += 8) {
        ctx.beginPath();
        ctx.moveTo(x, y + dy);
        ctx.lineTo(x + w, y + dy);
        ctx.stroke();
      }
      // 安全標線
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = '#FFC107';
      ctx.fillRect(x, y + h - 6, w, 3);
      ctx.fillRect(x, y + 3, w, 3);
      break;
    }
    case 'cleanroom': {
      // 無塵室 — 棋盤格
      for (let dy = 0; dy < h; dy += TILE_SIZE) {
        for (let dx = 0; dx < w; dx += TILE_SIZE) {
          if ((dx / TILE_SIZE + dy / TILE_SIZE) % 2 === 0) {
            ctx.fillStyle = '#90CAF9';
            ctx.fillRect(x + dx, y + dy, TILE_SIZE, TILE_SIZE);
          }
        }
      }
      break;
    }
    case 'concrete': {
      // 水泥地 — 粗糙
      ctx.strokeStyle = '#aaa';
      ctx.lineWidth = 0.3;
      for (let dy = 0; dy < h; dy += TILE_SIZE * 2) {
        ctx.beginPath();
        ctx.moveTo(x, y + dy);
        ctx.lineTo(x + w, y + dy);
        ctx.stroke();
      }
      for (let dx = 0; dx < w; dx += TILE_SIZE * 2) {
        ctx.beginPath();
        ctx.moveTo(x + dx, y);
        ctx.lineTo(x + dx, y + h);
        ctx.stroke();
      }
      break;
    }
    case 'carpet': {
      // 地毯 — 密集小點
      ctx.fillStyle = '#9C27B0';
      for (let dy = 4; dy < h; dy += 6) {
        for (let dx = 4; dx < w; dx += 6) {
          ctx.fillRect(x + dx, y + dy, 1, 1);
        }
      }
      break;
    }
  }

  ctx.restore();
}

/* ══════════════════════════════════════════
   危險區域邊框 — 動態黃黑條紋
   ══════════════════════════════════════════ */
function drawHazardBorder(ctx, x, y, w, h) {
  ctx.save();
  const offset = (animTime * 20) % 16;

  // 黃黑警戒線
  ctx.lineWidth = 5;
  ctx.setLineDash([8, 8]);
  ctx.lineDashOffset = -offset;
  ctx.strokeStyle = '#FFC107';
  ctx.strokeRect(x + 5, y + 5, w - 10, h - 10);
  ctx.lineDashOffset = -offset + 8;
  ctx.strokeStyle = 'rgba(0,0,0,0.4)';
  ctx.strokeRect(x + 5, y + 5, w - 10, h - 10);

  ctx.setLineDash([]);
  ctx.restore();
}

/* ══════════════════════════════════════════
   牆壁陰影 — 在地板上投射
   ══════════════════════════════════════════ */
function drawWallShadows(ctx, camera, sc, ec, sr, er) {
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.12)';
  for (let y = sr; y < er; y++) {
    for (let x = sc; x < ec; x++) {
      if (mapGrid[y][x] === TILE.WALL) {
        // 如果下方是地板，畫陰影
        if (y + 1 < MAP_ROWS && mapGrid[y + 1][x] !== TILE.WALL) {
          const sx = x * TILE_SIZE - camera.x;
          const sy = (y + 1) * TILE_SIZE - camera.y;
          ctx.fillRect(sx, sy, TILE_SIZE, 6);
        }
        // 如果右方是地板，畫陰影
        if (x + 1 < MAP_COLS && mapGrid[y][x + 1] !== TILE.WALL) {
          const sx = (x + 1) * TILE_SIZE - camera.x;
          const sy = y * TILE_SIZE - camera.y;
          ctx.fillRect(sx, sy, 4, TILE_SIZE);
        }
      }
    }
  }
  ctx.restore();
}

/* ══════════════════════════════════════════
   設備繪製 — 全新精緻版
   ══════════════════════════════════════════ */
function drawEquipment(ctx, camera) {
  for (const room of ROOMS) {
    for (const eq of room.equipment) {
      const ex = eq.x * TILE_SIZE - camera.x;
      const ey = eq.y * TILE_SIZE - camera.y;
      const ew = (eq.w || 1) * TILE_SIZE;
      const eh = (eq.h || 1) * TILE_SIZE;

      switch (eq.type) {
        case 'reception':
          drawReception(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'display_wall':
          drawDisplayWall(ctx, ex, ey, ew, (eq.h || 1) * TILE_SIZE, eq.label);
          break;
        case 'plant':
          drawPlant(ctx, ex, ey);
          break;
        case 'bench':
          drawBench(ctx, ex, ey, ew, eq.label);
          break;
        case 'smt_machine':
          drawSMTMachine(ctx, ex, ey, eq.label);
          break;
        case 'conveyor':
          drawConveyor(ctx, ex, ey, ew, eq.label);
          break;
        case 'status_light':
          drawStatusLight(ctx, ex, ey, eq.status);
          break;
        case 'aoi_machine':
          drawAOIMachine(ctx, ex, ey, ew, eq.label);
          break;
        case 'xray_machine':
          drawXrayMachine(ctx, ex, ey, ew, eq.label);
          break;
        case 'test_station':
          drawTestStation(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'monitor_rack':
          drawMonitorRack(ctx, ex, ey, ew, eq.label);
          break;
        case 'shelf_rack':
          drawShelfRack(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'forklift_lane':
          drawForkliftLane(ctx, ex, ey, (eq.h || 1) * TILE_SIZE);
          break;
        case 'loading_dock':
          drawLoadingDock(ctx, ex, ey, (eq.h || 1) * TILE_SIZE, eq.label);
          break;
        case 'conf_table':
          drawConfTable(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'projector':
          drawProjector(ctx, ex, ey, eq.label);
          break;
        case 'whiteboard':
          drawWhiteboard(ctx, ex, ey, (eq.h || 1) * TILE_SIZE, eq.label);
          break;
        // 向後相容舊設備
        case 'desk':
          drawReception(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'display':
          drawDisplayWall(ctx, ex, ey, ew, (eq.h || 1) * TILE_SIZE, eq.label);
          break;
        case 'machine':
          drawSMTMachine(ctx, ex, ey, eq.label);
          break;
        case 'xray':
          drawXrayMachine(ctx, ex, ey, TILE_SIZE, eq.label);
          break;
        case 'station':
          drawTestStation(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'shelf':
          drawShelfRack(ctx, ex, ey, ew, eh, eq.label);
          break;
        case 'table':
          drawConfTable(ctx, ex, ey, ew, eh, eq.label);
          break;
      }
    }
  }
}

/* ── 服務台 ── */
function drawReception(ctx, x, y, w, h, label) {
  ctx.save();
  // 桌面
  const grad = ctx.createLinearGradient(x, y, x, y + TILE_SIZE);
  grad.addColorStop(0, '#A1887F');
  grad.addColorStop(1, '#795548');
  ctx.fillStyle = grad;
  roundRect(ctx, x + 4, y + 10, w - 8, TILE_SIZE - 16, 3);
  ctx.fill();
  // 桌面反光
  ctx.fillStyle = 'rgba(255,255,255,0.15)';
  ctx.fillRect(x + 8, y + 12, w - 16, 4);
  // 標籤
  drawLabel(ctx, x + w / 2, y + TILE_SIZE / 2 + 2, label);
  ctx.restore();
}

/* ── 展示牆 ── */
function drawDisplayWall(ctx, x, y, w, h, label) {
  ctx.save();
  // 螢幕
  ctx.fillStyle = '#1a237e';
  roundRect(ctx, x + 6, y + 6, w - 12, h - 12, 3);
  ctx.fill();
  // 螢幕內容（閃爍效果）
  const brightness = 0.6 + Math.sin(animTime * 2) * 0.15;
  ctx.fillStyle = `rgba(33,150,243,${brightness})`;
  ctx.fillRect(x + 10, y + 10, w - 20, h - 20);
  // 文字
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('📊', x + w / 2, y + h / 2 + 3);
  drawLabel(ctx, x + w / 2, y + h + 10, label);
  ctx.restore();
}

/* ── 盆栽 ── */
function drawPlant(ctx, x, y) {
  ctx.save();
  // 花盆
  ctx.fillStyle = '#8D6E63';
  ctx.fillRect(x + 12, y + 22, 16, 14);
  // 植物
  ctx.fillStyle = '#4CAF50';
  ctx.beginPath();
  ctx.arc(x + 20, y + 16, 10, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#388E3C';
  ctx.beginPath();
  ctx.arc(x + 17, y + 14, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/* ── 等候長椅 ── */
function drawBench(ctx, x, y, w, label) {
  ctx.save();
  ctx.fillStyle = '#78909C';
  roundRect(ctx, x + 4, y + 14, w - 8, 16, 4);
  ctx.fill();
  ctx.fillStyle = '#546E7A';
  ctx.fillRect(x + 8, y + 10, 6, 6);
  ctx.fillRect(x + w - 14, y + 10, 6, 6);
  drawLabel(ctx, x + w / 2, y + 8, label);
  ctx.restore();
}

/* ── SMT 機台 ── */
function drawSMTMachine(ctx, x, y, label) {
  ctx.save();
  const T = TILE_SIZE;
  // 機身
  const grad = ctx.createLinearGradient(x, y, x, y + T);
  grad.addColorStop(0, '#78909C');
  grad.addColorStop(0.5, '#607D8B');
  grad.addColorStop(1, '#546E7A');
  ctx.fillStyle = grad;
  roundRect(ctx, x + 3, y + 3, T - 6, T - 6, 5);
  ctx.fill();
  // 邊框
  ctx.strokeStyle = '#455A64';
  ctx.lineWidth = 1.5;
  roundRect(ctx, x + 3, y + 3, T - 6, T - 6, 5);
  ctx.stroke();
  // 指示燈（呼吸效果）
  const ledBrightness = 0.5 + Math.sin(animTime * 3 + x) * 0.5;
  ctx.fillStyle = `rgba(76,175,80,${ledBrightness})`;
  ctx.beginPath();
  ctx.arc(x + T - 10, y + 10, 3, 0, Math.PI * 2);
  ctx.fill();
  // 螢幕
  ctx.fillStyle = '#263238';
  ctx.fillRect(x + 8, y + 8, T - 16, 12);
  ctx.fillStyle = '#4FC3F7';
  ctx.font = 'bold 8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(label, x + T / 2, y + 17);
  ctx.restore();
}

/* ── 輸送帶（動態）── */
function drawConveyor(ctx, x, y, w, label) {
  ctx.save();
  const beltY = y + 14;
  const beltH = 14;

  // 帶體
  ctx.fillStyle = '#455A64';
  roundRect(ctx, x + 2, beltY - 2, w - 4, beltH + 4, 3);
  ctx.fill();
  ctx.fillStyle = '#37474F';
  ctx.fillRect(x + 4, beltY, w - 8, beltH);

  // 動態滾輪線
  const offset = (animTime * 40) % 20;
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  for (let dx = -20 + offset; dx < w; dx += 20) {
    ctx.beginPath();
    ctx.moveTo(x + dx, beltY);
    ctx.lineTo(x + dx + 10, beltY + beltH);
    ctx.stroke();
  }

  // 流動箭頭
  const arrowOffset = (animTime * 60) % 40;
  ctx.fillStyle = '#FFE082';
  for (let i = -40 + arrowOffset; i < w; i += 40) {
    const ax = x + i;
    if (ax > x + 4 && ax < x + w - 14) {
      ctx.beginPath();
      ctx.moveTo(ax, beltY + 2);
      ctx.lineTo(ax + 10, beltY + beltH / 2);
      ctx.lineTo(ax, beltY + beltH - 2);
      ctx.closePath();
      ctx.fill();
    }
  }

  // 標籤
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  roundRect(ctx, x + w / 2 - 36, y + 1, 72, 13, 6);
  ctx.fill();
  ctx.fillStyle = '#FFE082';
  ctx.font = 'bold 9px "Microsoft JhengHei", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('🔥 ' + label, x + w / 2, y + 11);
  ctx.restore();
}

/* ── 狀態指示燈 ── */
function drawStatusLight(ctx, x, y, status) {
  const colors = { green: '#4CAF50', yellow: '#FFC107', red: '#f44336' };
  const color = colors[status] || colors.green;
  const pulse = 0.5 + Math.sin(animTime * 2) * 0.5;

  ctx.save();
  // 光暈
  ctx.fillStyle = color.replace(')', `,${pulse * 0.3})`).replace('rgb', 'rgba');
  ctx.beginPath();
  ctx.arc(x + 20, y + 20, 12, 0, Math.PI * 2);
  ctx.fill();
  // 燈
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x + 20, y + 20, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/* ── AOI 光學檢測機 ── */
function drawAOIMachine(ctx, x, y, w, label) {
  ctx.save();
  // 機身
  ctx.fillStyle = '#0277BD';
  roundRect(ctx, x + 4, y + 4, w - 8, TILE_SIZE - 8, 4);
  ctx.fill();
  // 鏡頭（閃爍）
  const lens = 0.4 + Math.sin(animTime * 4) * 0.3;
  ctx.fillStyle = `rgba(0,230,255,${lens})`;
  ctx.beginPath();
  ctx.arc(x + w / 2, y + TILE_SIZE / 2, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#01579B';
  ctx.lineWidth = 2;
  ctx.stroke();
  drawLabel(ctx, x + w / 2, y - 2, label);
  ctx.restore();
}

/* ── X-ray 檢測機 ── */
function drawXrayMachine(ctx, x, y, w, label) {
  ctx.save();
  // 機身
  ctx.fillStyle = '#37474F';
  roundRect(ctx, x + 3, y + 3, w - 6, TILE_SIZE - 6, 4);
  ctx.fill();
  // 放射符號
  const rot = animTime * 0.5;
  ctx.translate(x + w / 2, y + TILE_SIZE / 2);
  ctx.rotate(rot);
  ctx.fillStyle = '#FFC107';
  for (let i = 0; i < 3; i++) {
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, 12, i * Math.PI * 2 / 3, i * Math.PI * 2 / 3 + 0.8);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = '#37474F';
  ctx.beginPath();
  ctx.arc(0, 0, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  drawLabel(ctx, x + w / 2, y - 2, label);
  ctx.restore();
}

/* ── 功能測試站 ── */
function drawTestStation(ctx, x, y, w, h, label) {
  ctx.save();
  ctx.fillStyle = '#2E7D32';
  roundRect(ctx, x + 4, y + 4, w - 8, h - 8, 4);
  ctx.fill();
  // 螢幕
  ctx.fillStyle = '#1B5E20';
  ctx.fillRect(x + 8, y + 8, w - 16, h / 2 - 8);
  // 指示文字
  ctx.fillStyle = '#A5D6A7';
  ctx.font = 'bold 8px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('PASS ✓', x + w / 2, y + h / 4 + 4);
  drawLabel(ctx, x + w / 2, y - 2, label);
  ctx.restore();
}

/* ── 數據監控牆 ── */
function drawMonitorRack(ctx, x, y, w, label) {
  ctx.save();
  const screenW = (w - 12) / 3;
  for (let i = 0; i < 3; i++) {
    const sx = x + 4 + i * (screenW + 2);
    // 螢幕外框
    ctx.fillStyle = '#212121';
    ctx.fillRect(sx, y + 4, screenW, TILE_SIZE - 12);
    // 螢幕
    const hue = 180 + i * 40;
    ctx.fillStyle = `hsla(${hue},70%,30%,0.8)`;
    ctx.fillRect(sx + 2, y + 6, screenW - 4, TILE_SIZE - 16);
    // 模擬數據線
    ctx.strokeStyle = `hsla(${hue},80%,60%,0.7)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let px = 0; px < screenW - 6; px += 3) {
      const py = Math.sin(animTime * 2 + px * 0.3 + i) * 5 + (TILE_SIZE - 16) / 2;
      if (px === 0) ctx.moveTo(sx + 3 + px, y + 6 + py);
      else ctx.lineTo(sx + 3 + px, y + 6 + py);
    }
    ctx.stroke();
  }
  drawLabel(ctx, x + w / 2, y + TILE_SIZE + 2, label);
  ctx.restore();
}

/* ── 貨架 ── */
function drawShelfRack(ctx, x, y, w, h, label) {
  ctx.save();
  const cols = Math.max(1, Math.floor(w / TILE_SIZE));
  const rows = Math.max(1, Math.floor(h / TILE_SIZE));

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const sx = x + col * TILE_SIZE + 4;
      const sy = y + row * TILE_SIZE + 4;
      const sw = TILE_SIZE - 8;
      const sh = TILE_SIZE - 8;
      // 貨架本體
      ctx.fillStyle = '#6D4C41';
      ctx.fillRect(sx, sy, sw, sh);
      // 層板
      ctx.fillStyle = '#5D4037';
      for (let i = 1; i <= 2; i++) {
        ctx.fillRect(sx, sy + i * sh / 3, sw, 2);
      }
      // 箱子
      const boxColors = ['#FF8A65', '#FFB74D', '#A1887F'];
      for (let i = 0; i < 3; i++) {
        ctx.fillStyle = boxColors[i];
        const bx = sx + 2 + i * (sw / 3);
        const by = sy + 2 + i * sh / 3;
        ctx.fillRect(bx, by, sw / 3 - 3, sh / 3 - 4);
      }
    }
  }
  drawLabel(ctx, x + w / 2, y - 2, label);
  ctx.restore();
}

/* ── 堆高機通道 ── */
function drawForkliftLane(ctx, x, y, h) {
  ctx.save();
  const offset = (animTime * 30) % 20;
  ctx.strokeStyle = 'rgba(255,193,7,0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 10]);
  ctx.lineDashOffset = -offset;

  // 雙線
  ctx.beginPath();
  ctx.moveTo(x + 10, y);
  ctx.lineTo(x + 10, y + h);
  ctx.moveTo(x + TILE_SIZE - 10, y);
  ctx.lineTo(x + TILE_SIZE - 10, y + h);
  ctx.stroke();

  ctx.setLineDash([]);
  // 堆高機圖示
  ctx.font = '16px sans-serif';
  ctx.textAlign = 'center';
  const fy = y + h / 2 + Math.sin(animTime) * 10;
  ctx.fillText('🚜', x + TILE_SIZE / 2, fy);
  ctx.restore();
}

/* ── 出貨口 ── */
function drawLoadingDock(ctx, x, y, h, label) {
  ctx.save();
  // 大門
  ctx.fillStyle = '#455A64';
  ctx.fillRect(x + 4, y + 4, TILE_SIZE - 8, h - 8);
  // 捲門紋路
  ctx.strokeStyle = '#37474F';
  ctx.lineWidth = 1;
  for (let dy = 8; dy < h - 8; dy += 6) {
    ctx.beginPath();
    ctx.moveTo(x + 6, y + dy);
    ctx.lineTo(x + TILE_SIZE - 6, y + dy);
    ctx.stroke();
  }
  // 箭頭
  ctx.fillStyle = '#FFC107';
  ctx.font = 'bold 16px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('→', x + TILE_SIZE / 2, y + h / 2 + 5);
  drawLabel(ctx, x + TILE_SIZE / 2, y - 2, label);
  ctx.restore();
}

/* ── 會議桌 ── */
function drawConfTable(ctx, x, y, w, h, label) {
  ctx.save();
  const cx = x + w / 2;
  const cy = y + h / 2;

  // 桌面
  const tableGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, w / 2);
  tableGrad.addColorStop(0, '#6D4C41');
  tableGrad.addColorStop(1, '#4E342E');
  ctx.fillStyle = tableGrad;
  ctx.beginPath();
  ctx.ellipse(cx, cy, w / 2 - 10, h / 2 - 10, 0, 0, Math.PI * 2);
  ctx.fill();
  // 桌面反光
  ctx.fillStyle = 'rgba(255,255,255,0.1)';
  ctx.beginPath();
  ctx.ellipse(cx - 5, cy - 5, w / 4, h / 4, -0.3, 0, Math.PI * 2);
  ctx.fill();

  // 椅子
  const chairCount = 8;
  for (let i = 0; i < chairCount; i++) {
    const angle = (i / chairCount) * Math.PI * 2;
    const chairX = cx + Math.cos(angle) * (w / 2 + 2);
    const chairY = cy + Math.sin(angle) * (h / 2 + 2);
    // 椅面
    ctx.fillStyle = '#616161';
    ctx.beginPath();
    ctx.arc(chairX, chairY, 6, 0, Math.PI * 2);
    ctx.fill();
    // 椅背
    ctx.fillStyle = '#424242';
    const backX = chairX + Math.cos(angle) * 5;
    const backY = chairY + Math.sin(angle) * 5;
    ctx.beginPath();
    ctx.arc(backX, backY, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/* ── 投影機 ── */
function drawProjector(ctx, x, y, label) {
  ctx.save();
  ctx.fillStyle = '#37474F';
  roundRect(ctx, x + 8, y + 12, 24, 16, 3);
  ctx.fill();
  // 鏡頭
  const glow = 0.3 + Math.sin(animTime * 2) * 0.2;
  ctx.fillStyle = `rgba(255,255,255,${glow})`;
  ctx.beginPath();
  ctx.arc(x + 20, y + 20, 5, 0, Math.PI * 2);
  ctx.fill();
  // 光束
  ctx.fillStyle = `rgba(255,255,200,${glow * 0.3})`;
  ctx.beginPath();
  ctx.moveTo(x + 15, y + 20);
  ctx.lineTo(x - 30, y - 20);
  ctx.lineTo(x + 65, y - 20);
  ctx.lineTo(x + 25, y + 20);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

/* ── 白板 ── */
function drawWhiteboard(ctx, x, y, h, label) {
  ctx.save();
  // 板面
  ctx.fillStyle = '#FAFAFA';
  ctx.fillRect(x + 6, y + 4, TILE_SIZE - 12, h - 8);
  ctx.strokeStyle = '#BDBDBD';
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 6, y + 4, TILE_SIZE - 12, h - 8);
  // 模擬文字
  ctx.fillStyle = '#1565C0';
  ctx.font = '7px sans-serif';
  ctx.fillText('≡≡≡', x + TILE_SIZE / 2, y + h / 3);
  ctx.fillStyle = '#c62828';
  ctx.fillText('📈', x + TILE_SIZE / 2, y + h / 2);
  ctx.restore();
}

/* ══════════════════════════════════════════
   迷你地圖 — 升級版
   ══════════════════════════════════════════ */
export function drawMinimap(ctx, canvasW, canvasH, playerX, playerY, visitedAreas) {
  const scale = 0.12;
  const mw = MAP_COLS * TILE_SIZE * scale;
  const mh = MAP_ROWS * TILE_SIZE * scale;
  const mx = canvasW - mw - 12;
  const my = canvasH - mh - 12;

  ctx.save();
  // 背景 + 邊框
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  roundRect(ctx, mx - 4, my - 18, mw + 8, mh + 22, 6);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255,255,255,0.2)';
  ctx.lineWidth = 1;
  roundRect(ctx, mx - 4, my - 18, mw + 8, mh + 22, 6);
  ctx.stroke();

  // 標題
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.font = 'bold 8px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('MAP', mx + mw / 2, my - 8);

  // 牆壁底色
  ctx.fillStyle = '#333';
  ctx.fillRect(mx, my, mw, mh);

  // 房間
  for (const room of ROOMS) {
    const visited = visitedAreas.has(room.id);
    ctx.fillStyle = visited ? room.color : 'rgba(100,100,100,0.4)';
    ctx.fillRect(
      mx + room.x * TILE_SIZE * scale,
      my + room.y * TILE_SIZE * scale,
      room.w * TILE_SIZE * scale,
      room.h * TILE_SIZE * scale
    );
  }

  // 走廊
  for (const cor of CORRIDORS) {
    ctx.fillStyle = 'rgba(180,180,180,0.3)';
    ctx.fillRect(
      mx + cor.x * TILE_SIZE * scale,
      my + cor.y * TILE_SIZE * scale,
      cor.w * TILE_SIZE * scale,
      cor.h * TILE_SIZE * scale
    );
  }

  // 玩家點（脈衝效果）
  const pulseR = 3 + Math.sin(animTime * 4) * 1;
  ctx.fillStyle = '#ff4444';
  ctx.beginPath();
  ctx.arc(mx + playerX * scale, my + playerY * scale, pulseR, 0, Math.PI * 2);
  ctx.fill();
  // 外圈
  ctx.strokeStyle = 'rgba(255,68,68,0.4)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(mx + playerX * scale, my + playerY * scale, pulseR + 3, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
}

/* ══════════════════════════════════════════
   工具函式
   ══════════════════════════════════════════ */

function drawLabel(ctx, x, y, text) {
  if (!text) return;
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  const textW = Math.max(text.length * 7, 40);
  roundRect(ctx, x - textW / 2, y - 8, textW, 14, 7);
  ctx.fill();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 9px "Microsoft JhengHei", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(text, x, y + 2);
  ctx.restore();
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
