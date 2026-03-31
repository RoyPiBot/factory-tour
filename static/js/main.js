/**
 * main.js - 遊戲主迴圈
 */
import { SPAWN } from './config.js';
import { buildMap, drawMap, drawMinimap } from './map.js';
import { Player } from './player.js';
import { Camera } from './camera.js';
import { ChatPanel } from './chat.js';
import { TriggerSystem } from './triggers.js';
import { HUD } from './hud.js';

// ── 全域狀態 ──
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const keys = {};
let player, camera, chatPanel, triggers, hud;
let lastTime = 0;

// ── 初始化 ──
function init() {
  // 設定 canvas 大小
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // 建構地圖
  buildMap();

  // 建立玩家
  player = new Player(SPAWN.x, SPAWN.y);

  // 建立相機
  camera = new Camera(canvas.width, canvas.height);

  // 建立聊天面板
  chatPanel = new ChatPanel();

  // 建立觸發系統
  triggers = new TriggerSystem(chatPanel);

  // 建立 HUD
  hud = new HUD();

  // 鍵盤事件
  window.addEventListener('keydown', (e) => {
    keys[e.code] = true;
    // 防止方向鍵捲動頁面
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
      // 只在 canvas 焦點時阻止
      if (document.activeElement === document.body || document.activeElement === canvas) {
        e.preventDefault();
      }
    }
  });
  window.addEventListener('keyup', (e) => {
    keys[e.code] = false;
  });

  // 讓 canvas 可以取得焦點
  canvas.tabIndex = 1;
  canvas.focus();
  canvas.addEventListener('click', () => canvas.focus());

  // 開始遊戲迴圈
  requestAnimationFrame(gameLoop);
}

function resizeCanvas() {
  const container = document.getElementById('canvas-container');
  // 保持 4:3 比例，最大 800x600
  const maxW = Math.min(container.clientWidth, 900);
  const maxH = Math.min(Math.floor(maxW * 0.72), 650);
  canvas.width = maxW;
  canvas.height = maxH;
  if (camera) camera.resize(canvas.width, canvas.height);
}

// ── 遊戲迴圈 ──
function gameLoop(timestamp) {
  const dt = Math.min((timestamp - lastTime) / 1000, 0.05); // 限制 delta time
  lastTime = timestamp;

  // 更新
  player.update(keys, dt);
  camera.update(player.x, player.y);
  triggers.update(player.x, player.y);
  hud.update(dt);

  // 繪製
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 背景
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 地圖
  drawMap(ctx, camera, dt);

  // 玩家
  player.draw(ctx, camera);

  // 迷你地圖
  drawMinimap(ctx, canvas.width, canvas.height, player.x, player.y, triggers.getVisitedAreas());

  // HUD
  hud.draw(ctx, canvas.width, canvas.height, triggers);

  requestAnimationFrame(gameLoop);
}

// ── 啟動 ──
document.addEventListener('DOMContentLoaded', init);
