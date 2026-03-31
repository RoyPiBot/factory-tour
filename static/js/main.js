/**
 * main.js - 全螢幕 RPG 工廠導覽 — 遊戲主迴圈
 */
import { SPAWN } from './config.js';
import { buildMap, drawMap, drawMinimap } from './map.js';
import { Player } from './player.js';
import { Camera } from './camera.js';
import { NPCSystem } from './npc.js';
import { DialogSystem } from './dialog.js';
import { TriggerSystem } from './triggers.js';
import { HUD } from './hud.js';

// ── 全域狀態 ──
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const keys = {};
let player, camera, npcSystem, dialog, triggers, hud;
let lastTime = 0;
let gameStarted = false;
let chatInputOpen = false;

// ── 初始化 ──
function init() {
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  buildMap();

  player = new Player(SPAWN.x, SPAWN.y);
  camera = new Camera(canvas.width, canvas.height);
  npcSystem = new NPCSystem();
  dialog = new DialogSystem();
  triggers = new TriggerSystem(dialog, npcSystem);
  hud = new HUD();

  // 鍵盤事件
  window.addEventListener('keydown', (e) => {
    // 聊天輸入框開啟時，不攔截鍵盤
    if (chatInputOpen) {
      if (e.key === 'Escape') {
        closeChatInput();
        e.preventDefault();
      }
      return;
    }

    keys[e.code] = true;

    // 防止方向鍵/空白鍵捲動頁面
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
      e.preventDefault();
    }

    // E / 空白鍵 = 互動
    if (e.code === 'KeyE' || e.code === 'Space') {
      handleInteract();
      e.preventDefault();
    }

    // T = 開啟聊天輸入
    if (e.code === 'KeyT' && !dialog.isWaitingAPI) {
      openChatInput();
      e.preventDefault();
    }

    // Escape = 關閉對話
    if (e.code === 'Escape') {
      if (dialog.getIsOpen()) {
        dialog.close();
      }
    }
  });

  window.addEventListener('keyup', (e) => {
    if (chatInputOpen) return;
    keys[e.code] = false;
  });

  // 聊天輸入送出
  document.getElementById('chat-send').addEventListener('click', submitChatInput);
  document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitChatInput();
    }
    e.stopPropagation(); // 防止觸發遊戲鍵盤事件
  });

  // 開場畫面
  const startBtn = document.getElementById('start-btn');
  if (startBtn) {
    startBtn.addEventListener('click', startGame);
    // 也允許按任意鍵開始
    window.addEventListener('keydown', function onceStart(e) {
      if (!gameStarted && (e.code === 'Enter' || e.code === 'Space')) {
        startGame();
        window.removeEventListener('keydown', onceStart);
      }
    });
  } else {
    startGame();
  }
}

function startGame() {
  gameStarted = true;
  const titleScreen = document.getElementById('title-screen');
  if (titleScreen) titleScreen.classList.add('hidden');
  canvas.focus();
  requestAnimationFrame(gameLoop);
}

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  if (camera) camera.resize(canvas.width, canvas.height);
}

// ── 互動處理 ──
function handleInteract() {
  const npc = npcSystem.getInteractableNpc();
  dialog.handleInteractKey(npc);
}

// ── 聊天輸入 ──
function openChatInput() {
  chatInputOpen = true;
  // 清除所有按鍵狀態
  Object.keys(keys).forEach(k => keys[k] = false);
  const overlay = document.getElementById('chat-input-overlay');
  const input = document.getElementById('chat-input');
  overlay.classList.remove('hidden');
  input.value = '';
  input.focus();
}

function closeChatInput() {
  chatInputOpen = false;
  document.getElementById('chat-input-overlay').classList.add('hidden');
  canvas.focus();
}

async function submitChatInput() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  closeChatInput();
  await dialog.askFreeQuestion(msg);
}

// ── 遊戲迴圈 ──
function gameLoop(timestamp) {
  const dt = Math.min((timestamp - lastTime) / 1000, 0.05);
  lastTime = timestamp;

  // 更新（對話框開啟時停止移動）
  if (!dialog.getIsOpen() && !chatInputOpen) {
    player.update(keys, dt);
  }
  camera.update(player.x, player.y);
  npcSystem.update(player.x, player.y, dt);
  triggers.update(player.x, player.y);
  hud.update(dt);

  // 繪製
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 背景
  ctx.fillStyle = '#0a0a12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 地圖
  drawMap(ctx, camera, dt);

  // NPC（在玩家下方繪製，除非 NPC 在玩家後面）
  npcSystem.draw(ctx, camera);

  // 玩家
  player.draw(ctx, camera);

  // 小地圖
  drawMinimap(ctx, canvas.width, canvas.height, player.x, player.y, triggers.getVisitedAreas());

  // HUD
  hud.draw(ctx, canvas.width, canvas.height, triggers);

  requestAnimationFrame(gameLoop);
}

// ── 啟動 ──
document.addEventListener('DOMContentLoaded', init);
