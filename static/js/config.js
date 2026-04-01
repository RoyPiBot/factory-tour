/**
 * config.js - 遊戲常數與地圖資料（v3 — 大地圖版）
 */

export const TILE_SIZE = 40;
export const MAP_COLS = 40;
export const MAP_ROWS = 32;
export const PLAYER_SPEED = 180; // pixels per second（地圖變大，稍微加速）

// 地圖 tile 類型
export const TILE = {
  WALL: 0,
  FLOOR: 1,
  DOOR: 2,
  OUTSIDE: 3,
};

// 房間定義 (tile 座標) — v3 大地圖佈局
export const ROOMS = [
  {
    id: "lobby",
    x: 1, y: 1, w: 11, h: 9,
    name: "大廳 Lobby",
    emoji: "🏢",
    color: "#66bb6a",
    floorColor: "#e8f5e9",
    floorPattern: "marble",
    ambientColor: "rgba(102,187,106,0.06)",
    equipment: [
      { type: "reception", x: 4, y: 3, w: 4, label: "服務台" },
      { type: "display_wall", x: 10, y: 2, h: 5, label: "展示牆" },
      { type: "plant", x: 1, y: 1 },
      { type: "plant", x: 11, y: 1 },
      { type: "plant", x: 1, y: 9 },
      { type: "bench", x: 3, y: 7, w: 3, label: "等候區" },
      { type: "bench", x: 7, y: 7, w: 3, label: "等候區 B" },
    ],
  },
  {
    id: "assembly_a",
    x: 16, y: 1, w: 14, h: 9,
    name: "組裝線 A — SMT 產線",
    emoji: "⚙️",
    color: "#ffa726",
    floorColor: "#fff8e1",
    floorPattern: "industrial",
    ambientColor: "rgba(255,167,38,0.06)",
    hazard: true,
    equipment: [
      { type: "smt_machine", x: 17, y: 3, label: "印刷機" },
      { type: "smt_machine", x: 20, y: 3, label: "SMT-1" },
      { type: "smt_machine", x: 23, y: 3, label: "SMT-2" },
      { type: "smt_machine", x: 26, y: 3, label: "SMT-3" },
      { type: "conveyor", x: 17, y: 6, w: 12, label: "回焊爐 Reflow" },
      { type: "status_light", x: 16, y: 1, status: "green" },
      { type: "status_light", x: 29, y: 1, status: "green" },
    ],
  },
  {
    id: "qc_room",
    x: 16, y: 14, w: 14, h: 7,
    name: "品管室 QC Lab",
    emoji: "🔍",
    color: "#29b6f6",
    floorColor: "#e1f5fe",
    floorPattern: "cleanroom",
    ambientColor: "rgba(41,182,246,0.06)",
    hazard: true,
    equipment: [
      { type: "aoi_machine", x: 17, y: 15, w: 3, label: "AOI 光學檢測" },
      { type: "xray_machine", x: 21, y: 15, w: 3, label: "X-ray 檢測" },
      { type: "test_station", x: 25, y: 15, w: 3, h: 3, label: "功能測試站" },
      { type: "monitor_rack", x: 17, y: 18, w: 4, label: "數據監控牆" },
    ],
  },
  {
    id: "warehouse",
    x: 16, y: 24, w: 14, h: 7,
    name: "倉儲區 Warehouse",
    emoji: "📦",
    color: "#ef5350",
    floorColor: "#ffebee",
    floorPattern: "concrete",
    ambientColor: "rgba(239,83,80,0.06)",
    hazard: true,
    equipment: [
      { type: "shelf_rack", x: 17, y: 25, w: 4, h: 3, label: "貨架 A" },
      { type: "shelf_rack", x: 22, y: 25, w: 4, h: 3, label: "貨架 B" },
      { type: "forklift_lane", x: 21, y: 24, h: 7 },
      { type: "loading_dock", x: 28, y: 25, h: 4, label: "出貨口" },
    ],
  },
  {
    id: "conference",
    x: 1, y: 24, w: 11, h: 7,
    name: "會議室 Conference",
    emoji: "🪑",
    color: "#ab47bc",
    floorColor: "#f3e5f5",
    floorPattern: "carpet",
    ambientColor: "rgba(171,71,188,0.06)",
    equipment: [
      { type: "conf_table", x: 4, y: 25, w: 5, h: 4, label: "會議桌" },
      { type: "projector", x: 3, y: 24, label: "投影機" },
      { type: "whiteboard", x: 11, y: 25, h: 4, label: "白板" },
    ],
  },
];

// 走廊連接 (tile 座標的矩形) — v3 更寬更長的走廊
export const CORRIDORS = [
  // 大廳 → 組裝線A (水平走廊)
  { x: 12, y: 4, w: 4, h: 3 },
  // 組裝線A → 品管室 (垂直走廊)
  { x: 21, y: 10, w: 4, h: 4 },
  // 品管室 → 倉儲區 (垂直走廊)
  { x: 21, y: 21, w: 4, h: 3 },
  // 大廳 → 會議室 (垂直走廊)
  { x: 4, y: 10, w: 4, h: 14 },
  // 會議室 → 倉儲區 (水平走廊)
  { x: 12, y: 26, w: 4, h: 3 },
];

// 玩家出生點 (大廳中央)
export const SPAWN = { x: 6.5 * TILE_SIZE, y: 5.5 * TILE_SIZE };

// 門的位置（在走廊和房間交界處）
export const DOORS = [
  // 大廳 → 走廊1（水平）
  { x: 12, y: 4 }, { x: 12, y: 5 }, { x: 12, y: 6 },
  // 走廊1 → 組裝線A
  { x: 15, y: 4 }, { x: 15, y: 5 }, { x: 15, y: 6 },
  // 組裝線A → 走廊2（垂直）
  { x: 21, y: 10 }, { x: 22, y: 10 }, { x: 23, y: 10 }, { x: 24, y: 10 },
  // 走廊2 → 品管室
  { x: 21, y: 13 }, { x: 22, y: 13 }, { x: 23, y: 13 }, { x: 24, y: 13 },
  // 品管室 → 走廊3（垂直）
  { x: 21, y: 21 }, { x: 22, y: 21 }, { x: 23, y: 21 }, { x: 24, y: 21 },
  // 走廊3 → 倉儲區
  { x: 21, y: 23 }, { x: 22, y: 23 }, { x: 23, y: 23 }, { x: 24, y: 23 },
  // 大廳 → 走廊4（垂直）
  { x: 4, y: 10 }, { x: 5, y: 10 }, { x: 6, y: 10 }, { x: 7, y: 10 },
  // 走廊4 → 會議室
  { x: 4, y: 23 }, { x: 5, y: 23 }, { x: 6, y: 23 }, { x: 7, y: 23 },
  // 會議室 → 走廊5（水平）
  { x: 12, y: 26 }, { x: 12, y: 27 }, { x: 12, y: 28 },
  // 走廊5 → 倉儲區
  { x: 15, y: 26 }, { x: 15, y: 27 }, { x: 15, y: 28 },
];
