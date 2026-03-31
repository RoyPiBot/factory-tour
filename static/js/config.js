/**
 * config.js - 遊戲常數與地圖資料（v2 — 升級版工廠地圖）
 */

export const TILE_SIZE = 40;
export const MAP_COLS = 28;
export const MAP_ROWS = 22;
export const PLAYER_SPEED = 160; // pixels per second

// 地圖 tile 類型
export const TILE = {
  WALL: 0,
  FLOOR: 1,
  DOOR: 2,
  OUTSIDE: 3,
};

// 房間定義 (tile 座標) — 更寬敞的佈局
export const ROOMS = [
  {
    id: "lobby",
    x: 1, y: 1, w: 7, h: 6,
    name: "大廳 Lobby",
    emoji: "🏢",
    color: "#66bb6a",
    floorColor: "#e8f5e9",
    floorPattern: "marble",  // 大理石地板
    ambientColor: "rgba(102,187,106,0.06)",
    equipment: [
      { type: "reception", x: 3, y: 2, w: 3, label: "服務台" },
      { type: "display_wall", x: 6, y: 2, h: 4, label: "展示牆" },
      { type: "plant", x: 1, y: 1 },
      { type: "plant", x: 7, y: 1 },
      { type: "bench", x: 2, y: 5, w: 2, label: "等候區" },
    ],
  },
  {
    id: "assembly_a",
    x: 11, y: 1, w: 10, h: 6,
    name: "組裝線 A — SMT 產線",
    emoji: "⚙️",
    color: "#ffa726",
    floorColor: "#fff8e1",
    floorPattern: "industrial",  // 工業地板
    ambientColor: "rgba(255,167,38,0.06)",
    hazard: true,
    equipment: [
      { type: "smt_machine", x: 12, y: 2, label: "印刷機" },
      { type: "smt_machine", x: 14, y: 2, label: "SMT-1" },
      { type: "smt_machine", x: 16, y: 2, label: "SMT-2" },
      { type: "smt_machine", x: 18, y: 2, label: "SMT-3" },
      { type: "conveyor", x: 12, y: 4, w: 8, label: "回焊爐 Reflow" },
      { type: "status_light", x: 11, y: 1, status: "green" },
      { type: "status_light", x: 20, y: 1, status: "green" },
    ],
  },
  {
    id: "qc_room",
    x: 11, y: 9, w: 10, h: 5,
    name: "品管室 QC Lab",
    emoji: "🔍",
    color: "#29b6f6",
    floorColor: "#e1f5fe",
    floorPattern: "cleanroom",  // 無塵室地板
    ambientColor: "rgba(41,182,246,0.06)",
    hazard: true,
    equipment: [
      { type: "aoi_machine", x: 12, y: 10, w: 2, label: "AOI 光學檢測" },
      { type: "xray_machine", x: 15, y: 10, w: 2, label: "X-ray 檢測" },
      { type: "test_station", x: 18, y: 10, w: 2, h: 2, label: "功能測試站" },
      { type: "monitor_rack", x: 12, y: 12, w: 3, label: "數據監控牆" },
    ],
  },
  {
    id: "warehouse",
    x: 11, y: 16, w: 10, h: 5,
    name: "倉儲區 Warehouse",
    emoji: "📦",
    color: "#ef5350",
    floorColor: "#ffebee",
    floorPattern: "concrete",  // 水泥地
    ambientColor: "rgba(239,83,80,0.06)",
    hazard: true,
    equipment: [
      { type: "shelf_rack", x: 12, y: 17, w: 3, h: 2, label: "貨架 A" },
      { type: "shelf_rack", x: 16, y: 17, w: 3, h: 2, label: "貨架 B" },
      { type: "forklift_lane", x: 15, y: 16, h: 5 },
      { type: "loading_dock", x: 19, y: 17, h: 3, label: "出貨口" },
    ],
  },
  {
    id: "conference",
    x: 1, y: 16, w: 7, h: 5,
    name: "會議室 Conference",
    emoji: "🪑",
    color: "#ab47bc",
    floorColor: "#f3e5f5",
    floorPattern: "carpet",  // 地毯
    ambientColor: "rgba(171,71,188,0.06)",
    equipment: [
      { type: "conf_table", x: 3, y: 17, w: 3, h: 3, label: "會議桌" },
      { type: "projector", x: 2, y: 16, label: "投影機" },
      { type: "whiteboard", x: 7, y: 17, h: 3, label: "白板" },
    ],
  },
];

// 走廊連接 (tile 座標的矩形) — 更寬的走廊
export const CORRIDORS = [
  // 大廳 → 組裝線A (水平走廊)
  { x: 8, y: 3, w: 3, h: 2 },
  // 組裝線A → 品管室 (垂直走廊)
  { x: 14, y: 7, w: 3, h: 2 },
  // 品管室 → 倉儲區 (垂直走廊)
  { x: 14, y: 14, w: 3, h: 2 },
  // 大廳 → 會議室 (垂直走廊)
  { x: 3, y: 7, w: 3, h: 9 },
  // 會議室 → 倉儲區 (水平走廊)
  { x: 8, y: 17, w: 3, h: 2 },
];

// 玩家出生點 (大廳中央)
export const SPAWN = { x: 4.5 * TILE_SIZE, y: 4 * TILE_SIZE };

// 門的位置（在走廊和房間交界處）
export const DOORS = [
  { x: 8, y: 3 }, { x: 8, y: 4 },     // 大廳出口
  { x: 10, y: 3 }, { x: 10, y: 4 },    // 組裝線A入口
  { x: 14, y: 7 }, { x: 15, y: 7 }, { x: 16, y: 7 },   // 組裝線A下出口
  { x: 14, y: 8 }, { x: 15, y: 8 }, { x: 16, y: 8 },   // 品管室上入口
  { x: 14, y: 14 }, { x: 15, y: 14 }, { x: 16, y: 14 }, // 品管室下出口
  { x: 14, y: 15 }, { x: 15, y: 15 }, { x: 16, y: 15 }, // 倉儲區上入口
  { x: 3, y: 7 }, { x: 4, y: 7 }, { x: 5, y: 7 },       // 大廳下出口
  { x: 3, y: 15 }, { x: 4, y: 15 }, { x: 5, y: 15 },    // 會議室上入口
  { x: 8, y: 17 }, { x: 8, y: 18 },    // 會議室右出口
  { x: 10, y: 17 }, { x: 10, y: 18 },  // 倉儲區左入口
];
