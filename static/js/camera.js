/**
 * camera.js - 視角控制
 */
import { TILE_SIZE, MAP_COLS, MAP_ROWS } from './config.js';

export class Camera {
  constructor(canvasW, canvasH) {
    this.x = 0;
    this.y = 0;
    this.w = canvasW;
    this.h = canvasH;
    this.lerp = 0.08; // 平滑跟隨速度
  }

  update(playerX, playerY) {
    const targetX = playerX - this.w / 2;
    const targetY = playerY - this.h / 2;

    // 平滑插值
    this.x += (targetX - this.x) * this.lerp;
    this.y += (targetY - this.y) * this.lerp;

    // 限制在地圖範圍內
    const mapW = MAP_COLS * TILE_SIZE;
    const mapH = MAP_ROWS * TILE_SIZE;
    this.x = Math.max(0, Math.min(this.x, mapW - this.w));
    this.y = Math.max(0, Math.min(this.y, mapH - this.h));
  }

  resize(canvasW, canvasH) {
    this.w = canvasW;
    this.h = canvasH;
  }
}
