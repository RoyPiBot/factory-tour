/**
 * websocket.js - WebSocket 即時感測資料客戶端
 */

(function () {
  'use strict';

  class SensorWebSocket {
    constructor() {
      this.ws = null;
      this.areaData = {};
      this.callbacks = [];
      this.reconnectDelay = 1000;
      this.maxReconnectDelay = 30000;
      this.shouldReconnect = true;
      this.connected = false;
    }

    /**
     * 連接 WebSocket
     */
    connect() {
      this.shouldReconnect = true;

      try {
        const url = `ws://${location.host}/ws/sensors`;
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
          console.log('[SensorWS] 已連線');
          this.connected = true;
          this.reconnectDelay = 1000; // 重置延遲
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.area_id) {
              this.areaData[data.area_id] = data;
            }
            this.callbacks.forEach((cb) => {
              try {
                cb(data);
              } catch (e) {
                console.warn('[SensorWS] 回呼錯誤:', e);
              }
            });
          } catch (e) {
            console.warn('[SensorWS] 解析訊息失敗:', e);
          }
        };

        this.ws.onclose = (event) => {
          console.log('[SensorWS] 連線關閉:', event.code);
          this.connected = false;
          this._scheduleReconnect();
        };

        this.ws.onerror = (error) => {
          console.warn('[SensorWS] 連線錯誤');
          this.connected = false;
        };
      } catch (e) {
        console.warn('[SensorWS] 無法建立連線:', e.message);
        this._scheduleReconnect();
      }
    }

    /**
     * 斷開連線
     */
    disconnect() {
      this.shouldReconnect = false;
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      this.connected = false;
    }

    /**
     * 取得指定區域的最新感測資料
     */
    getAreaData(areaId) {
      return this.areaData[areaId] || null;
    }

    /**
     * 註冊感測資料更新回呼
     */
    onSensorUpdate(callback) {
      if (typeof callback === 'function') {
        this.callbacks.push(callback);
      }
    }

    /**
     * 排程自動重連（指數退避）
     */
    _scheduleReconnect() {
      if (!this.shouldReconnect) return;

      console.log(`[SensorWS] ${this.reconnectDelay / 1000}s 後重連...`);
      setTimeout(() => {
        if (this.shouldReconnect) {
          this.connect();
        }
      }, this.reconnectDelay);

      // 指數退避
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    }
  }

  window.SensorWebSocket = SensorWebSocket;
})();
