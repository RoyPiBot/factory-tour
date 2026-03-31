/**
 * editor.js — 知識庫管理前端邏輯
 * 提供文件 CRUD、上傳、Markdown 預覽、鍵盤快捷鍵等功能
 */

(function () {
  "use strict";

  // ── DOM 元素 ──
  const fileList = document.getElementById("file-list");
  const btnNew = document.getElementById("btn-new");
  const btnUpload = document.getElementById("btn-upload");
  const fileUpload = document.getElementById("file-upload");
  const editorEmpty = document.getElementById("editor-empty");
  const editorPanel = document.getElementById("editor-panel");
  const docName = document.getElementById("doc-name");
  const docContent = document.getElementById("doc-content");
  const previewArea = document.getElementById("preview-area");
  const btnSave = document.getElementById("btn-save");
  const btnPreview = document.getElementById("btn-preview");
  const btnDelete = document.getElementById("btn-delete");
  const saveIndicator = document.getElementById("save-indicator");
  const toastContainer = document.getElementById("toast-container");

  // ── 狀態 ──
  let currentDoc = null;       // 目前開啟的文件名稱（null = 尚未選擇）
  let isNewDoc = false;        // 是否為新建文件模式
  let isPreviewMode = false;   // 是否正在預覽
  let dirty = false;           // 是否有未儲存的修改
  let saveIndicatorTimer = null;

  // ══════════════════════════════════════
  //  初始化
  // ══════════════════════════════════════

  init();

  function init() {
    loadDocumentList();
    bindEvents();
  }

  function bindEvents() {
    btnNew.addEventListener("click", handleNew);
    btnUpload.addEventListener("click", () => fileUpload.click());
    fileUpload.addEventListener("change", handleUpload);
    btnSave.addEventListener("click", handleSave);
    btnPreview.addEventListener("click", togglePreview);
    btnDelete.addEventListener("click", handleDelete);

    // 追蹤修改
    docContent.addEventListener("input", () => {
      dirty = true;
    });
    docName.addEventListener("input", () => {
      dirty = true;
    });

    // 鍵盤快捷鍵
    document.addEventListener("keydown", (e) => {
      // Ctrl+S 儲存
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        handleSave();
      }
      // Ctrl+P 預覽切換
      if ((e.ctrlKey || e.metaKey) && e.key === "p") {
        e.preventDefault();
        togglePreview();
      }
    });

    // Tab 鍵支援（在 textarea 中插入 tab 而非跳離）
    docContent.addEventListener("keydown", (e) => {
      if (e.key === "Tab") {
        e.preventDefault();
        const start = docContent.selectionStart;
        const end = docContent.selectionEnd;
        docContent.value =
          docContent.value.substring(0, start) +
          "  " +
          docContent.value.substring(end);
        docContent.selectionStart = docContent.selectionEnd = start + 2;
        dirty = true;
      }
    });
  }

  // ══════════════════════════════════════
  //  文件列表
  // ══════════════════════════════════════

  async function loadDocumentList() {
    try {
      const res = await fetch("/documents");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const docs = await res.json();
      renderFileList(docs);
    } catch (err) {
      console.error("Failed to load document list:", err);
      fileList.innerHTML =
        '<li class="file-list-placeholder">無法載入文件列表</li>';
      showToast("無法載入文件列表", "error");
    }
  }

  function renderFileList(docs) {
    if (!docs || docs.length === 0) {
      fileList.innerHTML =
        '<li class="file-list-placeholder">尚無文件，點擊「新增文件」開始</li>';
      return;
    }

    fileList.innerHTML = "";
    // docs 可能是字串陣列或物件陣列（視後端而定）
    const names = docs.map((d) => (typeof d === "string" ? d : d.name || d.filename));

    names.sort((a, b) => a.localeCompare(b, "zh-TW"));

    for (const name of names) {
      const li = document.createElement("li");
      li.className = "file-item";
      if (name === currentDoc) li.classList.add("active");
      li.textContent = name;
      li.addEventListener("click", () => openDocument(name));
      fileList.appendChild(li);
    }
  }

  function setActiveFile(name) {
    const items = fileList.querySelectorAll(".file-item");
    items.forEach((item) => {
      item.classList.toggle("active", item.textContent === name);
    });
  }

  // ══════════════════════════════════════
  //  開啟文件
  // ══════════════════════════════════════

  async function openDocument(name) {
    if (dirty && currentDoc !== null) {
      const discard = confirm("目前的修改尚未儲存，確定要切換嗎？");
      if (!discard) return;
    }

    try {
      const res = await fetch(`/documents/${encodeURIComponent(name)}/content`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const content = typeof data === "string" ? data : data.content || "";

      currentDoc = name;
      isNewDoc = false;
      dirty = false;

      docName.value = name;
      docName.readOnly = true;
      docContent.value = content;

      showEditorPanel();
      setActiveFile(name);

      // 退出預覽模式
      if (isPreviewMode) {
        isPreviewMode = false;
        previewArea.classList.add("hidden");
        docContent.style.display = "";
        btnPreview.classList.remove("active");
      }
    } catch (err) {
      console.error("Failed to load document:", err);
      showToast(`無法載入文件：${name}`, "error");
    }
  }

  // ══════════════════════════════════════
  //  新增文件
  // ══════════════════════════════════════

  function handleNew() {
    if (dirty) {
      const discard = confirm("目前的修改尚未儲存，確定要新增嗎？");
      if (!discard) return;
    }

    currentDoc = null;
    isNewDoc = true;
    dirty = false;

    docName.value = "";
    docName.readOnly = false;
    docContent.value = "";

    showEditorPanel();
    setActiveFile(null);
    docName.focus();

    // 退出預覽模式
    if (isPreviewMode) {
      isPreviewMode = false;
      previewArea.classList.add("hidden");
      docContent.style.display = "";
      btnPreview.classList.remove("active");
    }
  }

  // ══════════════════════════════════════
  //  儲存
  // ══════════════════════════════════════

  async function handleSave() {
    const name = docName.value.trim();
    const content = docContent.value;

    if (!name) {
      showToast("請輸入文件名稱", "error");
      docName.focus();
      return;
    }

    try {
      let res;
      if (isNewDoc) {
        // 新建
        res = await fetch("/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, content }),
        });
      } else {
        // 更新
        res = await fetch(`/documents/${encodeURIComponent(name)}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ content }),
        });
      }

      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || `HTTP ${res.status}`);
      }

      currentDoc = name;
      isNewDoc = false;
      dirty = false;
      docName.readOnly = true;

      // 顯示儲存指示
      flashSaveIndicator();
      showToast("儲存成功", "success");

      // 重新載入列表
      await loadDocumentList();
      setActiveFile(name);
    } catch (err) {
      console.error("Failed to save:", err);
      showToast(`儲存失敗：${err.message}`, "error");
    }
  }

  function flashSaveIndicator() {
    saveIndicator.classList.remove("hidden");
    clearTimeout(saveIndicatorTimer);
    saveIndicatorTimer = setTimeout(() => {
      saveIndicator.classList.add("hidden");
    }, 2500);
  }

  // ══════════════════════════════════════
  //  刪除
  // ══════════════════════════════════════

  async function handleDelete() {
    if (isNewDoc || !currentDoc) {
      // 新建模式直接清除
      handleNew();
      return;
    }

    const confirmed = confirm(
      `確定要刪除「${currentDoc}」嗎？\n此操作無法復原。`
    );
    if (!confirmed) return;

    try {
      const res = await fetch(`/documents/${encodeURIComponent(currentDoc)}`, {
        method: "DELETE",
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      showToast(`已刪除「${currentDoc}」`, "info");
      currentDoc = null;
      isNewDoc = false;
      dirty = false;

      hideEditorPanel();
      await loadDocumentList();
    } catch (err) {
      console.error("Failed to delete:", err);
      showToast(`刪除失敗：${err.message}`, "error");
    }
  }

  // ══════════════════════════════════════
  //  上傳
  // ══════════════════════════════════════

  async function handleUpload() {
    const files = fileUpload.files;
    if (!files || files.length === 0) return;

    let successCount = 0;
    let failCount = 0;

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch("/documents/upload", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        successCount++;
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
        failCount++;
      }
    }

    // 重設 file input
    fileUpload.value = "";

    if (successCount > 0) {
      showToast(`已上傳 ${successCount} 個檔案`, "success");
      await loadDocumentList();
    }
    if (failCount > 0) {
      showToast(`${failCount} 個檔案上傳失敗`, "error");
    }
  }

  // ══════════════════════════════════════
  //  Markdown 預覽
  // ══════════════════════════════════════

  function togglePreview() {
    isPreviewMode = !isPreviewMode;

    if (isPreviewMode) {
      previewArea.innerHTML = renderMarkdown(docContent.value);
      previewArea.classList.remove("hidden");
      docContent.style.display = "none";
      btnPreview.classList.add("active");
    } else {
      previewArea.classList.add("hidden");
      docContent.style.display = "";
      btnPreview.classList.remove("active");
    }
  }

  /**
   * 簡易 Markdown → HTML 轉換
   * 支援：標題、粗體、斜體、行內程式碼、程式碼區塊、列表、引言、水平線、連結
   */
  function renderMarkdown(md) {
    if (!md) return '<p style="color:#555">（空白文件）</p>';

    let html = escapeHtml(md);

    // 程式碼區塊（需在行內程式碼之前處理）
    html = html.replace(
      /```(\w*)\n([\s\S]*?)```/g,
      (_, lang, code) => `<pre><code class="lang-${lang}">${code.trimEnd()}</code></pre>`
    );

    // 行內程式碼
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // 標題（h1–h6）
    html = html.replace(/^######\s+(.+)$/gm, "<h6>$1</h6>");
    html = html.replace(/^#####\s+(.+)$/gm, "<h5>$1</h5>");
    html = html.replace(/^####\s+(.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^###\s+(.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^##\s+(.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^#\s+(.+)$/gm, "<h1>$1</h1>");

    // 粗體 + 斜體
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // 水平線
    html = html.replace(/^---$/gm, "<hr>");
    html = html.replace(/^\*\*\*$/gm, "<hr>");

    // 引言
    html = html.replace(/^&gt;\s+(.+)$/gm, "<blockquote>$1</blockquote>");

    // 連結 [text](url)
    html = html.replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );

    // 無序列表
    html = html.replace(/^[\-\*]\s+(.+)$/gm, "<li>$1</li>");
    // 有序列表
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
    // 將連續 <li> 包裹為 <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // 段落：連續的非標籤行轉為 <p>
    html = html
      .split("\n\n")
      .map((block) => {
        block = block.trim();
        if (!block) return "";
        // 如果已經是 HTML 標籤開頭，不包裹
        if (/^<(h[1-6]|ul|ol|li|pre|blockquote|hr|p|div|table)/.test(block)) {
          return block;
        }
        return `<p>${block.replace(/\n/g, "<br>")}</p>`;
      })
      .join("\n");

    return html;
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // ══════════════════════════════════════
  //  UI 輔助
  // ══════════════════════════════════════

  function showEditorPanel() {
    editorEmpty.classList.add("hidden");
    editorPanel.classList.remove("hidden");
  }

  function hideEditorPanel() {
    editorPanel.classList.add("hidden");
    editorEmpty.classList.remove("hidden");
  }

  /**
   * 顯示 Toast 通知
   * @param {string} message
   * @param {"success"|"error"|"info"} type
   */
  function showToast(message, type = "info") {
    const el = document.createElement("div");
    el.className = `toast toast-${type}`;
    el.textContent = message;
    toastContainer.appendChild(el);

    // 自動消失
    setTimeout(() => {
      el.classList.add("toast-out");
      setTimeout(() => el.remove(), 300);
    }, 3000);
  }
})();
