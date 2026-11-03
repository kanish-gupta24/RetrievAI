/* RAGBench Studio — vanilla JS client for the RAGBench FastAPI backend. */
(() => {
  "use strict";

  const THEME_KEY = "ragbench-theme";
  const MAX_BYTES = 50 * 1024 * 1024;
  const HISTORY_LIMIT = 6;
  const NET_ERROR = "Can't reach the server. Check that the backend is running.";

  const mqMobile = window.matchMedia("(max-width: 760px)");
  const mqDrawer = window.matchMedia("(max-width: 1100px)");
  const mqReduce = window.matchMedia("(prefers-reduced-motion: reduce)");

  const $ = (sel, root = document) => root.querySelector(sel);

  // ------------------------------------------------------------
  // Icons
  // ------------------------------------------------------------
  const svg = (d, size = 16) =>
    `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
  const ICON = {
    file: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>', 18),
    trash: svg('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>'),
    x: svg('<path d="M6 6l12 12M18 6 6 18"/>', 14),
    copy: svg('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>', 13),
    check: svg('<path d="m5 12 5 5 9-10"/>', 13),
    list: svg('<path d="M4 6h16M4 12h16M4 18h10"/>', 13),
    alert: svg('<circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.5"/>', 18),
    retry: svg('<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>', 13),
    spark: svg('<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 17l.7 1.8L21.5 19.5l-1.8.7L19 22l-.7-1.8-1.8-.7 1.8-.7z"/>', 16),
    doc: svg('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>', 24),
    question: svg('<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17v.5"/>', 18),
  };

  // ------------------------------------------------------------
  // Elements
  // ------------------------------------------------------------
  const el = {
    app: $("#app"),
    statusPill: $("#status-pill"),
    statusLabel: $("#status-label"),
    themeToggle: $("#theme-toggle"),
    evToggle: $("#ev-toggle"),
    evToggleCount: $("#ev-toggle-count"),
    tabEvCount: $("#tab-ev-count"),
    tabs: Array.from(document.querySelectorAll(".tab")),
    backdrop: $("#backdrop"),
    paneEvidence: $("#pane-evidence"),

    dropzone: $("#dropzone"),
    fileInput: $("#file-input"),
    docList: $("#doc-list"),
    docEmpty: $("#doc-empty"),
    libCount: $("#lib-count"),
    processBtn: $("#process-btn"),
    processLabel: $("#process-label"),
    processHelp: $("#process-help"),
    progress: $("#progress"),
    summary: $("#summary"),
    systemList: $("#system-list"),
    resetBtn: $("#reset-btn"),

    scopeLabel: $("#scope-label"),
    clearBtn: $("#clear-btn"),
    messages: $("#messages"),
    empty: $("#empty"),
    thread: $("#thread"),
    composer: $("#composer"),
    input: $("#composer-input"),
    sendBtn: $("#send-btn"),
    hint: $("#composer-hint"),

    evClose: $("#ev-close"),
    evSub: $("#ev-sub"),
    evMeta: $("#ev-meta"),
    evEmpty: $("#ev-empty"),
    evList: $("#ev-list"),

    dialog: $("#confirm-dialog"),
    dialogTitle: $("#dialog-title"),
    dialogBody: $("#dialog-body"),
    dialogOk: $("#dialog-ok"),
    toasts: $("#toasts"),
    announce: $("#announce"),
  };

  // ------------------------------------------------------------
  // State
  // ------------------------------------------------------------
  const state = {
    status: null,
    online: null,
    docs: [],
    pending: [],
    selected: new Set(),
    seen: new Set(),
    busyDocs: new Set(),
    summary: null,
    processing: false,
    processStage: "",
    asking: false,
    controller: null,
    turns: new Map(),
    history: [],
    seq: 0,
    activeTurn: null,
    stick: true,
  };

  // ------------------------------------------------------------
  // Utilities
  // ------------------------------------------------------------
  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatSize(bytes) {
    if (!Number.isFinite(bytes)) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;
  const modelName = (m) => String(m || "—").replace(/^models\//, "");

  function h(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid);
    return node;
  }

  function fill(node, ...kids) {
    node.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
  }

  function errorDetail(data) {
    const d = data && data.detail;
    if (!d) return "";
    if (typeof d === "string") return d;
    if (Array.isArray(d)) return d.map((x) => (x && x.msg) || JSON.stringify(x)).join(" ");
    return String(d);
  }

  async function api(path, opts = {}) {
    let res;
    try {
      res = await fetch(path, opts);
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      throw new Error(NET_ERROR);
    }
    let data = null;
    try { data = await res.json(); } catch { data = null; }
    if (!res.ok) throw new Error(errorDetail(data) || `Request failed (${res.status}).`);
    return data || {};
  }

  function toast(message, type = "info", ms = type === "error" ? 7000 : 3800) {
    const node = h(
      "div",
      { class: `toast ${type}`, role: type === "error" ? "alert" : "status" },
      h("span", { class: "toast-msg", text: message }),
      h("button", { type: "button", class: "icon-btn sm", "aria-label": "Dismiss notification", html: ICON.x, onclick: () => node.remove() })
    );
    el.toasts.append(node);
    setTimeout(() => node.remove(), ms);
  }

  function confirmDialog({ title, body, confirmLabel = "Confirm", danger = false }) {
    const d = el.dialog;
    if (typeof d.showModal !== "function") return Promise.resolve(window.confirm(`${title}\n\n${body}`));
    return new Promise((resolve) => {
      el.dialogTitle.textContent = title;
      el.dialogBody.textContent = body;
      el.dialogOk.textContent = confirmLabel;
      el.dialogOk.classList.toggle("btn-danger", danger);
      el.dialogOk.classList.toggle("btn-primary", !danger);
      d.returnValue = "";
      d.addEventListener("close", () => resolve(d.returnValue === "ok"), { once: true });
      d.showModal();
    });
  }

  // ------------------------------------------------------------
  // Theme
  // ------------------------------------------------------------
  function applyTheme(theme, persist) {
    document.documentElement.dataset.theme = theme;
    if (persist) {
      try { localStorage.setItem(THEME_KEY, theme); } catch { /* storage unavailable */ }
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0b0d13" : "#f5f6fa");
    el.themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  // ------------------------------------------------------------
  // Derived state
  // ------------------------------------------------------------
  const indexedDocs = () => state.docs.filter((d) => d.indexed);
  const selectedNames = () => indexedDocs().filter((d) => state.selected.has(d.name)).map((d) => d.name);

  function selectedSources() {
    const all = indexedDocs();
    if (all.length === 0) return null;
    const sel = selectedNames();
    return sel.length === all.length ? null : sel;
  }

  function composerState() {
    const s = state.status;
    if (state.online === null) return { ok: false, hint: "Connecting to the backend…" };
    if (state.online === false) return { ok: false, warn: true, hint: "Can't reach the backend. Start the server, then click the status badge to retry." };
    if (s && !s.api_configured) return { ok: false, warn: true, hint: "GOOGLE_API_KEY is missing on the server." };
    if (!s || !s.faiss_ready) return { ok: false, hint: "Upload and process at least one PDF to start asking questions." };
    if (indexedDocs().length > 0 && selectedNames().length === 0) {
      return { ok: false, hint: "Select at least one indexed document in the Library to search." };
    }
    return { ok: true, hint: "Enter to send · Shift+Enter for a new line" };
  }

  function canProcess() {
    return !state.processing && state.online !== false && (state.pending.length > 0 || state.docs.some((d) => !d.indexed));
  }

  // ------------------------------------------------------------
  // Status
  // ------------------------------------------------------------
  async function refreshStatus() {
    try {
      state.status = await api("api/status");
      state.online = true;
    } catch {
      state.status = null;
      state.online = false;
    }
    renderStatus();
    renderComposer();
    renderEmpty();
  }

  function renderStatus() {
    const s = state.status;
    let st = "checking";
    let label = "Checking…";
    if (state.online === false) { st = "offline"; label = "Backend offline"; }
    else if (s && !s.api_configured) { st = "warn"; label = "API key missing"; }
    else if (s) { st = "online"; label = "Online"; }
    el.statusPill.dataset.state = st;
    el.statusLabel.textContent = label;

    const idx = (s && s.indexed_documents) || {};
    const chunkTotal = Object.values(idx).reduce((a, b) => a + (Number(b) || 0), 0);
    const rows = [];
    const row = (k, v, cls = "") => rows.push(h("dt", { text: k }), h("dd", { class: cls, text: v }));

    if (s) {
      row("API key", s.api_configured ? "Configured" : "Missing", s.api_configured ? "ok" : "bad");
      row("Index", s.faiss_ready ? `Ready · ${plural(chunkTotal, "chunk")}` : "Not built", s.faiss_ready ? "ok" : "");
      row("Embeddings", modelName(s.embedding_model), "mono");
      row("Chat model", modelName(s.chat_model), "mono");
    } else {
      row("Backend", state.online === false ? "Unreachable" : "Checking…", state.online === false ? "bad" : "");
    }
    el.systemList.replaceChildren(...rows);
  }

  // ------------------------------------------------------------
  // Documents
  // ------------------------------------------------------------
  async function refreshDocuments() {
    try {
      const data = await api("api/documents");
      state.docs = Array.isArray(data.documents) ? data.documents : [];
    } catch (err) {
      if (state.online !== false) toast(err.message, "error");
    }
    syncSelection();
    renderLibrary();
    renderComposer();
    renderEmpty();
  }

  function syncSelection() {
    const names = new Set(state.docs.map((d) => d.name));
    for (const n of Array.from(state.selected)) if (!names.has(n)) state.selected.delete(n);
    for (const n of Array.from(state.seen)) if (!names.has(n)) state.seen.delete(n);
    for (const d of state.docs) {
      if (!d.indexed) {
        state.seen.delete(d.name);
        state.selected.delete(d.name);
      } else if (!state.seen.has(d.name)) {
        state.seen.add(d.name);
        state.selected.add(d.name);
      }
    }
  }

  function addFiles(list) {
    const rejected = [];
    for (const f of Array.from(list || [])) {
      if (!/\.pdf$/i.test(f.name)) { rejected.push(`${f.name}: only PDF files are accepted`); continue; }
      if (f.size === 0) { rejected.push(`${f.name}: file is empty`); continue; }
      if (f.size > MAX_BYTES) { rejected.push(`${f.name}: larger than 50 MB`); continue; }
      state.pending = state.pending.filter((p) => p.name !== f.name);
      state.pending.push(f);
    }
    if (rejected.length) toast(rejected.join("\n"), "error");
    renderLibrary();
  }

  function renderLibrary() {
    const items = [];

    state.pending.forEach((f) => {
      const replaces = state.docs.some((d) => d.name === f.name);
      items.push(
        h("li", { class: "doc" },
          h("span", { class: "doc-lead", html: ICON.file }),
          h("div", {},
            h("div", { class: "doc-name", title: f.name, text: f.name }),
            h("div", { class: "doc-meta" },
              h("span", { class: "badge pending", text: replaces ? "Replaces existing" : "Not processed" }),
              h("span", { text: formatSize(f.size) }))),
          h("button", { type: "button", class: "icon-btn danger", "data-remove-pending": f.name, "aria-label": `Remove ${f.name} from queue`, title: "Remove from queue", html: ICON.x }))
      );
    });

    state.docs.forEach((d) => {
      const lead = d.indexed
        ? h("input", {
            type: "checkbox", "data-select": d.name,
            "aria-label": `Include ${d.name} in questions`,
            title: "Include in questions",
            checked: state.selected.has(d.name),
          })
        : h("span", { html: ICON.file });
      if (d.indexed && state.selected.has(d.name)) lead.checked = true;

      items.push(
        h("li", { class: "doc" + (state.busyDocs.has(d.name) ? " busy" : "") },
          h("span", { class: "doc-lead" }, lead),
          h("div", {},
            h("div", { class: "doc-name", title: d.name, text: d.name }),
            h("div", { class: "doc-meta" },
              h("span", { class: "badge " + (d.indexed ? "indexed" : "pending"), text: d.indexed ? "Indexed" : "Not processed" }),
              h("span", { text: formatSize(d.size) }),
              d.indexed ? h("span", { text: plural(d.chunks || 0, "chunk") }) : null)),
          h("button", { type: "button", class: "icon-btn danger", "data-delete": d.name, "aria-label": `Delete ${d.name}`, title: "Delete", html: ICON.trash }))
      );
    });

    el.docList.replaceChildren(...items);
    el.docEmpty.hidden = items.length > 0;

    const idx = indexedDocs().length;
    el.libCount.textContent = state.docs.length + state.pending.length === 0
      ? "No documents"
      : `${plural(state.docs.length, "document")} · ${idx} indexed`;

    // process button
    const busy = state.processing;
    el.processBtn.disabled = !canProcess();
    fill(el.processBtn, 
      busy ? h("span", { class: "spinner", "aria-hidden": "true" }) : null,
      h("span", { id: "process-label", text: busy ? state.processStage : "Process documents" })
    );
    el.processBtn.setAttribute("aria-busy", String(busy));

    const unindexed = state.docs.filter((d) => !d.indexed).length;
    let help = "";
    if (busy) help = "This can take a minute for large files.";
    else if (state.pending.length) help = `${plural(state.pending.length, "file")} ready to upload and index.`;
    else if (unindexed) help = `${plural(unindexed, "uploaded file")} not yet indexed.`;
    else if (state.docs.length) help = "Everything is indexed.";
    el.processHelp.textContent = help;

    el.progress.hidden = !busy;
    el.resetBtn.disabled = state.processing || !(state.status && state.status.faiss_ready);
    renderSummary();
    renderScope();
  }

  function renderScope() {
    const idx = indexedDocs().length;
    if (idx === 0) {
      el.scopeLabel.textContent = state.status && state.status.faiss_ready ? "Searching the existing index" : "No documents indexed yet";
    } else {
      el.scopeLabel.textContent = `Searching ${selectedNames().length} of ${plural(idx, "document")}`;
    }
  }

  function renderSummary() {
    const s = state.summary;
    const box = el.summary;
    if (!s) { box.hidden = true; box.replaceChildren(); return; }
    box.hidden = false;
    box.dataset.type = s.type;

    const head = h("div", { class: "summary-head" },
      h("strong", { text: s.type === "ok" ? "Processing complete" : "Processing failed" }),
      h("button", { type: "button", class: "icon-btn sm", "aria-label": "Dismiss summary", html: ICON.x, onclick: () => { state.summary = null; renderSummary(); } }));

    if (s.type === "error") {
      box.replaceChildren(head, h("p", { class: "err", text: s.message }));
      return;
    }

    const parts = [
      head,
      h("div", { class: "summary-stats" },
        h("div", {}, h("b", { text: String(s.documents ?? 0) }), h("span", { text: "documents" })),
        h("div", {}, h("b", { text: String(s.chunks ?? 0) }), h("span", { text: "chunks" }))),
    ];
    if (s.new_files && s.new_files.length) parts.push(h("p", { class: "summary-line" }, h("b", { text: "New: " }), s.new_files.join(", ")));
    if (s.removed_files && s.removed_files.length) parts.push(h("p", { class: "summary-line" }, h("b", { text: "Removed: " }), s.removed_files.join(", ")));
    if (s.new_files && !s.new_files.length && !(s.removed_files || []).length) parts.push(h("p", { class: "summary-line", text: "No changes — the index was already up to date." }));
    if (s.warnings && s.warnings.length) parts.push(h("ul", {}, s.warnings.map((w) => h("li", { text: w }))));
    box.replaceChildren(...parts);
  }

  async function processDocuments() {
    if (!canProcess()) return;
    state.processing = true;
    state.summary = null;
    state.processStage = state.pending.length ? "Uploading…" : "Indexing…";
    renderLibrary();

    try {
      if (state.pending.length) {
        const fd = new FormData();
        state.pending.forEach((f) => fd.append("files", f, f.name));
        await api("api/upload", { method: "POST", body: fd });
        state.pending = [];
        state.processStage = "Indexing…";
        renderLibrary();
      }
      const result = await api("api/process", { method: "POST" });
      state.summary = { type: "ok", ...result };
      toast(`Indexed ${plural(result.documents || 0, "document")} (${plural(result.chunks || 0, "chunk")}).`, "success");
      if (mqMobile.matches) setTab("chat");
    } catch (err) {
      state.summary = { type: "error", message: err.message || "Processing failed." };
      toast(err.message || "Processing failed.", "error");
    } finally {
      state.processing = false;
      await Promise.all([refreshStatus(), refreshDocuments()]);
    }
  }

  async function deleteDocument(name) {
    if (state.busyDocs.has(name) || state.processing) return;
    const ok = await confirmDialog({
      title: "Delete document?",
      body: `“${name}” will be removed from your uploads and from the search index.`,
      confirmLabel: "Delete",
      danger: true,
    });
    if (!ok) return;
    state.busyDocs.add(name);
    renderLibrary();
    try {
      await api(`api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast(`Deleted ${name}.`, "success");
    } catch (err) {
      toast(err.message, "error");
    } finally {
      state.busyDocs.delete(name);
      await Promise.all([refreshStatus(), refreshDocuments()]);
    }
  }

  async function resetIndex() {
    if (state.processing) return;
    const ok = await confirmDialog({
      title: "Reset the index?",
      body: "The search index will be cleared. Your uploaded PDFs are kept and can be processed again.",
      confirmLabel: "Reset index",
      danger: true,
    });
    if (!ok) return;
    try {
      await api("api/reset", { method: "POST" });
      state.summary = null;
      toast("Index cleared.", "success");
    } catch (err) {
      toast(err.message, "error");
    }
    await Promise.all([refreshStatus(), refreshDocuments()]);
  }

  // ------------------------------------------------------------
  // Markdown (escaped first, then safe markup)
  // ------------------------------------------------------------
  function inline(s, citeMax) {
    return s
      .split(/(`[^`]+`)/g)
      .map((seg, i) => {
        if (i % 2 === 1) return `<code>${seg.slice(1, -1)}</code>`;
        let t = seg
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/(^|[\s(])\*([^\s*][^*]*?)\*(?=$|[\s).,;:!?])/g, "$1<em>$2</em>");
        if (citeMax > 0) {
          t = t.replace(/\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g, (m, nums) => {
            const ids = nums.split(",").map((n) => parseInt(n, 10));
            if (ids.some((n) => n < 1 || n > citeMax)) return m;
            return ids
              .map((n) => `<button type="button" class="cite" data-n="${n}" aria-label="Open source ${n}">${n}</button>`)
              .join("");
          });
        }
        return t;
      })
      .join("");
  }

  function renderBlocks(text, citeMax) {
    const lines = text.replace(/\r/g, "").split("\n");
    const out = [];
    let para = [];
    let list = null;

    const flushPara = () => { if (para.length) { out.push(`<p>${para.join("<br>")}</p>`); para = []; } };
    const flushList = () => {
      if (list) {
        out.push(`<${list.type}>${list.items.map((i) => `<li>${i}</li>`).join("")}</${list.type}>`);
        list = null;
      }
    };
    const flush = () => { flushPara(); flushList(); };
    const cells = (t) => t.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

    for (let i = 0; i < lines.length; i++) {
      const t = escapeHtml(lines[i]).trim();

      if (t === "") { flush(); continue; }

      if (t.startsWith("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(lines[i + 1])) {
        flush();
        const head = cells(t);
        const rows = [];
        i += 2;
        while (i < lines.length && escapeHtml(lines[i]).trim().startsWith("|")) {
          rows.push(cells(escapeHtml(lines[i]).trim()));
          i++;
        }
        i--;
        out.push(
          `<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inline(c, citeMax)}</th>`).join("")}</tr></thead>` +
          `<tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c, citeMax)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`
        );
        continue;
      }

      const heading = t.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        flush();
        const level = heading[1].length + 2;
        out.push(`<h${level}>${inline(heading[2], citeMax)}</h${level}>`);
        continue;
      }

      const bullet = t.match(/^[-*•]\s+(.*)$/);
      const numbered = bullet ? null : t.match(/^\d+[.)]\s+(.*)$/);
      if (bullet || numbered) {
        flushPara();
        const type = bullet ? "ul" : "ol";
        if (list && list.type !== type) flushList();
        if (!list) list = { type, items: [] };
        list.items.push(inline((bullet || numbered)[1], citeMax));
        continue;
      }

      flushList();
      para.push(inline(t, citeMax));
    }
    flush();
    return out.join("");
  }

  function renderMarkdown(raw, citeMax = 0) {
    return String(raw)
      .split("```")
      .map((part, i) => {
        if (i % 2 === 0) return renderBlocks(part, citeMax);
        const code = part.replace(/^[\w+#.-]*\n/, "").replace(/\n$/, "");
        return `<pre><code>${escapeHtml(code)}</code></pre>`;
      })
      .join("");
  }

  const isNotFound = (raw) =>
    raw.trim().replace(/[.\s]+$/, "").toLowerCase() === "answer is not available in the context";

  const NOT_FOUND_HTML =
    `<div class="not-found">${ICON.question}<div><strong>Not found in your documents</strong>` +
    `<span>The selected documents don’t seem to contain an answer. Try rephrasing, or include more documents.</span></div></div>`;

  // ------------------------------------------------------------
  // Chat rendering
  // ------------------------------------------------------------
  function renderEmpty() {
    if (state.turns.size > 0) { el.empty.hidden = true; return; }
    el.empty.hidden = false;
    const ready = composerState().ok;

    if (!ready) {
      fill(el.empty, 
        h("div", { class: "empty-icon", html: ICON.doc }),
        h("h3", { text: "Add documents to get started" }),
        h("p", { text: "Upload PDFs in the Library, then process them. After that you can ask questions and see exactly which pages each answer comes from." }),
        mqMobile.matches ? h("button", { type: "button", class: "btn btn-primary", text: "Open library", onclick: () => setTab("library") }) : null
      );
      return;
    }

    const names = selectedNames();
    const suggestions = [
      names.length === 1 ? `Give me a short summary of ${names[0]}` : "Summarize the main points of these documents",
      "What are the key findings or conclusions?",
      "List the important terms and definitions",
    ];
    fill(el.empty, 
      h("div", { class: "empty-icon", html: ICON.spark }),
      h("h3", { text: "Ask your documents anything" }),
      h("p", { text: "Answers cite the exact passages they came from. Try one of these to start:" }),
      h("div", { class: "suggestions" }, suggestions.map((q) =>
        h("button", { type: "button", class: "suggestion", text: q, onclick: () => sendQuestion(q) })))
    );
  }

  function renderComposer() {
    const c = composerState();
    el.input.disabled = !c.ok;
    el.hint.textContent = state.asking ? "Generating… press Stop to cancel." : c.hint;
    el.hint.classList.toggle("warn", !!c.warn && !state.asking);

    if (state.asking) {
      el.sendBtn.dataset.mode = "stop";
      el.sendBtn.setAttribute("aria-label", "Stop generating");
      el.sendBtn.disabled = false;
    } else {
      el.sendBtn.dataset.mode = "send";
      el.sendBtn.setAttribute("aria-label", "Send question");
      el.sendBtn.disabled = !c.ok || el.input.value.trim() === "";
    }
    el.clearBtn.disabled = state.turns.size === 0;
    renderScope();
  }

  function autosize() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
  }

  const nearBottom = () => el.messages.scrollHeight - el.messages.scrollTop - el.messages.clientHeight < 96;
  const scrollBottom = () => { el.messages.scrollTop = el.messages.scrollHeight; };

  function createTurn(id, question) {
    const body = h("div", { class: "msg-body" });
    const statusText = h("span", { text: "Searching your documents…" });
    const status = h("div", { class: "msg-status" },
      h("span", { class: "dots", "aria-hidden": "true" }, h("i"), h("i"), h("i")), statusText);
    const errorBox = h("div", { hidden: true });
    const note = h("div", { class: "msg-note", hidden: true });
    const actions = h("div", { class: "msg-actions", hidden: true });

    const ai = h("article", { class: "msg msg-ai", "data-turn": String(id), "aria-label": "Answer" },
      h("div", { class: "avatar", "aria-hidden": "true", html: ICON.spark }),
      h("div", { class: "msg-content" },
        h("div", { class: "msg-role", text: "RAGBench" }),
        status, body, errorBox, note, actions));
    const user = h("article", { class: "msg msg-user", "aria-label": "Your question" },
      h("div", { class: "bubble", text: question }));
    return { root: h("div", { class: "turn" }, user, ai), body, status, statusText, errorBox, note, actions };
  }

  function paint(turn) {
    const { body } = turn.ui;
    if (turn.notFound) body.innerHTML = NOT_FOUND_HTML;
    else if (turn.raw.trim()) body.innerHTML = renderMarkdown(turn.raw, turn.sources.length);
    else body.innerHTML = "";
    if (state.stick) scrollBottom();
  }

  function scheduleRender(turn) {
    if (turn.raf) return;
    turn.raf = requestAnimationFrame(() => { turn.raf = 0; paint(turn); });
  }

  async function copyText(text, btn) {
    let ok = false;
    try {
      await navigator.clipboard.writeText(text);
      ok = true;
    } catch {
      const ta = h("textarea", { class: "sr-only", "aria-hidden": "true" });
      ta.value = text;
      document.body.append(ta);
      ta.select();
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      ta.remove();
    }
    if (!ok) { toast("Couldn't copy to the clipboard.", "error"); return; }
    const original = btn.innerHTML;
    btn.classList.add("done");
    btn.innerHTML = `${ICON.check} Copied`;
    setTimeout(() => { btn.classList.remove("done"); btn.innerHTML = original; }, 1600);
  }

  function finishTurn(turn, { stopped = false, error = null } = {}) {
    const ui = turn.ui;
    if (turn.raf) { cancelAnimationFrame(turn.raf); turn.raf = 0; }
    ui.status.hidden = true;
    ui.body.classList.remove("streaming");

    const hasText = turn.raw.trim().length > 0;
    turn.notFound = hasText && !error && !stopped && isNotFound(turn.raw);
    paint(turn);

    if (error) {
      ui.errorBox.hidden = false;
      ui.errorBox.className = "msg-error";
      ui.errorBox.setAttribute("role", "alert");
      fill(ui.errorBox, 
        h("span", { html: ICON.alert }),
        h("div", {},
          h("div", { text: error }),
          h("button", { type: "button", class: "link-btn", html: `${ICON.retry} Try again`, onclick: () => retryTurn(turn) }))
      );
    }

    const notes = [];
    if (stopped) notes.push(hasText ? "Generation stopped." : "Stopped before an answer was generated.");
    const asked = turn.question.trim().toLowerCase();
    if (turn.query && turn.query.trim().toLowerCase() !== asked && !error) notes.push(`Searched for: “${turn.query}”`);
    if (notes.length) { ui.note.hidden = false; ui.note.textContent = notes.join(" · "); }

    const acts = [];
    if (hasText && !turn.notFound) {
      acts.push(h("button", { type: "button", class: "chip-btn", "aria-label": "Copy answer", html: `${ICON.copy} Copy` , onclick: (e) => copyText(turn.raw.trim(), e.currentTarget) }));
    }
    if (turn.sources.length) {
      acts.push(h("button", { type: "button", class: "chip-btn", html: `${ICON.list} ${plural(turn.sources.length, "source")}`, onclick: () => openEvidence(turn.id) }));
    }
    if (acts.length) { ui.actions.hidden = false; ui.actions.replaceChildren(...acts); }

    el.announce.textContent = error ? "The answer failed." : stopped ? "Generation stopped." : `Answer ready. ${plural(turn.sources.length, "source")} available.`;
  }

  function retryTurn(turn) {
    if (state.asking) return;
    turn.ui.root.remove();
    state.turns.delete(turn.id);
    if (state.activeTurn === turn.id) { state.activeTurn = null; renderEvidence(); }
    renderEmpty();
    renderComposer();
    sendQuestion(turn.question);
  }

  // ------------------------------------------------------------
  // Streaming
  // ------------------------------------------------------------
  async function streamAnswer(payload, signal, onEvent) {
    const init = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    };

    let res;
    try {
      res = await fetch("api/ask/stream", init);
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      throw new Error(NET_ERROR);
    }
    if (!res.ok) {
      let data = null;
      try { data = await res.json(); } catch { data = null; }
      throw new Error(errorDetail(data) || `Request failed (${res.status}).`);
    }

    if (!res.body) {
      const data = await api("api/ask", init);
      onEvent({ type: "sources", sources: data.sources || [], query: data.query || "" });
      onEvent({ type: "token", text: data.answer || "" });
      onEvent({ type: "done" });
      return;
    }

    const emit = (line) => {
      let ev;
      try { ev = JSON.parse(line); } catch { return; }
      if (ev && ev.type) onEvent(ev);
    };

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) emit(line);
      }
    }
    buf += decoder.decode();
    if (buf.trim()) emit(buf.trim());
  }

  async function sendQuestion(raw) {
    const question = String(raw || "").trim();
    if (!question || state.asking || !composerState().ok) return;

    state.asking = true;
    state.stick = true;
    const id = ++state.seq;
    const ui = createTurn(id, question);
    const turn = { id, question, query: "", sources: [], raw: "", notFound: false, ui, raf: 0 };
    state.turns.set(id, turn);
    el.empty.hidden = true;
    el.thread.append(ui.root);
    scrollBottom();
    el.messages.setAttribute("aria-busy", "true");
    renderComposer();

    const controller = new AbortController();
    state.controller = controller;
    const payload = { question, history: state.history.slice(-HISTORY_LIMIT), sources: selectedSources() };

    let gotDone = false;
    const outcome = {};
    try {
      await streamAnswer(payload, controller.signal, (ev) => {
        if (ev.type === "sources") {
          turn.sources = Array.isArray(ev.sources) ? ev.sources : [];
          turn.query = ev.query || "";
          ui.statusText.textContent = turn.sources.length
            ? `Reading ${plural(turn.sources.length, "passage")}…`
            : "No matching passages found.";
          state.activeTurn = id;
          renderEvidence();
        } else if (ev.type === "token") {
          if (!ui.status.hidden) { ui.status.hidden = true; ui.body.classList.add("streaming"); }
          turn.raw += ev.text || "";
          scheduleRender(turn);
        } else if (ev.type === "error") {
          throw new Error(ev.detail || "The answer could not be generated.");
        } else if (ev.type === "done") {
          gotDone = true;
        }
      });
      if (!gotDone && !turn.raw.trim()) throw new Error("The response ended unexpectedly. Please try again.");
    } catch (err) {
      if (err && err.name === "AbortError") outcome.stopped = true;
      else outcome.error = (err && err.message) || "Something went wrong.";
    } finally {
      state.asking = false;
      state.controller = null;
      el.messages.removeAttribute("aria-busy");
    }

    if (!state.turns.has(id)) return;
    finishTurn(turn, outcome);

    if (!outcome.error && !outcome.stopped && turn.raw.trim()) {
      state.history.push({ role: "user", content: question }, { role: "assistant", content: turn.raw.trim() });
      state.history = state.history.slice(-HISTORY_LIMIT * 2);
    }

    renderComposer();
    const active = document.activeElement;
    if (!mqMobile.matches && (active === document.body || active === el.sendBtn)) el.input.focus();
  }

  function clearChat() {
    if (state.controller) state.controller.abort();
    state.turns.clear();
    state.history = [];
    state.activeTurn = null;
    el.thread.replaceChildren();
    renderEvidence();
    renderEmpty();
    renderComposer();
  }

  // ------------------------------------------------------------
  // Evidence
  // ------------------------------------------------------------
  function sourceCard(s) {
    const text = String(s.content || "").trim();
    const long = text.length > 420;
    const passage = h("div", { class: "passage" + (long ? " clamped" : ""), text });
    const toggle = long
      ? h("button", {
          type: "button", class: "link-btn", "aria-expanded": "false", text: "Show more",
          onclick: (e) => {
            const open = passage.classList.toggle("clamped") === false;
            e.currentTarget.setAttribute("aria-expanded", String(open));
            e.currentTarget.textContent = open ? "Show less" : "Show more";
          },
        })
      : null;

    return h("article", { class: "source", "data-index": String(s.index), tabindex: "-1", "aria-label": `Source ${s.index}` },
      h("header", { class: "source-head" },
        h("span", { class: "source-index", text: String(s.index) }),
        h("span", { class: "source-name", title: s.source || "Document", text: s.source || "Document" }),
        s.page != null ? h("span", { class: "source-page", text: `p. ${s.page}` }) : null),
      passage, toggle);
  }

  function updateEvidenceBadges(count) {
    for (const badge of [el.evToggleCount, el.tabEvCount]) {
      badge.hidden = count === 0;
      badge.textContent = String(count);
    }
  }

  function renderEvidence() {
    const turn = state.turns.get(state.activeTurn);
    if (!turn || turn.sources.length === 0) {
      el.evMeta.hidden = true;
      el.evList.replaceChildren();
      el.evEmpty.hidden = false;
      el.evSub.textContent = turn ? "No passages matched this question" : "Passages behind each answer";
      updateEvidenceBadges(0);
      return;
    }

    el.evEmpty.hidden = true;
    el.evSub.textContent = plural(turn.sources.length, "passage") + " retrieved";
    el.evMeta.hidden = false;
    fill(el.evMeta, 
      h("div", {}, "Sources for ", h("q", { text: turn.question })),
      turn.notFound ? h("div", { class: "ev-warn", text: "These were the closest matches, but they didn’t contain an answer." }) : null
    );
    el.evList.replaceChildren(...turn.sources.map(sourceCard));
    updateEvidenceBadges(turn.sources.length);
  }

  function openEvidence(turnId, index) {
    if (state.activeTurn !== turnId) {
      state.activeTurn = turnId;
      renderEvidence();
    }
    revealEvidence();

    el.evList.querySelectorAll(".source.active").forEach((n) => n.classList.remove("active"));
    if (index == null) return;
    const card = el.evList.querySelector(`.source[data-index="${index}"]`);
    if (!card) return;
    const toggle = card.querySelector(".link-btn");
    const passage = card.querySelector(".passage");
    if (toggle && passage.classList.contains("clamped")) toggle.click();
    card.classList.add("active");
    card.scrollIntoView({ block: "nearest", behavior: mqReduce.matches ? "auto" : "smooth" });
  }

  function revealEvidence() {
    if (mqMobile.matches) setTab("evidence");
    else if (mqDrawer.matches) setDrawer(true);
  }

  // ------------------------------------------------------------
  // Navigation (tabs, drawer)
  // ------------------------------------------------------------
  function setTab(tab) {
    el.app.dataset.tab = tab;
    el.tabs.forEach((b) => {
      if (b.dataset.tab === tab) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    if (tab === "chat") scrollBottom();
  }

  function setDrawer(open) {
    const was = el.app.classList.contains("drawer-open");
    el.app.classList.toggle("drawer-open", open);
    el.backdrop.hidden = !open;
    el.evToggle.setAttribute("aria-expanded", String(open));
    if (open && !was) el.evClose.focus({ preventScroll: true });
    if (!open && was && el.paneEvidence.contains(document.activeElement)) el.evToggle.focus();
  }

  // ------------------------------------------------------------
  // Events
  // ------------------------------------------------------------
  function bindEvents() {
    el.themeToggle.addEventListener("click", () => {
      applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
    });
    el.statusPill.addEventListener("click", () => { refreshStatus(); refreshDocuments(); });

    el.tabs.forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
    el.evToggle.addEventListener("click", () => setDrawer(!el.app.classList.contains("drawer-open")));
    el.evClose.addEventListener("click", () => setDrawer(false));
    el.backdrop.addEventListener("click", () => setDrawer(false));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && el.app.classList.contains("drawer-open") && !el.dialog.open) setDrawer(false);
    });
    mqDrawer.addEventListener("change", () => { if (!mqDrawer.matches) setDrawer(false); });
    mqMobile.addEventListener("change", () => { renderEmpty(); });

    // Library
    el.fileInput.addEventListener("change", (e) => { addFiles(e.target.files); el.fileInput.value = ""; });
    ["dragenter", "dragover"].forEach((t) =>
      el.dropzone.addEventListener(t, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((t) =>
      el.dropzone.addEventListener(t, () => el.dropzone.classList.remove("dragover")));
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
        addFiles(e.dataTransfer.files);
        if (mqMobile.matches) setTab("library");
      }
    });

    el.docList.addEventListener("click", (e) => {
      const del = e.target.closest("[data-delete]");
      if (del) { deleteDocument(del.dataset.delete); return; }
      const rm = e.target.closest("[data-remove-pending]");
      if (rm) {
        state.pending = state.pending.filter((f) => f.name !== rm.dataset.removePending);
        renderLibrary();
        renderComposer();
      }
    });
    el.docList.addEventListener("change", (e) => {
      const box = e.target.closest("[data-select]");
      if (!box) return;
      if (box.checked) state.selected.add(box.dataset.select);
      else state.selected.delete(box.dataset.select);
      renderScope();
      renderComposer();
      renderEmpty();
    });

    el.processBtn.addEventListener("click", processDocuments);
    el.resetBtn.addEventListener("click", resetIndex);

    // Chat
    el.clearBtn.addEventListener("click", clearChat);
    el.messages.addEventListener("scroll", () => { state.stick = nearBottom(); }, { passive: true });
    el.messages.addEventListener("click", (e) => {
      const cite = e.target.closest(".cite");
      if (!cite) return;
      const msg = cite.closest(".msg-ai");
      if (msg) openEvidence(Number(msg.dataset.turn), Number(cite.dataset.n));
    });

    el.input.addEventListener("input", () => { autosize(); renderComposer(); });
    el.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (!state.asking) el.composer.requestSubmit();
      }
    });
    el.composer.addEventListener("submit", (e) => {
      e.preventDefault();
      if (state.asking) { if (state.controller) state.controller.abort(); return; }
      const q = el.input.value.trim();
      if (!q || !composerState().ok) return;
      el.input.value = "";
      autosize();
      sendQuestion(q);
    });

    window.addEventListener("online", refreshStatus);
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refreshStatus(); });
  }

  // ------------------------------------------------------------
  // Init
  // ------------------------------------------------------------
  function init() {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "dark" : "light", false);
    bindEvents();
    setTab(el.app.dataset.tab || "chat");
    renderStatus();
    renderLibrary();
    renderComposer();
    renderEmpty();
    renderEvidence();
    refreshStatus();
    refreshDocuments();
  }

  init();
})();