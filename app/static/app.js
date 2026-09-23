const $ = (id) => document.getElementById(id);

let sessionId = null;
let sessionPromise = null;
let currentImageSrc = null;
let previousImageSrc = null;
let drawing = false;
let tool = "mask";
let lastPaintTool = "mask";
let lastPoint = null;
let refs = [];
let workflow = "edit";
let busy = false;

const imageCanvas = $("imageCanvas");
const maskCanvas = $("maskCanvas");
const sketchCanvas = $("sketchCanvas");
const imageCtx = imageCanvas.getContext("2d");
const maskCtx = maskCanvas.getContext("2d");
const sketchCtx = sketchCanvas.getContext("2d");

async function apiError(response) {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    detail = body.detail || body.error || JSON.stringify(body);
  } catch (_) {
    try { detail = await response.text() || detail; } catch (_) {}
  }
  return detail;
}

function setServerState(kind, text) {
  const el = $("serverState");
  el.className = `status-pill ${kind}`;
  el.textContent = text;
}

async function createSession() {
  setServerState("waiting", "connecting…");
  const r = await fetch("/api/sessions", { method: "POST", cache: "no-store" });
  if (!r.ok) throw new Error(await apiError(r));
  const data = await r.json();
  sessionId = data.session_id;
  setServerState("ready", "ready");
  return sessionId;
}

async function ensureSession() {
  if (sessionId) return sessionId;
  if (!sessionPromise) {
    sessionPromise = createSession().catch((err) => {
      sessionPromise = null;
      setServerState("error", "server error");
      throw err;
    });
  }
  return sessionPromise;
}

async function init() {
  try {
    await ensureSession();
    const r = await fetch("/api/health", { cache: "no-store" });
    if (r.ok) {
      const health = await r.json();
      if (health.default_provider && $("provider").querySelector(`option[value="${health.default_provider}"]`)) {
        $("provider").value = health.default_provider;
      }
      const gpu = health.gpu || {};
      if (gpu.available) {
        addMessage("system", `GPU: ${gpu.name} · ${gpu.total_vram_gb} GB · ${gpu.preferred_dtype}. Загрузите фото или создайте изображение с нуля.`);
      } else {
        addMessage("system", "CUDA GPU не обнаружена. Интерфейс работает, но генеративные операции потребуют GPU.");
      }
    }
  } catch (e) {
    addMessage("error", `Не удалось создать сессию: ${e.message}`);
  }
}

function addMessage(role, content) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = content;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
}

function setBusy(on, text = "Working…") {
  busy = on;
  $("primaryAction").disabled = on;
  $("imageUpload").disabled = on;
  $("newBtn").disabled = on;
  if (!$("stage").classList.contains("hidden")) {
    $("busy").classList.toggle("hidden", !on);
    $("busyText").textContent = text;
  }
  if (on) setServerState("waiting", text.toLowerCase());
  else if (sessionId) setServerState("ready", "ready");
  updatePrimaryAction();
}

function updatePrimaryAction() {
  const btn = $("primaryAction");
  if (workflow === "create") {
    btn.textContent = busy ? "Generating…" : "Generate image";
    $("hint").textContent = "FLUX.2 klein 4B создаёт изображение с нуля. После генерации можно сразу продолжить локальными или глобальными правками.";
    $("referencesButton").classList.add("hidden");
    $("refCount").classList.add("hidden");
  } else {
    btn.textContent = busy ? "Editing…" : "Apply edit";
    $("hint").textContent = currentImageSrc
      ? "Для локального изменения нарисуй маску. Sketch — отдельный слой подсказки формы и положения."
      : "Сначала загрузите фото слева или создайте изображение в режиме «Создать с нуля».";
    $("referencesButton").classList.remove("hidden");
    $("refCount").classList.remove("hidden");
  }
  btn.disabled = busy;
}

function setWorkflow(next) {
  workflow = next;
  $("editTab").classList.toggle("active", next === "edit");
  $("createTab").classList.toggle("active", next === "create");
  $("editSettings").classList.toggle("hidden", next !== "edit");
  $("createSettings").classList.toggle("hidden", next !== "create");
  $("prompt").placeholder = next === "create"
    ? "Например: cinematic photo of a glass observatory on a snowy mountain at blue hour…"
    : "Например: добавь в выделенную область старое кожаное кресло, сохрани перспективу и освещение…";
  updatePrimaryAction();
  if (next === "create") $("prompt").focus();
}

function resizeStage(img) {
  const stage = $("stage");
  const availableW = Math.max(240, $("dropZone").clientWidth - 48);
  const availableH = Math.max(360, window.innerHeight - 190);
  const scale = Math.min(1, availableW / img.naturalWidth, availableH / img.naturalHeight);
  stage.style.width = `${Math.round(img.naturalWidth * scale)}px`;
  stage.style.height = `${Math.round(img.naturalHeight * scale)}px`;
}

async function loadImage(src, keepPrevious = true) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      if (keepPrevious && currentImageSrc) previousImageSrc = currentImageSrc;
      currentImageSrc = src;
      [imageCanvas, maskCanvas, sketchCanvas].forEach((c) => {
        c.width = img.naturalWidth;
        c.height = img.naturalHeight;
      });
      imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
      imageCtx.drawImage(img, 0, 0);
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
      sketchCtx.clearRect(0, 0, sketchCanvas.width, sketchCanvas.height);
      resizeStage(img);
      $("stage").classList.remove("hidden");
      $("emptyState").classList.add("hidden");
      $("dropZone").classList.remove("empty-clickable");
      $("toolbar").classList.remove("disabled-toolbar");
      $("beforeBtn").disabled = !previousImageSrc;
      updatePrimaryAction();
      resolve();
    };
    img.onerror = () => reject(new Error("Браузер не смог открыть изображение, возвращённое сервером."));
    img.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  });
}

async function uploadPhoto(file) {
  if (!file || busy) return;
  if (!file.type.startsWith("image/")) {
    addMessage("error", "Выбранный файл не является изображением.");
    return;
  }

  try {
    await ensureSession();
    const fd = new FormData();
    fd.append("image", file, file.name || "upload.png");
    setBusy(true, "Uploading…");
    addMessage("system", `Загружаю ${file.name || "image"}…`);
    const r = await fetch(`/api/sessions/${sessionId}/upload`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await apiError(r));
    const data = await r.json();
    previousImageSrc = null;
    currentImageSrc = null;
    await loadImage(data.image_url, false);
    setWorkflow("edit");
    addMessage("assistant", `Фото загружено: ${data.width}×${data.height}. Можно выделять область и описывать правку.`);
  } catch (e) {
    addMessage("error", `Ошибка загрузки: ${e.message}`);
  } finally {
    $("imageUpload").value = "";
    setBusy(false);
  }
}

function pointFromEvent(e) {
  const rect = maskCanvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * maskCanvas.width / rect.width,
    y: (e.clientY - rect.top) * maskCanvas.height / rect.height,
  };
}

function activeContext() {
  if (tool === "sketch" || (tool === "erase" && lastPaintTool === "sketch")) return sketchCtx;
  return maskCtx;
}

function drawSegment(a, b) {
  const ctx = activeContext();
  const size = Number($("brushSize").value);
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = size;
  if (tool === "erase") {
    ctx.globalCompositeOperation = "destination-out";
    ctx.strokeStyle = "rgba(0,0,0,1)";
  } else if (tool === "sketch") {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = $("sketchColor").value;
  } else {
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = "rgba(255,255,255,1)";
  }
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function pointerDown(e) {
  if (!currentImageSrc || busy) return;
  drawing = true;
  lastPoint = pointFromEvent(e);
  drawSegment(lastPoint, { x: lastPoint.x + 0.01, y: lastPoint.y + 0.01 });
  maskCanvas.setPointerCapture?.(e.pointerId);
}

function pointerMove(e) {
  if (!drawing) return;
  const p = pointFromEvent(e);
  drawSegment(lastPoint, p);
  lastPoint = p;
}

function pointerUp() {
  drawing = false;
  lastPoint = null;
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Не удалось сериализовать слой canvas.")), "image/png");
  });
}

function canvasHasInk(canvas) {
  if (!canvas.width || !canvas.height) return false;
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) return true;
  return false;
}

async function pollJob(jobId, actionName, onDone = null) {
  while (true) {
    await new Promise((r) => setTimeout(r, 1000));
    let r;
    try {
      r = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!r.ok) throw new Error(await apiError(r));
    } catch (e) {
      addMessage("error", `Не удалось получить статус задачи: ${e.message}`);
      setBusy(false);
      return;
    }
    const job = await r.json();
    if (job.status === "done") {
      try {
        await loadImage(job.image_url, true);
        addMessage("assistant", `${actionName} · ${job.provider} · ${job.mode} · seed ${job.seed}`);
        if (onDone) onDone(job);
      } catch (e) {
        addMessage("error", e.message);
      }
      setBusy(false);
      return;
    }
    if (job.status === "error") {
      addMessage("error", job.error || "Generation failed");
      setBusy(false);
      return;
    }
    const label = job.status === "queued" ? "Queued on GPU…" : "Running model…";
    if (!$("stage").classList.contains("hidden")) $("busyText").textContent = label;
    setServerState("waiting", job.status === "queued" ? "queued" : "generating");
  }
}

async function applyEdit() {
  const prompt = $("prompt").value.trim();
  if (!currentImageSrc) {
    addMessage("error", "Сначала загрузите фото или создайте изображение с нуля.");
    return;
  }
  if (!prompt) {
    $("prompt").focus();
    return;
  }

  const mode = $("editMode").value;
  const hasMask = canvasHasInk(maskCanvas);
  if (mode === "local" && !hasMask) {
    addMessage("error", "Local mode требует нарисованную маску.");
    return;
  }

  try {
    await ensureSession();
    addMessage("user", prompt);
    const fd = new FormData();
    fd.append("prompt", prompt);
    fd.append("provider", $("provider").value);
    fd.append("mode", mode);
    fd.append("strict_local", $("strictLocal").checked ? "true" : "false");
    const seed = $("seed").value.trim();
    if (seed) fd.append("seed", seed);
    if (hasMask) fd.append("mask", await canvasBlob(maskCanvas), "mask.png");
    if (canvasHasInk(sketchCanvas)) fd.append("sketch", await canvasBlob(sketchCanvas), "sketch.png");
    refs.slice(0, 3).forEach((file) => fd.append("references", file, file.name));

    setBusy(true, "Queueing edit…");
    const r = await fetch(`/api/sessions/${sessionId}/edit`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await apiError(r));
    $("prompt").value = "";
    const data = await r.json();
    pollJob(data.job_id, "Edit complete");
  } catch (e) {
    addMessage("error", `Edit request failed: ${e.message}`);
    setBusy(false);
  }
}

function generationDimensions() {
  const size = Number($("generationSize").value);
  const ratio = $("aspectRatio").value;
  const pairs = {
    "1:1": [size, size],
    "4:3": [size, Math.round(size * 3 / 4 / 8) * 8],
    "3:4": [Math.round(size * 3 / 4 / 8) * 8, size],
    "16:9": [size, Math.round(size * 9 / 16 / 8) * 8],
    "9:16": [Math.round(size * 9 / 16 / 8) * 8, size],
  };
  return pairs[ratio] || [size, size];
}

async function generateNew() {
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("prompt").focus();
    return;
  }
  try {
    await ensureSession();
    const [width, height] = generationDimensions();
    const seedText = $("seed").value.trim();
    const seed = seedText ? Number(seedText) : null;
    addMessage("user", `[Создать ${width}×${height}] ${prompt}`);
    setBusy(true, "Queueing generation…");
    const r = await fetch(`/api/sessions/${sessionId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, width, height, seed }),
    });
    if (!r.ok) throw new Error(await apiError(r));
    $("prompt").value = "";
    const data = await r.json();
    pollJob(data.job_id, "Generation complete", () => setWorkflow("edit"));
  } catch (e) {
    addMessage("error", `Generation request failed: ${e.message}`);
    setBusy(false);
  }
}

async function resetSession() {
  if (busy) return;
  try {
    await ensureSession();
    const r = await fetch(`/api/sessions/${sessionId}/reset`, { method: "POST" });
    if (!r.ok) throw new Error(await apiError(r));
  } catch (e) {
    addMessage("error", `Reset failed: ${e.message}`);
    return;
  }
  currentImageSrc = null;
  previousImageSrc = null;
  refs = [];
  $("references").value = "";
  $("refCount").textContent = "0 refs";
  [imageCanvas, maskCanvas, sketchCanvas].forEach((c) => {
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
    c.width = 0;
    c.height = 0;
  });
  $("stage").classList.add("hidden");
  $("emptyState").classList.remove("hidden");
  $("dropZone").classList.add("empty-clickable");
  $("toolbar").classList.add("disabled-toolbar");
  $("beforeBtn").disabled = true;
  setWorkflow("edit");
  addMessage("system", "Новая сессия изображения. Загрузите фото или создайте изображение с нуля.");
}

function showBefore(on) {
  if (!previousImageSrc || !currentImageSrc) return;
  const img = new Image();
  img.onload = () => {
    imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
  };
  const src = on ? previousImageSrc : currentImageSrc;
  img.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

$("imageUpload").addEventListener("change", (e) => uploadPhoto(e.target.files?.[0]));
$("primaryAction").addEventListener("click", () => workflow === "create" ? generateNew() : applyEdit());
$("newBtn").addEventListener("click", resetSession);
$("editTab").addEventListener("click", () => setWorkflow("edit"));
$("createTab").addEventListener("click", () => setWorkflow("create"));
$("startGenerateBtn").addEventListener("click", (e) => { e.stopPropagation(); setWorkflow("create"); });

$("clearMask").addEventListener("click", () => maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height));
$("clearSketch").addEventListener("click", () => sketchCtx.clearRect(0, 0, sketchCanvas.width, sketchCanvas.height));
$("references").addEventListener("change", (e) => {
  refs = [...e.target.files].slice(0, 3);
  $("refCount").textContent = `${refs.length} refs`;
});

document.querySelectorAll(".tool").forEach((btn) => btn.addEventListener("click", () => {
  tool = btn.dataset.tool;
  if (tool === "mask" || tool === "sketch") lastPaintTool = tool;
  document.querySelectorAll(".tool").forEach((x) => x.classList.toggle("active", x === btn));
}));

maskCanvas.addEventListener("pointerdown", pointerDown);
maskCanvas.addEventListener("pointermove", pointerMove);
maskCanvas.addEventListener("pointerup", pointerUp);
maskCanvas.addEventListener("pointercancel", pointerUp);

$("beforeBtn").addEventListener("mousedown", () => showBefore(true));
$("beforeBtn").addEventListener("mouseup", () => showBefore(false));
$("beforeBtn").addEventListener("mouseleave", () => showBefore(false));
$("beforeBtn").addEventListener("touchstart", (e) => { e.preventDefault(); showBefore(true); });
$("beforeBtn").addEventListener("touchend", () => showBefore(false));

const drop = $("dropZone");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragging"); });
drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragging");
  const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
  if (file) uploadPhoto(file);
});
drop.addEventListener("click", (e) => {
  if (!currentImageSrc && !busy && !e.target.closest("button")) $("imageUpload").click();
});

$("prompt").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    workflow === "create" ? generateNew() : applyEdit();
  }
});

window.addEventListener("resize", () => {
  if (!currentImageSrc || !imageCanvas.width) return;
  const img = { naturalWidth: imageCanvas.width, naturalHeight: imageCanvas.height };
  resizeStage(img);
});

setWorkflow("edit");
init();
