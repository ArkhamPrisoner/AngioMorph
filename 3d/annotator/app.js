const TYPE_META = {
  stenosis: { label: "Стеноз", color: "#d94841", defaultShape: "segment" },
  bifurcation: { label: "Бифуркация", color: "#0c7c59", defaultShape: "point" },
  intersection: { label: "Пересечение", color: "#1f5aa6", defaultShape: "point" },
  uncertain: { label: "Спорный участок", color: "#a15c00", defaultShape: "box" },
};

const SHAPE_META = {
  point: { label: "Точка" },
  segment: { label: "Отрезок" },
  box: { label: "Область" },
};

const KEY_TYPE = {
  "1": "stenosis",
  "2": "bifurcation",
  "3": "intersection",
  "4": "uncertain",
};

const KEY_SHAPE = {
  q: "point",
  w: "segment",
  e: "box",
};

const state = {
  manifest: null,
  annotations: null,
  currentSeriesId: null,
  currentFrameKey: null,
  selectedType: "stenosis",
  selectedShape: "segment",
  selectedAnnotationId: null,
  image: null,
  mask: null,
  savedSerialized: "",
  draft: null,
  view: {
    scale: 1,
    offsetX: 0,
    offsetY: 0,
  },
  pan: {
    active: false,
    lastX: 0,
    lastY: 0,
  },
  history: {
    undoStack: [],
    redoStack: [],
  },
};

const el = {
  datasetName: document.getElementById("dataset-name"),
  seriesSelect: document.getElementById("series-select"),
  frameSelect: document.getElementById("frame-select"),
  prevFrame: document.getElementById("prev-frame"),
  nextFrame: document.getElementById("next-frame"),
  frameMeta: document.getElementById("frame-meta"),
  typeButtons: document.getElementById("type-buttons"),
  shapeButtons: document.getElementById("shape-buttons"),
  draftStatus: document.getElementById("draft-status"),
  showMask: document.getElementById("show-mask"),
  maskOpacity: document.getElementById("mask-opacity"),
  selectedStatus: document.getElementById("selected-status"),
  annotationNote: document.getElementById("annotation-note"),
  applyChanges: document.getElementById("apply-changes"),
  deleteAnnotation: document.getElementById("delete-annotation"),
  undoButton: document.getElementById("undo-button"),
  redoButton: document.getElementById("redo-button"),
  saveAll: document.getElementById("save-all"),
  reloadAll: document.getElementById("reload-all"),
  saveStatus: document.getElementById("save-status"),
  cursorStatus: document.getElementById("cursor-status"),
  annotationList: document.getElementById("annotation-list"),
  canvas: document.getElementById("viewer"),
};

const ctx = el.canvas.getContext("2d");

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function serializeAnnotations() {
  return JSON.stringify(state.annotations.frames);
}

function syncDirtyState() {
  const dirty = serializeAnnotations() !== state.savedSerialized;
  el.saveStatus.textContent = dirty ? "Есть несохранённые изменения" : "Изменений нет";
  return dirty;
}

function setSavedState() {
  state.savedSerialized = serializeAnnotations();
  syncDirtyState();
}

function pushHistory() {
  state.history.undoStack.push(deepClone(state.annotations));
  if (state.history.undoStack.length > 100) {
    state.history.undoStack.shift();
  }
  state.history.redoStack = [];
  updateHistoryButtons();
}

function updateHistoryButtons() {
  el.undoButton.disabled = state.history.undoStack.length === 0;
  el.redoButton.disabled = state.history.redoStack.length === 0;
}

function applySnapshot(snapshot) {
  state.annotations = deepClone(snapshot);
  if (!state.annotations.shape_types) {
    state.annotations.shape_types = Object.keys(SHAPE_META);
  }
  selectAnnotation(null);
  updateAnnotationList();
  updateHistoryButtons();
  syncDirtyState();
  draw();
}

function undo() {
  if (state.history.undoStack.length === 0) {
    return;
  }
  state.history.redoStack.push(deepClone(state.annotations));
  const snapshot = state.history.undoStack.pop();
  applySnapshot(snapshot);
}

function redo() {
  if (state.history.redoStack.length === 0) {
    return;
  }
  state.history.undoStack.push(deepClone(state.annotations));
  const snapshot = state.history.redoStack.pop();
  applySnapshot(snapshot);
}

function currentFrameAnnotations() {
  return state.annotations.frames[state.currentFrameKey].annotations;
}

function seriesFrames(seriesId) {
  const group = state.manifest.series_order.find((item) => item.series_id === seriesId);
  return group ? group.frame_keys : [];
}

function frameMeta(frameKey) {
  return state.manifest.frames.find((item) => item.frame_key === frameKey);
}

function setType(type, updateShape = true) {
  state.selectedType = type;
  if (updateShape) {
    state.selectedShape = TYPE_META[type].defaultShape;
  }
  updateTypeButtons();
  updateShapeButtons();
}

function setShape(shape) {
  state.selectedShape = shape;
  updateShapeButtons();
}

function updateTypeButtons() {
  el.typeButtons.innerHTML = "";
  for (const type of state.manifest.label_types) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `type-button${state.selectedType === type ? " active" : ""}`;
    button.textContent = TYPE_META[type].label;
    button.style.borderLeft = `0.5rem solid ${TYPE_META[type].color}`;
    button.addEventListener("click", () => setType(type, !state.selectedAnnotationId));
    el.typeButtons.appendChild(button);
  }
}

function updateShapeButtons() {
  el.shapeButtons.innerHTML = "";
  for (const shape of state.annotations.shape_types || Object.keys(SHAPE_META)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `shape-button${state.selectedShape === shape ? " active" : ""}`;
    button.textContent = SHAPE_META[shape].label;
    button.addEventListener("click", () => setShape(shape));
    el.shapeButtons.appendChild(button);
  }
  updateDraftStatus();
}

function resizeCanvas() {
  const rect = el.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  el.canvas.width = Math.max(1, Math.round(rect.width * dpr));
  el.canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function imageToScreen(x, y) {
  return {
    x: x * state.view.scale + state.view.offsetX,
    y: y * state.view.scale + state.view.offsetY,
  };
}

function screenToImage(clientX, clientY) {
  const rect = el.canvas.getBoundingClientRect();
  return {
    x: (clientX - rect.left - state.view.offsetX) / state.view.scale,
    y: (clientY - rect.top - state.view.offsetY) / state.view.scale,
  };
}

function fitImageToCanvas() {
  if (!state.image) {
    return;
  }
  const rect = el.canvas.getBoundingClientRect();
  const scale = Math.min(rect.width / state.image.width, rect.height / state.image.height) * 0.96;
  state.view.scale = scale;
  state.view.offsetX = (rect.width - state.image.width * scale) / 2;
  state.view.offsetY = (rect.height - state.image.height * scale) / 2;
}

function drawMask() {
  if (!state.mask || !el.showMask.checked) {
    return;
  }
  ctx.save();
  ctx.globalAlpha = Number(el.maskOpacity.value) / 100;
  ctx.drawImage(
    state.mask,
    state.view.offsetX,
    state.view.offsetY,
    state.mask.width * state.view.scale,
    state.mask.height * state.view.scale
  );
  ctx.globalCompositeOperation = "source-atop";
  ctx.fillStyle = "#34c6d3";
  ctx.fillRect(0, 0, el.canvas.clientWidth, el.canvas.clientHeight);
  ctx.restore();
}

function normalizeBox(geometry) {
  const x = Math.min(geometry.x, geometry.x + geometry.width);
  const y = Math.min(geometry.y, geometry.y + geometry.height);
  const width = Math.abs(geometry.width);
  const height = Math.abs(geometry.height);
  return { x, y, width, height };
}

function annotationAnchor(annotation) {
  const { geometry } = annotation;
  if (annotation.shape === "point") {
    return { x: geometry.x, y: geometry.y };
  }
  if (annotation.shape === "segment") {
    return geometry.points[0];
  }
  const box = normalizeBox(geometry);
  return { x: box.x, y: box.y };
}

function annotationCenter(annotation) {
  const { geometry } = annotation;
  if (annotation.shape === "point") {
    return { x: geometry.x, y: geometry.y };
  }
  if (annotation.shape === "segment") {
    return {
      x: (geometry.points[0].x + geometry.points[1].x) / 2,
      y: (geometry.points[0].y + geometry.points[1].y) / 2,
    };
  }
  const box = normalizeBox(geometry);
  return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

function distanceToSegment(point, a, b) {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const wx = point.x - a.x;
  const wy = point.y - a.y;
  const c1 = vx * wx + vy * wy;
  if (c1 <= 0) {
    return Math.hypot(point.x - a.x, point.y - a.y);
  }
  const c2 = vx * vx + vy * vy;
  if (c2 <= c1) {
    return Math.hypot(point.x - b.x, point.y - b.y);
  }
  const t = c1 / c2;
  const projX = a.x + t * vx;
  const projY = a.y + t * vy;
  return Math.hypot(point.x - projX, point.y - projY);
}

function hitTestAnnotation(imagePoint) {
  let best = null;
  let bestDistance = 16 / state.view.scale;

  for (const annotation of currentFrameAnnotations()) {
    let distance = Infinity;
    if (annotation.shape === "point") {
      distance = Math.hypot(annotation.geometry.x - imagePoint.x, annotation.geometry.y - imagePoint.y);
    } else if (annotation.shape === "segment") {
      distance = distanceToSegment(imagePoint, annotation.geometry.points[0], annotation.geometry.points[1]);
    } else if (annotation.shape === "box") {
      const box = normalizeBox(annotation.geometry);
      const inside =
        imagePoint.x >= box.x &&
        imagePoint.x <= box.x + box.width &&
        imagePoint.y >= box.y &&
        imagePoint.y <= box.y + box.height;
      if (inside) {
        distance = 0;
      } else {
        const cx = Math.max(box.x, Math.min(imagePoint.x, box.x + box.width));
        const cy = Math.max(box.y, Math.min(imagePoint.y, box.y + box.height));
        distance = Math.hypot(imagePoint.x - cx, imagePoint.y - cy);
      }
    }
    if (distance < bestDistance) {
      bestDistance = distance;
      best = annotation;
    }
  }
  return best;
}

function drawPoint(point, color, selected, label) {
  const p = imageToScreen(point.x, point.y);
  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, selected ? 8 : 6, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeStyle = selected ? "#fff" : "rgba(0,0,0,0.55)";
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(p.x - 10, p.y);
  ctx.lineTo(p.x + 10, p.y);
  ctx.moveTo(p.x, p.y - 10);
  ctx.lineTo(p.x, p.y + 10);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.font = "12px IBM Plex Sans, sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, p.x + 10, p.y - 10);
  ctx.restore();
}

function drawSegment(points, color, selected, label) {
  const a = imageToScreen(points[0].x, points[0].y);
  const b = imageToScreen(points[1].x, points[1].y);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.lineWidth = selected ? 6 : 4;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(a.x, a.y, selected ? 6 : 4, 0, Math.PI * 2);
  ctx.arc(b.x, b.y, selected ? 6 : 4, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  const center = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  ctx.font = "12px IBM Plex Sans, sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, center.x + 10, center.y - 10);
  ctx.restore();
}

function drawBox(geometry, color, selected, label) {
  const box = normalizeBox(geometry);
  const p = imageToScreen(box.x, box.y);
  ctx.save();
  ctx.fillStyle = `${color}22`;
  ctx.strokeStyle = color;
  ctx.lineWidth = selected ? 4 : 3;
  ctx.fillRect(p.x, p.y, box.width * state.view.scale, box.height * state.view.scale);
  ctx.strokeRect(p.x, p.y, box.width * state.view.scale, box.height * state.view.scale);
  ctx.font = "12px IBM Plex Sans, sans-serif";
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, p.x + 8, p.y - 8);
  ctx.restore();
}

function drawAnnotations() {
  for (const annotation of currentFrameAnnotations()) {
    const color = TYPE_META[annotation.type].color;
    const selected = annotation.id === state.selectedAnnotationId;
    const label = TYPE_META[annotation.type].label;
    if (annotation.shape === "point") {
      drawPoint(annotation.geometry, color, selected, label);
    } else if (annotation.shape === "segment") {
      drawSegment(annotation.geometry.points, color, selected, label);
    } else if (annotation.shape === "box") {
      drawBox(annotation.geometry, color, selected, label);
    }
  }
}

function drawDraft() {
  if (!state.draft) {
    return;
  }
  const color = TYPE_META[state.selectedType].color;
  if (state.draft.shape === "segment" && state.draft.start && state.draft.preview) {
    drawSegment([state.draft.start, state.draft.preview], color, false, "Черновик");
  } else if (state.draft.shape === "box" && state.draft.start && state.draft.preview) {
    drawBox(
      {
        x: state.draft.start.x,
        y: state.draft.start.y,
        width: state.draft.preview.x - state.draft.start.x,
        height: state.draft.preview.y - state.draft.start.y,
      },
      color,
      false,
      "Черновик"
    );
  }
}

function draw() {
  ctx.clearRect(0, 0, el.canvas.clientWidth, el.canvas.clientHeight);
  if (!state.image) {
    return;
  }
  ctx.drawImage(
    state.image,
    state.view.offsetX,
    state.view.offsetY,
    state.image.width * state.view.scale,
    state.image.height * state.view.scale
  );
  drawMask();
  drawAnnotations();
  drawDraft();
}

function annotationSummary(annotation) {
  if (annotation.shape === "point") {
    return `${annotation.geometry.x.toFixed(1)}, ${annotation.geometry.y.toFixed(1)}`;
  }
  if (annotation.shape === "segment") {
    const [a, b] = annotation.geometry.points;
    return `${a.x.toFixed(1)}, ${a.y.toFixed(1)} -> ${b.x.toFixed(1)}, ${b.y.toFixed(1)}`;
  }
  const box = normalizeBox(annotation.geometry);
  return `${box.x.toFixed(1)}, ${box.y.toFixed(1)} | ${box.width.toFixed(1)}x${box.height.toFixed(1)}`;
}

function updateAnnotationList() {
  el.annotationList.innerHTML = "";
  const annotations = currentFrameAnnotations();
  if (annotations.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "На этом кадре пока нет разметки";
    el.annotationList.appendChild(empty);
    return;
  }
  for (const annotation of annotations) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `annotation-chip${annotation.id === state.selectedAnnotationId ? " active" : ""}`;
    chip.textContent = `${TYPE_META[annotation.type].label} / ${SHAPE_META[annotation.shape].label}: ${annotationSummary(annotation)}`;
    chip.style.borderLeft = `0.5rem solid ${TYPE_META[annotation.type].color}`;
    chip.addEventListener("click", () => selectAnnotation(annotation.id));
    el.annotationList.appendChild(chip);
  }
}

function updateDraftStatus() {
  if (!state.draft) {
    el.draftStatus.textContent = `Текущий инструмент: ${SHAPE_META[state.selectedShape].label}`;
    return;
  }
  if (state.draft.shape === "segment") {
    el.draftStatus.textContent = "Отрезок: выберите вторую точку";
  } else if (state.draft.shape === "box") {
    el.draftStatus.textContent = "Область: выберите противоположный угол";
  }
}

function selectAnnotation(annotationId) {
  state.selectedAnnotationId = annotationId;
  const annotation = currentFrameAnnotations().find((item) => item.id === annotationId);
  if (!annotation) {
    el.selectedStatus.textContent = "Ничего не выбрано";
    el.annotationNote.value = "";
  } else {
    el.selectedStatus.textContent = `${TYPE_META[annotation.type].label} / ${SHAPE_META[annotation.shape].label}`;
    el.annotationNote.value = annotation.note || "";
    setType(annotation.type, false);
    setShape(annotation.shape);
  }
  updateAnnotationList();
  draw();
}

function clearDraft() {
  state.draft = null;
  updateDraftStatus();
  draw();
}

function createAnnotation(geometry) {
  pushHistory();
  const annotation = {
    id: `${state.currentFrameKey}_${Date.now()}`,
    type: state.selectedType,
    shape: state.selectedShape,
    geometry,
    note: "",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  currentFrameAnnotations().push(annotation);
  syncDirtyState();
  updateAnnotationList();
  selectAnnotation(annotation.id);
}

function createPoint(point) {
  createAnnotation({ shape: "point", x: Number(point.x.toFixed(2)), y: Number(point.y.toFixed(2)) });
}

function createSegment(a, b) {
  if (Math.hypot(a.x - b.x, a.y - b.y) < 2 / state.view.scale) {
    clearDraft();
    return;
  }
  createAnnotation({
    shape: "segment",
    points: [
      { x: Number(a.x.toFixed(2)), y: Number(a.y.toFixed(2)) },
      { x: Number(b.x.toFixed(2)), y: Number(b.y.toFixed(2)) },
    ],
  });
}

function createBox(a, b) {
  const width = b.x - a.x;
  const height = b.y - a.y;
  if (Math.abs(width) < 2 / state.view.scale || Math.abs(height) < 2 / state.view.scale) {
    clearDraft();
    return;
  }
  createAnnotation({
    shape: "box",
    x: Number(a.x.toFixed(2)),
    y: Number(a.y.toFixed(2)),
    width: Number(width.toFixed(2)),
    height: Number(height.toFixed(2)),
  });
}

function deleteSelectedAnnotation() {
  if (!state.selectedAnnotationId) {
    return;
  }
  const annotations = currentFrameAnnotations();
  const index = annotations.findIndex((item) => item.id === state.selectedAnnotationId);
  if (index >= 0) {
    pushHistory();
    annotations.splice(index, 1);
    syncDirtyState();
  }
  selectAnnotation(null);
  updateAnnotationList();
  draw();
}

function applySelectedChanges() {
  if (!state.selectedAnnotationId) {
    return;
  }
  const annotation = currentFrameAnnotations().find((item) => item.id === state.selectedAnnotationId);
  if (!annotation) {
    return;
  }
  pushHistory();
  annotation.type = state.selectedType;
  annotation.note = el.annotationNote.value.trim();
  annotation.updated_at = new Date().toISOString();
  syncDirtyState();
  selectAnnotation(annotation.id);
}

async function saveAll() {
  const response = await fetch("/api/annotations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state.annotations),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || "Ошибка сохранения");
  }
  setSavedState();
  el.saveStatus.textContent = `Сохранено: ${data.saved_to}`;
}

async function loadFrame(frameKey, resetView = false) {
  state.currentFrameKey = frameKey;
  clearDraft();
  const meta = frameMeta(frameKey);
  const image = new Image();
  const mask = new Image();
  image.src = `/data/${meta.image_rel_path}`;
  const maskPromise = meta.mask_rel_path
    ? new Promise((resolve) => {
        mask.onload = resolve;
        mask.onerror = resolve;
        mask.src = `/data/${meta.mask_rel_path}`;
      })
    : Promise.resolve();
  await image.decode();
  await maskPromise;
  state.image = image;
  state.mask = meta.mask_rel_path ? mask : null;
  if (resetView || state.view.scale <= 0) {
    fitImageToCanvas();
  }
  el.frameSelect.value = frameKey;
  el.frameMeta.textContent = `${meta.patient_id} / серия ${meta.series_id} / кадр ${meta.frame_id}`;
  selectAnnotation(null);
  updateAnnotationList();
  draw();
}

function renderFrameSelect() {
  el.frameSelect.innerHTML = "";
  for (const frameKey of seriesFrames(state.currentSeriesId)) {
    const option = document.createElement("option");
    option.value = frameKey;
    option.textContent = frameMeta(frameKey).frame_id;
    el.frameSelect.appendChild(option);
  }
}

async function setSeries(seriesId, preferredFrameKey = null) {
  state.currentSeriesId = seriesId;
  renderFrameSelect();
  const frames = seriesFrames(seriesId);
  const target = preferredFrameKey && frames.includes(preferredFrameKey) ? preferredFrameKey : frames[0];
  await loadFrame(target, true);
}

async function loadApp() {
  const [manifestResponse, annotationsResponse] = await Promise.all([
    fetch("/api/manifest"),
    fetch("/api/annotations"),
  ]);
  state.manifest = await manifestResponse.json();
  state.annotations = await annotationsResponse.json();
  if (!state.annotations.shape_types) {
    state.annotations.shape_types = Object.keys(SHAPE_META);
  }

  el.datasetName.textContent = state.manifest.dataset_dir;
  el.seriesSelect.innerHTML = "";
  for (const series of state.manifest.series_order) {
    const option = document.createElement("option");
    option.value = series.series_id;
    option.textContent = `Серия ${series.series_id}`;
    el.seriesSelect.appendChild(option);
  }

  setType(state.selectedType);
  updateHistoryButtons();
  await setSeries(state.manifest.series_order[0].series_id);
  resizeCanvas();
  setSavedState();
}

function moveFrame(step) {
  const frames = seriesFrames(state.currentSeriesId);
  const index = frames.indexOf(state.currentFrameKey);
  const target = frames[Math.max(0, Math.min(frames.length - 1, index + step))];
  if (target && target !== state.currentFrameKey) {
    loadFrame(target, false);
  }
}

function handleCanvasAction(imagePoint) {
  if (!state.image) {
    return;
  }
  if (imagePoint.x < 0 || imagePoint.y < 0 || imagePoint.x > state.image.width || imagePoint.y > state.image.height) {
    return;
  }

  if (!state.draft) {
    const hit = hitTestAnnotation(imagePoint);
    if (hit) {
      selectAnnotation(hit.id);
      return;
    }
  }

  if (state.selectedShape === "point") {
    createPoint(imagePoint);
    return;
  }

  if (!state.draft) {
    state.draft = {
      shape: state.selectedShape,
      start: { x: imagePoint.x, y: imagePoint.y },
      preview: { x: imagePoint.x, y: imagePoint.y },
    };
    updateDraftStatus();
    draw();
    return;
  }

  if (state.draft.shape === "segment") {
    createSegment(state.draft.start, imagePoint);
  } else if (state.draft.shape === "box") {
    createBox(state.draft.start, imagePoint);
  }
  clearDraft();
}

el.seriesSelect.addEventListener("change", () => setSeries(el.seriesSelect.value));
el.frameSelect.addEventListener("change", () => loadFrame(el.frameSelect.value, false));
el.prevFrame.addEventListener("click", () => moveFrame(-1));
el.nextFrame.addEventListener("click", () => moveFrame(1));
el.showMask.addEventListener("change", draw);
el.maskOpacity.addEventListener("input", draw);
el.applyChanges.addEventListener("click", applySelectedChanges);
el.deleteAnnotation.addEventListener("click", deleteSelectedAnnotation);
el.undoButton.addEventListener("click", undo);
el.redoButton.addEventListener("click", redo);
el.saveAll.addEventListener("click", async () => {
  try {
    await saveAll();
  } catch (error) {
    el.saveStatus.textContent = String(error);
  }
});
el.reloadAll.addEventListener("click", async () => {
  state.annotations = await (await fetch("/api/annotations")).json();
  if (!state.annotations.shape_types) {
    state.annotations.shape_types = Object.keys(SHAPE_META);
  }
  state.history.undoStack = [];
  state.history.redoStack = [];
  updateHistoryButtons();
  await loadFrame(state.currentFrameKey, false);
  setSavedState();
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", (event) => {
  if (syncDirtyState()) {
    event.preventDefault();
    event.returnValue = "";
  }
});

el.canvas.addEventListener("contextmenu", (event) => event.preventDefault());

el.canvas.addEventListener("mousedown", (event) => {
  if (event.button === 2 || event.shiftKey) {
    state.pan.active = true;
    state.pan.lastX = event.clientX;
    state.pan.lastY = event.clientY;
    return;
  }
  handleCanvasAction(screenToImage(event.clientX, event.clientY));
});

window.addEventListener("mouseup", () => {
  state.pan.active = false;
});

window.addEventListener("mousemove", (event) => {
  const imagePoint = screenToImage(event.clientX, event.clientY);
  el.cursorStatus.textContent = `x: ${imagePoint.x.toFixed(1)}, y: ${imagePoint.y.toFixed(1)}`;

  if (state.pan.active) {
    state.view.offsetX += event.clientX - state.pan.lastX;
    state.view.offsetY += event.clientY - state.pan.lastY;
    state.pan.lastX = event.clientX;
    state.pan.lastY = event.clientY;
    draw();
    return;
  }

  if (state.draft) {
    state.draft.preview = imagePoint;
    draw();
  }
});

el.canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = el.canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const preX = (mouseX - state.view.offsetX) / state.view.scale;
  const preY = (mouseY - state.view.offsetY) / state.view.scale;
  const factor = event.deltaY < 0 ? 1.1 : 0.9;
  state.view.scale = Math.max(0.1, Math.min(20, state.view.scale * factor));
  state.view.offsetX = mouseX - preX * state.view.scale;
  state.view.offsetY = mouseY - preY * state.view.scale;
  draw();
});

window.addEventListener("keydown", (event) => {
  if (event.target === el.annotationNote) {
    return;
  }

  const lowerKey = event.key.toLowerCase();
  if ((event.ctrlKey || event.metaKey) && lowerKey === "z" && !event.shiftKey) {
    event.preventDefault();
    undo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && (lowerKey === "y" || (lowerKey === "z" && event.shiftKey))) {
    event.preventDefault();
    redo();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && lowerKey === "s") {
    event.preventDefault();
    saveAll().catch((error) => {
      el.saveStatus.textContent = String(error);
    });
    return;
  }

  if (KEY_TYPE[event.key]) {
    setType(KEY_TYPE[event.key], !state.selectedAnnotationId);
    return;
  }
  if (KEY_SHAPE[lowerKey]) {
    setShape(KEY_SHAPE[lowerKey]);
    clearDraft();
    return;
  }

  if (event.key === "ArrowLeft") {
    moveFrame(-1);
  } else if (event.key === "ArrowRight") {
    moveFrame(1);
  } else if (event.key === "Delete" || event.key === "Backspace") {
    deleteSelectedAnnotation();
  } else if (event.key === "Escape") {
    clearDraft();
    selectAnnotation(null);
  } else if (lowerKey === "f") {
    fitImageToCanvas();
    draw();
  }
});

loadApp().catch((error) => {
  el.saveStatus.textContent = `Ошибка загрузки: ${error}`;
});
