import * as THREE from '/vendor/three.module.js';
import { OrbitControls } from '/vendor/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from '/vendor/examples/jsm/loaders/PLYLoader.js';

const state = {
  manifest: null,
  selected: null,
  currentObject: null,
  gridVisible: true,
  volumeMeta: null,
};

const viewerEl = document.getElementById('viewer');
const artifactListEl = document.getElementById('artifact-list');
const statusEl = document.getElementById('status');
const selectionTitleEl = document.getElementById('selection-title');
const volumeControlsEl = document.getElementById('volume-controls');
const thresholdRangeEl = document.getElementById('threshold-range');
const thresholdNumberEl = document.getElementById('threshold-number');
const maxPointsEl = document.getElementById('max-points');
const reloadVolumeEl = document.getElementById('reload-volume');
const volumeMetaEl = document.getElementById('volume-meta');
const filterInputEl = document.getElementById('filter-input');
const fitViewEl = document.getElementById('fit-view');
const toggleGridEl = document.getElementById('toggle-grid');
const uploadInputEl = document.getElementById('upload-input');
const dropZoneEl = document.getElementById('drop-zone');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewerEl.clientWidth || 800, viewerEl.clientHeight || 600);
viewerEl.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x09111a);

const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(1.8, 1.5, 1.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(2, 2, 3);
scene.add(dirLight);

const grid = new THREE.GridHelper(2.0, 20, 0x335577, 0x1c2a39);
scene.add(grid);
const axes = new THREE.AxesHelper(0.8);
scene.add(axes);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

function setStatus(text) {
  statusEl.textContent = text;
}

function clearCurrentObject() {
  if (!state.currentObject) return;
  scene.remove(state.currentObject);
  if (state.currentObject.geometry) state.currentObject.geometry.dispose();
  if (state.currentObject.material) {
    if (Array.isArray(state.currentObject.material)) state.currentObject.material.forEach((m) => m.dispose());
    else state.currentObject.material.dispose();
  }
  state.currentObject = null;
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const fov = camera.fov * (Math.PI / 180);
  let distance = maxDim / (2 * Math.tan(fov / 2));
  distance *= 2.1;
  camera.position.copy(center.clone().add(new THREE.Vector3(distance, distance, distance)));
  controls.target.copy(center);
  camera.near = Math.max(0.001, distance / 100);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

async function loadManifest() {
  const response = await fetch('/api/manifest');
  state.manifest = await response.json();
  renderArtifactList();
  setStatus(`Найдено файлов: ${state.manifest.artifacts.length}`);
}

function renderArtifactList() {
  const filter = filterInputEl.value.trim().toLowerCase();
  artifactListEl.innerHTML = '';
  const artifacts = state.manifest.artifacts.filter((item) => item.rel_path.toLowerCase().includes(filter));
  for (const item of artifacts) {
    const button = document.createElement('button');
    button.className = 'artifact-item';
    if (state.selected && state.selected.rel_path === item.rel_path) button.classList.add('active');
    button.innerHTML = `
      <div class="artifact-title">${item.rel_path}</div>
      <div class="artifact-meta">${item.kind} · ${item.size_mb} MB</div>
    `;
    button.addEventListener('click', () => selectArtifact(item));
    artifactListEl.appendChild(button);
  }
}

async function selectArtifact(item) {
  state.selected = item;
  renderArtifactList();
  selectionTitleEl.textContent = item.rel_path;
  clearCurrentObject();
  volumeMetaEl.textContent = '';
  if (item.kind === 'ply') {
    volumeControlsEl.hidden = true;
    await loadPly(item);
  } else if (item.kind === 'volume') {
    volumeControlsEl.hidden = false;
    await prepareVolume(item);
    await loadVolumePoints(item);
  } else {
    volumeControlsEl.hidden = true;
    setStatus(`Файл не поддержан: ${item.rel_path}`);
  }
}

async function loadPly(item) {
  setStatus(`Загрузка PLY: ${item.rel_path}`);
  const loader = new PLYLoader();
  const geometry = await loader.loadAsync(item.url);
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  const size = box ? box.getSize(new THREE.Vector3()) : new THREE.Vector3(1, 1, 1);
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const pointSize = Math.max(maxDim / 160, 0.06);
  const material = new THREE.PointsMaterial({
    size: pointSize,
    color: 0x6fd3ff,
    sizeAttenuation: true,
  });
  const object = new THREE.Points(geometry, material);
  state.currentObject = object;
  scene.add(object);
  fitCameraToObject(object);
  setStatus(`Открыт PLY: ${item.rel_path}`);
}

async function prepareVolume(item) {
  const response = await fetch(`/api/volume_meta?path=${encodeURIComponent(item.rel_path)}`);
  const payload = await response.json();
  state.volumeMeta = payload.summary;
  const min = payload.summary.min;
  const max = payload.summary.max;
  const q = payload.summary.quantiles || {};
  const suggested = q['0.995'] ?? ((min + max) / 2);
  thresholdRangeEl.min = String(min);
  thresholdRangeEl.max = String(max);
  thresholdRangeEl.step = String((max - min) / 1000 || 0.000001);
  thresholdRangeEl.value = String(suggested);
  thresholdNumberEl.value = String(suggested);
  volumeMetaEl.textContent = `shape: ${payload.summary.shape.join('x')}\ndtype: ${payload.summary.dtype}\nmin: ${min}\nmax: ${max}\np99.5: ${q['0.995'] ?? 'n/a'}`;
}

async function loadVolumePoints(item) {
  const threshold = Number(thresholdNumberEl.value);
  const maxPoints = Number(maxPointsEl.value || 120000);
  setStatus(`Пересчёт объёма: ${item.rel_path}`);
  const response = await fetch(`/api/volume_points?path=${encodeURIComponent(item.rel_path)}&threshold=${encodeURIComponent(threshold)}&max_points=${encodeURIComponent(maxPoints)}`);
  const payload = await response.json();
  if (payload.error) {
    setStatus(`Ошибка: ${payload.error}`);
    return;
  }
  const positions = new Float32Array(payload.num_points_returned * 3);
  const colors = new Float32Array(payload.num_points_returned * 3);
  const min = payload.min;
  const max = payload.max;
  const span = max - min || 1;
  payload.points.forEach((point, index) => {
    positions[index * 3 + 0] = point[0];
    positions[index * 3 + 1] = point[1];
    positions[index * 3 + 2] = point[2];
    const t = (point[3] - min) / span;
    colors[index * 3 + 0] = 1.0;
    colors[index * 3 + 1] = 0.35 + 0.5 * t;
    colors[index * 3 + 2] = 0.1 + 0.8 * (1.0 - t);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });
  const object = new THREE.Points(geometry, material);
  state.currentObject = object;
  scene.add(object);
  fitCameraToObject(object);
  volumeMetaEl.textContent += `\nthreshold: ${payload.threshold}\nточек: ${payload.num_points_returned} / ${payload.num_points_total}`;
  setStatus(`Открыт объём: ${item.rel_path}`);
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  for (const file of files) {
    setStatus(`Загрузка файла: ${file.name}`);
    const body = new FormData();
    body.append('file', file);
    const response = await fetch('/api/upload', { method: 'POST', body });
    const payload = await response.json();
    if (!response.ok || payload.error) {
      setStatus(`Ошибка загрузки ${file.name}: ${payload.error || response.statusText}`);
      continue;
    }
    state.manifest = payload.manifest;
    renderArtifactList();
    setStatus(`Файл загружен: ${payload.artifact.rel_path}`);
    await selectArtifact(payload.artifact);
  }
}

function attachDropZone() {
  const stop = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
    dropZoneEl.addEventListener(name, stop);
  });
  ['dragenter', 'dragover'].forEach((name) => {
    dropZoneEl.addEventListener(name, () => dropZoneEl.classList.add('active'));
  });
  ['dragleave', 'drop'].forEach((name) => {
    dropZoneEl.addEventListener(name, () => dropZoneEl.classList.remove('active'));
  });
  dropZoneEl.addEventListener('drop', async (event) => {
    if (event.dataTransfer?.files?.length) {
      await uploadFiles(event.dataTransfer.files);
    }
  });
}

thresholdRangeEl.addEventListener('input', () => {
  thresholdNumberEl.value = thresholdRangeEl.value;
});
thresholdNumberEl.addEventListener('change', () => {
  thresholdRangeEl.value = thresholdNumberEl.value;
});
reloadVolumeEl.addEventListener('click', () => {
  if (state.selected && state.selected.kind === 'volume') loadVolumePoints(state.selected);
});
filterInputEl.addEventListener('input', renderArtifactList);
fitViewEl.addEventListener('click', () => {
  if (state.currentObject) fitCameraToObject(state.currentObject);
});
toggleGridEl.addEventListener('click', () => {
  state.gridVisible = !state.gridVisible;
  grid.visible = state.gridVisible;
  axes.visible = state.gridVisible;
});
uploadInputEl.addEventListener('change', async () => {
  if (uploadInputEl.files?.length) {
    await uploadFiles(uploadInputEl.files);
    uploadInputEl.value = '';
  }
});
window.addEventListener('resize', () => {
  const width = viewerEl.clientWidth || window.innerWidth;
  const height = viewerEl.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
});

attachDropZone();
await loadManifest();
const preselect = state.manifest.artifacts.find((item) => item.rel_path.endsWith('point_cloud.ply')) || state.manifest.artifacts[0];
if (preselect) await selectArtifact(preselect);
