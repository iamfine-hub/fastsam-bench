'use strict';

/* ------------------------------------------------------------------ *
 * FastSAM 检测台前端：单画布合成（原图 + 服务端掩码叠加 + 本地标注）
 * 坐标约定：S.boxes / S.points 一律存「原图坐标」，显示时按视口缩放。
 * ------------------------------------------------------------------ */

const $ = (sel) => document.querySelector(sel);

const canvas = $('#view');
const ctx = canvas.getContext('2d');
const wrap = $('#wrap');

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 16;

const S = {
  imageId: null,
  img: null,
  overlay: null,
  imgW: 0,
  imgH: 0,
  mode: 'everything',
  boxes: [],        // {x,y,w,h} 原图坐标
  pendingBoxes: null, // 深链预置框，等图片载入后再应用
  points: [],       // {x,y,label}
  detections: [],
  busy: false,
  queued: null,     // 忙时排队的下一次调用
  drag: null,       // 正在拖的框
  pointer: null,    // pointerdown 状态
  view: { s: 1, tx: 0, ty: 0 }, // 用户缩放/平移，叠加在「适应窗口」之上
  pan: null,
  space: false,
  hover: null,      // 笔刷光标位置（画布坐标，用于画大小圈）
  brush: {          // 修掩码笔刷
    value: 1,       // 1 添加 / 0 擦除
    radius: 24,     // 原图像素
    strokes: [],    // 已提交的笔迹 {value, radius, points:[[x,y]]}
    current: null,  // 正在画的笔迹
  },
};

const HINTS = {
  everything: '点「分割」，模型把整张图切成若干实例。滚轮缩放，空格或中键拖动平移，双击复位。',
  box: '在图上拖动画框，松开即匹配；可画多个框，右键撤销上一个。需先在全图模式分割一次。',
  point: '左键点目标（正样本），右键或 Shift+左键点背景（负样本）；点旁的 ✕ 可删掉误加的点。需先分割一次。',
  brush: '直接在图上涂抹来修掩码：白色是补上、深色是擦掉，笔迹可逐笔撤销。需先有分割结果。',
};

/* ------------------------------ 工具 ------------------------------ */

function params() {
  return {
    imgsz: +$('#imgsz').value,
    conf: +$('#conf').value,
    iou: +$('#iou').value,
    contours: $('#contours').checked,
    better_quality: $('#better').checked,
  };
}

// 服务端缓存被手工清过时会报「图片已过期」。这时候不该让用户自己重选：
// 记住的来源（本地文件 / 示例图名）还在，重建一次 image_id 再重试就好。
async function revive() {
  if (!S.source) return false;
  try {
    if (S.source.kind === 'sample') {
      const info = await api('/api/sample', { name: S.source.name }, 'POST', false);
      S.imageId = info.image_id;
      return true;
    }
    const form = new FormData();
    form.append('file', S.source.file, S.source.file.name);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) return false;
    const info = await res.json();
    S.imageId = info.image_id;
    return true;
  } catch (_) {
    return false;
  }
}

async function api(path, body, method = 'POST', allowRevive = true) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = `请求失败（${res.status}）`;
    try {
      const data = await res.json();
      if (data && data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch (_) { /* ignore */ }
    if (allowRevive && String(detail).includes('过期') && (await revive())) {
      const next = body && body.image_id ? { ...body, image_id: S.imageId } : body;
      return api(path, next, method, false);
    }
    throw new Error(detail);
  }
  return res.json();
}

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add('hidden'), 3200);
}

function setBusy(on, text) {
  S.busy = on;
  $('#busy').classList.toggle('hidden', !on);
  if (text) $('#busy-text').textContent = text;
}

/* ---------------------------- 画布几何 ---------------------------- */

function viewRect() {
  const rect = wrap.getBoundingClientRect();
  const W = Math.max(1, rect.width);
  const H = Math.max(1, rect.height);
  if (!S.imgW || !S.imgH) return { x: 0, y: 0, w: W, h: H, s: 1 };
  const fit = Math.min(W / S.imgW, H / S.imgH);
  const s = fit * S.view.s;
  const w = S.imgW * s;
  const h = S.imgH * s;
  return { x: (W - w) / 2 + S.view.tx, y: (H - h) / 2 + S.view.ty, w, h, s };
}

function toImage(ev) {
  const rect = wrap.getBoundingClientRect();
  const v = viewRect();
  const px = ev.clientX - rect.left;
  const py = ev.clientY - rect.top;
  const x = (px - v.x) / (v.s || 1);
  const y = (py - v.y) / (v.s || 1);
  return {
    x: Math.max(0, Math.min(S.imgW, x)),
    y: Math.max(0, Math.min(S.imgH, y)),
  };
}

function resetView() {
  S.view = { s: 1, tx: 0, ty: 0 };
  draw();
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  const W = Math.max(1, rect.width);
  const H = Math.max(1, rect.height);

  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (!S.img) return;

  const v = viewRect();
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(S.img, v.x, v.y, v.w, v.h);

  if (S.overlay) {
    ctx.drawImage(S.overlay, v.x, v.y, v.w, v.h);
  }

  const sx = v.s;
  const sy = v.s;

  // 已确认的框：白色虚线 + 极淡填充
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  S.boxes.forEach((b, i) => {
    ctx.fillStyle = 'rgba(255,255,255,.06)';
    ctx.strokeStyle = 'rgba(255,255,255,.92)';
    ctx.fillRect(v.x + b.x * sx, v.y + b.y * sy, b.w * sx, b.h * sy);
    ctx.strokeRect(v.x + b.x * sx, v.y + b.y * sy, b.w * sx, b.h * sy);
    label(`#${i + 1}`, v.x + b.x * sx, v.y + b.y * sy - 6);
  });

  // 正在拖的框
  if (S.drag) {
    const b = normBox(S.drag);
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(255,255,255,.7)';
    ctx.fillStyle = 'rgba(255,255,255,.08)';
    ctx.fillRect(v.x + b.x * sx, v.y + b.y * sy, b.w * sx, b.h * sy);
    ctx.strokeRect(v.x + b.x * sx, v.y + b.y * sy, b.w * sx, b.h * sy);
  }
  ctx.setLineDash([]);

  // 提示点：正样本实心、负样本空心
  S.points.forEach((p, i) => {
    const cx = v.x + p.x * sx;
    const cy = v.y + p.y * sy;
    const positive = p.label === 1;
    ctx.beginPath();
    ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
    ctx.fillStyle = positive ? 'rgba(240,240,240,.95)' : 'rgba(13,13,13,.85)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = positive ? 'rgba(13,13,13,.85)' : 'rgba(240,240,240,.95)';
    ctx.stroke();

    ctx.beginPath();
    ctx.strokeStyle = positive ? '#111' : '#f0f0f0';
    ctx.lineWidth = 1.6;
    ctx.moveTo(cx - 3.2, cy);
    ctx.lineTo(cx + 3.2, cy);
    if (positive) {
      ctx.moveTo(cx, cy - 3.2);
      ctx.lineTo(cx, cy + 3.2);
    }
    ctx.stroke();
    label(String(i + 1), cx + 9, cy - 9);
  });

  // 修掩码：正在画的笔迹 + 笔刷大小圈
  if (S.brush.current) {
    const cur = S.brush.current;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(1, cur.radius * 2 * sx);
    ctx.strokeStyle = cur.value ? 'rgba(255,255,255,.55)' : 'rgba(8,8,8,.55)';
    ctx.beginPath();
    cur.points.forEach((pt, i) => {
      const x = v.x + pt[0] * sx;
      const y = v.y + pt[1] * sy;
      if (i === 0) {
        ctx.moveTo(x, y);
        ctx.lineTo(x + 0.01, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.lineCap = 'butt';
  }

  if (S.mode === 'brush' && S.hover) {
    const cx = v.x + S.hover.x * sx;
    const cy = v.y + S.hover.y * sy;
    const r = Math.max(2, S.brush.radius * sx);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(13,13,13,.9)';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, r - 1, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(240,240,240,.9)';
    ctx.stroke();
  }
}

function label(text, x, y) {
  // 必须保存/还原绘图状态：这里会把 fillStyle 改成不透明色，
  // 泄漏出去会让下一个框的 fillRect 用上实色，把整个框涂白。
  ctx.save();
  ctx.font = '11px ui-monospace, Consolas, monospace';
  const w = ctx.measureText(text).width + 8;
  ctx.fillStyle = 'rgba(8,8,8,.82)';
  ctx.fillRect(x, y - 12, w, 15);
  ctx.fillStyle = '#e6e6e6';
  ctx.fillText(text, x + 4, y);
  ctx.restore();
}

function normBox(d) {
  const x = Math.min(d.x0, d.x1);
  const y = Math.min(d.y0, d.y1);
  return { x, y, w: Math.abs(d.x1 - d.x0), h: Math.abs(d.y1 - d.y0) };
}

/* ---------------------------- 状态更新 ---------------------------- */

function updateStatus(res) {
  $('#size').textContent = S.imgW ? `${S.imgW} × ${S.imgH}` : '—';
  const count = res ? res.count : (S.detections.length || 0);
  $('#stat-count').textContent = res ? `${count} 个实例` : '—';
  if (res) {
    const ms = res.elapsed_ms || 0;
    const suffix = res.cached && res.elapsed_ms ? '（缓存）' : '';
    $('#stat-time').textContent = ms ? `${ms} ms${suffix}` : '即时匹配';
  } else {
    $('#stat-time').textContent = '—';
  }
  const ready = !!(S.imageId && S.detections.length);
  $('#export-boxes').disabled = !ready;
  $('#export-ann').disabled = !ready;
  $('#copy').disabled = !ready;
  $('#download').disabled = !ready;
}

/* ---------------------------- 提示点列表 ---------------------------- */

function removePoint(i) {
  if (i < 0 || i >= S.points.length) return;
  S.points.splice(i, 1);
  renderPoints();
  draw();
  if (S.points.length) {
    run('point');
  } else {
    S.overlay = null;
    S.detections = [];
    updateStatus(null);
  }
}

function renderPoints() {
  const box = $('#points');
  if (!box) return;
  box.innerHTML = '';
  box.classList.toggle('hidden', S.points.length === 0);
  S.points.forEach((p, i) => {
    const chip = document.createElement('span');
    chip.className = 'pt ' + (p.label === 1 ? 'pos' : 'neg');
    const tag = document.createElement('b');
    tag.textContent = (p.label === 1 ? '+' : '−') + (i + 1);
    const del = document.createElement('button');
    del.type = 'button';
    del.textContent = '✕';
    del.title = '删除这个提示点';
    del.addEventListener('click', () => removePoint(i));
    chip.append(tag, del);
    box.appendChild(chip);
  });
}

function setMode(mode) {
  S.mode = mode;
  S.drag = null;
  S.brush.current = null;
  S.pointer = null;
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === mode));
  $('#hint').textContent = HINTS[mode];
  canvas.classList.toggle('mode-box', mode === 'box');
  canvas.classList.toggle('mode-point', mode === 'point');
  canvas.classList.toggle('mode-brush', mode === 'brush');
  $('#brushbar').classList.toggle('hidden', mode !== 'brush');
  $('#run').style.display = mode === 'everything' ? '' : 'none';
  syncBrushButtons();
  draw();
}

/* ---------------------------- 修掩码笔刷 ---------------------------- */

function syncBrushButtons() {
  const has = S.brush.strokes.length > 0;
  $('#brush-undo').disabled = !has;
  $('#brush-clear').disabled = !has;
  $('#brush-add').classList.toggle('on', S.brush.value === 1);
  $('#brush-erase').classList.toggle('on', S.brush.value === 0);
  $('#v-brush').textContent = `${S.brush.radius} px`;
}

function clearBrush() {
  S.brush.strokes = [];
  S.brush.current = null;
  syncBrushButtons();
}

async function pushBrush() {
  if (!S.imageId) return;
  if (!S.brush.strokes.length) {
    // 没有笔迹 = 回到提示的原始结果，交给服务端按空笔迹重算
    S.brush.strokes = [];
  }
  try {
    const res = await api('/api/brush', {
      image_id: S.imageId,
      strokes: S.brush.strokes,
      contours: $('#contours').checked,
    });
    S.overlay = await loadOverlay(res.overlay);
    S.detections = res.detections || [];
    updateStatus(res);
  } catch (err) {
    toast(err.message);
  } finally {
    draw();
  }
}

/* ---------------------------- 推理调用 ---------------------------- */

async function run(kind) {
  if (!S.imageId) {
    toast('先选一张图片');
    return;
  }
  if (S.busy) {
    S.queued = kind;
    return;
  }

  S.busy = true;
  setBusy(true, kind === 'everything' ? '全图分割中…' : '匹配中…');

  // 新的提示结果会把之前的画笔修整覆盖掉，笔迹一起清空
  clearBrush();

  try {
    const p = params();
    let res;
    if (kind === 'everything') {
      res = await api('/api/everything', { image_id: S.imageId, ...p });
      S.overlay = await loadOverlay(res.overlay);
    } else if (kind === 'box') {
      if (!S.boxes.length) { setBusy(false); S.busy = false; return; }
      res = await api('/api/box', {
        image_id: S.imageId,
        boxes: S.boxes.map((b) => [b.x, b.y, b.x + b.w, b.y + b.h]),
        contours: p.contours,
        better_quality: p.better_quality,
      });
      S.overlay = res.hit ? await loadOverlay(res.overlay) : null;
      if (!res.hit) toast('框内没有命中任何实例');
    } else {
      if (!S.points.length) { setBusy(false); S.busy = false; return; }
      res = await api('/api/point', {
        image_id: S.imageId,
        points: S.points.map((pt) => [pt.x, pt.y]),
        labels: S.points.map((pt) => pt.label),
        contours: p.contours,
        better_quality: p.better_quality,
      });
      S.overlay = res.hit ? await loadOverlay(res.overlay) : null;
      if (!res.hit) toast('这些点没有落在任何实例上');
    }
    S.detections = res.detections || [];
    updateStatus(res);
  } catch (err) {
    toast(err.message);
  } finally {
    S.busy = false;
    setBusy(false);
    draw();
    const queued = S.queued;
    S.queued = null;
    if (queued) run(queued);
  }
}

function loadOverlay(dataUrl) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('叠加图加载失败'));
    im.src = dataUrl;
  });
}

/* ---------------------------- 图片加载 ---------------------------- */

async function loadImage(info) {
  S.imageId = info.image_id;
  S.imgW = info.width;
  S.imgH = info.height;
  S.boxes = [];
  S.points = [];
  S.detections = [];
  S.overlay = null;
  S.drag = null;
  S.view = { s: 1, tx: 0, ty: 0 };
  clearBrush();
  renderPoints();

  $('#filename').textContent = info.name || '未命名';
  updateStatus(null);

  const im = new Image();
  im.onload = async () => {
    S.img = im;
    $('#empty').classList.add('hidden');
    updateStatus(null);
    draw();
    // 载入图片只做载入，不自动分割：分割由用户点「分割」或直接画框/点选/修掩码触发
    if (S.pendingBoxes && S.pendingBoxes.length) {
      S.boxes = S.pendingBoxes;
      S.pendingBoxes = null;
      setMode('box');
      run('box');
    }
  };
  im.onerror = () => toast('图片加载失败');
  im.src = `/api/image/${info.image_id}?t=${Date.now()}`;
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    toast('只支持图片文件');
    return;
  }
  const form = new FormData();
  form.append('file', file, file.name);
  try {
    setBusy(true, '上传中…');
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`上传失败（${res.status}）`);
    const info = await res.json();
    S.source = { kind: 'file', file };
    await loadImage(info);
    if (info.scaled) toast(`图片过大，已缩放到 ${info.width}×${info.height}`);
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

/* ---------------------------- 事件绑定 ---------------------------- */

canvas.addEventListener('pointerdown', (ev) => {
  if (!S.img || !S.imageId) return;
  ev.preventDefault();
  canvas.setPointerCapture(ev.pointerId);

  // 中键或按住空格 → 平移画布
  if (ev.button === 1 || (ev.button === 0 && S.space)) {
    S.pan = { cx: ev.clientX, cy: ev.clientY, tx: S.view.tx, ty: S.view.ty };
    canvas.classList.add('panning');
    return;
  }

  const p = toImage(ev);
  S.pointer = { x: p.x, y: p.y, cx: ev.clientX, cy: ev.clientY, moved: false, button: ev.button };
  if (S.mode === 'box') {
    S.drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  } else if (S.mode === 'brush') {
    if (!S.detections.length && !S.overlay) {
      toast('先跑一次分割，再来修掩码');
      return;
    }
    const erase = ev.button === 2 || ev.shiftKey;
    S.brush.current = {
      value: erase ? 0 : S.brush.value,
      radius: S.brush.radius,
      points: [[p.x, p.y]],
    };
    draw();
  }
});

canvas.addEventListener('pointermove', (ev) => {
  if (S.pan) {
    S.view.tx = S.pan.tx + (ev.clientX - S.pan.cx);
    S.view.ty = S.pan.ty + (ev.clientY - S.pan.cy);
    draw();
    return;
  }

  if (S.mode === 'brush') {
    S.hover = toImage(ev);
    const cur = S.brush.current;
    if (cur) {
      const p = S.hover;
      const last = cur.points[cur.points.length - 1];
      if (!last || Math.hypot(p.x - last[0], p.y - last[1]) > 1.2) cur.points.push([p.x, p.y]);
    }
    draw();
    return;
  }

  if (!S.pointer) return;
  if (Math.abs(ev.clientX - S.pointer.cx) + Math.abs(ev.clientY - S.pointer.cy) > 4) {
    S.pointer.moved = true;
  }
  if (S.mode === 'box' && S.drag) {
    const p = toImage(ev);
    S.drag.x1 = p.x;
    S.drag.y1 = p.y;
    draw();
  }
});

canvas.addEventListener('pointerup', (ev) => {
  if (S.pan) {
    S.pan = null;
    canvas.classList.remove('panning');
    try { canvas.releasePointerCapture(ev.pointerId); } catch (_) { /* ignore */ }
    return;
  }

  if (S.mode === 'brush' && S.brush.current) {
    const cur = S.brush.current;
    S.brush.current = null;
    S.brush.strokes.push(cur);
    S.pointer = null;
    try { canvas.releasePointerCapture(ev.pointerId); } catch (_) { /* ignore */ }
    syncBrushButtons();
    draw();
    pushBrush();
    return;
  }

  if (!S.pointer) return;
  const p = toImage(ev);
  try { canvas.releasePointerCapture(ev.pointerId); } catch (_) { /* ignore */ }

  if (S.mode === 'box' && S.drag) {
    const b = normBox(S.drag);
    S.drag = null;
    if (b.w > 4 && b.h > 4) {
      S.boxes.push(b);
      draw();
      run('box');
    } else {
      draw();
    }
  } else if (S.mode === 'point' && !S.pointer.moved) {
    const negative = ev.button === 2 || ev.shiftKey;
    S.points.push({ x: p.x, y: p.y, label: negative ? 0 : 1 });
    renderPoints();
    draw();
    run('point');
  }
  S.pointer = null;
});

canvas.addEventListener('contextmenu', (ev) => {
  ev.preventDefault();
  if (S.mode === 'box' && S.boxes.length) {
    S.boxes.pop();
    if (S.boxes.length) run('box');
    else { S.overlay = null; S.detections = []; updateStatus(null); draw(); }
  }
});

// 滚轮缩放：以光标所在的图像像素为锚点
wrap.addEventListener('wheel', (ev) => {
  if (!S.img) return;
  ev.preventDefault();
  const rect = wrap.getBoundingClientRect();
  const W = Math.max(1, rect.width);
  const H = Math.max(1, rect.height);
  const v = viewRect();
  const px = ev.clientX - rect.left;
  const py = ev.clientY - rect.top;
  const ix = (px - v.x) / v.s;
  const iy = (py - v.y) / v.s;

  const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, S.view.s * Math.exp(-ev.deltaY * 0.0015)));
  if (next === S.view.s) return;

  const fit = Math.min(W / S.imgW, H / S.imgH);
  const s = fit * next;
  const w = S.imgW * s;
  const h = S.imgH * s;
  S.view.s = next;
  S.view.tx = px - (W - w) / 2 - ix * s;
  S.view.ty = py - (H - h) / 2 - iy * s;
  draw();
}, { passive: false });

canvas.addEventListener('dblclick', (ev) => {
  ev.preventDefault();
  resetView();
});

window.addEventListener('keydown', (ev) => {
  if (ev.code === 'Space' && !S.space) {
    S.space = true;
    canvas.classList.add('grab');
    ev.preventDefault();
  } else if (ev.key === 'Escape') {
    resetView();
  } else if ((ev.key === 'Backspace' || ev.key === 'Delete') && S.mode === 'point' && S.points.length) {
    ev.preventDefault();
    removePoint(S.points.length - 1);
  }
});

window.addEventListener('keyup', (ev) => {
  if (ev.code === 'Space') {
    S.space = false;
    canvas.classList.remove('grab');
  }
});

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => setMode(tab.dataset.mode));
});

$('#run').addEventListener('click', () => run('everything'));

// 修掩码笔刷控件
$('#brush-add').addEventListener('click', () => { S.brush.value = 1; syncBrushButtons(); });
$('#brush-erase').addEventListener('click', () => { S.brush.value = 0; syncBrushButtons(); });
$('#brush-size').addEventListener('input', (ev) => {
  S.brush.radius = +ev.target.value;
  syncBrushButtons();
  draw();
});
$('#brush-undo').addEventListener('click', () => {
  if (!S.brush.strokes.length) return;
  S.brush.strokes.pop();
  syncBrushButtons();
  draw();
  pushBrush();
});
$('#brush-clear').addEventListener('click', () => {
  if (!S.brush.strokes.length) return;
  clearBrush();
  draw();
  pushBrush();
});
canvas.addEventListener('pointerleave', () => {
  if (S.mode !== 'brush' || !S.hover) return;
  S.hover = null;
  draw();
});
syncBrushButtons();

$('#reset').addEventListener('click', () => {
  S.boxes = [];
  S.points = [];
  S.overlay = null;
  S.detections = [];
  S.drag = null;
  clearBrush();
  resetView();
  renderPoints();
  updateStatus(null);
  draw();
});

$('#pick').addEventListener('click', () => $('#file').click());
$('#file').addEventListener('change', (ev) => uploadFile(ev.target.files[0]));

// 拖拽上传
['dragenter', 'dragover'].forEach((type) => {
  wrap.addEventListener(type, (ev) => {
    ev.preventDefault();
    $('#dropzone').classList.add('drag');
  });
});
['dragleave', 'drop'].forEach((type) => {
  wrap.addEventListener(type, (ev) => {
    ev.preventDefault();
    if (type === 'dragleave' && ev.relatedTarget && wrap.contains(ev.relatedTarget)) return;
    $('#dropzone').classList.remove('drag');
  });
});
wrap.addEventListener('drop', (ev) => {
  const file = ev.dataTransfer?.files?.[0];
  if (file) uploadFile(file);
});

// 参数显示
const sliders = [
  ['imgsz', 'v-imgsz', (v) => v],
  ['conf', 'v-conf', (v) => Number(v).toFixed(2)],
  ['iou', 'v-iou', (v) => Number(v).toFixed(2)],
];
sliders.forEach(([id, out, fmt]) => {
  const el = $('#' + id);
  const sync = () => { $('#' + out).textContent = fmt(el.value); };
  el.addEventListener('input', sync);
  sync();
});

$('#copy').addEventListener('click', async () => {
  if (!S.detections.length) { toast('还没有检测结果'); return; }
  const payload = {
    image: $('#filename').textContent,
    width: S.imgW,
    height: S.imgH,
    mode: S.mode,
    count: S.detections.length,
    detections: S.detections,
  };
  const text = JSON.stringify(payload, null, 2);
  try {
    await navigator.clipboard.writeText(text);
    toast(`已复制 ${S.detections.length} 个实例的坐标`);
  } catch (_) {
    // 剪贴板不可用时退化为下载 JSON
    const blob = new Blob([text], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `fastsam-${S.imageId || 'result'}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('剪贴板不可用，已下载 JSON');
  }
});

// 导出掩码：黑白二值 PNG（白=实例，黑=背景，尺寸与原图一致）
$('#download').addEventListener('click', async () => {
  if (!S.imageId) { toast('先选一张图片'); return; }
  try {
    const res = await fetch(`/api/mask/${S.imageId}`);
    if (!res.ok) {
      let detail = `导出失败（${res.status}）`;
      try {
        const data = await res.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `fastsam-${S.imageId}-mask.png`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('已导出二值掩码');
  } catch (err) {
    toast(err.message);
  }
});

// 导出框：把当前掩码转成框标注（每个掩码取最小外接矩形）
$('#export-boxes').addEventListener('click', async () => {
  if (!S.imageId) { toast('先选一张图片'); return; }
  try {
    const res = await fetch(`/api/boxes/${S.imageId}`);
    if (!res.ok) {
      let detail = `导出失败（${res.status}）`;
      try {
        const data = await res.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `fastsam-${S.imageId}-boxes.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('已导出框标注 ZIP');
  } catch (err) {
    toast(err.message);
  }
});

// 保存标注：后端打包 ZIP（标注 JSON + 索引掩码 + 叠加图 + 原图）
$('#export-ann').addEventListener('click', async () => {
  if (!S.imageId) { toast('先选一张图片'); return; }
  try {
    const res = await fetch(`/api/export/${S.imageId}`);
    if (!res.ok) {
      let detail = `导出失败（${res.status}）`;
      try {
        const data = await res.json();
        if (data && data.detail) detail = data.detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `fastsam-${S.imageId}.zip`;
    a.click();
    URL.revokeObjectURL(a.href);
    toast('已导出标注 ZIP');
  } catch (err) {
    toast(err.message);
  }
});

window.addEventListener('resize', draw);
new ResizeObserver(draw).observe(wrap);

/* ---------------------------- 初始化 ---------------------------- */

async function boot() {
  setMode('everything');
  try {
    const health = await fetch('/api/health').then((r) => r.json());

    const badge = $('#mode-badge');
    if (health.mode === 'real') {
      badge.textContent = '真实模型';
      badge.className = 'badge badge-real';
      $('#device-badge').textContent = health.device.toUpperCase();
    } else {
      badge.textContent = '模拟模式';
      badge.className = 'badge badge-mock';
      $('#device-badge').textContent = 'CPU';
      const banner = $('#banner');
      banner.textContent = `模拟模式：${health.reason}。当前掩码由 OpenCV 颜色聚类生成，仅供界面验证；装好 torch 后重启后端即自动切到真实推理。`;
      banner.classList.remove('hidden');
    }

    const box = $('#samples');
    const groups = {};
    (health.samples || []).forEach((s) => {
      const dir = s.name.includes('/') ? s.name.split('/')[0] : '';
      (groups[dir] = groups[dir] || []).push(s.name);
    });
    Object.entries(groups).forEach(([dir, names]) => {
      names.forEach((name) => {
        const chip = document.createElement('button');
        chip.className = 'chip';
        chip.textContent = name.split('/').pop();
        chip.title = name;
        chip.addEventListener('click', async () => {
          document.querySelectorAll('.chip').forEach((c) => c.classList.remove('on'));
          chip.classList.add('on');
          try {
            const info = await api('/api/sample', { name });
            S.source = { kind: 'sample', name };
            await loadImage(info);
          } catch (err) {
            toast(err.message);
          }
        });
        box.appendChild(chip);
      });
    });

    // 深链：#sample=dogs.jpg&mode=brush 直接载入示例并切到指定提示方式
    const link = new URLSearchParams(location.hash.slice(1));
    const want = link.get('sample');
    const mode = link.get('mode');
    if (want) {
      const hit = [...box.querySelectorAll('.chip')]
        .find((c) => c.title === want || c.title.endsWith('/' + want));
      if (hit) hit.click();
      else toast(`没有名为「${want}」的示例图`);
    }
    if (mode && HINTS[mode]) setMode(mode);
    // 调试用：&boxes=[[x,y,w,h],...]（原图坐标）直接预置框，便于复现截图
    const dbg = link.get('boxes');
    if (dbg) {
      try {
        S.pendingBoxes = JSON.parse(decodeURIComponent(dbg)).map(([x, y, w, h]) => ({ x, y, w, h }));
      } catch (err) {
        toast('boxes 参数无法解析');
      }
    }
  } catch (err) {
    toast('后端未响应：' + err.message);
  }
}

boot();
