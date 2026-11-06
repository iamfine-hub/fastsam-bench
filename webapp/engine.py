"""FastSAM 推理封装：真实引擎 + 模拟引擎。

真实引擎需要 torch / torchvision 以及仓库自带的 ultralytics 8.0.120。
缺依赖时自动降级为「模拟引擎」—— 用 OpenCV 颜色聚类造出结构类似的掩码，
让前端的上传、画框、点选、叠加渲染在没有 torch 的机器上也能完整验证。

两种引擎对外暴露同一套接口，返回 Scene：
  masks  : (N, mh, mw) bool，掩码分辨率为推理分辨率（已裁到 MASK_SIDE 以内）
  boxes  : (N, 4) float32，原图坐标 x1,y1,x2,y2
  scores : (N,) float32
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_FILE = REPO_ROOT / "FastSAM-x.pt"
MAX_UPLOAD_SIDE = 1600   # 上传图最长边上限
MASK_SIDE = 1024         # 掩码最长边上限（控制显存/内存）
MAX_MASKS = 60           # 最多保留多少个实例

# 黑白渲染：所有实例共用同一个中性灰填充，靠白色发丝描边区分边界。
# 单色下没有色相可用，实例之间只能靠轮廓线分辨——这也是二值掩码工具的通行做法。
MASK_FILL = (140, 140, 140)   # BGR，中性灰：亮背景压暗、暗背景提亮，不会把照片洗白
MASK_EDGE = (255, 255, 255)
FOCUS_FILL = (245, 245, 245)  # 画框/点选命中的单个区域，比全图叠加亮得多


@dataclass
class Scene:
    masks: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    width: int
    height: int
    elapsed_ms: int
    key: tuple

    @property
    def n(self) -> int:
        return int(len(self.masks))

    @property
    def mh(self) -> int:
        return int(self.masks.shape[1])

    @property
    def mw(self) -> int:
        return int(self.masks.shape[2])


# --------------------------------------------------------------------------- #
# 渲染与几何工具（纯 numpy / cv2，不依赖 matplotlib）
# --------------------------------------------------------------------------- #
def encode_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _fit_masks(masks: np.ndarray) -> np.ndarray:
    """把 (N,H,W) 掩码缩到最长边 <= MASK_SIDE，控制内存。"""
    if masks.size == 0:
        return masks
    n, h, w = masks.shape
    side = max(h, w)
    if side <= MASK_SIDE:
        return masks
    scale = MASK_SIDE / side
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    out = np.zeros((n, nh, nw), dtype=bool)
    for i in range(n):
        out[i] = cv2.resize(
            masks[i].astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return out


def refine_masks(masks: np.ndarray) -> np.ndarray:
    """better_quality：形态学去毛刺。"""
    if masks.size == 0:
        return masks
    kernel = np.ones((3, 3), np.uint8)
    out = np.empty_like(masks)
    for i in range(len(masks)):
        m = cv2.morphologyEx(masks[i].astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        out[i] = m.astype(bool)
    return out


def render_overlay(
    bgr: np.ndarray,
    masks: np.ndarray,
    contours: bool = True,
    alpha: float | None = None,
    dim: float = 0.0,
    focus: bool = False,
) -> np.ndarray:
    """在原图上叠加半透明掩码；bgr 需与掩码同分辨率。

    黑白模式：所有实例共用同一中性灰填充，边界靠白色发丝线区分；
    focus=True（画框/点选命中的单个区域）时填充更亮，让命中区域自己跳出来。
    """
    if alpha is None:
        # 全图模式靠轮廓线读结构，填充会把整张照片糊成一片灰；只有关掉边缘线时才补填充。
        alpha = 0.12 if focus else (0.0 if contours else 0.4)

    canvas = bgr.copy()

    if masks.size == 0:
        if dim > 0:
            canvas = cv2.addWeighted(canvas, 1.0 - dim, np.zeros_like(canvas), dim, 0)
        return canvas

    h, w = canvas.shape[:2]
    mh, mw = masks.shape[1], masks.shape[2]

    union = np.zeros((mh, mw), dtype=bool)
    for m in masks:
        union |= m

    if (h, w) != (mh, mw):
        # 填充只放大一次并集，代价小、边缘软；
        # 轮廓不放大掩码，而是把顶点坐标乘上比例——直接放大掩码会让轮廓变成锯齿块。
        union_full = cv2.resize(
            union.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        sx, sy = w / mw, h / mh
    else:
        union_full = union
        sx = sy = 1.0

    if focus:
        # 命中区域保持原亮度、其余压暗（聚光效果）。
        # 反过来给命中区域刷白色填充的话，选中一大片实例时整个框看起来就是一块白板。
        dimmed = cv2.addWeighted(canvas, 1.0 - dim, np.zeros_like(canvas), dim, 0)
        canvas = dimmed
        canvas[union_full] = bgr[union_full]
    elif dim > 0:
        canvas = cv2.addWeighted(canvas, 1.0 - dim, np.zeros_like(canvas), dim, 0)

    if alpha > 0:
        tint = canvas.copy()
        tint[union_full] = FOCUS_FILL if focus else MASK_FILL
        blended = cv2.addWeighted(canvas, 1.0 - alpha, tint, alpha, 0)
        canvas[union_full] = blended[union_full]

    if contours:
        # 先描一道黑底衬再压白线：白线在亮天空和暗物体上都读得出来，
        # 纯白线遇到沙滩/天空会直接消失。
        # 轮廓用双线性放大后重新取边界，否则 512 掩码直接乘比例会留下一格一格的台阶。
        for m in masks:
            if sx != 1.0 or sy != 1.0:
                smooth = cv2.resize(
                    m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_LINEAR
                )
                binary = (smooth > 127).astype(np.uint8)
            else:
                binary = m.astype(np.uint8)
            cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cs, -1, (0, 0, 0), 2, lineType=cv2.LINE_AA)
            cv2.drawContours(canvas, cs, -1, MASK_EDGE, 1, lineType=cv2.LINE_AA)
    return canvas


def boxes_from_masks(masks: np.ndarray, mw: int, mh: int, width: int, height: int):
    """掩码 → 原图坐标的框。"""
    out = []
    sx, sy = width / mw, height / mh
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        out.append(
            [
                float(xs.min()) * sx,
                float(ys.min()) * sy,
                float(xs.max() + 1) * sx,
                float(ys.max() + 1) * sy,
            ]
        )
    return np.asarray(out, dtype=np.float32).reshape(-1, 4)


def area_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.size) if mask.size else 0.0


# --------------------------------------------------------------------------- #
# 提示匹配：在已有掩码上做选择，不重新推理（CPU 上这是毫秒级的关键）
# --------------------------------------------------------------------------- #
def single_region_scene(scene: Scene, region: np.ndarray) -> Scene:
    """把「提示命中的单个区域」包装成 Scene，方便复用 detections_of。"""
    boxes = boxes_from_masks(region, scene.mw, scene.mh, scene.width, scene.height)
    return Scene(
        masks=region,
        boxes=boxes,
        scores=np.ones(len(region), dtype=np.float32),
        width=scene.width,
        height=scene.height,
        elapsed_ms=0,
        key=(),
    )


def box_prompt(scene: Scene, boxes: list[list[float]]) -> np.ndarray:
    """多框提示：每个框各取 IoU 最大的实例，返回去重后的「实例本身」。

    注意不要把多个框的结果合并成一个并集——那样第二个框只会让计数还是 1，
    看起来就像第二个框失效了。这里保持每个实例独立，交给上层各自出框。
    """
    if scene.n == 0:
        return np.zeros((0, scene.mh or 1, scene.mw or 1), dtype=bool)
    mh, mw = scene.mh, scene.mw
    sx, sy = mw / max(scene.width, 1), mh / max(scene.height, 1)
    picked: list[int] = []
    for box in boxes:
        x1, x2 = sorted((int(box[0] * sx), int(box[2] * sx)))
        y1, y2 = sorted((int(box[1] * sy), int(box[3] * sy)))
        x1, y1 = max(0, min(x1, mw - 1)), max(0, min(y1, mh - 1))
        x2, y2 = max(x1 + 1, min(x2, mw)), max(y1 + 1, min(y2, mh))

        box_area = float((x2 - x1) * (y2 - y1))
        inside = scene.masks[:, y1:y2, x1:x2].sum(axis=(1, 2)).astype(np.float32)
        total = scene.masks.sum(axis=(1, 2)).astype(np.float32)
        union = box_area + total - inside
        iou = np.divide(inside, union, out=np.zeros_like(inside), where=union > 0)
        # 按 IoU 选实例：框和实例互相覆盖得最完整的那一个。
        # 曾经试过再乘 contain（实例落在框内的比例），结果框里只圈到小狗叼的树枝
        # 这种小物件，反而选不中狗本身，所以回到纯 IoU。
        if float(iou.max()) > 0.0:
            idx = int(np.argmax(iou))
            if idx not in picked:
                picked.append(idx)

    if not picked:
        return np.zeros((0, mh, mw), dtype=bool)
    return scene.masks[picked]


def point_prompt(scene: Scene, points: list[list[float]], labels: list[int]) -> np.ndarray:
    """正点加入命中实例，负点从结果中剔除（>=2 个点时按面积从大到小叠加）。"""
    if scene.n == 0:
        return np.zeros((0, 1, 1), dtype=bool)
    mh, mw = scene.mh, scene.mw
    sx, sy = mw / max(scene.width, 1), mh / max(scene.height, 1)
    pts = [(int(p[0] * sx), int(p[1] * sy)) for p in points]

    order = np.argsort(-scene.masks.sum(axis=(1, 2)))
    out = np.zeros((mh, mw), dtype=bool)
    for idx in order:
        m = scene.masks[idx]
        add = sub = False
        for (px, py), lab in zip(pts, labels):
            if 0 <= px < mw and 0 <= py < mh and m[py, px]:
                if lab == 1:
                    add = True
                else:
                    sub = True
        if add:
            out |= m
        if sub:
            out &= ~m
    return out[None, ...]


def apply_brush(
    masks: np.ndarray,
    strokes: list[dict],
    width: int,
    height: int,
) -> np.ndarray:
    """把画笔笔迹应用到掩码上。

    strokes[i] = {"value": 1 加 / 0 擦, "radius": 原图像素半径, "points": [[x, y], ...]}
    笔迹按原图坐标给出，内部换算到掩码分辨率；后画的笔迹覆盖先画的，
    所以「擦」可以修掉刚「加」出来的地方。加出来的部分作为一个新实例追加。
    """
    k, mh, mw = masks.shape
    sx, sy = mw / max(width, 1), mh / max(height, 1)
    paint = np.zeros((mh, mw), dtype=np.uint8)  # 0 没涂 / 1 加 / 2 擦

    for st in strokes:
        pts = [
            (int(round(p[0] * sx)), int(round(p[1] * sy)))
            for p in st.get("points", [])
        ]
        if not pts:
            continue
        value = 1 if int(st.get("value", 1)) else 2
        radius = max(1, int(round(float(st.get("radius", 12)) * (sx + sy) / 2.0)))
        thickness = max(1, radius * 2)
        if len(pts) == 1:
            cv2.circle(paint, pts[0], radius, value, -1, lineType=cv2.LINE_8)
        else:
            for i in range(1, len(pts)):
                cv2.line(paint, pts[i - 1], pts[i], value, thickness, lineType=cv2.LINE_8)

    erase = paint == 2
    add = paint == 1

    kept = [m & ~erase for m in masks]
    painted = add & ~erase
    if painted.any():
        kept.append(painted)
    if not kept:
        return np.zeros((0, mh, mw), dtype=bool)

    stacked = np.stack(kept)
    nonempty = stacked.reshape(len(kept), -1).any(axis=1)
    return stacked[nonempty]


# --------------------------------------------------------------------------- #
# 真实引擎
# --------------------------------------------------------------------------- #
class RealEngine:
    mode = "real"

    def __init__(self, model_path: Path, device: str = "cpu"):
        from fastsam import FastSAM  # 仓库内 vendored 包

        self.device = device
        self.model = FastSAM(str(model_path))

    def everything(self, bgr, imgsz, conf, iou, better_quality=False) -> Scene:
        from PIL import Image

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.model(
            Image.fromarray(rgb),
            device=self.device,
            retina_masks=True,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            max_det=MAX_MASKS,
        )
        h, w = bgr.shape[:2]
        r = results[0]
        if r.masks is None or len(r.masks.data) == 0:
            masks = np.zeros((0, 1, 1), dtype=bool)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            masks = r.masks.data.cpu().numpy().astype(bool)[:MAX_MASKS]
            masks = _fit_masks(masks)
            if better_quality:
                masks = refine_masks(masks)
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)[: len(masks)]

        mh, mw = masks.shape[1], masks.shape[2]
        boxes = boxes_from_masks(masks, mw, mh, w, h)
        return Scene(
            masks=masks,
            boxes=boxes,
            scores=scores,
            width=w,
            height=h,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            key=("real", imgsz, conf, iou, better_quality),
        )


# --------------------------------------------------------------------------- #
# 模拟引擎：颜色聚类 + 连通域，产出「像那么回事」的实例掩码
# --------------------------------------------------------------------------- #
class MockEngine:
    mode = "mock"

    def everything(self, bgr, imgsz, conf, iou, better_quality=False) -> Scene:
        t0 = time.perf_counter()
        h0, w0 = bgr.shape[:2]
        scale = min(1.0, float(imgsz) / max(h0, w0))
        w, h = max(8, int(round(w0 * scale))), max(8, int(round(h0 * scale)))
        small = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        data = small.reshape(-1, 3).astype(np.float32)
        k = 6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        cv2.setRNGSeed(7)
        _, labels, _ = cv2.kmeans(
            data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.reshape(h, w)

        kernel = np.ones((5, 5), np.uint8)
        masks, scores = [], []
        min_area = 0.002 * h * w
        for cluster in range(k):
            m = (labels == cluster).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n_cc):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                masks.append(cc == i)
                scores.append(float(min(0.99, 0.35 + area / (h * w))))

        if not masks:
            masks_arr = np.zeros((0, h, w), dtype=bool)
            scores_arr = np.zeros((0,), dtype=np.float32)
        else:
            order = np.argsort([-m.sum() for m in masks])[:MAX_MASKS]
            masks_arr = np.stack([masks[i] for i in order])
            scores_arr = np.asarray([scores[i] for i in order], dtype=np.float32)

        masks_arr = _fit_masks(masks_arr)
        if better_quality:
            masks_arr = refine_masks(masks_arr)

        mh, mw = masks_arr.shape[1], masks_arr.shape[2]
        boxes = boxes_from_masks(masks_arr, mw, mh, w0, h0)
        return Scene(
            masks=masks_arr,
            boxes=boxes,
            scores=scores_arr,
            width=w0,
            height=h0,
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            key=("mock", imgsz, conf, iou, better_quality),
        )


# --------------------------------------------------------------------------- #
# 引擎管理：探测依赖 → 懒加载 → 失败自动降级
# --------------------------------------------------------------------------- #
def probe(model_path: Path = MODEL_FILE) -> tuple[bool, str]:
    if not model_path.exists():
        return False, f"权重文件不存在：{model_path.name}"
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "未安装 torch（真实推理不可用）"
    try:
        import fastsam  # noqa: F401
    except Exception as exc:  # pragma: no cover - 依赖环境
        return False, f"fastsam 包导入失败：{type(exc).__name__}: {exc}"
    return True, ""


def pick_device(explicit: str | None = None) -> str:
    """决定推理设备。

    优先显式参数，其次环境变量 FASTSAM_DEVICE，最后自动判断：装了 torch 且
    CUDA 可用就用 cuda，否则 cpu。前端只认 mode/device 两个字段，所以装 CPU 版
    还是 GPU 版 torch 都不影响界面，换机器也不用改代码。
    """
    if explicit:
        return explicit
    env = os.environ.get("FASTSAM_DEVICE", "").strip()
    if env:
        return env
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


class Manager:
    """全局单例：模型只加载一次，CPU 推理串行执行。"""

    def __init__(self, model_path: Path = MODEL_FILE, device: str | None = None):
        self.model_path = Path(model_path)
        self.device = pick_device(device)
        self._engine = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.reason = ""
        self.mode = "unprobed"

    def status(self) -> dict:
        ok, why = probe(self.model_path)
        mode = "real" if (ok or self.mode == "real") else "mock"
        return {
            "mode": mode,
            "reason": "" if ok else (why or self.reason),
            "device": self.device,
            "model": self.model_path.name,
            "model_present": self.model_path.exists(),
        }

    def engine(self):
        with self._lock:
            if self._engine is None:
                ok, why = probe(self.model_path)
                if ok:
                    try:
                        self._engine = RealEngine(self.model_path, self.device)
                    except Exception as exc:
                        self.reason = f"模型加载失败：{type(exc).__name__}: {exc}"
                        self._engine = MockEngine()
                else:
                    self.reason = why
                    self._engine = MockEngine()
                self.mode = self._engine.mode
                self.reason = self.reason or why
            return self._engine

    def everything(self, bgr, imgsz, conf, iou, better_quality=False) -> Scene:
        with self._infer_lock:
            return self.engine().everything(bgr, imgsz, conf, iou, better_quality)
