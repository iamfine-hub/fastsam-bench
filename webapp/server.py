"""FastSAM 检测台 —— FastAPI 后端。

启动：
    D:\\python12\\python.exe D:\\FastSAM_clone\\webapp\\server.py
浏览器打开 http://127.0.0.1:8000

接口：
    GET  /api/health             引擎状态（真实 / 模拟）、示例图列表
    POST /api/upload             上传图片 → image_id
    POST /api/sample             载入仓库自带示例图 → image_id
    GET  /api/image/{image_id}   取回原图 PNG
    POST /api/everything         全图分割
    POST /api/box                框选提示（复用已缓存的掩码，不重推）
    POST /api/point              点选提示（同上）
    POST /api/clear/{image_id}   丢弃该图缓存
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import sys
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import engine as eng  # noqa: E402

REPO_ROOT = eng.REPO_ROOT
STATIC_DIR = WEBAPP_DIR / "static"
SAMPLE_DIRS = ["examples", "images", "test_img"]
SAMPLE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

manager = eng.Manager()
app = FastAPI(title="FastSAM 检测台", version="1.0")


# --------------------------------------------------------------------------- #
# 图像仓库（带 LRU，避免长跑吃内存）
# --------------------------------------------------------------------------- #
class Entry:
    __slots__ = (
        "image_id",
        "name",
        "bgr",
        "png",
        "scene",
        "width",
        "height",
        "last_masks",
        "base_masks",
    )

    def __init__(self, image_id: str, name: str, bgr: np.ndarray):
        self.image_id = image_id
        self.name = name
        self.bgr = bgr
        self.height, self.width = bgr.shape[:2]
        self.png = cv2.imencode(".png", bgr)[1].tobytes()
        self.scene: eng.Scene | None = None
        # 最近一次「当前提示」的掩码（全图 / 框 / 点都可能更新它），导出标注时用它。
        self.last_masks: np.ndarray | None = None
        # 画笔修整之前的原始提示结果；每次重画笔迹都从它重新算，撤销才是可逆的。
        self.base_masks: np.ndarray | None = None


class ImageStore:
    """内存 LRU + 磁盘缓存。

    内存只保留最近 cap 张（免得多开几张图就把内存吃掉），但 image_id 不因此失效：
    被挤出去的图在磁盘上还有一份，get() 时会自动读回来。这样「图片已过期」只在
    手动清过缓存目录或换了机器时才会出现。
    """

    def __init__(
        self,
        cap: int = 6,
        disk_dir: Path | None = None,
        disk_cap: int = 200,
        disk_bytes: int = 512 * 1024 * 1024,
    ):
        self._cap = cap
        self._items: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._dir = disk_dir or (WEBAPP_DIR / ".cache" / "images")
        self._disk_cap = disk_cap
        self._disk_bytes = disk_bytes
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._dir = None  # 磁盘不可写就退化成纯内存

    # ---- 磁盘 ----
    def _disk_png(self, image_id: str) -> Path | None:
        return None if self._dir is None else self._dir / f"{image_id}.png"

    def _persist(self, entry: Entry) -> None:
        path = self._disk_png(entry.image_id)
        if path is None:
            return
        try:
            tmp = path.with_suffix(".png.tmp")
            tmp.write_bytes(entry.png)
            tmp.replace(path)
            path.with_suffix(".json").write_text(
                json.dumps({"name": entry.name}, ensure_ascii=False), encoding="utf-8"
            )
            self._prune_disk()
        except OSError:
            pass

    def _prune_disk(self) -> None:
        if self._dir is None:
            return
        files = sorted(
            (p for p in self._dir.glob("*.png") if p.suffix == ".png"),
            key=lambda p: p.stat().st_mtime,
        )
        sizes = {}
        for p in files:
            try:
                sizes[p] = p.stat().st_size
            except OSError:
                sizes[p] = 0
        total = sum(sizes.values())
        # 先按张数、再按总字节，从最旧的开始删
        for i, old in enumerate(files):
            if len(files) - i <= self._disk_cap and total <= self._disk_bytes:
                break
            for p in (old, old.with_suffix(".json")):
                try:
                    p.unlink()
                except OSError:
                    pass
            total -= sizes.get(old, 0)

    def _restore(self, image_id: str) -> Entry | None:
        path = self._disk_png(image_id)
        if path is None or not path.is_file():
            return None
        data = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        name = image_id
        meta = path.with_suffix(".json")
        if meta.is_file():
            try:
                name = json.loads(meta.read_text(encoding="utf-8")).get("name") or name
            except (OSError, ValueError):
                pass
        entry = Entry(image_id, name, bgr)
        entry.png = data
        return entry

    # ---- 公开接口 ----
    def put(self, entry: Entry) -> Entry:
        self._persist(entry)
        with self._lock:
            self._items[entry.image_id] = entry
            self._items.move_to_end(entry.image_id)
            while len(self._items) > self._cap:
                self._items.popitem(last=False)
        return entry

    def get(self, image_id: str) -> Entry:
        with self._lock:
            entry = self._items.get(image_id)
            if entry is not None:
                self._items.move_to_end(image_id)
                return entry
        restored = self._restore(image_id)
        if restored is None:
            raise HTTPException(status_code=404, detail="图片已过期，请重新上传")
        return self.put(restored)

    def drop(self, image_id: str) -> None:
        with self._lock:
            self._items.pop(image_id, None)
        for p in (self._disk_png(image_id),):
            if p is None:
                continue
            for f in (p, p.with_suffix(".json")):
                try:
                    f.unlink()
                except (OSError, AttributeError):
                    pass


store = ImageStore()


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #
class EverythingReq(BaseModel):
    image_id: str
    imgsz: int = 1024
    conf: float = 0.25
    iou: float = 0.7
    contours: bool = True
    better_quality: bool = False


class BoxReq(BaseModel):
    image_id: str
    boxes: list[list[float]]
    contours: bool = True
    better_quality: bool = False


class PointReq(BaseModel):
    image_id: str
    points: list[list[float]]
    labels: list[int]
    contours: bool = True
    better_quality: bool = False


class StrokeIn(BaseModel):
    value: int = 1            # 1 = 加，0 = 擦
    radius: float = 12.0      # 原图像素半径
    points: list[list[float]]


class BrushReq(BaseModel):
    image_id: str
    strokes: list[StrokeIn] = []
    contours: bool = True


class SampleReq(BaseModel):
    name: str


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def decode_image(data: bytes, name: str) -> tuple[np.ndarray, bool]:
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"图片解码失败：{exc}") from exc

    scaled = False
    if max(im.size) > eng.MAX_UPLOAD_SIDE:
        im.thumbnail((eng.MAX_UPLOAD_SIDE, eng.MAX_UPLOAD_SIDE), Image.LANCZOS)
        scaled = True

    arr = np.asarray(im)
    return arr[:, :, ::-1].copy(), scaled


def disposition(filename: str) -> str:
    """生成 Content-Disposition，同时给 ASCII 回退名和 RFC 5987 的 UTF-8 名。

    HTTP 头只能是 latin-1，直接往里塞中文文件名会抛 UnicodeEncodeError，
    表现就是「上传中文文件名的图片后，导出掩码偶尔 500」。
    """
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "download"
    if not ascii_name[0].isalnum():
        # 纯中文名会被压成 "-mask.png" 这种，补个前缀好看一点
        ascii_name = f"image{ascii_name}"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename, safe='')}"


def overlay_of(
    entry: Entry,
    masks: np.ndarray,
    contours: bool,
    dim: float = 0.0,
    focus: bool = False,
) -> str:
    # 黑白呈现：整张分析图先转灰阶，再叠白线/灰填充。
    # 若想保留照片原色，把下面两行换成 `base = entry.bgr` 即可。
    gray = cv2.cvtColor(entry.bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if masks.size == 0:
        return eng.encode_png(base)

    # 掩码保持自己的分辨率，render_overlay 内部负责按原图尺寸缩放填充与轮廓点。
    rendered = eng.render_overlay(base, masks, contours=contours, dim=dim, focus=focus)
    return eng.encode_png(rendered)


def detections_of(scene: eng.Scene, limit: int = 200) -> list[dict]:
    out = []
    for i in range(min(scene.n, limit)):
        box = scene.boxes[i] if i < len(scene.boxes) else [0, 0, 0, 0]
        ratio = eng.area_ratio(scene.masks[i])
        out.append(
            {
                "id": i,
                "bbox": [round(float(v), 1) for v in box],
                "score": round(float(scene.scores[i]) if i < len(scene.scores) else 0.0, 3),
                "area_ratio": round(float(ratio), 4),
                "area": round(float(ratio) * 100, 2),
            }
        )
    return out


def ensure_scene(entry: Entry, imgsz: int, conf: float, iou: float, better: bool):
    scene = entry.scene
    key = ("mock" if manager.status()["mode"] == "mock" else "real", int(imgsz), float(conf), float(iou), bool(better))
    if scene is not None and scene.key == key:
        return scene, True
    scene = manager.everything(entry.bgr, int(imgsz), float(conf), float(iou), bool(better))
    entry.scene = scene
    return scene, False


def list_samples() -> list[dict]:
    out = []
    for folder in SAMPLE_DIRS:
        directory = REPO_ROOT / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in SAMPLE_EXT:
                out.append({"name": str(path.relative_to(REPO_ROOT)).replace("\\", "/")})
    return out


# --------------------------------------------------------------------------- #
# 接口
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    status = manager.status()
    status["samples"] = list_samples()
    status["images_loaded"] = len(store._items)  # noqa: SLF001 - 只读统计
    return status


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    bgr, scaled = decode_image(data, file.filename or "upload")
    entry = store.put(Entry(uuid.uuid4().hex[:12], file.filename or "upload", bgr))
    return {
        "image_id": entry.image_id,
        "width": entry.width,
        "height": entry.height,
        "name": entry.name,
        "scaled": scaled,
    }


@app.post("/api/sample")
def sample(req: SampleReq):
    path = (REPO_ROOT / req.name).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法路径") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="示例图不存在")
    bgr, scaled = decode_image(path.read_bytes(), path.name)
    entry = store.put(Entry(uuid.uuid4().hex[:12], path.name, bgr))
    return {
        "image_id": entry.image_id,
        "width": entry.width,
        "height": entry.height,
        "name": entry.name,
        "scaled": scaled,
    }


@app.get("/api/image/{image_id}")
def image(image_id: str):
    return Response(content=store.get(image_id).png, media_type="image/png")


@app.post("/api/everything")
def everything(req: EverythingReq):
    entry = store.get(req.image_id)
    scene, cached = ensure_scene(entry, req.imgsz, req.conf, req.iou, req.better_quality)
    entry.last_masks = scene.masks
    entry.base_masks = scene.masks
    return {
        "overlay": overlay_of(entry, scene.masks, req.contours),
        "detections": detections_of(scene),
        "count": scene.n,
        "elapsed_ms": scene.elapsed_ms,
        "cached": cached,
        "mode": manager.status()["mode"],
    }


@app.post("/api/box")
def box(req: BoxReq):
    entry = store.get(req.image_id)
    if not req.boxes:
        raise HTTPException(status_code=400, detail="至少需要一个框")
    scene, _ = ensure_scene(entry, 1024, 0.25, 0.7, req.better_quality)
    region = eng.box_prompt(scene, req.boxes)
    if region.size == 0 or not region.any():
        entry.last_masks = None
        entry.base_masks = None
        return {
            "overlay": overlay_of(entry, np.zeros((0, 1, 1), dtype=bool), req.contours, dim=0.45),
            "hit": False,
            "detections": [],
            "count": 0,
            "elapsed_ms": 0,
            "cached": True,
        }
    entry.last_masks = region
    entry.base_masks = region
    return {
        "overlay": overlay_of(entry, region, req.contours, dim=0.5, focus=True),
        "hit": True,
        "detections": detections_of(eng.single_region_scene(scene, region)),
        "count": int(region.shape[0]),
        "elapsed_ms": 0,
        "cached": True,
    }


@app.post("/api/point")
def point(req: PointReq):
    entry = store.get(req.image_id)
    if not req.points:
        raise HTTPException(status_code=400, detail="至少需要一个点")
    labels = req.labels or [1] * len(req.points)
    if len(labels) != len(req.points):
        raise HTTPException(status_code=400, detail="points 与 labels 数量不一致")
    scene, _ = ensure_scene(entry, 1024, 0.25, 0.7, req.better_quality)
    region = eng.point_prompt(scene, req.points, labels)
    if region.size == 0 or not region.any():
        entry.last_masks = None
        entry.base_masks = None
        return {
            "overlay": overlay_of(entry, np.zeros((0, 1, 1), dtype=bool), req.contours, dim=0.45),
            "hit": False,
            "detections": [],
            "count": 0,
            "elapsed_ms": 0,
            "cached": True,
        }
    entry.last_masks = region
    entry.base_masks = region
    return {
        "overlay": overlay_of(entry, region, req.contours, dim=0.5, focus=True),
        "hit": True,
        "detections": detections_of(eng.single_region_scene(scene, region)),
        "count": int(region.shape[0]),
        "elapsed_ms": 0,
        "cached": True,
    }


@app.post("/api/brush")
def brush(req: BrushReq):
    """用画笔修当前掩码：每次从「提示原始结果」重算，所以撤销笔迹是干净的。"""
    entry = store.get(req.image_id)
    base = entry.base_masks
    if base is None or base.size == 0 or not base.any():
        raise HTTPException(status_code=400, detail="还没有分割结果，先跑一次分割")

    strokes = [
        {"value": s.value, "radius": s.radius, "points": s.points} for s in req.strokes
    ]
    masks = eng.apply_brush(base, strokes, entry.width, entry.height)

    if masks.size == 0 or not masks.any():
        entry.last_masks = None
        return {
            "overlay": overlay_of(
                entry, np.zeros((0, 1, 1), dtype=bool), req.contours, dim=0.45
            ),
            "hit": False,
            "detections": [],
            "count": 0,
            "elapsed_ms": 0,
            "cached": True,
        }

    entry.last_masks = masks
    return {
        "overlay": overlay_of(entry, masks, req.contours),
        "hit": True,
        "detections": detections_of(eng.single_region_scene(entry.scene, masks)),
        "count": int(masks.shape[0]),
        "elapsed_ms": 0,
        "cached": True,
    }


@app.get("/api/mask/{image_id}")
def mask(image_id: str):
    """黑白二值掩码 PNG：白 = 当前结果的所有实例，黑 = 背景，尺寸与原图一致。"""
    entry = store.get(image_id)
    masks = entry.last_masks
    if masks is None or masks.size == 0 or not masks.any():
        raise HTTPException(status_code=400, detail="还没有分割结果，先跑一次分割")

    union = masks.any(axis=0).astype(np.uint8) * 255
    mh, mw = union.shape[:2]
    if (mh, mw) != (entry.height, entry.width):
        # 纯二值图不能用插值放大，否则边缘会变成灰阶过渡
        union = cv2.resize(
            union, (entry.width, entry.height), interpolation=cv2.INTER_NEAREST
        )
    ok, buf = cv2.imencode(".png", union)
    if not ok:
        raise HTTPException(status_code=500, detail="掩码编码失败")
    stem = Path(entry.name).stem or "mask"
    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={"Content-Disposition": disposition(f"{stem}-mask.png")},
    )


@app.get("/api/export/{image_id}")
def export(image_id: str):
    """把当前这一版掩码打成 ZIP：annotations.json + masks.png（索引图）+ overlay.png + image.png。"""
    entry = store.get(image_id)
    masks = entry.last_masks
    if masks is None or masks.size == 0 or not masks.any():
        raise HTTPException(status_code=400, detail="还没有分割结果，先跑一次分割")

    mh, mw = masks.shape[1], masks.shape[2]
    n = int(masks.shape[0])
    areas = masks.reshape(n, -1).sum(axis=1)
    order = np.argsort(-areas)[:255]  # 索引图是 8 位，最多 255 个实例

    # 索引掩码：0 = 背景，i = 第 i 个实例（按面积从大到小编号，方便人工核对）
    index = np.zeros((mh, mw), dtype=np.uint8)
    for rank, idx in enumerate(order, start=1):
        index[masks[idx]] = rank
    if (mh, mw) != (entry.height, entry.width):
        index = cv2.resize(
            index, (entry.width, entry.height), interpolation=cv2.INTER_NEAREST
        )

    boxes = eng.boxes_from_masks(masks, mw, mh, entry.width, entry.height)
    px_scale = (entry.width / max(mw, 1)) * (entry.height / max(mh, 1))
    total = float(entry.width * entry.height) or 1.0

    instances = []
    for rank, idx in enumerate(order, start=1):
        x1, y1, x2, y2 = boxes[idx]
        area = float(areas[idx]) * px_scale
        instances.append(
            {
                "id": rank,
                "label": "instance",
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2 - x1), 2),
                    round(float(y2 - y1), 2),
                ],
                "area_px": int(round(area)),
                "area_ratio": round(area / total, 6),
            }
        )

    ann = {
        "image": entry.name,
        "image_id": entry.image_id,
        "width": entry.width,
        "height": entry.height,
        "mask_size": [mw, mh],
        "count": len(instances),
        "bbox_format": "xywh，原图像素坐标",
        "instances": instances,
    }

    overlay_b64 = overlay_of(entry, masks, True, dim=0.0, focus=False)
    ok, png = cv2.imencode(".png", index)
    if not ok:
        raise HTTPException(status_code=500, detail="索引掩码编码失败")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations.json", json.dumps(ann, ensure_ascii=False, indent=2))
        zf.writestr("masks.png", png.tobytes())
        zf.writestr("overlay.png", base64.b64decode(overlay_b64.split(",", 1)[-1]))
        zf.writestr("image.png", entry.png)
        zf.writestr(
            "README.txt",
            "masks.png 是 8 位索引掩码：0 表示背景，第 i 个实例的像素值为 i。\n"
            "编号顺序与 annotations.json 里的 instances[].id 一致，按面积从大到小。\n"
            "bbox 为 xywh，坐标基于原图 image.png。\n",
        )

    stem = Path(entry.name).stem
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": disposition(f"{stem}-annot.zip")},
    )


@app.get("/api/boxes/{image_id}")
def boxes(image_id: str, format: str = "zip"):
    """把当前掩码转成框标注：每个实例取掩码的最小外接矩形，坐标是原图像素。

    format：
      zip   （默认）一次给全套：boxes.json / labels_yolo.txt / boxes.csv / boxes.png / README.txt
      json  COCO 结构的框标注
      yolo  YOLO 训练用的 `class cx cy w h`（归一化）
      csv   人看的表格
    """
    entry = store.get(image_id)
    masks = entry.last_masks
    if masks is None or masks.size == 0 or not masks.any():
        raise HTTPException(status_code=400, detail="还没有分割结果，先跑一次分割")

    fmt = (format or "zip").lower()
    if fmt == "coco":
        fmt = "json"
    if fmt not in {"zip", "json", "yolo", "csv"}:
        raise HTTPException(status_code=400, detail="format 只支持 zip / json / yolo / csv")

    mh, mw = masks.shape[1], masks.shape[2]
    n = int(masks.shape[0])
    areas = masks.reshape(n, -1).sum(axis=1)
    order = np.argsort(-areas)  # 按面积从大到小，和 masks.png 的编号一致
    xyxy_all = eng.boxes_from_masks(masks, mw, mh, entry.width, entry.height)
    px_scale = (entry.width / max(mw, 1)) * (entry.height / max(mh, 1))
    W, H = float(entry.width), float(entry.height)
    total = (W * H) or 1.0

    instances = []
    for rank, idx in enumerate(order, start=1):
        x1, y1, x2, y2 = (float(v) for v in xyxy_all[idx])
        bw, bh = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
        area = float(areas[idx]) * px_scale
        instances.append(
            {
                "id": rank,
                "label": "instance",
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "center": [round(x1 + bw / 2, 2), round(y1 + bh / 2, 2)],
                "area_px": int(round(area)),
                "area_ratio": round(area / total, 6),
                "fill_ratio": round((bw * bh) / total, 6),
            }
        )

    doc = {
        "image": entry.name,
        "image_id": entry.image_id,
        "width": entry.width,
        "height": entry.height,
        "count": len(instances),
        "source": "masks",
        "note": "每个框是掩码的最小外接矩形；bbox 为 xywh，原图像素坐标",
        "instances": instances,
    }
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": Path(entry.name).name,
                "width": entry.width,
                "height": entry.height,
            }
        ],
        "annotations": [
            {
                "id": f["id"],
                "image_id": 1,
                "category_id": 1,
                "bbox": f["bbox"],
                "area": f["area_px"],
                "iscrowd": 0,
            }
            for f in instances
        ],
        "categories": [{"id": 1, "name": "instance", "supercategory": "object"}],
    }

    yolo_txt = "".join(
        f"0 {f['center'][0] / W:.6f} {f['center'][1] / H:.6f} "
        f"{f['bbox'][2] / W:.6f} {f['bbox'][3] / H:.6f}\n"
        for f in instances
    )

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, lineterminator="\n")
    writer.writerow(
        ["id", "label", "x1", "y1", "x2", "y2", "w", "h", "area_px", "area_ratio", "fill_ratio"]
    )
    for f in instances:
        x1, y1, x2, y2 = f["bbox_xyxy"]
        writer.writerow(
            [
                f["id"],
                f["label"],
                x1,
                y1,
                x2,
                y2,
                f["bbox"][2],
                f["bbox"][3],
                f["area_px"],
                f["area_ratio"],
                f["fill_ratio"],
            ]
        )
    csv_txt = csv_buf.getvalue()

    stem = Path(entry.name).stem or "image"
    if fmt == "json":
        body = json.dumps(coco, ensure_ascii=False, indent=2).encode("utf-8")
        name = f"{stem}-boxes.json"
        media = "application/json"
    elif fmt == "yolo":
        body = yolo_txt.encode("utf-8")
        name = f"{stem}-labels.txt"
        media = "text/plain; charset=utf-8"
    elif fmt == "csv":
        body = csv_txt.encode("utf-8")
        name = f"{stem}-boxes.csv"
        media = "text/csv; charset=utf-8"
    else:
        gray = cv2.cvtColor(entry.bgr, cv2.COLOR_BGR2GRAY)
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for f in instances:
            x1, y1, x2, y2 = (int(round(v)) for v in f["bbox_xyxy"])
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        ok, png = cv2.imencode(".png", preview)
        if not ok:
            raise HTTPException(status_code=500, detail="预览图编码失败")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("boxes.json", json.dumps(coco, ensure_ascii=False, indent=2))
            zf.writestr("labels_yolo.txt", yolo_txt)
            zf.writestr("boxes.csv", csv_txt)
            zf.writestr("boxes.png", png.tobytes())
            zf.writestr("annotations_full.json", json.dumps(doc, ensure_ascii=False, indent=2))
            zf.writestr(
                "README.txt",
                "由掩码转出的框标注，共 %d 个实例。\n"
                "\n"
                "boxes.json            COCO 结构：images / annotations(bbox=xywh) / categories\n"
                "annotations_full.json 同样的数据，带 xyxy、中心点、面积占比、框内填充率\n"
                "labels_yolo.txt       YOLO 训练格式：`0 cx cy w h`，全部归一化到 0~1\n"
                "boxes.csv             表格，方便 Excel 打开\n"
                "boxes.png             框画在原图上的预览\n"
                "\n"
                "每个框 = 对应掩码的最小外接矩形，坐标基于原图（%d x %d）。\n"
                % (len(instances), entry.width, entry.height),
            )
        body = buf.getvalue()
        name = f"{stem}-boxes.zip"
        media = "application/zip"

    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": disposition(name)},
    )


@app.post("/api/clear/{image_id}")
def clear(image_id: str):
    store.drop(image_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
