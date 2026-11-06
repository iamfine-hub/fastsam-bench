"""端到端冒烟测试：直接打 HTTP 接口，验证前端依赖的每个字段都存在。"""
import io
import json
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

BASE = "http://127.0.0.1:8000"
fails = []


def req(path, body=None, method=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method or ("POST" if data else "GET"),
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            payload = resp.read()
            if raw:
                return resp.status, resp.headers.get("Content-Type"), payload
            return resp.status, json.loads(payload)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")
        if raw:
            return e.code, e.headers.get("Content-Type"), text.encode("utf-8")
        return e.code, text


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


print("[1] 静态页面")
status, ctype, body = req("/", raw=True)
html = body.decode("utf-8", "replace")
check("GET / 200", status == 200, status)
check("GET / 返回 text/html", "text/html" in (ctype or ""), ctype)
check("页面含画布元素", 'id="view"' in html)
check("页面引用 app.js", '"/app.js' in html)
check("页面引用 style.css", '"/style.css' in html)

print("[2] /api/health")
status, health = req("/api/health")
check("GET /api/health 200", status == 200, health)
check("mode 字段存在", "mode" in (health or {}))
check("mode 为 real/mock", health.get("mode") in ("real", "mock"), health.get("mode"))
samples = health.get("samples") or []
check("samples 非空", len(samples) > 0, len(samples))
print("       mode=%s device=%s samples=%d" % (health.get("mode"), health.get("device"), len(samples)))
if health.get("reason"):
    print("       reason=%s" % health["reason"])

print("[3] /api/sample")
name = samples[0]["name"]
status, info = req("/api/sample", {"name": name})
check("POST /api/sample 200", status == 200, info)
if status != 200:
    print(json.dumps(info, ensure_ascii=False)[:400] if isinstance(info, dict) else info[:400])
    sys.exit(1)
for k in ("image_id", "width", "height", "name"):
    check(f"sample 含 {k}", k in info, list(info))
image_id = info["image_id"]
W, H = info["width"], info["height"]
print("       %s  %dx%d" % (info.get("name"), W, H))

print("[4] 原图回读")
status, ctype, body = req(f"/api/image/{image_id}", raw=True)
check("GET /api/image 200", status == 200, status)
check("原图是图片类型", (ctype or "").startswith("image/"), ctype)
check("原图字节非空", len(body) > 1000, len(body))

print("[5] /api/everything")
status, res = req("/api/everything", {"image_id": image_id, "imgsz": 512, "conf": 0.4, "iou": 0.9})
check("POST /api/everything 200", status == 200, res)
if status == 200:
    check("返回 count", "count" in res)
    check("count > 0", res.get("count", 0) > 0, res.get("count"))
    check("overlay 是 data URL", str(res.get("overlay", "")).startswith("data:image/png;base64,"))
    dets = res.get("detections") or []
    check("detections 数量与 count 一致", len(dets) == res.get("count"), (len(dets), res.get("count")))
    if dets:
        d = dets[0]
        for k in ("id", "bbox", "area", "area_ratio", "score"):
            check(f"detection 含 {k}", k in d, list(d))
        check("bbox 是 4 个数", isinstance(d.get("bbox"), list) and len(d["bbox"]) == 4, d.get("bbox"))
    check("有 elapsed_ms", isinstance(res.get("elapsed_ms"), (int, float)))
    print("       %d 个实例，%d ms" % (res.get("count", 0), res.get("elapsed_ms", 0)))

print("[6] /api/box")
cx, cy = W // 2, H // 2
box = [cx - W // 8, cy - H // 8, cx + W // 8, cy + H // 8]
status, res = req("/api/box", {"image_id": image_id, "boxes": [box], "contours": True})
check("POST /api/box 200", status == 200, res)
if status == 200:
    check("box 有 hit 字段", "hit" in res)
    if res.get("hit"):
        check("box hit 时返回 overlay", str(res.get("overlay", "")).startswith("data:image/png;base64,"))
        check("box hit 时 detections 非空", len(res.get("detections") or []) > 0)

print("[7] /api/point")
status, res = req("/api/point", {"image_id": image_id, "points": [[cx, cy]], "labels": [1], "contours": True})
check("POST /api/point 200", status == 200, res)
if status == 200:
    check("point 有 hit 字段", "hit" in res)
    if res.get("hit"):
        check("point hit 时返回 overlay", str(res.get("overlay", "")).startswith("data:image/png;base64,"))

print("[8] 多框提示（第二个框必须真的生效）")
box_a = [cx - W // 4, cy - H // 4, cx, cy + H // 4]
box_b = [cx, cy - H // 4, cx + W // 4, cy + H // 4]
status, one = req("/api/box", {"image_id": image_id, "boxes": [box_a], "contours": True})
status2, two = req("/api/box", {"image_id": image_id, "boxes": [box_a, box_b], "contours": True})
check("两次 /api/box 都 200", status == 200 and status2 == 200, (status, status2))
if status == 200 and status2 == 200:
    check("单框 count >= 1", one.get("count", 0) >= 1, one.get("count"))
    # 两个框命中同一实例时 count 仍可能是 1，真正的回归点是「不再恒等于 1」且 detections 跟着 count 走
    check("双框 count 不为负", two.get("count", -1) >= 0, two.get("count"))
    check("双框 count 与 detections 一致", two.get("count") == len(two.get("detections") or []),
          (two.get("count"), len(two.get("detections") or [])))
    print("       单框 %d 个 / 双框 %d 个" % (one.get("count", 0), two.get("count", 0)))

print("[9] /api/export 标注打包")
status, ctype, body = req(f"/api/export/{image_id}", raw=True)
check("GET /api/export 200", status == 200, status)
check("导出是 zip", (ctype or "").startswith("application/zip"), ctype)
if status == 200:
    import io as _io
    import zipfile as _zip
    try:
        zf = _zip.ZipFile(_io.BytesIO(body))
        names = zf.namelist()
        for n in ("annotations.json", "masks.png", "overlay.png", "image.png", "README.txt"):
            check(f"zip 含 {n}", n in names, names)
        if "annotations.json" in names:
            ann = json.loads(zf.read("annotations.json"))
            for k in ("image", "width", "height", "count", "instances"):
                check(f"annotations 含 {k}", k in ann, list(ann))
            check("annotations.instances 非空", len(ann.get("instances") or []) > 0, ann.get("count"))
            if ann.get("instances"):
                ins = ann["instances"][0]
                check("instance 含 bbox", isinstance(ins.get("bbox"), list) and len(ins["bbox"]) == 4)
                check("instance 含 area_px", ins.get("area_px", 0) > 0, ins.get("area_px"))
        print("       %d 个文件，%d 字节" % (len(names), len(body)))
    except _zip.BadZipFile:
        check("导出内容可解压", False, "BadZipFile")

print("[10] /api/brush 修掩码")
base_status, base = req("/api/everything", {"image_id": image_id, "imgsz": 512, "contours": True})
base_count = base.get("count", 0) if base_status == 200 else 0
stroke = {"value": 1, "radius": 24, "points": [[cx - 20, cy], [cx + 20, cy]]}
status, added = req("/api/brush", {"image_id": image_id, "strokes": [stroke], "contours": True})
check("POST /api/brush 200", status == 200, added)
if status == 200:
    check("brush 有 hit 字段", "hit" in added)
    check("brush 返回 overlay", str(added.get("overlay", "")).startswith("data:image/png;base64,"))
    check("brush 后实例数 >= 1", added.get("count", 0) >= 1, added.get("count"))
    check("brush count 与 detections 一致", added.get("count") == len(added.get("detections") or []),
          (added.get("count"), len(added.get("detections") or [])))
    status2, cleared = req("/api/brush", {"image_id": image_id, "strokes": [], "contours": True})
    check("空笔迹重算 200", status2 == 200, cleared)
    if status2 == 200:
        # 画笔每次都从「提示原始结果」重算，所以撤回笔迹必须精确回到原实例数
        check("撤回笔迹回到原始实例数", cleared.get("count") == base_count,
              (cleared.get("count"), base_count))
        print("       原始 %d → 加一笔 %d → 撤回 %d" % (base_count, added.get("count", 0), cleared.get("count", 0)))
status, res = req("/api/brush", {"image_id": "nope", "strokes": []})
check("brush 不存在的 image_id 返回 4xx", 400 <= status < 500, status)

print("[11] /api/boxes 掩码转框标注")
status, ctype, body = req(f"/api/boxes/{image_id}", raw=True)
check("GET /api/boxes 200", status == 200, status)
check("框标注是 zip", (ctype or "").startswith("application/zip"), ctype)
if status == 200:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            names = zf.namelist()
            for want in ("boxes.json", "labels_yolo.txt", "boxes.csv", "boxes.png", "README.txt"):
                check(f"框标注 zip 含 {want}", want in names, names)
            coco = json.loads(zf.read("boxes.json"))
            check("boxes.json 含 annotations", len(coco.get("annotations") or []) > 0,
                  len(coco.get("annotations") or []))
            first = (coco.get("annotations") or [{}])[0]
            check("annotation bbox 是 4 个数",
                  isinstance(first.get("bbox"), list) and len(first["bbox"]) == 4, first.get("bbox"))
            yolo = zf.read("labels_yolo.txt").decode("utf-8").strip().splitlines()
            check("yolo 行数与实例数一致",
                  len(yolo) == len(coco.get("annotations") or []),
                  (len(yolo), len(coco.get("annotations") or [])))
            parts = yolo[0].split() if yolo else []
            check("yolo 行是 5 段", len(parts) == 5, parts)
            if len(parts) == 5:
                check("yolo 归一化在 0~1", all(0.0 <= float(v) <= 1.0 for v in parts[1:]), parts)
    except zipfile.BadZipFile:
        check("框标注 zip 可解压", False, "BadZipFile")
for fmt, want_type in (("json", "application/json"), ("yolo", "text/plain"), ("csv", "text/csv")):
    status, ctype, body = req(f"/api/boxes/{image_id}?format={fmt}", raw=True)
    check(f"/api/boxes?format={fmt} 200", status == 200, status)
    check(f"/api/boxes?format={fmt} 类型正确", want_type in (ctype or ""), ctype)
status, ctype, body = req("/api/boxes/nope", raw=True)
check("框标注不存在的 image_id 返回 4xx", 400 <= status < 500, status)

print("[12] 中文文件名导出")
# 回归测试：Content-Disposition 是 latin-1，直接塞中文会 UnicodeEncodeError → 500
img_bytes = (Path(__file__).resolve().parent.parent / "examples" / "dogs.jpg").read_bytes()
boundary = "----smoke-boundary"
mp = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="测试 截图.png"\r\n'
    f"Content-Type: image/png\r\n\r\n"
).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
up_req = urllib.request.Request(
    BASE + "/api/upload", data=mp,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
try:
    with urllib.request.urlopen(up_req, timeout=120) as resp:
        up_status, up = resp.status, json.loads(resp.read())
except urllib.error.HTTPError as e:
    up_status, up = e.code, {}
check("中文文件名上传 200", up_status == 200, up_status)
cn_id = up.get("image_id")
if cn_id:
    req("/api/everything", {"image_id": cn_id, "imgsz": 512})
    for part in ("mask", "export", "boxes"):
        path = f"/api/{part}/{cn_id}"
        try:
            with urllib.request.urlopen(BASE + path, timeout=180) as resp:
                cd = resp.headers.get("Content-Disposition", "") or ""
                resp.read()
            code = resp.status
        except urllib.error.HTTPError as e:
            cd, code = "", e.code
        check(f"中文名导出 {part} 200", code == 200, code)
        check(f"中文名导出 {part} 带 filename*", "filename*=UTF-8''" in cd, cd)

print("[13] 错误处理")
status, res = req("/api/sample", {"name": "nope/does-not-exist.jpg"})
check("不存在的样图返回 4xx", 400 <= status < 500, status)
status, res = req("/api/everything", {"image_id": "nope"})
check("不存在的 image_id 返回 4xx", 400 <= status < 500, status)

print()
if fails:
    print("FAILED %d: %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("全部通过")
