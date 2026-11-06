"""前端一致性检查：app.js 引用的元素 / class 是否真的存在于 HTML / CSS。"""
import pathlib
import re

base = pathlib.Path(__file__).parent / "static"
js = (base / "app.js").read_text(encoding="utf-8")
html = (base / "index.html").read_text(encoding="utf-8")
css = (base / "style.css").read_text(encoding="utf-8")

ids = set(re.findall(r"""['"]#([A-Za-z0-9_-]+)""", js))
ids |= set(re.findall(r"""getElementById\(['"]([A-Za-z0-9_-]+)['"]\)""", js))
html_ids = set(re.findall(r"""id="([A-Za-z0-9_-]+)\"""", html))

print(f"app.js 引用 id 数: {len(ids)}")
print(f"index.html 定义 id 数: {len(html_ids)}")
missing = sorted(ids - html_ids)
print("JS 引用但 HTML 缺失:", missing or "无")
unused = sorted(html_ids - ids)
print("HTML 有但 JS 未用:", unused or "无")

cls = set(re.findall(r"""querySelector(?:All)?\(['"]\.([A-Za-z0-9_-]+)""", js))
classes_in_html = set()
for attr in re.findall(r"""class="([^"]+)\"""", html):
    classes_in_html.update(attr.split())
css_rules = set(re.findall(r"\.([A-Za-z][A-Za-z0-9_-]*)", css))

print("JS 用到的 class:", sorted(cls))
print("HTML 里没有的 class（JS 选中）:", sorted(c for c in cls if c not in classes_in_html) or "无")
print("CSS 里没有样式的 class（HTML+JS）:", sorted(c for c in (cls | classes_in_html) if c not in css_rules) or "无")

# JS 里 classList.add/remove/toggle 用到的类名
dyn = set(re.findall(r"""classList\.(?:add|remove|toggle)\(['"]([A-Za-z0-9_-]+)""", js))
print("动态 class:", sorted(dyn))
print("动态 class 缺 CSS:", sorted(c for c in dyn if c not in css_rules) or "无")
print("style.css 字节数:", len(css))
