"""Publish tracked research documents and source text without importing research code.

Run with Python-Markdown 3.9 installed. The output is a static GitHub Pages site.
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "KennyMcSimpson/measurement-scene-state"
GITHUB = f"https://github.com/{REPOSITORY}"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/main"
CORE = (
    "STATUS.md",
    "EXPERIMENTS.md",
    "README.md",
    "ROADMAP.md",
    "docs/research_summary_20261003.md",
    "docs/handoff/REPOSITORY_MAP.md",
    "docs/handoff/SHARED_WORKSPACE.md",
    "docs/handoff/3D_REPRODUCTION.md",
    "docs/provenance/CLAIMS.md",
    "docs/provenance/DATA.md",
    "docs/PROTOCOL.md",
    "docs/DATASETS.md",
)
CSS = """
:root{color-scheme:light;--ink:#182b40;--muted:#576879;--accent:#096b73}
*{box-sizing:border-box}body{margin:0;background:#fafaf7;color:var(--ink);
font:17px/1.75 system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif}
header,main,footer{max-width:1080px;margin:auto;padding:28px 32px}
header{border-bottom:1px solid #dae0e2}nav{display:flex;gap:20px;flex-wrap:wrap}
a{color:var(--accent);text-underline-offset:3px}h1,h2,h3{line-height:1.3}
h1{font-size:2.2rem;letter-spacing:-.025em}h2{margin-top:2em}
.eyebrow,.meta,footer{color:var(--muted)}.eyebrow{font-size:.85rem;letter-spacing:.1em}
.lead{font-size:1.12rem;max-width:820px}.meta{font-size:.88rem;overflow-wrap:anywhere}
.links{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:22px}
.links h2{font-size:1.1rem;margin-bottom:4px}.links p{margin-top:4px}
article{overflow-wrap:anywhere}table{display:block;overflow-x:auto;border-collapse:collapse}
td,th{padding:10px 14px;border:1px solid #d8dfe2;text-align:left}
pre{background:#edf1f2;padding:20px;border-radius:5px;overflow:auto;line-height:1.55}
pre code{white-space:pre;font-size:.86rem}code{font-size:.9em}
img{max-width:100%;height:auto}blockquote{margin-left:0;padding-left:20px;
border-left:3px solid #9eb6bc;color:var(--muted)}li{margin:5px 0}
.directory li{font-size:.92rem}footer{border-top:1px solid #dae0e2;margin-top:45px}
@media(max-width:650px){header,main,footer{padding:22px 18px}h1{font-size:1.75rem}}
"""
ALLOWED_TAGS = set(
    "a p br hr h1 h2 h3 h4 h5 h6 pre code ul ol li blockquote strong em b i "
    "table thead tbody tr td th img div span details summary sup sub del dl dt dd".split()
)
VOID_TAGS = {"br", "hr", "img"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def text_path(path: str) -> str:
    return "text/" + (path[:-3] + ".txt" if path.endswith(".md") else path + ".txt")


class DocumentHTML(HTMLParser):
    """Keep passive Markdown HTML and resolve links against the original source path."""

    def __init__(self, source: str, pages: dict[str, str], directories: set[str]) -> None:
        super().__init__(convert_charrefs=True)
        self.source = source
        self.pages = pages
        self.directories = directories
        self.output: list[str] = []
        self.blocked = 0

    def target(self, value: str, image: bool = False) -> str:
        parsed = urlsplit(value)
        if parsed.netloc and not parsed.scheme:
            return "https:" + value
        if parsed.scheme or parsed.netloc:
            return value if parsed.scheme in {"http", "https", "mailto"} else "#"
        if not parsed.path:
            return value
        source = posixpath.normpath(
            posixpath.join(posixpath.dirname(self.source), unquote(parsed.path))
        )
        if source.startswith("outputs/"):
            archived = "docs/experiments/" + source[len("outputs/") :]
            if archived in self.pages:
                source = archived
        if source in self.pages and not image:
            value = posixpath.relpath(
                self.pages[source], posixpath.dirname(self.pages[self.source])
            )
        else:
            prefix = (
                RAW
                if image
                else f"{GITHUB}/{'tree' if source in self.directories else 'blob'}/main"
            )
            value = prefix + "/" + quote(source, safe="/")
        if parsed.query:
            value += "?" + parsed.query
        if parsed.fragment:
            value += "#" + parsed.fragment
        return value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "iframe"}:
            self.blocked += 1
        if self.blocked or tag not in ALLOWED_TAGS:
            return
        safe = []
        for key, value in attrs:
            if value is None or key not in {"href", "src", "alt", "title", "id", "class", "open"}:
                continue
            if key in {"href", "src"}:
                value = self.target(value, image=key == "src")
            safe.append(f' {key}="{html.escape(value, quote=True)}"')
        self.output.append(f"<{tag}{''.join(safe)}>")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "iframe"} and self.blocked:
            self.blocked -= 1
            return
        if not self.blocked and tag in ALLOWED_TAGS and tag not in VOID_TAGS:
            self.output.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self.blocked:
            self.output.append(html.escape(data))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    parser.add_argument(
        "--base-url", default="https://kennymcsimpson.github.io/measurement-scene-state/"
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/") + "/"
    if urlsplit(base).scheme != "https":
        parser.error("--base-url must be an HTTPS URL")
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents:
        parser.error("--output must not overwrite the repository or its parents")
    output.mkdir(parents=True, exist_ok=True)
    tracked = git("ls-files", "-z").split("\0")
    documents = sorted(
        path
        for path in tracked
        if path.endswith(".md")
        and ("/" not in path or path.startswith(("docs/", "research/ideas/")))
    )
    sources = sorted(
        path
        for path in tracked
        if (
            path.startswith(("src/", "scripts/", "configs/", "docs/", "tests/"))
            and path.endswith((".py", ".yaml", ".yml", ".toml"))
        )
        or path == "pyproject.toml"
    )
    pages = {path: "pages/" + path[:-3] + ".html" for path in documents}
    pages.update({path: "code/" + path + ".html" for path in sources})
    directories = {str(parent) for path in tracked for parent in Path(path).parents}
    commit = git("rev-parse", "HEAD")
    updated = git("show", "-s", "--format=%cI", "HEAD")

    def write(path: str, content: str) -> None:
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")

    def shell(title: str, body: str, path: str, description: str) -> str:
        return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} | measurement-scene-state</title>
<meta name="description" content="{html.escape(description, quote=True)}">
<meta name="robots" content="index,follow">
<link rel="canonical" href="{base}{quote(path, safe="/")}">
<link rel="stylesheet" href="{base}style.css">
<link rel="alternate" type="text/plain" href="{base}llms.txt" title="Text index">
</head><body><header><nav>
<a href="{base}">Measurement Scene State</a>
<a href="{base}pages/STATUS.html">Status</a>
<a href="{base}pages/EXPERIMENTS.html">Experiments</a>
<a href="{base}sources.html">Source index</a>
<a href="{GITHUB}">GitHub</a></nav></header><main>{body}</main>
<footer>Public research archive · <a href="{GITHUB}/commit/{commit}">{commit[:12]}</a>
 · <time datetime="{updated}">{updated[:10]}</time>
 · <a href="{base}llms.txt">Text index</a> · <a href="{base}sitemap.xml">Sitemap</a>
</footer></body></html>'''

    entries = []
    for path, page in pages.items():
        text = (ROOT / path).read_bytes().decode("utf-8")
        raw = text_path(path)
        write(raw, text)
        if path in documents:
            rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
            resolver = DocumentHTML(path, pages, directories)
            resolver.feed(rendered)
            content = "".join(resolver.output)
        else:
            content = f"<h1>{html.escape(path)}</h1><pre><code>{html.escape(text)}</code></pre>"
        meta = (
            f'<p class="meta">Source: <a href="{GITHUB}/blob/main/{quote(path, safe="/")}">'
            f'{html.escape(path)}</a> · <a href="{base}{quote(raw, safe="/")}">Plain text</a></p>'
        )
        write(page, shell(path, meta + "<article>" + content + "</article>", page, path))
        entries.append(
            {"path": path, "html": page, "text": raw, "raw_url": RAW + "/" + quote(path, safe="/")}
        )

    cards = "".join(
        f'<section><h2><a href="{base}{pages[path]}">{html.escape(path)}</a></h2>'
        f'<p><a href="{base}{text_path(path)}">Plain text / 原文</a></p></section>'
        for path in CORE
        if path in pages
    )
    landing = f'''
<p class="eyebrow">PUBLIC RESEARCH ARCHIVE · KennyMcSimpson</p>
<h1>Measurement Scene State</h1>
<p class="lead">Inverse-JEPA、Dynamic-TTT 与场景状态的研究档案。
本页提供原始文档、实验记录和源代码的公开阅读入口。</p>
<p>Repository: <a href="{GITHUB}">KennyMcSimpson/measurement-scene-state</a></p>
<p>阅读研究结论时，先查看 STATUS.md 和 EXPERIMENTS.md 的证据等级、数据划分与限制。
这些页面同步仓库原文，保留历史记录；具体结论以对应实验协议和报告为准。</p>
<h2>直接读取 / Direct reading</h2>
<p><a href="{base}llms-full.txt">核心文档合并原文 / Core documents in one text file</a> ·
<a href="{base}llms.txt">研究文档与源码的文本索引</a></p>
<p>可把上述链接直接交给具备网页读取功能的 AI。
直接读取公开 URL 不依赖关键词搜索是否已有结果。</p>
<h2>关键文档 / Key documents</h2><div class="links">{cards}</div>
<h2>完整导航 / Browse</h2><p><a href="{base}documents.html">研究文档目录</a> ·
<a href="{base}sources.html">源代码与配置</a></p>'''
    write(
        "index.html",
        shell(
            "Measurement Scene State",
            landing,
            "",
            "KennyMcSimpson/measurement-scene-state: Inverse-JEPA and Dynamic-TTT research "
            "archive, experimental status, protocols, reproduction notes, and source code.",
        ),
    )
    for filename, title, items in (
        ("documents.html", "Research documents / 研究文档", documents),
        ("sources.html", "Source code and configurations / 源码与配置", sources),
    ):
        links = "".join(
            f'<li><a href="{base}{quote(pages[path], safe="/")}">{html.escape(path)}</a>'
            f' · <a href="{base}{quote(text_path(path), safe="/")}">text</a></li>'
            for path in items
        )
        write(
            filename,
            shell(title, f'<h1>{title}</h1><ul class="directory">{links}</ul>', filename, title),
        )
    full = [f"# {REPOSITORY}\n\nRepository: {GITHUB}\nSource commit: {commit}\n"]
    for path in CORE:
        full.append(f"\n\n---\n\nSource file: {path}\nRaw URL: {RAW}/{path}\n\n")
        full.append((ROOT / path).read_bytes().decode("utf-8"))
    write("llms-full.txt", "".join(full))
    index = [
        f"# Measurement Scene State\n\n> Public research archive: {REPOSITORY}.\n",
        f"\nRepository: {GITHUB}\nSource commit: {commit}\n",
        f"\n## Core documents\n\n- [Combined core text]({base}llms-full.txt)\n",
    ]
    for path in CORE:
        index.append(f"- [{path}]({base}{text_path(path)})\n")
    index.append("\n## Published document and source text\n\n")
    for entry in entries:
        index.append(f"- [{entry['path']}]({base}{quote(entry['text'], safe='/')})\n")
    write("llms.txt", "".join(index))
    write(
        "file-index.json",
        json.dumps(
            {"repository": REPOSITORY, "commit": commit, "files": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    urls = ["", "documents.html", "sources.html", *pages.values()]
    locations = "".join(
        f"<url><loc>{html.escape(base + quote(path, safe='/'))}</loc>"
        f"<lastmod>{updated[:10]}</lastmod></url>"
        for path in urls
    )
    write(
        "sitemap.xml",
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + locations + "</urlset>\n",
    )
    write("style.css", CSS)
    write(".nojekyll", "")
    print(f"Built {len(documents)} documents and {len(sources)} source files in {output}")


if __name__ == "__main__":
    main()
