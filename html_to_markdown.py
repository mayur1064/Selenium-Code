#!/usr/bin/env python3
"""
Confluence HTML to Markdown Converter
=====================================

A standalone, self-contained script that converts Confluence HTML (body.view format)
into clean, well-formatted Markdown.  All Confluence-specific conversion logic is
built-in — no Confluence API connection needed.

Features
--------
* Panels → GitHub-style alerts (NOTE, TIP, IMPORTANT, WARNING, CAUTION)
* PlantUML diagrams → `` ```plantuml `` code fences
* DrawIO diagrams → Mermaid `` ```mermaid `` code fences
* Tables → pipe tables with colspan / rowspan support
* Expand macros → ``<details><summary>`` HTML
* Column layouts → Markdown tables
* Task lists → GitHub checkboxes
* Code blocks with language detection
* Image captions, emoticons, status badges
* Text highlights, font colours
* Jira issue links, user @-mentions
* Include / excerpt-include macros
* Markdown macros (pass-through)
* Page-properties → front-matter YAML or dataview fields
* Unresolved template-placeholder escaping (Obsidian compatibility)
* Whitespace / ``&nbsp;`` normalisation

Requirements
------------
    pip install markdownify beautifulsoup4 tabulate pyyaml

Usage
-----
.. code-block:: bash

    # Basic conversion
    python confluence_html_to_md.py page_body_view.html -o output.md

    # With title, editor2 XML and storage XML for richer conversion
    python confluence_html_to_md.py page.html \\
        --title "My Page" \\
        --editor2 editor2.xml \\
        --storage body_storage.xml \\
        -o output.md

    # Pipe HTML from stdin
    cat page.html | python confluence_html_to_md.py -o output.md

    # Read from stdin, print to stdout
    cat page.html | python confluence_html_to_md.py

    # With attachment and page link paths
    python confluence_html_to_md.py page.html \\
        --attachment-base attachments/ \\
        --page-base pages/ \\
        --page-href relative \\
        -o output.md

Options
-------
============================ =======================================================
Option                        Description
============================ =======================================================
``-o, --output``              Output Markdown file (default: stdout)
``--title``                   Page title (prepended as H1)
``--editor2``                 Path to editor2 XML (Fabric Editor v2 source)
``--storage``                 Path to body.storage XML
``--base-url``                Confluence base URL for link resolution
``--space-key``               Confluence space key
``--include-title / --no-include-title``
                              Prepend page title as H1 (default: True)
``--page-href``               Link style for page refs: absolute|relative|wiki
                              (default: relative)
``--attachment-href``         Attachment link style (default: relative)
``--attachment-base``         Base dir for attachment link paths
``--page-base``               Base dir for page link paths
``--frontmatter / --no-frontmatter``
                              Include YAML front-matter (default: True)
``--breadcrumbs / --no-breadcrumbs``
                              Include breadcrumbs (default: False)
``--toc / --no-toc``          Render TOC macros (default: True)
``--text-highlights / --no-text-highlights``
                              Convert cell / text highlights (default: True)
``--font-colors / --no-font-colors``
                              Convert font colours (default: True)
``--status-badges / --no-status-badges``
                              Convert status lozenges (default: True)
``--image-captions / --no-image-captions``
                              Render image captions (default: True)
``--include-macro``           Handling for include/excerpt macros:
                              transclusion|inline (default: inline)
``--page-properties-format``  Format for page properties: table|frontmatter|
                              frontmatter_and_table|dataview-inline-field|
                              meta-bind-view-fields (default: frontmatter_and_table)
``--page-properties-report-format``
                              Format for property reports: dataview|table
                              (default: dataview)
``-h, --help``                Show this help message and exit
============================ =======================================================
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import logging
import os
import re
import sys
from collections.abc import Set
from pathlib import Path
from typing import ClassVar
from typing import cast
from urllib.parse import unquote
from urllib.parse import urlparse

import yaml
from bs4 import BeautifulSoup
from bs4 import Tag
from markdownify import ATX
from markdownify import MarkdownConverter
from tabulate import tabulate

__version__ = "1.0.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & compiled regexes
# ---------------------------------------------------------------------------

_MAX_UNICODE_CODEPOINT = 0x10FFFF

_RE_RGB_BG = re.compile(r"background-color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)")
_RE_RGB_COLOR = re.compile(r"(?<![a-z-])color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)")
_RE_COLORID_CSS = re.compile(r"(?<![>\w])\[data-colorid=(\w+)\]\{color:(#[0-9a-fA-F]+)\}")
_RE_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_RE_LEADING_BR_OR_WS = re.compile(r"^(?:\s|<br\s*/?>)+")
_RE_TRAILING_BR_OR_WS = re.compile(r"(?:\s|<br\s*/?>)+$")
_ANGLE_BRACKET_RE = re.compile(r"<([^<>\n]*)>")
_CODE_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_AUTOLINK_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]{1,31}:[^\s<>]*$")
_AUTOLINK_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~\-]+@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*$"
)

_DEFAULT_HEADER_BGS = frozenset({"#f4f5f7", "#f2f2f2"})

# Background colours for Confluence status-badge lozenges
_LOZENGE_COLORS: dict[str, str] = {
    "aui-lozenge-complete": "#cce0ff",
    "aui-lozenge-success": "#baf3db",
    "aui-lozenge-current": "#f8e6a0",
    "aui-lozenge-error": "#ffd5d2",
    "aui-lozenge-progress": "#dfd8fd",
}

_ATLASSIAN_EMOTICONS: dict[str, str] = {
    "atlassian-check_mark": "✅",
    "atlassian-cross_mark": "❌",
    "atlassian-yes": "👍",
    "atlassian-no": "👎",
    "atlassian-information": "\u2139\ufe0f",
    "atlassian-warning": "⚠️",
    "atlassian-forbidden": "🚫",
    "atlassian-plus": "\u2795",
    "atlassian-minus": "\u2796",
    "atlassian-question": "❓",
    "atlassian-exclamation": "❗",
    "atlassian-light_on": "💡",
    "atlassian-light_off": "💡",
    "atlassian-star_yellow": "⭐",
    "atlassian-blue_star": "🔵",
    "atlassian-smile": "😊",
    "atlassian-sad": "😞",
    "atlassian-tongue": "😛",
    "atlassian-biggrin": "😁",
    "atlassian-wink": "😉",
}

_HTML_ELEMENTS = frozenset(
    {
        "a", "abbr", "acronym", "address", "area", "article", "aside", "audio",
        "b", "base", "bdi", "bdo", "blockquote", "body", "br", "button",
        "canvas", "caption", "cite", "code", "col", "colgroup", "data", "datalist",
        "dd", "del", "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed",
        "fieldset", "figcaption", "figure", "font", "footer", "form",
        "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hgroup", "hr",
        "html", "i", "iframe", "img", "input", "ins", "kbd", "keygen",
        "label", "legend", "li", "link", "main", "map", "mark", "menu",
        "menuitem", "meta", "meter", "nav", "noscript", "object", "ol",
        "optgroup", "option", "output", "p", "picture", "pre", "progress",
        "q", "rp", "rt", "ruby", "s", "samp", "script", "section", "select",
        "small", "source", "span", "strong", "style", "sub", "summary",
        "sup", "table", "tbody", "td", "template", "textarea", "tfoot",
        "th", "thead", "time", "title", "tr", "track", "u", "ul", "var",
        "video", "wbr",
    }
)

MACROS_TO_IGNORE: Set[str] = frozenset({"qc-read-and-understood-signature-box"})


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_cell_highlight_hex(el: Tag) -> str | None:
    """Return Confluence cell background hex from data-highlight-colour, or None."""
    val = el.get("data-highlight-colour")
    if not isinstance(val, str):
        return None
    val = val.strip().lower()
    if not val or val == "transparent" or val in _DEFAULT_HEADER_BGS:
        return None
    if _RE_HEX_COLOR.match(val):
        return val
    return None


def _get_int_attr(cell: Tag, attr: str, default: str = "1") -> int:
    """Safely get an integer attribute from a tag."""
    val = cell.get(attr, default)
    if isinstance(val, list):
        val = val[0] if val else default
    try:
        return int(str(val))
    except (ValueError, TypeError):
        return int(default)


def make_empty_cell() -> Tag:
    """Return an empty <td> Tag."""
    return Tag(name="td")


def _normalize_table_cell_text(text: str) -> str:
    text = text.replace("|", "\\|").replace("\n", "<br/>")
    text = _RE_LEADING_BR_OR_WS.sub("", text)
    return _RE_TRAILING_BR_OR_WS.sub("", text)


def _parse_image_captions(storage_xml: str) -> dict[str, str]:
    """Return {filename: caption} parsed from Confluence storage-format XML."""
    captions: dict[str, str] = {}
    if not storage_xml:
        return captions
    for block in re.findall(r"<ac:image[^>]*>.*?</ac:image>", storage_xml, re.DOTALL):
        filename_m = re.search(r'ri:filename="([^"]+)"', block)
        if not filename_m:
            continue
        caption_m = re.search(r"<ac:caption[^>]*>(.*?)</ac:caption>", block, re.DOTALL)
        if not caption_m:
            continue
        caption_content = caption_m.group(1)
        cdata_m = re.search(
            r"<ac:plain-text-body>\s*<!\[CDATA\[(.*?)\]\]>\s*</ac:plain-text-body>",
            caption_content,
            re.DOTALL,
        )
        if cdata_m:
            caption = cdata_m.group(1).strip()
        else:
            caption = BeautifulSoup(caption_content, "html.parser").get_text().strip()
        if caption:
            captions[filename_m.group(1)] = caption
    return captions


def github_heading_slug(text: str) -> str:
    """Generate a GitHub-compatible heading anchor slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-{2,}", "-", text)


def sanitize_key(s: str, connector: str = "_") -> str:
    """Convert an input string to a valid YAML front-matter key."""
    s = s.lower()
    s = re.sub(f"[^a-z0-9{connector}]", connector, s)
    s = re.sub(f"{connector}+", connector, s)
    s = s.strip(connector)
    if not re.match(r"^[a-z]", s):
        s = f"key{connector}{s}"
    return s


# ---------------------------------------------------------------------------
# DrawIO → Mermaid conversion
# ---------------------------------------------------------------------------

def load_and_parse_drawio(file_path: str | Path) -> str | None:
    """Load a DrawIO file and extract Mermaid diagram as Markdown code fence."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    xml_content = file_path.read_text(encoding="utf-8")
    try:
        soup = BeautifulSoup(xml_content, "xml")
        user_object = soup.find("UserObject")
        if user_object is None:
            return None
        try:
            attrs = cast("dict[str, str]", user_object.attrs)
            mermaid_data_attr = attrs.get("mermaidData")
            if mermaid_data_attr is None:
                return None
            raw = html_module.unescape(mermaid_data_attr)
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "data" in parsed:
                    raw = parsed["data"]
            except (json.JSONDecodeError, TypeError):
                pass
            return f"```mermaid\n{raw}\n```"
        except AttributeError:
            return None
    except Exception:
        logger.exception("Error extracting mermaid data from DrawIO XML")
        return None


# ---------------------------------------------------------------------------
# Table conversion (handles colspan / rowspan)
# ---------------------------------------------------------------------------

def pad_rows(rows: list[list[Tag]]) -> list[list[Tag]]:
    """Pad table rows to handle rowspan and colspan for Markdown conversion."""
    padded: list[list[Tag]] = []
    occ: dict[tuple[int, int], Tag] = {}
    for r, row in enumerate(rows):
        if not row:
            continue
        cur: list[Tag] = []
        c = 0
        for cell in row:
            while (r, c) in occ:
                cur.append(occ.pop((r, c)))
                c += 1
            rs = _get_int_attr(cell, "rowspan", "1")
            cs = _get_int_attr(cell, "colspan", "1")
            cur.append(cell)
            if cs > 1:
                cur.extend(make_empty_cell() for _ in range(1, cs))
            for i in range(rs):
                for j in range(cs):
                    if i or j:
                        occ[(r + i, c + j)] = make_empty_cell()
            c += cs
        while (r, c) in occ:
            cur.append(occ.pop((r, c)))
            c += 1
        padded.append(cur)
    return padded


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class ConfluenceMarkdownConverter(MarkdownConverter):
    """Custom MarkdownConverter for Confluence HTML → Markdown conversion.

    This is a fully offline converter — it does not connect to any Confluence
    API.  Pass the HTML bodies (view, editor2, storage) and optional metadata
    directly.
    """

    class Options(MarkdownConverter.DefaultOptions):
        bullets = "-"
        heading_style = ATX

    def __init__(self, **options) -> None:
        super().__init__(**options)
        # Inputs
        self._page_html: str = ""
        self._editor2: str = ""
        self._body_storage: str = ""
        self._page_title: str = ""
        self._base_url: str = ""
        self._space_key: str = ""
        # Path bases for link generation
        self._page_base: str = ""
        self._attachment_base: str = ""
        # Caches
        self._colorid_map_cache: dict[str, str] | None = None
        self._image_captions_cache: dict[str, str] | None = None
        self._panel_icon_map_cache: dict[str, str] | None = None
        self._plantuml_index: int = 0
        self._storage_plantuml_macros_cache: list[Tag] | None = None
        # Page properties for front-matter
        self.page_properties: dict[str, object] = {}
        # --- Configuration flags ---
        self.include_title: bool = True
        self.page_href: str = "relative"  # absolute | relative | wiki
        self.attachment_href: str = "relative"
        self.include_frontmatter: bool = True
        self.include_breadcrumbs: bool = False
        self.include_toc: bool = True
        self.convert_text_highlights: bool = True
        self.convert_font_colors: bool = True
        self.convert_status_badges: bool = True
        self.image_captions: bool = True
        self.include_macro: str = "inline"  # transclusion | inline
        self.comments_export: str = "none"  # none | inline | footer | all
        self.page_properties_format: str = "frontmatter_and_table"
        self.page_properties_report_format: str = "dataview"
        self.attachment_path_template: str = "{attachment_title}"
        self.page_path_template: str = "{page_title}"
        # Accumulator for inline comment markers
        self._marked_texts: dict[str, str] = {}
        # Ancestor breadcrumbs
        self._ancestor_titles: list[str] = []

    # ---- Public API ----

    @property
    def html(self) -> str:
        """The HTML that will be converted. Optionally prepends a title H1."""
        if self.include_title and self._page_title:
            return f"<h1>{self._page_title}</h1>{self._page_html}"
        return self._page_html

    def convert_html(self) -> str:
        """Run the full conversion pipeline and return Markdown."""
        html = self._strip_excerpt_include_panel_titles(self.html)
        md_body = self.convert(html)
        md_body = self._escape_template_placeholders(md_body)
        parts: list[str] = []
        if self.include_frontmatter:
            fm = self._build_frontmatter()
            if fm:
                parts.append(fm)
        if self.include_breadcrumbs and self._ancestor_titles:
            parts.append(self._build_breadcrumbs())
        parts.append(md_body)
        return "\n".join(parts) + "\n"

    # ---- Configuration setters ----

    def set_html(self, html: str) -> None:
        self._page_html = html

    def set_editor2(self, editor2: str) -> None:
        self._editor2 = editor2

    def set_body_storage(self, body_storage: str) -> None:
        self._body_storage = body_storage

    def set_title(self, title: str) -> None:
        self._page_title = title

    def set_base_url(self, url: str) -> None:
        self._base_url = url

    def set_space_key(self, key: str) -> None:
        self._space_key = key

    def set_ancestor_titles(self, titles: list[str]) -> None:
        self._ancestor_titles = titles

    # ---- Cached properties ----

    @property
    def _colorid_map(self) -> dict[str, str]:
        if self._colorid_map_cache is None:
            cache: dict[str, str] = {}
            soup = BeautifulSoup(self.html, "html.parser")
            for style_tag in soup.find_all("style"):
                css = style_tag.get_text()
                for m in _RE_COLORID_CSS.finditer(css):
                    color_id = m.group(1)
                    if color_id not in cache:
                        cache[color_id] = m.group(2)
            self._colorid_map_cache = cache
        return self._colorid_map_cache

    @property
    def _storage_plantuml_macros(self) -> list[Tag]:
        if self._storage_plantuml_macros_cache is None:
            macros: list[Tag] = []
            if self._body_storage:
                wrapped = f"<root>{self._body_storage}</root>"
                soup = BeautifulSoup(wrapped, "xml")
                macros.extend(
                    macro
                    for macro in soup.find_all("structured-macro")
                    if isinstance(macro, Tag) and macro.get("name") == "plantuml"
                )
            self._storage_plantuml_macros_cache = macros
        return self._storage_plantuml_macros_cache

    @property
    def _image_captions(self) -> dict[str, str]:
        if self._image_captions_cache is None:
            self._image_captions_cache = _parse_image_captions(self._body_storage)
        return self._image_captions_cache

    @property
    def _panel_icon_map(self) -> dict[str, str]:
        if self._panel_icon_map_cache is None:
            cache: dict[str, str] = {}
            if self._editor2:
                wrapped = f"<root>{self._editor2}</root>"
                soup = BeautifulSoup(wrapped, "xml")
                panel_names = {"panel", "info", "note", "tip", "warning"}
                for macro in soup.find_all("structured-macro"):
                    if not isinstance(macro, Tag):
                        continue
                    if macro.get("name") not in panel_names:
                        continue
                    macro_id = macro.get("macro-id")
                    if not macro_id:
                        continue
                    emoji = self._extract_panel_emoji(macro)
                    if emoji:
                        cache[str(macro_id)] = emoji
            self._panel_icon_map_cache = cache
        return self._panel_icon_map_cache

    # ---- Front-matter & breadcrumbs ----

    def _build_frontmatter(self) -> str:
        if not self.page_properties:
            return ""
        yml = yaml.dump(self.page_properties, indent=2).strip()
        yml = re.sub(r"^( *)(- )", r"\1  \2", yml, flags=re.MULTILINE)
        return f"---\n{yml}\n---\n"

    def _build_breadcrumbs(self) -> str:
        return " > ".join(self._ancestor_titles) + "\n"

    def set_page_properties(self, **props: object) -> None:
        for key, value in props.items():
            if value:
                self.page_properties[sanitize_key(key)] = value

    # ---- Static helpers ----

    @staticmethod
    def _extract_panel_emoji(macro: Tag) -> str | None:
        params: dict[str, str] = {}
        for p in macro.find_all("parameter", recursive=False):
            if not isinstance(p, Tag):
                continue
            name = p.get("name")
            if name:
                params[str(name)] = p.get_text(strip=True)
        if text := params.get("panelIconText"):
            return text
        if icon_id := params.get("panelIconId"):
            try:
                cps = [int(cp, 16) for cp in icon_id.split("-")]
                if all(0 <= cp <= _MAX_UNICODE_CODEPOINT for cp in cps):
                    return "".join(chr(cp) for cp in cps)
            except (OverflowError, ValueError):
                pass
        return None

    # ---- Path helpers ----

    def _get_path_for_href(self, name: str, style: str) -> str:
        """Build a link path for a file/attachment based on link style."""
        if style == "absolute":
            base = self._attachment_base if self._attachment_base else ""
            return "/" + str(Path(base) / name).lstrip("/").replace("\\", "/")
        elif style == "wiki":
            return name
        else:  # relative
            return name

    # ---- Escape helpers ----

    def escape(self, text: str, parent_tags: list[str]) -> str:
        escaped: str = cast("Any", MarkdownConverter).escape(self, text, parent_tags)
        return escaped.replace("[", r"\[").replace("]", r"\]")

    def _escape_template_placeholders(self, text: str) -> str:
        r"""Escape <placeholder> patterns that Obsidian misparses as HTML tags."""

        def _escape_if_placeholder(m: re.Match) -> str:
            inner = m.group(1)
            if _AUTOLINK_URI_RE.match(inner) or _AUTOLINK_EMAIL_RE.match(inner):
                return m.group(0)
            stripped = inner.strip().lstrip("/")
            tag_name = re.split(r"[\s/]", stripped)[0].lower() if stripped else ""
            if tag_name in _HTML_ELEMENTS or inner.startswith("!"):
                return m.group(0)
            return f"\\<{inner}\\>"

        lines = text.split("\n")
        result = []
        in_fence = False
        for line in lines:
            if _CODE_FENCE_RE.match(line):
                in_fence = not in_fence
                result.append(line)
                continue
            if in_fence:
                result.append(line)
                continue
            parts = _INLINE_CODE_RE.split(line)
            codes = _INLINE_CODE_RE.findall(line)
            processed = []
            for i, part in enumerate(parts):
                processed.append(_ANGLE_BRACKET_RE.sub(_escape_if_placeholder, part))
                if i < len(codes):
                    processed.append(codes[i])
            result.append("".join(processed))
        return "\n".join(result)

    def _normalize_unicode_whitespace(self, text: str) -> str:
        """Normalize Unicode whitespace to regular spaces (fixes &nbsp; issues)."""
        normalized = text
        for char in text:
            if char.isspace() and char not in " \n\r\t":
                normalized = normalized.replace(char, " ")
        return normalized

    # ==================================================================
    #  Element-level converters
    # ==================================================================

    # -- Text formatting (whitespace-normalised) --

    def convert_em(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = self._normalize_unicode_whitespace(text)
        return super().convert_em(el, text, parent_tags)

    def convert_strong(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = self._normalize_unicode_whitespace(text)
        return super().convert_strong(el, text, parent_tags)

    def convert_code(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = self._normalize_unicode_whitespace(text)
        return super().convert_code(el, text, parent_tags)

    def convert_i(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = self._normalize_unicode_whitespace(text)
        return super().convert_i(el, text, parent_tags)

    def convert_b(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = self._normalize_unicode_whitespace(text)
        return super().convert_b(el, text, parent_tags)

    # -- Pre / code blocks --

    def convert_pre(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if not text:
            return ""
        code_language = ""
        if el.has_attr("data-syntaxhighlighter-params"):
            match = re.search(r"brush:\s*([^;]+)", str(el["data-syntaxhighlighter-params"]))
            if match:
                code_language = match.group(1)
        if "@startuml" in text:
            code_language = "plantuml"
        return f"\n\n```{code_language}\n{text}\n```\n\n"

    # -- Subscript / superscript --

    def convert_sub(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        return f"<sub>{text}</sub>"

    def convert_sup(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if el.previous_sibling is None:
            return f"[^{text}]:"  # Footnote definition
        return f"[^{text}]"

    # -- Time --

    def convert_time(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if el.has_attr("datetime"):
            return f"{el['datetime']}"
        return f"{text}"

    # -- Lists (task lists handled in the unified convert_li below) --

    def _convert_emoticon(self, el: BeautifulSoup) -> str | None:
        classes = el.get("class") or []
        if "emoticon" not in classes:
            return None
        emoji_id = str(el.get("data-emoji-id", ""))
        fallback = str(el.get("data-emoji-fallback", ""))
        if fallback and not fallback.startswith(":"):
            return fallback
        if emoji_id:
            try:
                codepoints = [int(cp, 16) for cp in emoji_id.split("-")]
                if all(0 <= cp <= _MAX_UNICODE_CODEPOINT for cp in codepoints):
                    return "".join(chr(cp) for cp in codepoints)
            except (OverflowError, ValueError):
                pass
            if emoji_id in _ATLASSIAN_EMOTICONS:
                return _ATLASSIAN_EMOTICONS[emoji_id]
        shortname = str(el.get("data-emoji-shortname", ""))
        return shortname or fallback or str(el.get("alt", "")) or None

    def convert_img(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if emoticon := self._convert_emoticon(el):
            return emoticon

        url_src = str(el.get("src", ""))
        alt = str(el.get("alt", text))

        # DrawIO embedded in PNG
        if ".drawio.png" in url_src:
            filename = urlparse(url_src).path.split("/")[-1]
            drawio_path = Path(self._attachment_base) / filename.replace(".png", ".drawio") if self._attachment_base else None
            if drawio_path and drawio_path.exists():
                mermaid_md = load_and_parse_drawio(str(drawio_path))
                if mermaid_md:
                    return mermaid_md

        href = el.get("href") or text
        src = url_src if url_src else ""

        # For offline use, try to build a path from attachment filename
        if self._attachment_base and src:
            parsed = urlparse(src)
            filename = unquote(parsed.path.split("/")[-1]) if parsed.path else ""
            if filename:
                src = self._get_path_for_href(filename, self.attachment_href)

        if src:
            caption = self._image_captions.get(alt, "") if self.image_captions else ""
            img_md = f"![{alt}]({src.replace(' ', '%20')})"
            return f"{img_md}\n*{caption}*" if caption else img_md

        if href:
            return f"![{text}]({href})"
        return text

    # -- Links --

    def convert_a(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        # User mentions
        if "user-mention" in str(el.get("class")):
            return text.removeprefix("@")

        # Broken Confluence links
        if "createpage.action" in str(el.get("href")) or "createlink" in str(el.get("class")):
            return f"[[{text}]]"

        href_str = str(el.get("href", ""))

        # Attachment links via download path
        if href_str and "/download/attachments/" in href_str:
            parsed = urlparse(href_str)
            filename = unquote(parsed.path.split("/")[-1]) if parsed.path else ""
            if filename:
                path = self._get_path_for_href(filename, self.attachment_href)
                return f"[{text}]({path.replace(' ', '%20')})"

        # Attachment links via data attributes
        if "attachment" in str(el.get("data-linked-resource-type")):
            filename = str(el.get("data-linked-resource-filename", ""))
            if filename:
                path = self._get_path_for_href(filename, self.attachment_href)
                return f"[{text}]({path.replace(' ', '%20')})"
            return f"[{text}]({href_str})"

        # Page links via data attributes
        if "page" in str(el.get("data-linked-resource-type")):
            page_title = str(el.get("data-linked-resource-content-title", text))
            page_id = str(el.get("data-linked-resource-id", ""))
            if self.page_href == "wiki":
                return f"[[{page_title}]]"
            if self._page_base:
                path = f"{self._page_base}/{sanitize_filename(page_title)}.md"
            else:
                path = f"{sanitize_filename(page_title)}.md"
            return f"[{page_title}]({path.replace(' ', '%20')})"

        # Anchor links
        if href_str.startswith("#"):
            if self.page_href == "wiki":
                return f"[[#{text}]]"
            return f"[{text}](#{github_heading_slug(href_str[1:])})"

        return super().convert_a(el, text, parent_tags)

    # -- Spans (highlights, colours, status, inline macros) --

    def _span_highlight(self, style: str, text: str) -> str | None:
        bg_m = _RE_RGB_BG.search(style)
        if not bg_m:
            return None
        hex_color = _rgb_to_hex(int(bg_m.group(1)), int(bg_m.group(2)), int(bg_m.group(3)))
        return f'<mark style="background: {hex_color};">{text}</mark>'

    def _span_font_color(self, el: BeautifulSoup, style: str, text: str) -> str | None:
        color_m = _RE_RGB_COLOR.search(style)
        if color_m:
            hex_color = _rgb_to_hex(
                int(color_m.group(1)), int(color_m.group(2)), int(color_m.group(3))
            )
            return f'<font style="color: {hex_color};">{text}</font>'
        color_id = el.get("data-colorid")
        if isinstance(color_id, str):
            hex_color = self._colorid_map.get(color_id)
            if hex_color:
                return f'<font style="color: {hex_color};">{text}</font>'
        return None

    def _span_status_badge(self, el: BeautifulSoup, text: str) -> str | None:
        if not self.convert_status_badges:
            return None
        classes = el.get("class") or []
        if not isinstance(classes, list):
            return None
        if "status-macro" not in classes:
            return None
        bg = "#dfe1e6"
        for cls, color in _LOZENGE_COLORS.items():
            if cls in classes:
                bg = color
                break
        return f'<mark style="background: {bg};">{text.strip()}</mark>'

    def convert_span(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if el.has_attr("data-macro-name"):
            if el["data-macro-name"] == "jira":
                return self.convert_jira_issue(el, text, parent_tags)
            if el["data-macro-name"] == "status":
                result = self._span_status_badge(el, text)
                if result is not None:
                    return result
            if el["data-macro-name"] == "plantuml":
                return self.convert_plantuml(el, text, parent_tags)

        if el.has_attr("class") and "inline-comment-marker" in el["class"]:
            return self._convert_inline_comment_marker(el, text)

        raw_style = el.get("style", "")
        style = raw_style if isinstance(raw_style, str) else ""
        if self.convert_text_highlights:
            result = self._span_highlight(style, text)
            if result is not None:
                return result

        if self.convert_font_colors:
            result = self._span_font_color(el, style, text)
            if result is not None:
                return result

        return text

    def _convert_inline_comment_marker(self, el: BeautifulSoup, text: str) -> str:
        if self.comments_export in ("inline", "all"):
            ref = el.get("data-ref", "")
            if isinstance(ref, str) and ref and ref not in self._marked_texts:
                self._marked_texts[ref] = text
        return text

    # -- Jira issue links --

    def convert_jira_issue(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        issue_key = el.get("data-jira-key")
        link = cast("BeautifulSoup", el.find("a", {"class": "jira-issue-key"}))
        if not link:
            return text
        if not issue_key:
            return self.process_tag(link, parent_tags)
        return f"[[{issue_key}]]({link.get('href')})"

    # -- PlantUML --

    def _extract_uml_from_editor2(self, macro_id: str) -> str | None:
        if not self._editor2:
            return None
        wrapped = f"<root>{self._editor2}</root>"
        soup = BeautifulSoup(wrapped, "xml")
        for macro in soup.find_all("structured-macro"):
            if not isinstance(macro, Tag):
                continue
            if macro.get("name") != "plantuml" or macro.get("macro-id") != macro_id:
                continue
            plain_text_body = macro.find("plain-text-body")
            if not isinstance(plain_text_body, Tag):
                continue
            cdata = plain_text_body.get_text(strip=True)
            if not cdata:
                continue
            try:
                return json.loads(cdata).get("umlDefinition") or None
            except json.JSONDecodeError:
                return None
        return None

    def _extract_uml_from_storage(self) -> str | None:
        storage_macros = self._storage_plantuml_macros
        idx = self._plantuml_index
        self._plantuml_index += 1
        if idx >= len(storage_macros):
            return None
        plain_text_body = storage_macros[idx].find("plain-text-body")
        if not isinstance(plain_text_body, Tag):
            return None
        uml = plain_text_body.get_text(strip=True)
        return uml or None

    def convert_plantuml(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        macro_id = el.get("data-macro-id")
        if macro_id:
            uml = self._extract_uml_from_editor2(str(macro_id))
            if uml:
                return f"\n```plantuml\n{uml}\n```\n\n"
        uml = self._extract_uml_from_storage()
        if uml:
            return f"\n```plantuml\n{uml}\n```\n\n"
        return "\n<!-- PlantUML diagram (source not found) -->\n\n"

    # -- DrawIO --

    def _convert_drawio_embedded_mermaid(self, filename: str) -> str | None:
        drawio_title = filename.removesuffix(".png")
        drawio_path = Path(self._attachment_base) / f"{drawio_title}.drawio" if self._attachment_base else None
        if drawio_path and drawio_path.exists():
            return load_and_parse_drawio(str(drawio_path))
        return None

    def convert_drawio(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if match := re.search(r"\|diagramName=(.+?)\|", str(el)):
            drawio_name = match.group(1)
            preview_name = f"{drawio_name}.png"
            preview_path = self._get_path_for_href(preview_name, self.attachment_href)
            drawio_path = self._get_path_for_href(f"{drawio_name}.drawio", self.attachment_href)
            img_embed = f"![{drawio_name}]({preview_path.replace(' ', '%20')})"
            return f"\n[{img_embed}]({drawio_path.replace(' ', '%20')})\n\n"
        return ""

    # -- Div dispatch (macros & special containers) --

    def convert_div(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        # Confluence macros
        if el.has_attr("data-macro-name"):
            macro_name = str(el["data-macro-name"])
            if macro_name in MACROS_TO_IGNORE:
                return ""

            macro_handlers = {
                "panel": self.convert_alert,
                "info": self.convert_alert,
                "note": self.convert_alert,
                "tip": self.convert_alert,
                "warning": self.convert_alert,
                "details": self.convert_page_properties,
                "drawio": self.convert_drawio,
                "plantuml": self.convert_plantuml,
                "scroll-ignore": self.convert_hidden_content,
                "toc": self.convert_toc,
                "jira": self.convert_jira_table,
                "attachments": self.convert_attachments,
                "markdown": self.convert_markdown,
                "mohamicorp-markdown": self.convert_markdown,
                "include": self.convert_include,
                "excerpt-include": self.convert_include,
            }
            if macro_name in macro_handlers:
                return macro_handlers[macro_name](el, text, parent_tags)

        # Special class-based containers
        class_handlers = {
            "expand-container": self.convert_expand_container,
            "columnLayout": self.convert_column_layout,
        }
        for class_name, handler in class_handlers.items():
            if class_name in str(el.get("class", "")):
                return handler(el, text, parent_tags)

        return super().convert_div(el, text, parent_tags)

    # -- Alerts (panel → GitHub alert) --

    def convert_alert(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        alert_type_map = {
            "info": "IMPORTANT",
            "panel": "NOTE",
            "tip": "TIP",
            "note": "WARNING",
            "warning": "CAUTION",
        }
        alert_emoji_map = {
            "NOTE": "\U0001f4dd",
            "TIP": "\U0001f4a1",
            "IMPORTANT": "❗",
            "WARNING": "⚠️",
            "CAUTION": "\U0001f6d1",
        }

        alert_type = alert_type_map.get(str(el["data-macro-name"]), "NOTE")

        macro_id = el.get("data-macro-id")
        custom_emoji = self._panel_icon_map.get(str(macro_id)) if macro_id else None
        emoji = custom_emoji or alert_emoji_map[alert_type]

        tags = parent_tags if isinstance(parent_tags, list | set) else set()
        if "td" in tags or "th" in tags:
            return f"{emoji} {text.strip()}"

        blockquote = super().convert_blockquote(el, text, parent_tags)
        return f"\n> [!{alert_type}]{blockquote}"

    # -- Expand container --

    def convert_expand_container(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        summary_element = el.find("span", class_="expand-control-text")
        summary_text = summary_element.get_text().strip() if summary_element else "Click here to expand..."
        content_element = el.find("div", class_="expand-content")
        content = self.process_tag(content_element, parent_tags).strip() if content_element else ""
        return f"\n<details>\n<summary>{summary_text}</summary>\n\n{content}\n\n</details>\n\n"

    # -- Column layout → table --

    def convert_column_layout(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        cells = el.find_all("div", {"class": "cell"})
        if len(cells) < 2:
            return super().convert_div(el, text, parent_tags)
        html = f"<table><tr>{''.join([f'<td>{cell!s}</td>' for cell in cells])}</tr></table>"
        return self.convert_table(BeautifulSoup(html, "html.parser"), text, parent_tags)

    # -- Page properties --

    def convert_page_properties(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str | None:
        fmt = self.page_properties_format
        if fmt == "table":
            return text

        rows = [
            cast("list[Tag]", tr.find_all(["th", "td"]))
            for tr in cast("list[Tag]", el.find_all("tr"))
            if tr
        ]
        if not rows:
            return None

        props: dict[str, str] = {}
        key_counts: dict[str, int] = {}
        for row in rows:
            if len(row) == 2:
                raw_key = row[0].get_text(strip=True)
                count = key_counts.get(raw_key, 0) + 1
                key_counts[raw_key] = count
                unique_key = raw_key if count == 1 else f"{raw_key} {count}"
                props[unique_key] = self.convert(str(row[1])).strip()

        if fmt in ("frontmatter", "frontmatter_and_table", "meta-bind-view-fields"):
            self.set_page_properties(**props)

        if fmt == "frontmatter":
            return None
        if fmt == "frontmatter_and_table":
            return text
        if fmt == "dataview-inline-field":
            lines = "\n".join(f"{k}:: {v}" for k, v in props.items())
            return f"\n{lines}\n"
        # meta-bind-view-fields
        table_data = [(f"**{k}**", f"`VIEW[{{{sanitize_key(k)}}}][text(renderMarkdown)]`") for k in props]
        return "\n\n" + tabulate(table_data, headers=["", ""], tablefmt="pipe") + "\n"

    def convert_page_properties_report(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        data_cql = el.get("data-cql")
        if not data_cql:
            return ""
        if self.page_properties_report_format == "dataview":
            dql = self._cql_to_dataview(el, str(data_cql))
            if dql is not None:
                return f"\n```dataview\n{dql}\n```\n"
        soup = BeautifulSoup(self._page_html, "html.parser")
        table = soup.find("table", {"data-cql": data_cql})
        if not table:
            return ""
        return self.convert_table(table, "", parent_tags)

    def _cql_to_dataview(self, el: BeautifulSoup, cql: str) -> str | None:
        current_content_id = str(el.get("data-current-content-id", ""))
        headings_raw = str(el.get("data-headings", ""))
        first_col = str(el.get("data-first-column-heading", "Title"))
        sort_by = str(el.get("data-sort-by", first_col))
        reverse_sort = str(el.get("data-reverse-sort", "false")).lower() == "true"

        label_conditions = [
            m.group(1) for m in re.finditer(r'label\s*=\s*"([^"]+)"', cql, re.IGNORECASE)
        ]
        parent_match = re.search(r'parent\s*=\s*"?(\d+)"?', cql, re.IGNORECASE)
        current_content_match = re.search(
            r'(?:ancestor|parent)\s*=\s*currentContent\s*\(\s*\)', cql, re.IGNORECASE
        )

        from_clause: str | None = None
        if current_content_match or (parent_match and parent_match.group(1) == current_content_id):
            from_clause = f'"{self._page_base or "."}"'

        if from_clause is None and not label_conditions:
            return None

        lines: list[str] = []
        if headings_raw:
            headings = [h.strip() for h in headings_raw.split(",") if h.strip()]
            col_names = ", ".join(f'{sanitize_key(h)} AS "{h}"' for h in headings)
            lines.append(f"TABLE {col_names}")
        else:
            lines.append("TABLE")

        from_parts = ([from_clause] if from_clause else []) + [f"#{lbl}" for lbl in label_conditions]
        if from_parts:
            lines.append("FROM " + " AND ".join(from_parts))

        sort_col = sanitize_key(sort_by)
        sort_dir = "DESC" if reverse_sort else "ASC"
        lines.append(f"SORT {sort_col} {sort_dir}")
        return "\n".join(lines)

    # -- Attachments macro --

    def convert_attachments(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        return "\n<!-- Attachments macro (requires Confluence API to resolve) -->\n\n"

    # -- TOC --

    def convert_toc(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if not self.include_toc:
            return ""
        return "\n<!-- Table of Contents (rendered by Confluence) -->\n\n"

    # -- Hidden content --

    def convert_hidden_content(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        content = super().convert_p(el, text, parent_tags)
        if not content.strip():
            return ""
        return f"\n<!--{content}-->\n"

    # -- Jira table --

    def convert_jira_table(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        jira_tables = BeautifulSoup(self._page_html, "html.parser").find_all("div", {"class": "jira-table"})
        if not jira_tables:
            return text
        return self.process_tag(jira_tables[0], parent_tags)

    # -- Include / Excerpt-Include --

    def convert_include(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        macro_name = str(el.get("data-macro-name", ""))
        macro_id = el.get("data-macro-id")
        target_title: str | None = None
        if macro_id and isinstance(macro_id, str):
            target_title = self._extract_include_target_title(macro_id)

        if self.include_macro == "transclusion" and target_title:
            return f"\n![[{target_title}]]\n\n"

        if self.include_macro == "transclusion":
            logger.warning(
                "%s macro found but target page title could not be resolved; "
                "falling back to inline content",
                macro_name,
            )

        inline = super().convert_div(el, text, parent_tags)
        if macro_name == "excerpt-include":
            title_note = f" from page '{target_title}'" if target_title else ""
            return (
                f"\n<!-- excerpt start{title_note} -->\n"
                f"{inline}"
                f"\n<!-- excerpt end{title_note} -->\n\n"
            )
        return inline

    def _strip_excerpt_include_panel_titles(self, html_str: str) -> str:
        soup = BeautifulSoup(html_str, "html.parser")
        for el in soup.find_all(attrs={"data-macro-name": "excerpt-include"}):
            self._unwrap_excerpt_include_panel(el)
        return str(soup)

    def _unwrap_excerpt_include_panel(self, el: Tag) -> None:
        classes = el.get("class") or []
        if not isinstance(classes, list) or "panel" not in classes:
            return
        header = el.find("div", class_="panelHeader")
        if isinstance(header, Tag):
            header.decompose()
        content = el.find("div", class_="panelContent")
        if isinstance(content, Tag):
            content.unwrap()

    def _extract_include_target_title(self, macro_id: str) -> str | None:
        wrapped_editor2 = f"<root>{self._editor2}</root>"
        soup_editor2 = BeautifulSoup(wrapped_editor2, "xml")
        for macro in soup_editor2.find_all("structured-macro"):
            if not isinstance(macro, Tag):
                continue
            if macro.get("name") not in ("include", "excerpt-include"):
                continue
            if macro.get("macro-id") != macro_id:
                continue
            ri_page = macro.find("page")
            if isinstance(ri_page, Tag):
                title = ri_page.get("content-title")
                if isinstance(title, str) and title:
                    return title
        return None

    # -- Markdown macro (pass-through) --

    def _extract_markdown_from_body(self, el: BeautifulSoup) -> str | None:
        # Try plain-text-body
        plain_text_body = el.find("plain-text-body") or el.find("ac:plain-text-body")
        if isinstance(plain_text_body, Tag):
            return plain_text_body.get_text()

        # Check structured-macro child
        structured_macro = el.find("structured-macro") or el.find("ac:structured-macro")
        if isinstance(structured_macro, Tag):
            ptb = structured_macro.find("plain-text-body") or structured_macro.find("ac:plain-text-body")
            if isinstance(ptb, Tag):
                return ptb.get_text()

        # Try parameter for mohamicorp-markdown
        param = el.find("parameter", {"name": "markdown"}) or el.find("ac:parameter", {"ac:name": "markdown"})
        if isinstance(param, Tag):
            return param.get_text()

        if isinstance(structured_macro, Tag):
            param = structured_macro.find("parameter", {"name": "markdown"}) or structured_macro.find("ac:parameter", {"ac:name": "markdown"})
            if isinstance(param, Tag):
                return param.get_text()

        return None

    def _extract_markdown_from_editor2(self, macro_id: str) -> str | None:
        if not self._editor2:
            return None
        wrapped_editor2 = f"<root>{self._editor2}</root>"
        soup_editor2 = BeautifulSoup(wrapped_editor2, "xml")
        for macro in soup_editor2.find_all("structured-macro"):
            if not isinstance(macro, Tag):
                continue
            if macro.get("name") in ("markdown", "mohamicorp-markdown") and macro.get("macro-id") == macro_id:
                plain_text_body = macro.find("plain-text-body")
                if isinstance(plain_text_body, Tag):
                    return plain_text_body.get_text(strip=True)
                param = macro.find("parameter", {"name": "markdown"})
                if isinstance(param, Tag):
                    return param.get_text(strip=True)
        return None

    def convert_markdown(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        macro_name = el.get("data-macro-name", "")
        markdown_content = self._extract_markdown_from_body(el)
        if not markdown_content:
            macro_id = el.get("data-macro-id")
            if macro_id and isinstance(macro_id, str):
                markdown_content = self._extract_markdown_from_editor2(macro_id)
        if not markdown_content:
            logger.warning("Markdown macro (%s) found but no content could be extracted", macro_name)
            return f"\n<!-- Markdown macro ({macro_name}) content not found -->\n\n"
        return f"\n{markdown_content}\n\n"

    # ---- Table conversion (with colspan / rowspan) ----

    def convert_table(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        if el.has_attr("class") and "metadata-summary-macro" in el["class"]:
            return self.convert_page_properties_report(el, text, parent_tags)

        if "table" in parent_tags:
            return str(el)

        row_tags: list[Tag] = []
        for child in el.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
            if child.name == "tr":
                row_tags.append(cast("Tag", child))
            else:
                row_tags.extend(cast("list[Tag]", child.find_all("tr", recursive=False)))
        rows = [
            cast("list[Tag]", tr.find_all(["td", "th"], recursive=False))
            for tr in row_tags
            if tr
        ]
        if not rows:
            return ""

        padded_rows = pad_rows(rows)
        converted = [
            [self.process_tag(cell, parent_tags={"table"}) for cell in row]
            for row in padded_rows
        ]

        has_header = all(cell.name == "th" for cell in rows[0])
        if has_header:
            return tabulate(converted[1:], headers=converted[0], tablefmt="pipe")

        return tabulate(converted, headers=[""] * len(converted[0]), tablefmt="pipe")

    def _wrap_cell_highlight(self, el: BeautifulSoup, text: str) -> str:
        if not self.convert_text_highlights:
            return text
        bg = _extract_cell_highlight_hex(el)
        if bg is None:
            return text
        inner = text or "&nbsp;"
        return f'<mark style="background: {bg};">{inner}</mark>'

    def convert_td(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = _normalize_table_cell_text(text)
        return self._wrap_cell_highlight(el, text)

    def convert_th(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        text = _normalize_table_cell_text(text)
        return self._wrap_cell_highlight(el, text)

    def convert_tr(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        return text

    def convert_thead(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        return text

    def convert_tbody(self, el: BeautifulSoup, text: str, parent_tags: list[str]) -> str:
        return text

    # -- Lists inside table cells --

    ParentTags = list[str] | set[str]

    @staticmethod
    def _normalize_parent_tags(parent_tags: ParentTags | bool) -> ParentTags:
        return parent_tags if isinstance(parent_tags, list | set) else set()

    def convert_ol(self, el: BeautifulSoup, text: str, parent_tags: ParentTags | bool) -> str:
        tags = self._normalize_parent_tags(parent_tags)
        if "td" in tags:
            lines = text.splitlines()
            if not lines:
                return ""
            start = int(el.get("start") or 1)
            numbered = [
                f"{start + i}. {item}".rstrip() if item.strip() else str(start + i)
                for i, item in enumerate(lines)
            ]
            return "<br>".join(n for n in numbered if n)
        return super().convert_ol(el, text, tags)

    def convert_li(self, el: BeautifulSoup, text: str, parent_tags: ParentTags | bool) -> str:
        tags = self._normalize_parent_tags(parent_tags)
        if "td" in tags:
            return text.strip().removesuffix("<br/>") + "\n"
        # Delegate to the task-list-aware convert_li above
        bullet = self.options["bullets"][0]
        md = MarkdownConverter.convert_li(self, el, text, tags)
        if el.has_attr("data-inline-task-id"):
            is_checked = el.has_attr("class") and "checked" in el["class"]
            return md.replace(f"{bullet} ", f"{bullet} {'[x]' if is_checked else '[ ]'} ", 1)
        return md

    def convert_ul(self, el: BeautifulSoup, text: str, parent_tags: ParentTags | bool) -> str:
        tags = self._normalize_parent_tags(parent_tags)
        if "td" in tags:
            items = [item for item in text.splitlines() if item.strip()]
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            return "- " + "<br>- ".join(items)
        return super().convert_ul(el, text, tags)

    def convert_p(self, el: BeautifulSoup, text: str, parent_tags: ParentTags | bool) -> str:
        tags = self._normalize_parent_tags(parent_tags)
        md = super().convert_p(el, text, tags)
        if "td" in tags:
            md = md.replace("\n", "") + "<br/>"
        return md


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str) -> str:
    """Basic filename sanitization for page titles used in link paths."""
    sanitized = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    sanitized = sanitized.rstrip(" .")
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    name = Path(sanitized).stem.upper()
    if name in reserved:
        sanitized = f"{sanitized}_"
    return sanitized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Confluence HTML (body.view) to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python confluence_html_to_md.py page.html -o output.md
  python confluence_html_to_md.py page.html --title "My Page" --editor2 e2.xml --storage s.xml -o out.md
  cat page.html | python confluence_html_to_md.py -o output.md
        """,
    )
    parser.add_argument("html_file", nargs="?", help="Path to Confluence body.view HTML file (or stdin if omitted)")
    parser.add_argument("-o", "--output", help="Output Markdown file (default: stdout)")
    parser.add_argument("--title", default="", help="Page title (prepended as H1)")
    parser.add_argument("--editor2", help="Path to editor2 XML (Fabric Editor v2 source)")
    parser.add_argument("--storage", help="Path to body.storage XML")
    parser.add_argument("--base-url", default="", help="Confluence base URL")
    parser.add_argument("--space-key", default="", help="Confluence space key")
    parser.add_argument("--include-title", action=argparse.BooleanOptionalAction, default=True,
                        help="Prepend page title as H1 (default: True)")
    parser.add_argument("--page-href", choices=["absolute", "relative", "wiki"], default="relative",
                        help="Link style for page references (default: relative)")
    parser.add_argument("--attachment-href", choices=["absolute", "relative", "wiki"], default="relative",
                        help="Attachment link style (default: relative)")
    parser.add_argument("--attachment-base", default="", help="Base directory for attachment link paths")
    parser.add_argument("--page-base", default="", help="Base directory for page link paths")
    parser.add_argument("--frontmatter", action=argparse.BooleanOptionalAction, default=True,
                        help="Include YAML front-matter (default: True)")
    parser.add_argument("--breadcrumbs", action=argparse.BooleanOptionalAction, default=False,
                        help="Include breadcrumbs (default: False)")
    parser.add_argument("--toc", action=argparse.BooleanOptionalAction, default=True,
                        help="Render TOC macros (default: True)")
    parser.add_argument("--text-highlights", action=argparse.BooleanOptionalAction, default=True,
                        help="Convert text/cell highlights (default: True)")
    parser.add_argument("--font-colors", action=argparse.BooleanOptionalAction, default=True,
                        help="Convert font colours (default: True)")
    parser.add_argument("--status-badges", action=argparse.BooleanOptionalAction, default=True,
                        help="Convert status badges (default: True)")
    parser.add_argument("--image-captions", action=argparse.BooleanOptionalAction, default=True,
                        help="Render image captions (default: True)")
    parser.add_argument("--include-macro", choices=["transclusion", "inline"], default="inline",
                        help="Include/excerpt macro handling (default: inline)")
    parser.add_argument("--page-properties-format",
                        choices=["table", "frontmatter", "frontmatter_and_table",
                                 "dataview-inline-field", "meta-bind-view-fields"],
                        default="frontmatter_and_table",
                        help="Format for page properties (default: frontmatter_and_table)")
    parser.add_argument("--page-properties-report-format", choices=["dataview", "table"],
                        default="dataview",
                        help="Format for property reports (default: dataview)")
    parser.add_argument("--ancestors", nargs="*", default=[],
                        help="Ancestor page titles for breadcrumbs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    # Read HTML input
    if args.html_file:
        html_content = Path(args.html_file).read_text(encoding="utf-8")
    else:
        html_content = sys.stdin.read()

    # Read optional bodies
    editor2 = ""
    body_storage = ""
    if args.editor2:
        editor2 = Path(args.editor2).read_text(encoding="utf-8")
    if args.storage:
        body_storage = Path(args.storage).read_text(encoding="utf-8")

    # Build and configure converter
    conv = ConfluenceMarkdownConverter()
    conv.set_html(html_content)
    conv.set_editor2(editor2)
    conv.set_body_storage(body_storage)
    conv.set_title(args.title)
    conv.set_base_url(args.base_url)
    conv.set_space_key(args.space_key)
    conv.set_ancestor_titles(args.ancestors)

    conv.include_title = args.include_title
    conv.page_href = args.page_href
    conv.attachment_href = args.attachment_href
    conv._attachment_base = args.attachment_base
    conv._page_base = args.page_base
    conv.include_frontmatter = args.frontmatter
    conv.include_breadcrumbs = args.breadcrumbs
    conv.include_toc = args.toc
    conv.convert_text_highlights = args.text_highlights
    conv.convert_font_colors = args.font_colors
    conv.convert_status_badges = args.status_badges
    conv.image_captions = args.image_captions
    conv.include_macro = args.include_macro
    conv.page_properties_format = args.page_properties_format
    conv.page_properties_report_format = args.page_properties_report_format

    # Convert
    markdown_output = conv.convert_html()

    # Write output
    if args.output:
        Path(args.output).write_text(markdown_output, encoding="utf-8")
        print(f"✅ Markdown written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(markdown_output)


if __name__ == "__main__":
    main()
