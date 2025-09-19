# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, uuid, pickle, hashlib, base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional, Tuple
from datetime import datetime, timezone

import regex as re2  # 仍保留，如未用可移除
from dateutil import parser as dtparser
from dateutil import tz
import pytz
from tqdm import tqdm  # 如未用可移除
import ftfy

# ---- Groq 安全匯入（若未安裝則退化為規則式摘要） ----
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import orjson as fastjson
    def dumps(o: Any) -> str:
        return fastjson.dumps(o, option=fastjson.OPT_NON_STR_KEYS|fastjson.OPT_SERIALIZE_NUMPY).decode()
except Exception:
    def dumps(o: Any) -> str:
        return json.dumps(o, ensure_ascii=False)

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import numpy as np
except Exception as e:
    raise SystemExit("NumPy required: pip install numpy")

# ------------------- DEFAULTS -------------------
DEFAULTS = {
    "timezone_parse": "Asia/Taipei",
    "stores": {"zh": "indices/store_zh", "en": "indices/store_en"},
    "chunking": {
        "pdf": {"size": 500, "overlap": 80},
        "blog": {"size": 500, "overlap": 80},
        # 新增：摘要 txt 切塊
        "summary": {"size": 900, "overlap": 120}
    },
    "embedding": {
        "zh": "BAAI/bge-large-zh-v1.5",
        "en": "BAAI/bge-large-en-v1.5"
    },
    "weights": {
        "default": 1.0,
        "executive_summary": 1.3,
        "conclusion": 1.2,
        "qa": 1.1,
        "mdna": 1.35,
        "risk_factors": 1.25,
        "market_risk": 1.15,
        "financials": 1.10
    },
    "multimodal": {
        "enable": True,
        "table": {"max_rows": 60},
        "vlm": {"enable": True, "max_tokens": 180},
        "vision": {"enable": True, "max_tokens": 300}
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "text_model": "llama-3.3-70b",
        "vision_model": "llama-4-scout"     # 依 Groq 控制台實際可用的 vision 型號
    }
}

SECTION_VALUES = {"executive_summary","conclusion","methodology","prepared_remarks","qa","other"}
TAIPEI = tz.gettz(DEFAULTS["timezone_parse"]) or pytz.timezone(DEFAULTS["timezone_parse"])  # type: ignore
UTC = tz.UTC

# ------------------- UTILS -------------------
FNAME_DATE_PAT = re.compile(r"(20\d{2})[-/._\s]?(0[1-9]|1[0-2])[-/._\s]?(0[1-9]|[12]\d|3[01])")

def date_from_name_to_utc_iso(name: str, *, hour_local: int = 9) -> Optional[str]:
    m = FNAME_DATE_PAT.search(name or "")
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    local_dt = datetime(y, mo, d, hour_local, 0, 0, tzinfo=TAIPEI)
    return local_dt.astimezone(UTC).isoformat().replace("+00:00", "Z")

def has_han(txt: str) -> bool:
    try:
        import regex as _rx
        return _rx.search(r"\p{Han}", txt) is not None
    except Exception:
        return any('\u4e00' <= ch <= '\u9fff' for ch in txt)

def now_utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00','Z')

def sha1_of_text(text: str) -> str:
    h = hashlib.sha1(); h.update(text.encode('utf-8','ignore')); return f"sha1:{h.hexdigest()}"

def sha1_of_bytes(b: bytes) -> str:
    h = hashlib.sha1(); h.update(b); return f"sha1:{h.hexdigest()}"

def to_utc_iso(dt_like: str|datetime|None, assume_tz=TAIPEI) -> Optional[str]:
    if dt_like is None: return None
    if isinstance(dt_like, datetime):
        if dt_like.tzinfo is None: dt_like = dt_like.replace(tzinfo=assume_tz)
        return dt_like.astimezone(UTC).isoformat().replace('+00:00','Z')
    try:
        d = dtparser.parse(str(dt_like))
        if d.tzinfo is None: d = d.replace(tzinfo=assume_tz)
        return d.astimezone(UTC).isoformat().replace('+00:00','Z')
    except Exception:
        return None

def norm_lang(v: str) -> str:
    v = (v or "").lower()
    if v.startswith("zh"):
        return "zh"
    return "en"

def df_to_markdown_csv(df):
    """盡量輸出 Markdown（若沒 tabulate 則 fallback），並永遠輸出 CSV。"""
    csv = df.to_csv(index=False)
    try:
        md = df.to_markdown(index=False)  # 依賴 tabulate
    except Exception:
        # fallback：手刻 Markdown 表格
        headers = list(map(str, df.columns))
        rows = df.astype(str).values.tolist()
        sep = "|" + "|".join("---" for _ in headers) + "|"
        head = "|" + "|".join(headers) + "|"
        body = "\n".join("|" + "|".join(r) + "|" for r in rows)
        md = "\n".join([head, sep, body]) if rows else "\n".join([head, sep])
    return md, csv

def summarize_table_rule(md: str, title: str = "", lang: str = "zh") -> str:
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    n_rows = max(0, len(lines) - 2)
    bullet = "•" if lang.startswith("zh") else "-"
    return f"{bullet} 表格：{title or '（無題）'}\n{bullet} 欄位列：{header}\n{bullet} 約 {n_rows} 行資料"

def analyze_page_for_objects(plumber_page, fitz_page):
    """回傳 (tables, images)"""
    try:
        tables = plumber_page.find_tables() or []
    except Exception:
        tables = []
    images = []
    try:
        for xref, *_ in fitz_page.get_images(full=True):
            images.append({"xref": xref})
    except Exception:
        pass
    return tables, images

class GroqHelper:
    def __init__(self, api_key: Optional[str], text_model: Optional[str], vision_model: Optional[str] = None):
        self.client = Groq(api_key=api_key) if (Groq and api_key) else None
        self.text_model = text_model
        self.vision_model = vision_model

    def summarize_table(self, md: str, title: str, lang: str = "zh", max_tokens: int = 180) -> str:
        if not self.client or not self.text_model:
            return summarize_table_rule(md, title, lang)
        sys = (
            "You are a precise data summarizer. Given a table in GitHub Markdown, "
            "return 3-6 bullets with key columns, extremes, and notable trends. "
            "If you include numbers, COPY them exactly from the table. "
            "Do not invent values. Keep within ~120 tokens."
        )
        user = f"Language: {lang}\nTitle: {title}\nTable (GitHub Markdown):\n{md}"
        try:
            resp = self.client.chat.completions.create(
                model=self.text_model,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            out = (resp.choices[0].message.content or "").strip()
            return out or summarize_table_rule(md, title, lang)
        except Exception:
            return summarize_table_rule(md, title, lang)

    def vision_extract(self, image_bytes: bytes, title: str, *, max_tokens: int = 300, lang: str = "zh") -> Dict[str, Any]:
        """用 Vision LLM 讓模型自行判斷圖片類型，並保留可驗證數據。"""
        if not self.client or not self.vision_model:
            return {"type": "other", "caption": f"{title} (vision disabled)"}
        sys = "You convert images of tables/charts into precise, verifiable text."
        prompt = (
            "Identify whether the image is (1) a TABLE, (2) a CHART/GRAPH, or (3) OTHER.\n"
            "If TABLE: transcribe EXACTLY into GitHub Markdown and CSV. Keep signs, % and units; do not round or invent values.\n"
            "If CHART: output compact JSON with numeric points you can confidently read (no guessing):\n"
            "{ \"series\": [{\"name\": \"...\", \"points\": [{\"x\": <number or string>, \"y\": <number>}]}],"
            "  \"axes\": {\"x_label\": \"...\", \"y_label\": \"...\", \"unit\": \"...\"} }\n"
            "Also include a short natural-language caption.\n"
            "If OTHER: just return a short caption.\n"
            "Return a single JSON object with keys: type, and corresponding fields.\n"
            f"Language: {lang}"
        )
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "system", "content": sys
                },{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + f"\nTitle: {title}"},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }],
                temperature=0.0, max_tokens=max_tokens
            )
            txt = (resp.choices[0].message.content or "").strip()
            try:
                return json.loads(txt)
            except Exception:
                return {"type": "other", "caption": txt}
        except Exception:
            return {"type": "other", "caption": f"{title} (vision failed)"}

# ------------------- CHUNKERS -------------------
def split_text_by_tokens(text: str, size: int, overlap: int) -> List[str]:
    words = text.split(); chunks=[]
    if not words: return chunks
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i+size]
        chunks.append(' '.join(chunk_words))
    return chunks

# ====== 新增：摘要判斷與切塊 ======
SEC_PATTERNS = [
    (re.compile(r'^\s*(Executive Summary|摘要)[:：]?', re.I), "executive_summary"),
    (re.compile(r'^\s*(Conclusion|結論)[:：]?', re.I), "conclusion"),
    (re.compile(r'^\s*(Q&A|Q＆A|QA|問答|問與答)\b', re.I), "qa"),
    (re.compile(r'^\s*(Prepared Remarks|開場|主旨|說明)\b', re.I), "prepared_remarks"),
]

def guess_section_from_chunk(text: str) -> str:
    """根據 chunk 的第一行粗略判斷章節，對應 DEFAULTS['weights']。"""
    head = (text or "").strip().splitlines()[0][:160]
    for pat, lab in SEC_PATTERNS:
        if pat.search(head):
            return lab if lab in SECTION_VALUES else "other"
    return "other"

def chunk_summary_text(text: str, size: int, overlap: int) -> List[str]:
    """摘要 txt 的切塊：先以空行分段，再做聚合，最後 fallback token-ish 切法。"""
    if not text.strip():
        return []
    paras = [ftfy.fix_text(p.strip()) for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    if paras:
        buf = ""
        hard_limit = size * 6  # 與 blog 區塊一致的聚合邏輯
        for para in paras:
            if len(buf) + len(para) < hard_limit:
                buf += ("\n\n" if buf else "") + para
            else:
                chunks.append(buf); buf = para
        if buf:
            chunks.append(buf)
    if not chunks:
        chunks = split_text_by_tokens(text, size, overlap)
    return chunks

# ------------------- IO -------------------
def blog_jsonl_iter(path: Path) -> Iterable[Dict[str,Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def pdf_to_dict(pdf_path: Path) -> Dict[str, Any]:
    """
    將 PDF 轉成 dict：
    {
      "source": ".../x.pdf",
      "filename": "x.pdf",
      "type": "pdf",
      "pages": [
        {"page": 1, "text": "...", "tables": [
           {"columns": [...], "rows": [[...], ...], "bbox": (x0,y0,x1,y1), "table_index": 0},
           ...
        ]},
        ...
      ]
    }
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed")
    try:
        import pdfplumber
    except Exception:
        raise RuntimeError("pdfplumber not installed: pip install pdfplumber")

    pages: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        plumber_doc = pdfplumber.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"PDF open failed: {pdf_path}: {e}")

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            text = ftfy.fix_text(page.get_text("text") or "").strip()

            p2 = plumber_doc.pages[i]
            tables_out = []
            tcount = 0
            for t in (p2.find_tables() or []):
                rows = t.extract() or []
                if len(rows) < 2:
                    continue
                header = [str(c).strip() for c in rows[0]]
                data_rows = [[str(c).strip() for c in r] for r in rows[1:]]
                tables_out.append({
                    "columns": header,
                    "rows": data_rows,
                    "bbox": getattr(t, "bbox", None),
                    "table_index": tcount
                })
                tcount += 1

            pages.append({
                "page": i + 1,
                "text": text,
                "tables": tables_out
            })
    finally:
        try:
            doc.close()
        except Exception:
            pass
        try:
            plumber_doc.close()
        except Exception:
            pass

    return {
        "source": str(pdf_path),
        "filename": pdf_path.name,
        "type": "pdf",
        "pages": pages
    }

class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def encode(self, texts: List[str]):
        import numpy as np
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype='float32')

def save_faiss(index_path: Path, vectors, payloads: List[Dict[str,Any]]):
    import faiss, numpy as np
    index_path.parent.mkdir(parents=True, exist_ok=True)
    d = int(vectors.shape[1])
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    faiss.write_index(index, str(index_path))
    with open(index_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(payloads, f)

def pack_payloads(metas: List[Dict[str, Any]], texts: List[str]) -> List[Dict[str, Any]]:
    out = []
    n = min(len(metas), len(texts))
    for i in range(n):
        m = dict(metas[i]); t = texts[i]
        m["page_content"] = t
        m["create_at"] = m.get("published_date") or m.get("event_date") or m.get("ingested_at")
        m["source"] = m.get("source") or m.get("source_type") or ""
        m["title"] = (m.get("title") or "無標題") or "無標題"
        out.append(m)
    return out

# ------------------- PIPELINE -------------------
@dataclass
class BuildContext:
    data_root: Path
    out_meta: Path
    config: Dict[str,Any]
    version: str

def build(ctx: BuildContext):
    tz_name = ctx.config.get("timezone_parse", DEFAULTS["timezone_parse"]) or "Asia/Taipei"
    global TAIPEI
    TAIPEI = tz.gettz(tz_name) or pytz.timezone(tz_name)  # type: ignore
    stores = ctx.config.get("stores", DEFAULTS["stores"])

    payloads_by_lang: Dict[str, List[dict]] = {"zh": [], "en": []}
    texts_by_lang:    Dict[str, List[str]]  = {"zh": [], "en": []}
    ingested_at = now_utc_iso()

    groq_cfg = ctx.config.get("groq", {})
    groq_helper = GroqHelper(
        api_key=os.getenv(groq_cfg.get("api_key_env", "GROQ_API_KEY")),
        text_model=groq_cfg.get("text_model"),
        vision_model=groq_cfg.get("vision_model"),
    )
    mm_cfg = ctx.config.get("multimodal", {})

    # Blogs
    for lang_code, rel in (("en","blogs_en/blog_posts_en.jsonl"),("zh","blogs_zh/blog_posts_zh.jsonl")):
        p = ctx.data_root / rel
        if not p.exists():
            continue
        for row in blog_jsonl_iter(p):
            post_id = str(row.get("post_id") or row.get("id") or uuid.uuid4())
            content = str(row.get("content",""))
            if not content.strip():
                continue

            lang_norm = norm_lang(row.get("lang") or lang_code)

            title = str(row.get("title") or "").strip()
            create_at_iso = to_utc_iso(row.get("create_at") or row.get("created_at"))
            tags = row.get("tags") or []
            if isinstance(tags, str):
                tags = [t for t in re.split(r"[,;\s]+", tags) if t]
            tags = [t.strip().lower() for t in tags if t]

            size = int(ctx.config["chunking"]["blog"]["size"])
            overlap = int(ctx.config["chunking"]["blog"]["overlap"])
            paras = [ftfy.fix_text(p.strip()) for p in re.split(r"\n\s*\n+", content) if p.strip()]
            if not paras:
                paras = split_text_by_tokens(content, size, overlap)

            chunks: List[str] = []
            buf = ""
            for para in paras:
                if len(buf) + len(para) < size*6:
                    buf += ("\n" if buf else "") + para
                else:
                    chunks.append(buf); buf = para
            if buf: chunks.append(buf)
            if not chunks:
                chunks = split_text_by_tokens(content, size, overlap)

            doc_id = f"blog:{post_id}"
            for i, ch in enumerate(chunks):
                chunk_id = f"{doc_id}#p{i:04d}"
                payload = {
                    "page_content": ch,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": "blog",
                    "source_type": "blog",
                    "lang": lang_norm,
                    "title": title or "無標題",
                    "document_title": title or "無標題",
                    "create_at": create_at_iso,
                    "created_at": create_at_iso,
                    "published_date": create_at_iso,
                    "tags": tags,
                    "weight": float(ctx.config["weights"].get("default", 1.0)),
                    "vector_model": ctx.config["embedding"][lang_norm],
                    "chunking": {"size": size, "overlap": overlap},
                    "ingested_at": ingested_at,
                    "version": ctx.version,
                }
                payloads_by_lang[lang_norm].append(payload)
                texts_by_lang[lang_norm].append(ch)

    # PDFs EN
    pdf_en_dir = ctx.data_root / "pdf_en"
    if pdf_en_dir.exists():
        if fitz is None:
            print("[WARN] PyMuPDF not installed; skip pdf_en")
        else:
            import pdfplumber, pandas as pd
            for pdf_path in sorted(pdf_en_dir.glob("*.pdf")):
                try:
                    doc = fitz.open(pdf_path)
                    plumber_doc = pdfplumber.open(str(pdf_path))
                except Exception as e:
                    print(f"[WARN] PDF open failed: {pdf_path}: {e}")
                    continue

                base = pdf_path.stem
                event_date = date_from_name_to_utc_iso(base)
                report_type = "special_topic"
                size = int(ctx.config["chunking"]["pdf"]["size"])
                overlap = int(ctx.config["chunking"]["pdf"]["overlap"])
                doc_id = f"pdf_report:{base}"
                LANG = "en"

                for i in range(len(doc)):
                    page = doc.load_page(i)
                    text = ftfy.fix_text(page.get_text("text") or "").strip()

                    # === 文字 chunk ===
                    if text:
                        for j, ch in enumerate(split_text_by_tokens(text, size, overlap)):
                            payload = {
                                "page_content": ch,
                                "doc_id": doc_id,
                                "chunk_id": f"{doc_id}#p{i+1:03d}-{j:02d}",
                                "source": "pdf_report",
                                "source_type": "pdf_report",
                                "lang": LANG,
                                "title": base,
                                "filename": pdf_path.name,
                                "report_type": report_type,
                                "page": i + 1,
                                "published_date": event_date,
                                "event_date": event_date,
                                "published_at": event_date,
                                "created_at": event_date,
                                "create_at": event_date,
                                "tags": [],
                                "weight": float(ctx.config["weights"].get("default", 1.0)),
                                "vector_model": ctx.config["embedding"][LANG],
                                "chunking": {"size": size, "overlap": overlap},
                                "ingested_at": ingested_at,
                                "version": ctx.version,
                            }
                            payloads_by_lang[LANG].append(payload)
                            texts_by_lang[LANG].append(ch)

                    # === 表格抽取 + 摘要 ===
                    if mm_cfg.get("enable", True):
                        try:
                            plumber_page = plumber_doc.pages[i]
                            for t_idx, t in enumerate(plumber_page.find_tables() or []):
                                rows = t.extract()
                                if not rows or len(rows) < 2:
                                    continue
                                header, data_rows = rows[0], rows[1:]
                                if not any(data_rows):
                                    continue
                                import pandas as pd
                                df = pd.DataFrame(data_rows, columns=header)
                                mx = int(mm_cfg["table"]["max_rows"])
                                if len(df) > mx:
                                    df = df.head(mx)
                                md, csv = df_to_markdown_csv(df)
                                bbox = getattr(t, "bbox", None)
                                base_id = f"{doc_id}#p{i+1:03d}-table{t_idx:02d}"
                                
                                table_payload = {
                                    "page_content": md,
                                    "doc_id": doc_id,
                                    "chunk_id": base_id,
                                    "source": "pdf_report",
                                    "source_type": "table",
                                    "lang": LANG,
                                    "title": base,
                                    "filename": pdf_path.name,
                                    "page": i + 1,
                                    "table_id": t_idx,
                                    "bbox": bbox,
                                    "view": "table_markdown",
                                    "table_csv": csv,
                                    "published_date": event_date,
                                    "event_date": event_date,
                                    "vector_model": ctx.config["embedding"][LANG],
                                    "weight": float(ctx.config["weights"].get("default", 1.0)),
                                    "ingested_at": ingested_at,
                                    "version": ctx.version,
                                }
                                payloads_by_lang[LANG].append(table_payload)
                                texts_by_lang[LANG].append(md)
                        except Exception as e:
                            print(f"[WARN] table extract failed on {pdf_path.name} p{i+1}: {e}")

                    # === 圖片 → Vision 自動判斷（表格/圖表/其他） ===
                    if mm_cfg.get("vision", {}).get("enable", True) and groq_helper.vision_model:
                        try:
                            plumber_page = plumber_doc.pages[i]
                            _, images_found = analyze_page_for_objects(plumber_page, page)
                            for k, im in enumerate(images_found or []):
                                try:
                                    pix = fitz.Pixmap(doc, im["xref"])
                                    if pix.alpha:
                                        pix = fitz.Pixmap(fitz.csRGB, pix)
                                    img_bytes = pix.tobytes("png")
                                except Exception:
                                    continue

                                vis = groq_helper.vision_extract(
                                    image_bytes=img_bytes,
                                    title=f"{base} p{i+1} fig{k}",
                                    max_tokens=int(mm_cfg["vision"]["max_tokens"]),
                                    lang=LANG
                                )
                                base_id = f"{doc_id}#p{i+1:03d}-fig{k:02d}"
                                img_sha = sha1_of_bytes(img_bytes)

                                if vis.get("type") == "table":
                                    md = vis.get("markdown", "") or ""
                                    csv = vis.get("csv", "") or ""
                                    table_payload = {
                                        "page_content": md or "(empty table)",
                                        "doc_id": doc_id,
                                        "chunk_id": base_id.replace("fig", "table"),
                                        "source": "pdf_report",
                                        "source_type": "table",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "table_id": f"img-{k}",
                                        "view": "table_markdown",
                                        "table_csv": csv,
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(table_payload)
                                    texts_by_lang[LANG].append(table_payload["page_content"])

                                elif vis.get("type") == "chart":
                                    data_json = vis.get("data") or {}
                                    caption = vis.get("caption", "") or ""
                                    chart_payload = {
                                        "page_content": json.dumps(data_json, ensure_ascii=False),
                                        "doc_id": doc_id,
                                        "chunk_id": base_id,
                                        "source": "pdf_report",
                                        "source_type": "chart",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "figure_id": k,
                                        "view": "chart_data",
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(chart_payload)
                                    texts_by_lang[LANG].append(chart_payload["page_content"])

                                    cap_payload = dict(chart_payload)
                                    cap_payload["page_content"] = caption
                                    cap_payload["chunk_id"] = base_id + ":caption"
                                    cap_payload["view"] = "chart_caption"
                                    payloads_by_lang[LANG].append(cap_payload)
                                    texts_by_lang[LANG].append(caption)

                                else:
                                    cap = vis.get("caption", "") or "Figure"
                                    fig_payload = {
                                        "page_content": cap,
                                        "doc_id": doc_id,
                                        "chunk_id": base_id,
                                        "source": "pdf_report",
                                        "source_type": "figure",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "figure_id": k,
                                        "view": "figure_caption",
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(fig_payload)
                                    texts_by_lang[LANG].append(cap)
                        except Exception as e:
                            print(f"[WARN] vision parse failed on {pdf_path.name} p{i+1}: {e}")
                doc.close()
                plumber_doc.close()

    # Research ZH PDFs
    research_zh_dir = ctx.data_root / "research_zh"
    if research_zh_dir.exists():
        if fitz is None:
            print("[WARN] PyMuPDF not installed; skip research_zh")
        else:
            import pdfplumber, pandas as pd
            for pdf_path in sorted(research_zh_dir.glob("*.pdf")):
                try:
                    doc = fitz.open(pdf_path)
                    plumber_doc = pdfplumber.open(str(pdf_path))
                except Exception as e:
                    print(f"[WARN] PDF open failed: {pdf_path}: {e}")
                    continue

                base = pdf_path.stem
                event_date = date_from_name_to_utc_iso(base)
                report_type = "special_topic"
                size = int(ctx.config["chunking"]["pdf"]["size"])
                overlap = int(ctx.config["chunking"]["pdf"]["overlap"])
                doc_id = f"pdf_report:{base}"
                LANG = "zh"

                for i in range(len(doc)):
                    page = doc.load_page(i)
                    text = ftfy.fix_text(page.get_text("text") or "").strip()

                    # === 文字 chunk ===
                    if text:
                        for j, ch in enumerate(split_text_by_tokens(text, size, overlap)):
                            payload = {
                                "page_content": ch,
                                "doc_id": doc_id,
                                "chunk_id": f"{doc_id}#p{i+1:03d}-{j:02d}",
                                "source": "pdf_report",
                                "source_type": "pdf_report",
                                "lang": LANG,
                                "title": base,
                                "filename": pdf_path.name,
                                "report_type": report_type,
                                "page": i + 1,
                                "published_date": event_date,
                                "event_date": event_date,
                                "published_at": event_date,
                                "created_at": event_date,
                                "create_at": event_date,
                                "tags": [],
                                "weight": float(ctx.config["weights"].get("default", 1.0)),
                                "vector_model": ctx.config["embedding"][LANG],
                                "chunking": {"size": size, "overlap": overlap},
                                "ingested_at": ingested_at,
                                "version": ctx.version,
                            }
                            payloads_by_lang[LANG].append(payload)
                            texts_by_lang[LANG].append(ch)

                    # === 表格抽取 + 摘要 ===
                    if mm_cfg.get("enable", True):
                        try:
                            plumber_page = plumber_doc.pages[i]
                            for t_idx, t in enumerate(plumber_page.find_tables() or []):
                                rows = t.extract()
                                if not rows or len(rows) < 2:
                                    continue
                                header, data_rows = rows[0], rows[1:]
                                if not any(data_rows):
                                    continue
                                import pandas as pd
                                df = pd.DataFrame(data_rows, columns=header)
                                mx = int(mm_cfg["table"]["max_rows"])
                                if len(df) > mx:
                                    df = df.head(mx)
                                md, csv = df_to_markdown_csv(df)
                                bbox = getattr(t, "bbox", None)
                                base_id = f"{doc_id}#p{i+1:03d}-table{t_idx:02d}"
                               
                                table_payload = {
                                    "page_content": md,
                                    "doc_id": doc_id,
                                    "chunk_id": base_id,
                                    "source": "pdf_report",
                                    "source_type": "table",
                                    "lang": LANG,
                                    "title": base,
                                    "filename": pdf_path.name,
                                    "page": i + 1,
                                    "table_id": t_idx,
                                    "bbox": bbox,
                                    "view": "table_markdown",
                                    "table_csv": csv,
                                    "published_date": event_date,
                                    "event_date": event_date,
                                    "vector_model": ctx.config["embedding"][LANG],
                                    "weight": float(ctx.config["weights"].get("default", 1.0)),
                                    "ingested_at": ingested_at,
                                    "version": ctx.version,
                                }
                                payloads_by_lang[LANG].append(table_payload)
                                texts_by_lang[LANG].append(md)

                        except Exception as e:
                            print(f"[WARN] table extract failed on {pdf_path.name} p{i+1}: {e}")

                    # === 圖片 → Vision 自動判斷 ===
                    if mm_cfg.get("vision", {}).get("enable", True) and groq_helper.vision_model:
                        try:
                            plumber_page = plumber_doc.pages[i]
                            _, images_found = analyze_page_for_objects(plumber_page, page)
                            for k, im in enumerate(images_found or []):
                                try:
                                    pix = fitz.Pixmap(doc, im["xref"])
                                    if pix.alpha:
                                        pix = fitz.Pixmap(fitz.csRGB, pix)
                                    img_bytes = pix.tobytes("png")
                                except Exception:
                                    continue
                                vis = groq_helper.vision_extract(
                                    image_bytes=img_bytes,
                                    title=f"{base} p{i+1} 圖{k}",
                                    max_tokens=int(mm_cfg["vision"]["max_tokens"]),
                                    lang=LANG
                                )
                                base_id = f"{doc_id}#p{i+1:03d}-fig{k:02d}"
                                img_sha = sha1_of_bytes(img_bytes)

                                if vis.get("type") == "table":
                                    md = vis.get("markdown", "") or ""
                                    csv = vis.get("csv", "") or ""
                                    table_payload = {
                                        "page_content": md or "(empty table)",
                                        "doc_id": doc_id,
                                        "chunk_id": base_id.replace("fig", "table"),
                                        "source": "pdf_report",
                                        "source_type": "table",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "table_id": f"img-{k}",
                                        "view": "table_markdown",
                                        "table_csv": csv,
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(table_payload)
                                    texts_by_lang[LANG].append(table_payload["page_content"])                                   
                                elif vis.get("type") == "chart":
                                    data_json = vis.get("data") or {}
                                    caption = vis.get("caption", "") or ""
                                    chart_payload = {
                                        "page_content": json.dumps(data_json, ensure_ascii=False),
                                        "doc_id": doc_id,
                                        "chunk_id": base_id,
                                        "source": "pdf_report",
                                        "source_type": "chart",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "figure_id": k,
                                        "view": "chart_data",
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(chart_payload)
                                    texts_by_lang[LANG].append(chart_payload["page_content"])
                                    cap_payload = dict(chart_payload)
                                    cap_payload["page_content"] = caption
                                    cap_payload["chunk_id"] = base_id + ":caption"
                                    cap_payload["view"] = "chart_caption"
                                    payloads_by_lang[LANG].append(cap_payload)
                                    texts_by_lang[LANG].append(caption)
                                else:
                                    cap = vis.get("caption", "") or "圖像"
                                    fig_payload = {
                                        "page_content": cap,
                                        "doc_id": doc_id,
                                        "chunk_id": base_id,
                                        "source": "pdf_report",
                                        "source_type": "figure",
                                        "lang": LANG,
                                        "title": base,
                                        "filename": pdf_path.name,
                                        "page": i + 1,
                                        "figure_id": k,
                                        "view": "figure_caption",
                                        "published_date": event_date,
                                        "event_date": event_date,
                                        "vector_model": ctx.config["embedding"][LANG],
                                        "weight": float(ctx.config["weights"].get("default", 1.0)),
                                        "ingested_at": ingested_at,
                                        "version": ctx.version,
                                        "image_sha1": img_sha,
                                    }
                                    payloads_by_lang[LANG].append(fig_payload)
                                    texts_by_lang[LANG].append(cap)
                        except Exception as e:
                            print(f"[WARN] vision parse failed on {pdf_path.name} p{i+1}: {e}")
                doc.close()
                plumber_doc.close()

    # ===== Finance PDFs (取代原本 Markdown 區塊) =====
    finance_dir = ctx.data_root / "finance"
    if finance_dir.exists():
        import pandas as pd

        size = int(ctx.config["chunking"]["pdf"]["size"])       # 500
        overlap = int(ctx.config["chunking"]["pdf"]["overlap"]) # 80
        default_w = float(ctx.config["weights"].get("default", 1.0))

        for pdf_path in sorted(finance_dir.glob("*.pdf")):
            try:
                pdfd = pdf_to_dict(pdf_path)
            except Exception as e:
                print(f"[WARN] Finance PDF parse failed: {pdf_path.name}: {e}")
                continue

            base = pdf_path.stem
            event_date = date_from_name_to_utc_iso(base)
            doc_id = f"finance_pdf:{base}"

            for page_obj in pdfd.get("pages", []):
                page_no = int(page_obj.get("page") or 0)
                text = page_obj.get("text") or ""

                # --- 文字切 chunk ---
                if text.strip():
                    for j, ch in enumerate(split_text_by_tokens(text, size, overlap)):
                        lang_val = "zh" if has_han(ch) else "en"
                        payload = {
                            "page_content": ch,
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}#p{page_no:03d}-{j:02d}",
                            "source": "finance_pdf",
                            "source_type": "pdf_report",
                            "lang": lang_val,
                            "title": base,
                            "filename": pdf_path.name,
                            "report_type": "finance_report",
                            "page": page_no,
                            "published_date": event_date,
                            "event_date": event_date,
                            "published_at": event_date,
                            "created_at": event_date,
                            "create_at": event_date,
                            "tags": [],
                            "weight": default_w,
                            "vector_model": ctx.config["embedding"][lang_val],
                            "chunking": {"size": size, "overlap": overlap},
                            "ingested_at": ingested_at,
                            "version": ctx.version,
                        }
                        payloads_by_lang[lang_val].append(payload)
                        texts_by_lang[lang_val].append(ch)

                # --- 表格轉 DataFrame -> Markdown -> chunk ---
                for t in page_obj.get("tables", []) or []:
                    cols = t.get("columns") or []
                    rows = t.get("rows") or []
                    try:
                        df = pd.DataFrame(rows, columns=cols)
                    except Exception:
                        # 欄列不齊時做一次補齊
                        max_len = max((len(r) for r in rows), default=0)
                        cols = (cols + [f"col_{i}" for i in range(len(cols), max_len)])[:max_len]
                        norm_rows = [(r + [""] * max(0, max_len - len(r)))[:max_len] for r in rows]
                        df = pd.DataFrame(norm_rows, columns=cols)

                    md_tbl, csv_tbl = df_to_markdown_csv(df)
                    # 語言用本頁文字偵測（沒有文字則預設 en）
                    page_lang = "zh" if has_han((text or "")[:200]) else "en"

                    base_id = f"{doc_id}#p{page_no:03d}-table{int(t.get('table_index', 0)):02d}"
                    table_payload = {
                        "page_content": md_tbl,
                        "doc_id": doc_id,
                        "chunk_id": base_id,
                        "source": "finance_pdf",
                        "source_type": "table",
                        "lang": page_lang,
                        "title": base,
                        "filename": pdf_path.name,
                        "report_type": "finance_report",
                        "page": page_no,
                        "table_id": t.get("table_index"),
                        "bbox": t.get("bbox"),
                        "view": "table_markdown",
                        "table_csv": csv_tbl,
                        "published_date": event_date,
                        "event_date": event_date,
                        "vector_model": ctx.config["embedding"][page_lang],
                        "weight": default_w,
                        "ingested_at": ingested_at,
                        "version": ctx.version,
                    }
                    payloads_by_lang[page_lang].append(table_payload)
                    texts_by_lang[page_lang].append(md_tbl)
    else:
        print("[INFO] finance dir not found; skip")

    # ------------------- Transcripts Summaries ONLY（純摘要 txt，無時間戳） -------------------
    trans_dir = ctx.data_root / "transcripts_summary"
    if trans_dir.exists():
        cfg_s = ctx.config["chunking"].get("summary", {"size": 900, "overlap": 120})

        for txt_path in sorted(trans_dir.glob("*.txt")):
            base = txt_path.stem
            event_date = date_from_name_to_utc_iso(base)

            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_summary_text(text, int(cfg_s.get("size", 900)), int(cfg_s.get("overlap", 120)))
            if not chunks:
                continue

            doc_id = f"transcript_summary:{base}"

            for i, ch in enumerate(chunks):
                lang_val = "zh" if has_han(ch) else "en"
                sec = guess_section_from_chunk(ch)  # executive_summary / conclusion / qa / prepared_remarks / other
                w = float(ctx.config["weights"].get(sec, ctx.config["weights"].get("default", 1.0)))

                payload = {
                    "page_content": ch,
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}#s{i:04d}",
                    "source": "transcripts",
                    "source_type": "transcripts_summary",
                    "lang": lang_val,
                    "title": base,
                    "filename": txt_path.name,
                    "document_title": base,
                    "section": sec,
                    "published_date": event_date,
                    "event_date": event_date,
                    "published_at": event_date,
                    "created_at": event_date,
                    "create_at": event_date,
                    "tags": [],
                    "weight": w,
                    "vector_model": ctx.config["embedding"][lang_val],
                    "chunking": {
                        "mode": "summary",
                        "size": int(cfg_s.get("size", 900)),
                        "overlap": int(cfg_s.get("overlap", 120)),
                    },
                    "ingested_at": ingested_at,
                    "version": ctx.version,
                }
                payloads_by_lang[lang_val].append(payload)
                texts_by_lang[lang_val].append(ch)

    # ====== 寫出 meta 並建立向量 ======
    out_meta = ctx.out_meta
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open('w', encoding='utf-8') as f:
        for lang in ("zh", "en"):
            for pl in payloads_by_lang[lang]:
                f.write(dumps(pl) + "\n")
    print(f"[OK] Wrote {out_meta}")

    for lang in ("zh","en"):
        texts = texts_by_lang[lang]
        if not texts:
            print(f"[WARN] no texts for {lang}"); continue
        model = ctx.config["embedding"][lang]
        print(f"[EMB] {lang}: {len(texts)} chunks with {model}")
        vecs = Embedder(model).encode(texts)

        store_dir = Path(stores[lang]); store_dir.mkdir(parents=True, exist_ok=True)
        save_faiss(store_dir/"index.faiss", vecs, payloads_by_lang[lang])
        print(f"[OK] Saved {store_dir/'index.faiss'} and {store_dir/'index.pkl'}")

# ------------------- CLI -------------------

def load_settings_yaml(maybe_path: Path) -> Dict[str,Any]:
    if not maybe_path.exists(): return {}
    try:
        import yaml
        return yaml.safe_load(maybe_path.read_text(encoding='utf-8')) or {}
    except Exception as e:
        print(f"[WARN] settings.yaml parse failed: {e}"); return {}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build vector stores (summary-only transcripts)")
    ap.add_argument('--data-root', default='data_raw')
    ap.add_argument('--out-meta', default='metadata/meta_min.jsonl')
    ap.add_argument('--settings', default='config/settings.yaml')
    ap.add_argument('--version', default=datetime.utcnow().strftime('v%Y%m%d%H%M'))
    args = ap.parse_args()

    conf = json.loads(dumps(DEFAULTS))
    y = load_settings_yaml(Path(args.settings))
    for k, v in (y or {}).items():
        if isinstance(v, dict) and k in conf:
            conf[k] = {**conf[k], **v}
        else:
            conf[k] = v

    ctx = BuildContext(data_root=Path(args.data_root), out_meta=Path(args.out_meta), config=conf, version=args.version)
    build(ctx)

if __name__ == '__main__':
    main()

'''
python3 build_vectorstores.py \
  --data-root data_raw \
  --out-meta metadata/meta_min.jsonl \
  --settings settings.yaml
'''
