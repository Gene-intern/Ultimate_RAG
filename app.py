# -*- coding: utf-8 -*-
import os, re, json, time, pickle
from typing import List, Dict, Any, Tuple, Optional, Iterable, Set, Callable
from datetime import datetime, timedelta

import gradio as gr
import numpy as np
import torch
import requests
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import threading
try:
    from rapidfuzz import fuzz  # 可用則使用；不可用時以 None 表示
except Exception:
    fuzz = None
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from dateutil import tz
from dateutil.parser import parse as dtparse
from dateutil.relativedelta import relativedelta

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ===== 共用：請求器 + 迷你快取 =====
_REQ_TIMEOUT = 3
def _req_get(
    url: str,
    params: dict = None,
    headers: dict = None,
    timeout: float = _REQ_TIMEOUT,
    retries: int = 1,
    backoff: float = 0.6,
):
    base_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    if headers:
        base_headers.update(headers)
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=base_headers, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r
        except Exception:
            pass
        time.sleep(backoff * (2 ** i))
    return None

# 超輕量記憶體快取（TTL 秒）
_CACHE: Dict[str, Dict[str, Any]] = {}
_RESOLVE_CACHE_TTL = 1800  # seconds
_RECENT_RESOLVE_TTL = 1800  # seconds

_RECENT_RESOLVED: Dict[str, Tuple[str, float]] = {}
_BG_EXECUTOR_LOCK = threading.Lock()
_BG_EXECUTOR: Optional[ThreadPoolExecutor] = None
_TW_LOAD_LOCK = threading.Lock()
_TW_LOAD_FUTURE: Optional[Future] = None

def _cache_get(key: str) -> Optional[Any]:
    it = _CACHE.get(key)
    if not it:
        return None
    if it["exp"] < time.time():
        _CACHE.pop(key, None)
        return None
    return it["val"]

def _cache_set(key: str, val: Any, ttl: int = 300) -> None:
    _CACHE[key] = {"val": val, "exp": time.time() + ttl}


def _get_bg_executor() -> ThreadPoolExecutor:
    global _BG_EXECUTOR
    with _BG_EXECUTOR_LOCK:
        if _BG_EXECUTOR is None:
            _BG_EXECUTOR = ThreadPoolExecutor(max_workers=3, thread_name_prefix="rag-bg")
        return _BG_EXECUTOR


def _recent_token_key(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    norm = _normalize_name(token)
    norm = norm.strip() if isinstance(norm, str) else ""
    if norm:
        return norm.lower()
    t = (token or "").strip().lower()
    return t or None


def _remember_resolved_symbol(token: Optional[str], symbol: Optional[str]) -> None:
    if not token or not symbol:
        return
    key = _recent_token_key(token)
    if not key:
        return
    _RECENT_RESOLVED[key] = (symbol, time.time())
    # 也記住 ticker 自身與去掉小數點的形式
    sym_key = _recent_token_key(symbol)
    if sym_key:
        _RECENT_RESOLVED[sym_key] = (symbol, time.time())
    digits = re.sub(r"\D", "", symbol)
    if digits:
        digits_key = digits.lower()
        _RECENT_RESOLVED[digits_key] = (symbol, time.time())


def _lookup_recent_symbol(token: Optional[str]) -> Optional[str]:
    key = _recent_token_key(token)
    if not key:
        return None
    item = _RECENT_RESOLVED.get(key)
    if not item:
        return None
    sym, ts = item
    if (time.time() - ts) > _RECENT_RESOLVE_TTL:
        _RECENT_RESOLVED.pop(key, None)
        return None
    return sym


def _warm_tw_lists_async(force: bool = False) -> None:
    global _TW_LOAD_FUTURE
    fresh = _is_tw_cache_fresh()
    if fresh and not force:
        return
    with _TW_LOAD_LOCK:
        if _TW_LOAD_FUTURE and not _TW_LOAD_FUTURE.done():
            return
        executor = _get_bg_executor()
        _TW_LOAD_FUTURE = executor.submit(_load_tw_lists, True)

# =========================
# FAISS 兼容載入
# =========================
def load_faiss_compat(folder_path: str, embeddings, index_name: str = "index"):
    """
    兼容不同年代的 LangChain FAISS 儲存格式。
    """
    import faiss

    from langchain_community.vectorstores.faiss import FAISS as _FAISS
    try:
        return _FAISS.load_local(
            folder_path, embeddings, index_name=index_name,
            allow_dangerous_deserialization=True
        )
    except Exception:
        pass

    faiss_path = os.path.join(folder_path, f"{index_name}.faiss")
    pkl_path   = os.path.join(folder_path, f"{index_name}.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(pkl_path)):
        raise FileNotFoundError(f"Missing {faiss_path} or {pkl_path}")

    faiss_index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    InMemoryDocstore = None
    for modpath, clsname in [
        ("langchain_community.docstore.in_memory", "InMemoryDocstore"),
        ("langchain_community.docstores", "InMemoryDocstore"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[clsname])
            InMemoryDocstore = getattr(mod, clsname, None)
            if InMemoryDocstore: break
        except Exception:
            continue
    if InMemoryDocstore is None:
        try:
            mod = __import__("langchain.docstore.in_memory", fromlist=["InMemoryDocstore"])
            InMemoryDocstore = getattr(mod, "InMemoryDocstore")
        except Exception:
            pass

    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document  # 兼容舊版

    def is_doclike(x):
        if hasattr(x, "page_content") and hasattr(x, "metadata"): return True
        if isinstance(x, dict) and ("page_content" in x or "metadata" in x): return True
        return False

    def to_document(x):
        if hasattr(x, "page_content") and hasattr(x, "metadata"):
            return x
        if isinstance(x, dict):
            if "page_content" in x and "metadata" not in x:
                meta = {k: v for k, v in x.items() if k != "page_content"}
                return Document(page_content=x.get("page_content", ""), metadata=meta)
            if "metadata" in x:
                return Document(page_content=x.get("page_content", ""), metadata=x.get("metadata") or {})
            return Document(page_content="", metadata=x)
        return Document(page_content=str(x), metadata={})

    def coerce_docstore(x):
        if hasattr(x, "search") or hasattr(x, "_dict"): return x
        if isinstance(x, dict) and x:
            conv = {k: to_document(v) for k, v in x.items()}
            return InMemoryDocstore(conv) if InMemoryDocstore else conv
        return None

    def coerce_index_map(x):
        if isinstance(x, list):
            return {i: v for i, v in enumerate(x)}
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                try:
                    ki = int(k) if not isinstance(k, int) else k
                    out[ki] = v
                except Exception:
                    pass
            if out: return out
            rev = {}
            for k, v in x.items():
                try:
                    vi = int(v) if not isinstance(v, int) else v
                    rev[vi] = k
                except Exception:
                    continue
            if rev: return rev
        return None

    docstore = None
    index_to_docstore_id = None

    if isinstance(obj, dict):
        for k in ["docstore", "_docstore"]:
            if k in obj and docstore is None:
                ds = coerce_docstore(obj[k])
                if ds is not None: docstore = ds
        for k in ["index_to_docstore_id", "_index_to_docstore_id", "index_map"]:
            if k in obj and index_to_docstore_id is None:
                m = coerce_index_map(obj[k])
                if m is not None: index_to_docstore_id = m

    elif isinstance(obj, tuple) or (isinstance(obj, list) and len(obj) <= 3):
        items = list(obj)
        if len(items) == 3:
            items = items[1:]
        if len(items) == 2:
            a, b = items
            a_ds = coerce_docstore(a); b_ds = coerce_docstore(b)
            a_map = coerce_index_map(a); b_map = coerce_index_map(b)
            if a_ds is not None and b_map is not None:
                docstore, index_to_docstore_id = a_ds, b_map
            elif b_ds is not None and a_map is not None:
                docstore, index_to_docstore_id = b_ds, a_map

    elif isinstance(obj, list) and len(obj) > 3:
        docs, ids = [], []
        for i, e in enumerate(obj):
            if isinstance(e, tuple) and len(e) == 2 and is_doclike(e[1]):
                did, dd = e[0], e[1]
                ids.append(str(did))
                docs.append(to_document(dd))
            elif is_doclike(e):
                ids.append(str(i))
                docs.append(to_document(e))
            else:
                ids.append(str(i))
                docs.append(to_document(e))

        if InMemoryDocstore is None:
            raise RuntimeError("InMemoryDocstore unavailable in this environment.")
        doc_map = {did: doc for did, doc in zip(ids, docs)}
        n_vec = faiss_index.ntotal
        n_doc = len(doc_map)
        n = min(n_vec, n_doc)
        ordered_ids = ids[:n]
        docstore = InMemoryDocstore({did: doc_map[did] for did in ordered_ids})
        index_to_docstore_id = {i: did for i, did in enumerate(ordered_ids)}

    else:
        raise RuntimeError(f"Unsupported pickle type: {type(obj)}")

    from langchain_community.vectorstores.faiss import FAISS as _FAISS
    try:
        return _FAISS(embeddings, faiss_index, docstore, index_to_docstore_id)
    except TypeError:
        return _FAISS(getattr(embeddings, "embed_query", embeddings), faiss_index, docstore, index_to_docstore_id)
# =========================
# 基本設定
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# 優先使用 CUDA，其次使用 Apple Silicon 的 MPS，最後退回 CPU
try:
    _HAS_MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
except Exception:
    _HAS_MPS = False
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if _HAS_MPS else "cpu")
STORE_ZH = "indices/store_zh"
STORE_EN = "indices/store_en"
TZ_TW = tz.gettz("Asia/Taipei")
TW_TICKER_RE = re.compile(r"^\d{4}\.(TW|TWO)$")

SPECIAL_OVERVIEW_LABEL = "🗓️ 產出該月整體概覽（點我生成）"

# ==== BGE reranker ====
try:
    from FlagEmbedding import FlagReranker
    _HAS_BGE_RERANK = True
except Exception:
    FlagReranker = None
    _HAS_BGE_RERANK = False

BGE_RERANK_MODEL = os.getenv("BGE_RERANK_MODEL", "BAAI/bge-reranker-large")
bge_reranker = None
if _HAS_BGE_RERANK:
    try:
        # 依裝置設定：CUDA 用 FP16；MPS/CPU 用 FP32，並嘗試顯式指定 device
        rerank_kwargs = {"use_fp16": (DEVICE == "cuda")}
        try:
            # 新版 FlagReranker 支援 device 參數
            rerank_kwargs["device"] = DEVICE
            bge_reranker = FlagReranker(BGE_RERANK_MODEL, **rerank_kwargs)
        except TypeError:
            # 舊版不支援 device 參數，退回不帶 device
            rerank_kwargs.pop("device", None)
            bge_reranker = FlagReranker(BGE_RERANK_MODEL, **rerank_kwargs)
    except Exception as e:
        print(f"[WARN] init bge reranker failed (device={DEVICE}): {e}")
        _HAS_BGE_RERANK = False

# ===== LLM 請求大小保護（避免 413 / 過長 prompt）=====
# 字元粗估：英文約 4 chars/token，中文約 1–2 chars/token，這裡保守抓上限
LLM_CTX_TOTAL_CHAR_SOFT = 7000   # 單次回答 context 總長度上限（軟性）
LLM_CTX_PER_DOC_CHAR    = 1000   # 每段最多保留字元
LLM_CTX_MAX_DOCS        = 8      # 最多帶入的段落數

YAHOO_FIRST = True  # True=Yahoo-first + 名單冷備援
SYMBOL_MAP_PATH = os.getenv("SYMBOL_MAP_PATH", "cache/symbol_map.json")
SYMBOL_MAP_TTL_DAYS = int(os.getenv("SYMBOL_MAP_TTL_DAYS", "1"))

_SYM_KV = {"ts": 0, "map": {}}  # 內存鏡像

def _sym_norm_key(s: str) -> str:
    # 對中文名用你現成的 _normalize_name，對其它用 lower+strip
    if re.search(r"[\u4e00-\u9fff]", s or ""):
        return _normalize_name(s or "")
    return (s or "").strip().lower()

def _symcache_load():
    try:
        if os.path.exists(SYMBOL_MAP_PATH):
            with open(SYMBOL_MAP_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, dict) and "map" in obj:
                    _SYM_KV.update(obj)
    except Exception:
        pass

def _symcache_flush():
    try:
        os.makedirs(os.path.dirname(SYMBOL_MAP_PATH), exist_ok=True)
        with open(SYMBOL_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(_SYM_KV, f, ensure_ascii=False)
    except Exception:
        pass

def _symcache_get_record(name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    if not _SYM_KV["map"]:
        _symcache_load()
    key = _sym_norm_key(name)
    rec = _SYM_KV["map"].get(key)
    if not rec:
        return None
    try:
        ts = float(rec.get("ts", 0))
        if time.time() - ts > SYMBOL_MAP_TTL_DAYS * 86400:
            _SYM_KV["map"].pop(key, None)
            _symcache_flush()
            return None
    except Exception:
        return None
    return rec


def _symcache_get(name: str) -> Optional[str]:
    rec = _symcache_get_record(name)
    if not rec:
        return None
    return rec.get("symbol") or None

def _symcache_put(name: str, symbol: str) -> None:
    if not (name and symbol): return
    if not _SYM_KV["map"]:
        _symcache_load()
    key = _sym_norm_key(name)
    _SYM_KV["map"][key] = {"symbol": symbol, "ts": time.time()}
    _symcache_flush()
# =========================
# Embeddings & Vectorstores
# =========================
embed_zh = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
embed_en = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore_zh = load_faiss_compat(STORE_ZH, embed_zh, index_name="index")
vectorstore_en = load_faiss_compat(STORE_EN, embed_en, index_name="index")

def _set_thread_env_if_unset(n_threads: int | None = None):
    n = n_threads or (os.cpu_count() or 4)
    # 只在未設定時賦值，避免覆寫你已經在 shell 設的參數
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))  # macOS Accelerate/vecLib

_set_thread_env_if_unset()

# ---- 之後再 import faiss ----
try:
    import faiss
    # 某些 macOS/arm64 的 wheel 沒有這些 API；所以要先判斷
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 4)))
    if hasattr(faiss, "omp_get_max_threads"):
        print(f"[FAISS] max threads = {faiss.omp_get_max_threads()}")
    else:
        print("[FAISS] OpenMP control API not exposed; using env vars only.")
except Exception as e:
    print(f"[WARN] faiss thread setup failed: {e}")
# =========================
# LLMs
# =========================
# === 使用者可選的「生成模型」白名單（只影響最終回答）===
GEN_MODEL_WHITELIST = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    #"openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
GEN_MODEL_DEFAULT = "llama-3.3-70b-versatile"

# === 內部固定模型（不受前端控制）===
CLS_MODEL    = "llama-3.1-8b-instant"  # 分類 / 抽取
PARA_MODEL   = "llama-3.1-8b-instant"  # 多查詢改寫
RERANK_MODEL = "llama-3.1-8b-instant"  # LLM-based rerank（若你有 BGE，就當備援）

# 全域句柄（先給 None，稍後初始化）
llm = None             # 生成用（前端可選）
classifier_llm = None  # 固定
paraphrase_llm = None  # 固定
rerank_llm = None      # 固定

def init_internal_llms():
    """啟動時呼叫一次：把內部三位固定到輕模型。"""
    global classifier_llm, paraphrase_llm, rerank_llm
    classifier_llm = ChatGroq(temperature=0, model=CLS_MODEL)
    paraphrase_llm = ChatGroq(temperature=0, model=PARA_MODEL)
    rerank_llm     = ChatGroq(temperature=0, model=RERANK_MODEL)

def set_gen_llm(user_choice: str) -> str:
    """
    每次回答前呼叫：只切換『生成用』模型。
    若前端傳來的型號不在白名單，則回退成預設。
    回傳：實際生效的型號（可放到 Debug 顯示）。
    """
    global llm
    chosen = (user_choice or "").strip()
    model = chosen if chosen in GEN_MODEL_WHITELIST else GEN_MODEL_DEFAULT
    llm = ChatGroq(temperature=0, model=model)
    return model

# =========================
# 常數 / 工具
# =========================
FOLLOWUP_PLACEHOLDER = "— 點選延伸問題（會自動填入上方輸入框） —"
_ZH_MAP = str.maketrans({"臺": "台"})
_FULL2HALF = str.maketrans({
    "，": ",", "。": ".", "！": "!", "？": "?",
    "【": "[", "】": "]", "（": "(", "）": ")",
    "％": "%", "＋": "+", "－": "-", "＝": "=",
    "：": ":", "；": ";", "、": ",", "　": " ",
})
_ZWS = re.compile(r"[\u200B-\u200D\uFEFF\u00A0\s]+")
NEWS_KWS = ["新聞","快訊","消息","報導","報道","更新","頭條","press release","headline","breaking","news"]
STOCK_KWS = ["股票","股價","報價","行情","走勢","收盤","開盤","目標價","估值","本益比","price","quote","stock","target price"]
FINREP_KWS = ["財報","營收","EPS","毛利率","獲利","指引","guidance","年報","季報",
              "資產負債表","現金流量表","income statement","balance sheet","cash flow"]

_QWORDS_RE = re.compile(
    r"^(?:是)?(?:哪支|哪檔|哪家|哪間|哪個|哪一(?:支|檔|家)|什麼|甚麼|是什麼)(?:呢|嗎)?$"
)

DATE_KEYS_PRIMARY   = ("published_date", "event_date", "published_at", "pub_date")
DATE_KEYS_FALLBACK  = ("created_at", "create_at", "date", "ingested_at")

CHI_NUM = {"零":0,"〇":0,"○":0,"Ｏ":0,"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}

ALIAS_TICKER_MAP = {
    "蘋果": ("AAPL", ["Apple", "Apple Inc", "蘋果公司"]),
    "微軟": ("MSFT", ["Microsoft", "微軟公司"]),
    "輝達": ("NVDA", ["NVIDIA", "Nvidia", "輝達"]),
    "谷歌": ("GOOGL", ["Alphabet", "Google", "谷歌"]),
    "亞馬遜": ("AMZN", ["Amazon", "亞馬遜"]),
    "特斯拉": ("TSLA", ["Tesla", "特斯拉"]),
    "美光": ("MU", ["Micron", "Micron Technology", "美光"]),
}

def _tw_find_name_for_code(sym: str) -> Optional[str]:
    _load_tw_lists()
    return (_TW_CACHE.get("by_code") or {}).get((sym or "").upper())

def _cjk_overlap_score(a: str, b: str) -> int:
    import re
    A = set(re.findall(r"[\u4e00-\u9fff]", _normalize_name(a or "")))
    B = set(re.findall(r"[\u4e00-\u9fff]", _normalize_name(b or "")))
    return len(A & B)

def _norm_for_kw(s: str) -> str:
    s = (s or "").translate(_FULL2HALF).translate(_ZH_MAP).lower()
    return _ZWS.sub("", s)

def contains_any(text: str, kws: Iterable[str]) -> bool:
    t = _norm_for_kw(text)
    return any(_norm_for_kw(kw) in t for kw in kws)

def is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

def extract_json_from_text(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    for m in re.findall(r"(\{[\s\S]*?\})", text or ""):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None

def parse_any_date(s: str) -> Optional[datetime]:
    try:
        d = dtparse(s)
        return d if d.tzinfo else d.replace(tzinfo=tz.tzutc())
    except Exception:
        return None

_FNAME_DATE_PAT = re.compile(
    r"(20\d{2})[-/._\s]?(0[1-9]|1[0-2])[-/._\s]?(0[1-9]|[12]\d|3[01])"
)

def _infer_date_from_name(md: Dict[str, Any]) -> Optional[datetime]:
    for key in ("filename", "title"):
        s = (md or {}).get(key) or ""
        m = _FNAME_DATE_PAT.search(s)
        if m:
            y, mo, d = map(int, m.groups())
            return datetime(y, mo, d).replace(tzinfo=tz.tzutc())
    return None

def get_doc_date_dt(md: Dict[str, Any]) -> Optional[datetime]:
    # 先按優先序找
    for k in (*DATE_KEYS_PRIMARY, *DATE_KEYS_FALLBACK):
        v = (md or {}).get(k)
        if v:
            dt = parse_any_date(v)
            if dt:
                return dt
    # 都沒有時，嘗試檔名/標題
    return _infer_date_from_name(md)

# --- A) 嚴格的公司 token 清洗（不靠 map） ---
def clean_company_token(raw: str, *, original_text: str = "") -> Optional[str]:
    s = (raw or "").strip()

    # 🔧 新增：去口語前綴（避免「請跟我說融程電」被當成公司）
    s = re.sub(
        r"^(?:請問|請給我|請提供|請幫我|幫我|幫忙|麻煩|可不可以|可以給我|可以|能不能|是否|"
        r"幫我整理|跟我說|請跟我說|告訴我)\s*",
        "", s
    )

    # 標點與空白
    s = re.sub(r"[，,。．.、；;：:！!？?\s\u3000「」『』（）()［］\[\]【】<>〈〉…]+", "", s)
    s = re.sub(r"(嗎|呢|吧|呀|唷|喔|哦|啦|耶)+$", "", s)  # 句尾語氣詞
    s = re.sub(r"(?:是什麼|什麼是)$", "", s)

    # 先把常見「查詢尾碼」去掉：的財報/年報/法說/新聞/股價/重點/摘要/走勢…（含可選「的」）
    s = re.sub(
        r"(?:的)?(?:財報|年報|季報|法說(?:會)?|新聞|快訊|報導|更新|公告|重訊|營收|EPS|毛利率|展望|guidance|"
        r"重點|摘要|概況|介紹|報告|overview|股價|報價|目標價|走勢|今天|今日|現在|目前)+$",
        "", s, flags=re.I
    )
    # 🔧 新增：去掉「的股 / 股票 / 個股 / 代號 / 代碼」這類尾綴
    s = re.sub(r"(?:的|之)?(?:股票代號|股票|個股|本股|股|代號|代碼)+$", "", s, flags=re.I)

    # 去掉尾端助詞（避免「微軟的」「台積電之」）
    s = re.sub(r"(?:的|之)+$", "", s)
    
    s = re.sub(r"(?:的)?(?:介紹|簡介|概述|overview|是什麼|什麼是|定義)+$", "", s, flags=re.I)
    if re.search(r"(?:產業|行業|市場)$", s):  
        return None

    if _QWORDS_RE.fullmatch(s):
        return None
    
    if re.fullmatch(r"(?:是|嗎|呢|嘛|啥|何|什麼|甚麼)+", s):
        return None
    
    if not s:
        return None
    # 過長/過短或殘渣
    if len(s) < 2 or len(s) > 20:
        return None
    if re.search(r"(可以給我|請問|多少|價格|新聞|最新|重點|摘要|財報|法說|在哪|怎麼|為何|為什麼|今天|今日|現在|目前)", s):
        return None
    if re.fullmatch(r"\d{4}", s) and looks_like_year(s, original_text or ""):
        return None

    # 合法型態：ticker / 英文名 / 中文名
    is_ticker = bool(re.fullmatch(r"[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?|\d{4}(?:\.[A-Za-z]{2,4})?", s))
    is_en     = bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9&\-.]{1,19}", s))
    is_cjk    = bool(re.fullmatch(r"[\u4e00-\u9fff]{2,10}", s))
    if not (is_ticker or is_en or is_cjk):
        return None
    return s

# --- B) 以 Yahoo Autocomplete / resolve_symbol 做「可攜式典範化」 ---
def canonicalize_company_needles(
    token: str,
    *,
    fast_mode: bool = False,
    autoc_timeout: Optional[float] = None,
    autoc_max_langs: Optional[int] = None,
) -> Tuple[Optional[str], Set[str]]:
    """
    輸入：一個乾淨 token（可能是中文名/英文名/代號）
    輸出：(ticker or None, 需要匹配的 needles set[小寫])
    - 完全不使用手寫 map
    - 先用 resolve_symbol() 確認最終有效 ticker
    - 再用 yahoo_autoc_all() 抓到該 ticker 的 display name（中英混），補進 needles
    """
    needles: Set[str] = set()
    if not token:
        return None, needles

    # ✅ 先走極小別名保險絲（高頻 Top N）
    alias = ALIAS_TICKER_MAP.get(token)
    if alias:
        sym, extra = alias
        needles.update({token.lower(), sym.lower(), *[x.lower() for x in extra]})
        return sym, needles

    # 1) 先嘗試解析出有效 ticker
    sym = resolve_symbol(token, fast_mode=fast_mode, autoc_timeout=autoc_timeout, autoc_max_langs=autoc_max_langs)
    if sym:
        needles.add(sym.lower())
        import re
        digits = re.sub(r"\D", "", sym)
        if digits:
            needles.add(digits)
        # 2) 從 autocomplete 抓顯示名稱（跨語系/跨區）
        cands = yahoo_autoc_all(
            token,
            regions=("US","TW","HK"),
            langs=("zh-TW","zh-Hant-TW","en-US"),
            timeout=autoc_timeout or 5.0,
            max_langs=autoc_max_langs,
        )
        for c in cands:
            name = (c.get("name") or "").strip()
            if name:
                # 去除公司尾綴（Inc., Corp., Co., Ltd.等）
                name_norm = re.sub(r"\b(inc|inc\.|corp|corp\.|co|co\.|ltd|ltd\.|plc|sa|ag|nv|kk)\b\.?", "", name, flags=re.I)
                needles.add(name_norm.lower())
        # 再補一次用 ticker 反查（避免 token 不是原始查詢語言）
        cands2 = yahoo_autoc_all(
            sym,
            regions=("US","TW","HK"),
            langs=("zh-TW","en-US"),
            timeout=autoc_timeout or 5.0,
            max_langs=autoc_max_langs,
        )
        for c in cands2:
            if c.get("symbol") and c.get("symbol").lower() == sym.lower():
                name = (c.get("name") or "").strip()
                if name:
                    name_norm = re.sub(r"\b(inc|inc\.|corp|corp\.|co|co\.|ltd|ltd\.|plc|sa|ag|nv|kk)\b\.?", "", name, flags=re.I)
                    needles.add(name_norm.lower())
        return sym, {n for n in needles if n}

    # 3) 還原失敗時，試著讓 LLM 翻成英文公司名再走一次
    en_name = to_english_company_name(token) or token
    if fast_mode:
        return None, {token.lower()}
    cands = yahoo_autoc_all(
        en_name,
        regions=("US","TW","HK"),
        langs=("en-US","zh-TW"),
        timeout=autoc_timeout or 5.0,
        max_langs=autoc_max_langs,
    )
    best = pick_best_yahoo_candidate(en_name, cands) if cands else None
    if best and is_valid_symbol(best):
        needles.add(best.lower())
        for c in cands:
            if (c.get("symbol") or "").lower() == best.lower():
                name = (c.get("name") or "").strip()
                if name:
                    name_norm = re.sub(r"\b(inc|inc\.|corp|corp\.|co|co\.|ltd|ltd\.|plc|sa|ag|nv|kk)\b\.?", "", name, flags=re.I)
                    needles.add(name_norm.lower())
        return best, needles

    # 4) 實在找不到，就至少把 token 本身當 needle（低階保障）
    needles.add(token.lower())
    return None, needles

# --- C) 文檔是否提及公司（不靠 map，僅做小寫包含） ---
def doc_mentions_any_company(md: Dict[str, Any], text: str, needles: Set[str]) -> bool:
    title = (md or {}).get("title", "") or ""
    source = (md or {}).get("source", "") or ""
    url = ((md or {}).get("doc_url") or (md or {}).get("url") or "") or ""
    fname = (md or {}).get("filename", "") or ""
    hay = " ".join([title, source, url, text or ""]).lower()

    # ✅ Apple 的 SEC CIK（避免「apple pay」之類誤命中）
    if "sec.gov" in url.lower() and ("data/320193" in url.lower() or "cik=0000320193" in url.lower()):
        return True

    for k in needles or []:
        k = (k or "").lower().strip()
        if not k:
            continue
        # 英文詞：用字母數字邊界；中文詞：直接包含即可
        if re.search(r"[a-z]", k):
            if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", hay):
                return True
        else:
            if k in hay:
                return True
    return False

# ===== 年份解析 / 文件年判定 =====
# 支援 FY24 / FY'24 / FY’24 / FY2024
_FY_PAT = re.compile(r"\bFY\s*['’]?\s*(?:20)?(\d{2})\b", re.I)

# 支援 "Fiscal 2024" 與 "Fiscal Year 2024"
_FISCAL_PAT = re.compile(r"\bFISCAL(?:\s+YEAR)?\s+(20\d{2})\b", re.I)

# 1Q24 / 3Q FY2024
_Q_PAT = re.compile(r"\b([1-4])Q(?:\s*|\s*FY\s*)?(?:20)?(\d{2})\b", re.I)

# 四位數年份（若你已有 _Y4_PAT 就保留原名即可）
_Y4_PAT = re.compile(r"\b((?:19|20)\d{2})\b")

def _fy_to_year(s: str) -> int:
    # 兩位數 → 2000+yy；四位數就直回
    return int(s) if len(s) == 4 else 2000 + int(s)

def detect_doc_report_year_from_md_or_text(md: dict, text: str = "") -> int | None:
    hay = " ".join(str(x) for x in [
        md.get("title",""), md.get("filename",""), md.get("doc_url",""),
        md.get("url",""), md.get("source",""), " ".join((md.get("tags") or [])), text or ""
    ] if x)

    m = _FY_PAT.search(hay)
    if m:
        return _fy_to_year(m.group(1))

    m2 = _FISCAL_PAT.search(hay)
    if m2:
        return int(m2.group(1))

    m_q = _Q_PAT.search(hay)
    if m_q:
        return 2000 + int(m_q.group(2))

    m4 = _Y4_PAT.search(hay)
    if m4:
        return int(m4.group(1))

    dt = get_doc_date_dt(md)
    return dt.year if dt else None

def extract_report_year_targets(user_q: str) -> set[int]:
    """
    從使用者問題抽取可能的財報年份（會計年度）。支援：
    - 2023 / FY2023 / 1Q23 / 4Q2023 等
    回傳可能年份集合（整數），若抓不到則回空集合。
    """
    text = (user_q or "").strip()
    yrs: set[int] = set()

    # FY2023
    for m in _FY_PAT.finditer(text):
        yrs.add(int("20" + m.group(1)))

    # 1Q23 / 3Q FY2023
    for m in _Q_PAT.finditer(text):
        yy = int(m.group(2))
        yrs.add(2000 + yy)

    # 直接寫 2023
    for m in _Y4_PAT.finditer(text):
        yrs.add(int(m.group(1)))

    return yrs

def detect_doc_report_year(md: dict) -> int | None:
    md = md or {}
    hay = " ".join(
        str(x) for x in [
            md.get("title",""), md.get("filename",""), md.get("doc_url",""),
            md.get("url",""), md.get("source",""), " ".join((md.get("tags") or []))
        ] if x
    )

    # 先嘗試 FY（支援 FY25/FY 2025）
    m = _FY_PAT.search(hay)
    if m:
        yy = int(m.group(1))  # 兩位數
        return 2000 + yy

    # 再來嘗試 Qx FYyy（例如 Q4 FY25）
    m_q = _Q_PAT.search(hay)
    if m_q:
        yy = int(m_q.group(2))
        return 2000 + yy

    # 例如 "Microsoft 2025 Annual Report"
    m2 = _Y4_PAT.search(hay)
    if m2:
        return int(m2.group(1))

    dt = get_doc_date_dt(md)
    return dt.year if dt else None

def get_doc_date_str(md: Dict[str, Any]) -> str:
    dt = get_doc_date_dt(md)
    return dt.strftime("%Y-%m-%d") if dt else "未知日期"

def _is_pdf_md(md: Dict[str, Any]) -> bool:
    s = (md.get("source") or md.get("filename") or md.get("doc_url") or "").lower()
    st = (md.get("source_type") or "").lower()
    tags = md.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    return s.endswith(".pdf") or "pdf" in st or any("pdf" in str(t).lower() for t in tags)

def extract_page_number(md: Dict[str, Any]) -> Optional[int]:
    """
    從 metadata 取頁碼；常見鍵：
      - 'page'（PyPDFLoader/Unstructured 常見，通常 0-based）
      - 'page_number'（有些 loader 用 1-based）
      - 'loc': {'page': ...}
    對 PDF 預設將 0-based 轉為 1-based（更符合讀者直覺）。
    """
    cand = md.get("page", None)
    if cand is None:
        cand = md.get("page_number", None)
    if cand is None:
        loc = md.get("loc") or md.get("location")
        if isinstance(loc, dict):
            cand = loc.get("page") or loc.get("page_number")

    try:
        p = int(cand)
    except Exception:
        return None

    # 多數 PDF loader 的 page 是 0-based，統一轉 1-based 顯示
    if _is_pdf_md(md) and p >= 0:
        # 如果來源本來就是 1-based（少數情況），會多加 1；如需嚴格判斷可再加旗標控制
        p = p + 1
    return p if p > 0 else None

def _compress_pages_to_ranges(pages: Iterable[int]) -> List[Tuple[int, int]]:
    """把頁碼去重、排序、壓成連續範圍 [(start,end), ...]"""
    uniq = sorted(set(int(x) for x in pages if isinstance(x, int)))
    if not uniq:
        return []
    ranges = []
    start = prev = uniq[0]
    for p in uniq[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    return ranges

def _format_pages_suffix(pages: Iterable[int]) -> str:
    """回傳適合放在行尾的頁碼字串，例如：' ｜pp. 5–7, 12'；若空則回空字串"""
    ranges = _compress_pages_to_ranges(pages)
    if not ranges:
        return ""
    parts = []
    for a, b in ranges:
        parts.append(f"p. {a}" if a == b else f"pp. {a}–{b}")
    return " ｜" + ", ".join(parts)

def get_bucket(md: Dict[str, Any]) -> str:
    s = ((md or {}).get("source") or (md or {}).get("source_type") or "").lower()
    title = (md or {}).get("title", "").lower()
    url = ((md or {}).get("doc_url") or (md or {}).get("url") or "").lower()
    tags = md.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags_l = [str(t).lower() for t in tags]

    # filing
    if (
        "sec_filing" in s or re.search(r"\bsec(_|$|\b)", s) or "sec.gov" in url
        or any(k in title for k in _FILING_HINTS)
        or any(k in tags_l for k in ("filing","10-k","10q","8-k","20-f","6-k","s-1","f-1","def 14a","sec"))
    ):
        return "filing"

    # ✅ 統一為 transcript（含 transcript_summary/summary）
    transcript_hints = ("transcript", "transcripts", "transcripts_summary",
                        "earnings call", "conference call", "prepared remarks", "q&a", "q & a",
                        "法說", "法說會", "電話會議")
    if any(h in s for h in transcript_hints) or any(h in title for h in transcript_hints) \
       or any("transcript" in t for t in tags_l):
        return "transcript"


    if "blog" in s or "blogs" in s or "post" in s: return "blog"
    if "research" in s or "research_report" in s: return "research"
    if "pdf" in s or "report" in s or "pdf_report" in s: return "pdf"
    if any("research" in t for t in tags_l): return "research"
    if any("blog" in t for t in tags_l): return "blog"
    if any(("pdf" in t) or ("report" in t) for t in tags_l): return "pdf"
    return "other"

_FILING_HINTS = (
    "form 10-k","10-k","form 10-q","10-q","8-k",
    "20-f","form 20-f","6-k","form 6-k",
    "s-1","form s-1","f-1","form f-1",
    "def 14a","schedule 14a",
    "annual report"  # 仍保留
)
def is_filing_like(md: Dict[str, Any]) -> bool:
    s = ((md or {}).get("source") or (md or {}).get("source_type") or "").lower()
    title = (md or {}).get("title", "").lower()
    url = ((md or {}).get("doc_url") or (md or {}).get("url") or "").lower()
    tags = (md or {}).get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tags_l = [str(t).lower() for t in tags]

    # 更精準的 SEC 來源判斷：避免 "security" 誤擊
    if "sec.gov" in url or re.search(r"\bsec(_|$|\b)", s):
        return True

    if any(h in title for h in _FILING_HINTS): return True
    if any(h in url   for h in _FILING_HINTS): return True
    if any(h in tags_l for h in _FILING_HINTS): return True

    return get_bucket(md) == "filing"

def _recency_weight(md: Dict[str, Any], half_life_days: int = 45) -> float:
    """
    指數衰減的新鮮度分數：越新越接近 1，越舊越接近 0。
    half_life_days 可依需求調整（例如 30~90）。
    """
    dt = get_doc_date_dt(md or {})
    if not dt:
        return 1.0
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=tz.tzutc())
    now = datetime.now(tz=tz.tzutc())
    days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return float(np.exp(-days / float(half_life_days)))

def filter_by_recent(docs: List[Any], cos_scores: List[float], days: int = 120, min_keep: int = 6):
    """
    先嘗試保留近 X 天的候選；若剩太少（< min_keep），則回退為原串列。
    """
    now = datetime.now(tz=tz.tzutc())
    kept_docs, kept_scores = [], []
    for d, sc in zip(docs, cos_scores):
        dt = get_doc_date_dt(d.metadata or {})
        if not dt:
            continue
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=tz.tzutc())
        age_days = (now - dt).total_seconds() / 86400.0
        if age_days <= days:
            kept_docs.append(d); kept_scores.append(sc)
    if len(kept_docs) >= max(1, min_keep):
        return kept_docs, kept_scores
    return docs, cos_scores  # 回退

# =========================
# Yahoo / 代號解析 / 股價 / 新聞（強化）
# =========================
@lru_cache(maxsize=2048)
def _yahoo_v7_quote(symbol: str) -> dict:
    try:
        r = _req_get("https://query1.finance.yahoo.com/v7/finance/quote",
                     params={"symbols": symbol}, timeout=5)
        items = (r.json().get("quoteResponse", {}) or {}).get("result") or []
        return items[0] if items else {}
    except Exception:
        return {}
    
def _is_tradable_via_quote(symbol: str) -> bool:
    """當 v8 chart 還沒有資料（新股常見）時，用 v7 quote 檢查是否有可用報價。"""
    q = _yahoo_v7_quote(symbol)
    if not q:
        return False
    # 任何一個價格欄位有數值即可視為有效
    price_fields = [q.get("regularMarketPrice"), q.get("postMarketPrice"), q.get("preMarketPrice")]
    has_price = any(isinstance(x, (int, float)) for x in price_fields)
    # 交易所也要合理（台股優先）
    ex_ok = _tw_exchange_ok(symbol, q) or bool(q.get("exchange"))
    return bool(has_price and ex_ok)

def _tw_exchange_ok(symbol: str, quote: Optional[dict] = None) -> bool:
    s = (symbol or "").upper()
    if re.match(r"^\d{4}\.(TW|TWO)$", s):
        return True
    q = quote or _yahoo_v7_quote(s)
    ex = (q.get("fullExchangeName") or q.get("exchange") or "").lower()
    if ("taiwan" in ex) or ("taipei" in ex):
        return True
    if s.endswith(".TW") or s.endswith(".TWO"):
        return True
    return False

@lru_cache(maxsize=2048)
def yahoo_search_symbols_by_keyword(q: str, limit: int = 3) -> list[str]:
    """
    用 Yahoo 股市的站內搜尋頁當 fallback，直接從 HTML 抓 /quote/<symbol>。
    回傳已通過 chart 驗證的 symbol（優先 TW/TWO）。
    """
    q = (q or "").strip()
    if not q:
        return []

    # 試幾個常見 query 參數（Yahoo 不定期更動）
    endpoints = [
        ("https://tw.stock.yahoo.com/s",      {"k": q}),
        ("https://tw.stock.yahoo.com/search", {"query": q}),
        ("https://tw.stock.yahoo.com/search", {"p": q}),
    ]

    out, seen = [], set()
    for url, params in endpoints:
        r = _req_get(url, params=params, timeout=6)
        if not r:
            continue
        html = r.text or ""
        # 抓所有 /quote/<symbol> 片段（含 TW/TWO 與美股）
        for sym in re.findall(r"/quote/([A-Za-z0-9\.-]{2,15})", html):
            s = sym.upper()
            # 先偏好台股樣式；再放寬到一般 symbol
            ok = bool(re.match(r"^\d{4}\.(?:TW|TWO)$", s) or re.match(r"^[A-Z]{1,5}(?:\.[A-Z]{2,4})?$", s))
            if not ok or s in seen:
                continue
            seen.add(s)
            if _is_valid_symbol_cached(s, lookback_days=60):
                out.append(s)
                if len(out) >= limit:
                    return out
        if out:
            return out
    return out

def fetch_realtime_quote_batch(symbols: List[str], cache_ttl: int = 30, timeout: Optional[float] = None) -> List[dict]:
    syms = [s for s in (symbols or []) if s]
    if not syms:
        return []
    key = "rtq:" + ",".join(sorted(syms))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    r = _req_get(
        "https://query1.finance.yahoo.com/v7/finance/quote",
        params={"symbols": ",".join(syms)},
        timeout=timeout or _REQ_TIMEOUT,
    )
    out: List[dict] = []
    if not r:
        _cache_set(key, out, cache_ttl)
        return out

    tz_tw = TZ_TW
    try:
        items = r.json().get("quoteResponse", {}).get("result") or []
        for it in items:
            epoch = it.get("regularMarketTime") or it.get("postMarketTime") or it.get("preMarketTime")
            t_local = datetime.fromtimestamp(epoch, tz=tz_tw).strftime("%Y-%m-%d %H:%M:%S") if epoch else None
            price = it.get("regularMarketPrice")
            chg = it.get("regularMarketChange")
            chg_pct = it.get("regularMarketChangePercent")
            pm_price = it.get("postMarketPrice") or it.get("preMarketPrice")
            pm_chg = it.get("postMarketChange") or it.get("preMarketChange")
            pm_chg_pct = it.get("postMarketChangePercent") or it.get("preMarketChangePercent")
            out.append({
                "symbol": it.get("symbol"),
                "name": it.get("shortName") or it.get("longName") or it.get("displayName") or it.get("symbol"),
                "currency": it.get("currency"),
                "exchange": it.get("fullExchangeName") or it.get("exchange"),
                "marketState": it.get("marketState"),
                "time": t_local,
                "price": price,
                "change": chg,
                "changePercent": chg_pct,
                "pm_price": pm_price,
                "pm_change": pm_chg,
                "pm_changePercent": pm_chg_pct,
            })
    except Exception:
        out = []

    _cache_set(key, out, cache_ttl)
    return out

def format_realtime_quotes(quotes: List[dict]) -> str:
    if not quotes:
        return "抱歉，暫時無法取得即時報價。"

    lines = []
    for q in quotes:
        sym = q.get("symbol") or "?"
        # 名稱優先用 v7 回來的 name；沒有時用我們的備援查找
        nm = (q.get("name") or "").strip() or _display_name_for_symbol(sym) or ""
        head = f"{sym}（{nm}）" if nm else sym

        # 沒有價格 → 直接顯示「暫無報價」
        if q.get("price") is None:
            lines.append(f"{head}: 暫無報價")
            continue

        chg_pct = f"{q['changePercent']:+.2f}%" if isinstance(q.get("changePercent"), (int, float)) else "—"
        chg = f"{q['change']:+.3f}" if isinstance(q.get("change"), (int, float)) else "—"
        tail = f"（{q.get('currency','')}，{q.get('exchange','')}，{q.get('marketState','')}，{q.get('time','')}）"

        line = f"{head} 現價 {q['price']:.3f}，漲跌 {chg}（{chg_pct}）{tail}"

        # 盤後 / 盤前
        if q.get("pm_price") is not None:
            pm_chg_pct = f"{q['pm_changePercent']:+.2f}%" if isinstance(q.get("pm_changePercent"), (int, float)) else "—"
            line += f"\n  ↳ 盤後/盤前 {q['pm_price']:.3f}（{pm_chg_pct}）"

        lines.append(line)

    return "\n".join(lines)

@lru_cache(maxsize=512)
def to_english_company_name(name: str) -> Optional[str]:
    """用 LLM 產生常見英文官方名稱（只回名稱），再去打 en-US 的補全。"""
    try:
        txt = classifier_llm.invoke(
            f"把以下公司名稱翻成常見英文官方名，僅輸出名稱，不要解釋：{name}"
        ).content.strip()
        # 只允許簡短、安全的名字
        if 1 <= len(txt) <= 60 and not re.search(r"[^0-9A-Za-z .,&-]", txt):
            return txt
    except Exception:
        pass
    return None

@lru_cache(maxsize=512)
def guess_tickers_via_llm(name: str, max_n: int = 5) -> List[str]:
    """最後一層：請 LLM 直接給可能的 Yahoo 代號清單（中/英/數），之後逐一以 chart 驗證。"""
    prompt = (
        "只輸出一行合法 JSON：{\"tickers\":[...]}。"
        "給我這家公司最可能的 Yahoo Finance 代號（可含 .TW/.TWO/.HK），"
        "不要附任何解釋或其它欄位：\n公司：" + name
    )
    try:
        raw = classifier_llm.invoke(prompt).content.strip()
        obj = extract_json_from_text(raw) or {}
        items = obj.get("tickers", []) if isinstance(obj, dict) else []
        out = []
        for s in items:
            if isinstance(s, str) and 1 <= len(s) <= 15 and re.fullmatch(r"[A-Za-z0-9\.]+", s):
                out.append(s.strip())
        return out[:max_n]
    except Exception:
        return []

def _is_valid_symbol_cached(symbol: str, lookback_days: int = 30) -> bool:
    return is_valid_symbol(symbol, lookback_days=lookback_days)

@lru_cache(maxsize=4096)
def is_valid_symbol(symbol: str, lookback_days: int = 10) -> bool:
    try:
        now = int(datetime.now().timestamp())
        p1 = now - lookback_days * 86400
        r = _req_get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": p1, "period2": now, "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=5
        )
        res = (r.json().get("chart", {}) if r else {}).get("result")
        ts = (res[0] or {}).get("timestamp") if res else []
        if ts:
            return True
        # 後援：台股新股
        if re.match(r"^\d{4}\.(TW|TWO)$", (symbol or "").upper()):
            return _is_tradable_via_quote(symbol)
        return False
    except Exception:
        if re.match(r"^\d{4}\.(TW|TWO)$", (symbol or "").upper()):
            return _is_tradable_via_quote(symbol)
        return False

def _parse_jsonp(text: str) -> Optional[dict]:
    m = re.search(r"\((\{.*\})\)\s*$", text or "")
    if m:
        try: return json.loads(m.group(1))
        except Exception: return None
    try: return json.loads(text)
    except Exception: return None

def pick_best_yahoo_candidate(q: str, cands: List[dict]) -> Optional[str]:
    # 盡量用中文重疊 + 模糊度排出最可能；若全是英文名，也會回第一個有效候選
    qn = _normalize_name(q)
    has_rf = bool(fuzz)
    ranked: List[Tuple[float, dict]] = []
    for c in cands or []:
        name = _normalize_name(c.get("name","") or "")
        base = 0.0
        if has_rf and name:
            base = float(fuzz.token_set_ratio(qn, name))  # 0..100
        bias = 12.0 if c.get("region")=="TW" else (8.0 if c.get("region")=="US" else 0.0)
        ranked.append((base + bias, c))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    sym = ranked[0][1].get("symbol")
    return sym if sym and _is_valid_symbol_cached(sym, lookback_days=30) else None


@lru_cache(maxsize=4096)
def _yahoo_autoc_collect(q: str, regions, lang: str, timeout: float, *, legacy: bool) -> List[dict]:
    """Internal helper for Yahoo autocomplete; preserves original behavior."""
    if not (q or "").strip():
        return []
    out: List[dict] = []
    base = "https://s.yimg.com/aq/autoc" if legacy else "https://autoc.finance.yahoo.com/autoc"
    for rg in regions:
        try:
            params = {"query": q, "region": rg, "lang": lang}
            if legacy:
                params["callback"] = ""
            r = _req_get(base, params=params,
                         headers={"User-Agent":"Mozilla/5.0"}, timeout=timeout)
            data = _parse_jsonp(r.text if r else "") or {}
            items = ((data.get("ResultSet") or {}).get("Result") or [])
            for it in items:
                typ = (it.get("type") or it.get("Type") or "").upper()
                if typ not in ("S","E","EQUITY","ETF","FUND"): continue
                out.append({
                    "symbol": it.get("symbol") or it.get("Symbol"),
                    "name": it.get("name") or it.get("Name"),
                    "exch": (it.get("exch") or it.get("Exch") or "").upper(),
                    "region": rg,
                })
        except Exception:
            pass
        finally:
            time.sleep(0.05)
    def _score(x: dict) -> Tuple[int,int,int]:
        rg_score = {"TW":3,"US":2,"HK":1}.get(x.get("region","").upper(),0)
        ex_ok = int(any(k in (x.get("exch") or "") for k in ("TAI","TWO","NYQ","NMS","NQ","NAS")))
        return (rg_score, ex_ok, -len(x.get("symbol") or ""))
    out.sort(key=_score, reverse=True)
    seen, uniq = set(), []
    for c in out:
        s = c.get("symbol")
        if s and s not in seen:
            uniq.append(c); seen.add(s)
    return uniq


@lru_cache(maxsize=4096)
def yahoo_autoc_legacy(q: str, regions=("TW","US","HK"), lang="zh-TW", timeout: float=3.0) -> List[dict]:
    # Reuse shared collector to avoid duplication
    return _yahoo_autoc_collect(q, regions, lang, timeout, legacy=True)

@lru_cache(maxsize=4096)
def yahoo_autoc(q: str, regions=("TW","US","HK"), lang="zh-TW", timeout: float=3.0) -> List[dict]:
    return _yahoo_autoc_collect(q, regions, lang, timeout, legacy=False)

@lru_cache(maxsize=4096)
def yahoo_autoc_all(
    q: str,
    regions=("TW","US","HK"),
    langs=("zh-TW","zh-Hant-TW","zh-HK","en-US"),
    timeout: float = 5.0,
    *,
    max_langs: Optional[int] = None,
) -> List[dict]:
    """同時嘗試 legacy 與新版 autoc，並輪詢多語系；整合去重。

    max_langs: 限制最多嘗試幾個語系，None 表示全部。
    """
    if not (q or "").strip():
        return []
    out: List[dict] = []
    lang_iter = list(langs)
    if max_langs is not None:
        lang_iter = lang_iter[:max_langs]
    for lang in lang_iter:
        try:
            out += yahoo_autoc_legacy(q, regions=regions, lang=lang, timeout=timeout)
        except Exception:
            pass
        try:
            out += yahoo_autoc(q, regions=regions, lang=lang, timeout=timeout)
        except Exception:
            pass
        # 若已取得候選，提早停止後續語系輪詢以降低延遲
        if out:
            break
        time.sleep(0.05)
    # 去重
    seen, uniq = set(), []
    for c in out:
        sym = c.get("symbol")
        if sym and sym not in seen:
            uniq.append(c); seen.add(sym)
    return uniq

# --- TWSE/TPEX 名稱→代號（每日快取） ---
_TW_CACHE: Dict[str, Any] = {"ts": 0, "by_name": {}, "by_code": {}}


def _is_tw_cache_fresh(max_age: float = 24 * 3600) -> bool:
    if not _TW_CACHE.get("by_name"):
        return False
    ts = float(_TW_CACHE.get("ts") or 0)
    return (time.time() - ts) < max_age

def _normalize_name(name: str) -> str:
    s = (name or "").translate(_ZH_MAP)
    s = re.sub(r"\s+|　+", "", s)
    s = re.sub(r"[()（）【】『』「」《》〈〉·．・･．．，,。．\\.]", "", s)
    s = s.replace("股份有限公司", "").replace("控股", "").replace("集團", "").replace("公司", "").replace("股份", "").replace("有限", "")
    s = re.sub(r"-?\s*KY$", "", s, flags=re.I)
    s = re.sub(r"-?\s*ＫＹ$", "", s, flags=re.I)
    return s

def _ordered_subseq(a: str, b: str) -> bool:
    """判斷 a 是否為 b 的有序子序列；處理「台積電」 vs 「台灣積體電路製造」這類縮寫。"""
    it = iter(b)
    return all(ch in it for ch in a)

# rapidfuzz 已於檔頭集中處理；此處不再重複導入

def _load_tw_lists(force: bool=False) -> None:
    if not force and (time.time() - _TW_CACHE["ts"] < 24*3600) and _TW_CACHE["by_name"]:
        return
    by_name: Dict[str, set] = {}
    sources = [
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", ".TW"),
        ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_L", ".TWO"),
    ]
    headers = {"User-Agent":"Mozilla/5.0","Accept":"application/json, */*"}
    for url, suf in sources:
        try:
            r = _req_get(url, headers=headers, timeout=8)
            data = r.json() if (r and r.ok) else []
            for row in data:
                code = (row.get("公司代號") or row.get("有價證券代號") or row.get("Code") or "").strip()
                name = (row.get("公司名稱") or row.get("有價證券名稱") or row.get("Name") or "").strip()
                if re.fullmatch(r"\d{4}", code) and name:
                    norm = _normalize_name(name)
                    by_name.setdefault(norm, set()).add(code + suf)
                    norm_no_ky = re.sub(r"-?\s*KY$", "", norm, flags=re.I)
                    if norm_no_ky != norm:
                        by_name.setdefault(norm_no_ky, set()).add(code + suf)

                    _TW_CACHE["by_code"][code + suf] = name  # ← 新增：代號→中文名反查**
        except Exception:
            continue
    _TW_CACHE["by_name"] = by_name
    _TW_CACHE["ts"] = time.time()

def _resolve_via_yahoo_pipeline(
    q_raw: str,
    *,
    fast_mode: bool = False,
    autoc_timeout: Optional[float] = None,
    autoc_max_langs: Optional[int] = None,
    validate_symbols: bool = True,
) -> Optional[str]:
    q_raw = (q_raw or "").strip()
    if not q_raw:
        return None

    timeout_val = autoc_timeout or (1.2 if fast_mode else 5.0)
    max_langs_val = autoc_max_langs if autoc_max_langs is not None else (2 if fast_mode else None)
    default_langs = ("zh-TW", "zh-Hant-TW", "zh-HK", "en-US") if not fast_mode else ("zh-TW", "en-US")

    def _prefer_tw_first(cands_syms: list[str]) -> list[str]:
        return sorted(set(cands_syms), key=lambda s: (0 if re.match(r"^\d{4}\.(TW|TWO)$", s.upper()) else 1, len(s)))

    def _calc_max_langs(lang_seq: Iterable[str]) -> Optional[int]:
        if max_langs_val is None:
            return None
        seq = list(lang_seq)
        return min(len(seq), max_langs_val)

    def _autoc_candidates(query: str, *, regions=("TW", "US", "HK"), langs: Optional[Iterable[str]] = None) -> List[dict]:
        lang_seq = tuple(langs or default_langs)
        ml = _calc_max_langs(lang_seq)
        return yahoo_autoc_all(query, regions=regions, langs=lang_seq, timeout=timeout_val, max_langs=ml)

    def _is_valid(sym: str) -> bool:
        if not validate_symbols:
            return True
        return _is_valid_symbol_cached(sym, lookback_days=30)

    # 1) 直接用 autocomplete（多語多區）
    syms = []
    for c in _autoc_candidates(q_raw):
        s = (c.get("symbol") or "").upper()
        if s and _is_valid(s):
            syms.append(s)
    for s in _prefer_tw_first(syms):
        return s

    if fast_mode:
        return None

    # 2) 英譯再試一次（常見於中文公司名）
    en_name = to_english_company_name(q_raw)
    if en_name:
        syms2 = []
        for c in _autoc_candidates(en_name, langs=("en-US", "zh-TW")):
            s = (c.get("symbol") or "").upper()
            if s and _is_valid(s):
                syms2.append(s)
        for s in _prefer_tw_first(syms2):
            return s

    # 3) LLM 猜 ticker → 逐一驗證（需通過中文名重疊門檻）
    for s in _prefer_tw_first(guess_tickers_via_llm(q_raw, max_n=6)):
        if not _is_valid(s):
            continue

        ok = False

        if re.match(r"^\d{4}\.(TW|TWO)$", s.upper()):
            nm = _tw_find_name_for_code(s) or ""
            if _cjk_overlap_score(q_raw, nm) >= 2:
                ok = True

        if not ok:
            for c in _autoc_candidates(s, langs=("zh-TW", "en-US")):
                if (c.get("symbol", "").upper() == s.upper()):
                    nm = c.get("name") or ""
                    if _cjk_overlap_score(q_raw, nm) >= 2:
                        ok = True
                        break

        if ok:
            return s

    # 4) 站內搜尋（HTML）→ 二次中文名重疊驗證
    for s in yahoo_search_symbols_by_keyword(q_raw, limit=5):
        ok = False
        for c in _autoc_candidates(s, langs=("zh-TW", "en-US")):
            if (c.get("symbol", "").upper() != s.upper()):
                continue
            qn = _normalize_name(q_raw)
            cn = _normalize_name(c.get("name", ""))
            A = set(re.findall(r"[\u4e00-\u9fff]", qn))
            B = set(re.findall(r"[\u4e00-\u9fff]", cn))
            overlap = len(A & B)

            sim_ok = False
            if fuzz:
                try:
                    sim_ok = fuzz.partial_ratio(qn, cn) >= 85
                except Exception:
                    pass

            if overlap >= 2 or sim_ok:
                ok = True
                break
        if ok:
            return s

    return None

def resolve_symbol_by_name(
    name: str,
    *,
    fast_mode: bool = False,
    autoc_timeout: Optional[float] = None,
    autoc_max_langs: Optional[int] = None,
) -> Optional[str]:
    q_raw = (name or "").strip()
    if not q_raw:
        return None
    timeout_val = autoc_timeout or (1.2 if fast_mode else 5.0)
    max_langs_val = autoc_max_langs if autoc_max_langs is not None else (2 if fast_mode else None)
    default_langs = ("zh-TW", "zh-Hant-TW", "en-US") if not fast_mode else ("zh-TW", "en-US")

    def _calc_max_langs(lang_seq: Iterable[str]) -> Optional[int]:
        if max_langs_val is None:
            return None
        return min(len(list(lang_seq)), max_langs_val)

    def _autoc_all(query: str, *, regions=("TW", "US", "HK"), langs: Optional[Iterable[str]] = None) -> List[dict]:
        lang_seq = tuple(langs or default_langs)
        ml = _calc_max_langs(lang_seq)
        return yahoo_autoc_all(query, regions=regions, langs=lang_seq, timeout=timeout_val, max_langs=ml)
    
    # A) 先查持久快取（命中就回，避免重打多個端點）
    hit_rec = _symcache_get_record(q_raw)
    if hit_rec:
        hit = (hit_rec.get("symbol") or "").strip()
        if hit:
            try:
                age = max(0.0, time.time() - float(hit_rec.get("ts", 0)))
            except Exception:
                age = None
            ttl_days = max(1, SYMBOL_MAP_TTL_DAYS)
            revalidate_after = ttl_days * 43200  # 0.5 * 86400
            if age is None or age <= revalidate_after:
                _remember_resolved_symbol(q_raw, hit)
                return hit
            cands = _autoc_all(q_raw, regions=("TW","US","HK"), langs=("zh-TW","en-US"))
            syms = { (c.get("symbol") or "").upper() for c in cands }
            _load_tw_lists()
            norm = _normalize_name(q_raw)
            tw_ok = hit.upper() in set((_TW_CACHE.get("by_name") or {}).get(norm, []))
            if hit.upper() in syms or tw_ok:
                _symcache_put(q_raw, hit)
                _remember_resolved_symbol(q_raw, hit)
                return hit
            _SYM_KV["map"].pop(_sym_norm_key(q_raw), None)
            _symcache_flush()

    # B) Yahoo-first
    sym = _resolve_via_yahoo_pipeline(
        q_raw,
        fast_mode=fast_mode,
        autoc_timeout=autoc_timeout,
        autoc_max_langs=autoc_max_langs,
        validate_symbols=not fast_mode,
    )
    if sym:
        if is_zh(q_raw):
            ok = False
            if re.match(r"^\d{4}\.(TW|TWO)$", sym.upper()):
                nm = _tw_find_name_for_code(sym) or ""
                ok = _cjk_overlap_score(q_raw, nm) >= 2
            if not ok:
                for c in _autoc_all(q_raw, regions=("TW","US","HK"), langs=("zh-TW","en-US")):
                    if (c.get("symbol", "").upper() == sym.upper()):
                        ok = True
                        break
            if ok:
                _symcache_put(q_raw, sym)
                _remember_resolved_symbol(q_raw, sym)
            return sym
        else:
            _symcache_put(q_raw, sym)
            _remember_resolved_symbol(q_raw, sym)
            return sym

    # C) 名單冷備援（TWSE/TPEX，只有 Yahoo 全線失敗才用）
    _load_tw_lists()
    by_name: Dict[str, set] = _TW_CACHE["by_name"]
    q = _normalize_name(q_raw)

    def _bias(s: str) -> float:
        return 2.0 if s.endswith(".TW") else (1.0 if s.endswith(".TWO") else 0.0)

    if q in by_name:
        for s in sorted(by_name[q], key=_bias, reverse=True):
            if _is_valid_symbol_cached(s, lookback_days=30):
                _symcache_put(q_raw, s); _remember_resolved_symbol(q_raw, s); return s

    prefix_pool, contain_pool = [], []
    for k, syms in by_name.items():
        if k == q:
            continue
        if k.startswith(q) or q.startswith(k):
            prefix_pool.extend(syms)
        elif q and (q in k or k in q):
            contain_pool.extend(syms)
    for pool in (prefix_pool, contain_pool):
        seen = set()
        for s in sorted((x for x in pool if not (x in seen or seen.add(x))), key=_bias, reverse=True):
            if _is_valid_symbol_cached(s, lookback_days=30):
                _symcache_put(q_raw, s); _remember_resolved_symbol(q_raw, s); return s

    subs = []
    for k, syms in by_name.items():
        if _ordered_subseq(q, k):
            subs.extend(syms)
    for s in sorted(set(subs), key=_bias, reverse=True):
        if _is_valid_symbol_cached(s, lookback_days=30):
            _symcache_put(q_raw, s); _remember_resolved_symbol(q_raw, s); return s

    return None

def resolve_symbol(
    t: str,
    *,
    fast_mode: bool = False,
    autoc_timeout: Optional[float] = None,
    autoc_max_langs: Optional[int] = None,
) -> Optional[str]:
    raw = (t or "").strip()
    if not raw:
        return None
    cache_key = f"symres:{raw.lower()}"
    cached = _cache_get(cache_key)
    if isinstance(cached, str) and cached:
        return cached

    s = raw
    timeout_val = autoc_timeout or (1.2 if fast_mode else 5.0)
    max_langs_val = autoc_max_langs if autoc_max_langs is not None else (2 if fast_mode else None)

    def _cache_and_return(sym: Optional[str]) -> Optional[str]:
        if sym:
            _remember_resolved_symbol(raw, sym)
            _cache_set(cache_key, sym, _RESOLVE_CACHE_TTL)
        return sym

    # 含中文 → 先走中文名解析
    if re.search(r"[\u4e00-\u9fff]", s):
        hit = resolve_symbol_by_name(
            s,
            fast_mode=fast_mode,
            autoc_timeout=autoc_timeout,
            autoc_max_langs=autoc_max_langs,
        )
        if hit:
            return _cache_and_return(hit)

    # 純代碼樣式
    if re.fullmatch(r"[A-Za-z.]+", s):
        ticker_candidate = s.upper()
        is_ticker_like = bool(re.fullmatch(r"[A-Z]{1,6}(?:\.[A-Z]{2,4})?", ticker_candidate))
        if is_ticker_like and fast_mode:
            return _cache_and_return(ticker_candidate)
        if is_ticker_like and not fast_mode:
            if is_valid_symbol(ticker_candidate):
                return _cache_and_return(ticker_candidate)
        autoc_query = ticker_candidate if is_ticker_like else s
        # fast_mode 下直接依賴 autocomplete；正常模式補 chart 驗證
        if is_ticker_like and not fast_mode:
            cands = yahoo_autoc_all(
                autoc_query,
                regions=("US","HK","TW"),
                langs=("en-US","zh-TW","zh-Hant-TW"),
                timeout=timeout_val,
                max_langs=max_langs_val,
            )
            for c in cands:
                sym = c.get("symbol")
                if sym and _is_valid_symbol_cached(sym, lookback_days=30):
                    return _cache_and_return(sym)
        # fast_mode 或前面未命中 → 直接依賴 Yahoo autocomplete，多語多區域
        autoc_query = ticker_candidate if is_ticker_like else s
        cands = yahoo_autoc_all(
            autoc_query,
            regions=("US","HK","TW"),
            langs=("en-US","zh-TW","zh-Hant-TW"),
            timeout=timeout_val,
            max_langs=max_langs_val,
        )
        for c in cands:
            sym = c.get("symbol")
            if sym and _is_valid_symbol_cached(sym, lookback_days=30):
                return _cache_and_return(sym)
        # 3) 最後再用 LLM 猜測幾個 ticker 並驗證
        if not fast_mode:
            for cand in guess_tickers_via_llm(s, max_n=6):
                if _is_valid_symbol_cached(cand, lookback_days=30):
                    return _cache_and_return(cand)
    # 若以上都沒中，才真的回 None（落到函式最後的 return None）
    if re.fullmatch(r"\d{4}", s):
        for suf in (".TW",".TWO"):
            cand = s + suf
            if is_valid_symbol(cand):
                return _cache_and_return(cand)
        return None
    if re.fullmatch(r"\d{4}\.[A-Za-z]{2,4}", s):
        return _cache_and_return(s if is_valid_symbol(s) else None)

    # Autocomplete 備援（排名器）：統一使用 yahoo_autoc_all，並套用排名器
    for regs in (("TW",), ("US","HK")):
        cands = yahoo_autoc_all(
            s,
            regions=regs,
            langs=("zh-TW","en-US"),
            timeout=timeout_val,
            max_langs=max_langs_val,
        )
        sym = pick_best_yahoo_candidate(s, cands)
        if sym:
            return _cache_and_return(sym)
    return None

def fetch_latest_close_via_chart(symbol: str, lookback_days: int = 7, timeout: Optional[float] = None) -> Optional[dict]:
    """
    以 Yahoo v8 chart 在過去 lookback_days 天內尋找最近一筆「日線收盤」。
    用於 v7 quote 擋掉時的 fallback。
    """
    try:
        tz_tw = TZ_TW
        now = datetime.now(tz_tw)
        p1 = int((now - timedelta(days=lookback_days)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        p2 = int(now.timestamp())

        r = _req_get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": p1, "period2": p2, "interval": "1d", "includePrePost": "false"},
            timeout=timeout or _REQ_TIMEOUT,
        )
        if not r:
            return None

        res = (r.json().get("chart", {}) or {}).get("result") or []
        if not res:
            return None
        node = res[0] or {}
        ts = node.get("timestamp") or []
        q0 = ((node.get("indicators") or {}).get("quote") or [{}])[0] or {}
        opens  = q0.get("open")  or []
        highs  = q0.get("high")  or []
        lows   = q0.get("low")   or []
        closes = q0.get("close") or []
        vols   = q0.get("volume") or []

        # 從最近往回找第一筆有 close 的 K
        for i in range(len(ts) - 1, -1, -1):
            c = closes[i] if i < len(closes) else None
            if c is None:
                continue
            o = opens[i]  if i < len(opens)  else None
            h = highs[i]  if i < len(highs)  else None
            l = lows[i]   if i < len(lows)   else None
            v = vols[i]   if i < len(vols)   else None
            dstr = datetime.fromtimestamp(ts[i], tz_tw).date().isoformat()
            return {
                "symbol": symbol, "date": dstr,
                "open": o, "high": h, "low": l, "close": c,
                "volume": int(v) if v is not None else None
            }
        return None
    except Exception:
        return None

def fetch_stock_price_on_date(symbol: str, date_str: str) -> Optional[dict]:
    """用 Yahoo v8 chart 取得『某日』日線資料（找不到就回 None）。"""
    try:
        dt = dtparse(date_str)
        tz_tw = TZ_TW
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz_tw)
        p1 = int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        p2 = int((dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        r = _req_get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": p1, "period2": p2, "interval": "1d", "includePrePost": "false"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
        )
        res = (r.json().get("chart", {}) if r else {}).get("result")
        if not res:
            return None
        q = (res[0] or {}).get("indicators", {}).get("quote") or []
        if not q or not (q[0] or {}):
            return None
        q0 = q[0]
        def first(a): return a[0] if isinstance(a, list) and a else None
        o, h, l, c, v = map(first, (q0.get("open"), q0.get("high"), q0.get("low"), q0.get("close"), q0.get("volume")))
        if None in (o, h, l, c): return None
        return {"symbol": symbol, "date": date_str, "open": o, "high": h, "low": l, "close": c, "volume": int(v) if v is not None else None}
    except Exception:
        return None

def _display_name_for_symbol(symbol: str) -> Optional[str]:
    """盡量給出可讀名稱：先取 Yahoo v7，台股用 TW 名單補；最後用 Yahoo Autocomplete 對照。"""
    s = (symbol or "").upper()
    if not s:
        return None

    # (A) Yahoo v7 quote
    q = _yahoo_v7_quote(s) or {}
    for k in ("shortName", "longName", "displayName"):
        name = (q.get(k) or "").strip()
        if name:
            return name

    # (B) 台股：用 TWSE/TPEX 名單反查
    if re.match(r"^\d{4}\.(TW|TWO)$", s):
        nm = _tw_find_name_for_code(s)
        if nm:
            return nm

    # (C) Yahoo Autocomplete：用代號本身 & 純數字再比一次
    for cand in yahoo_autoc_all(
        s,
        regions=("TW","US","HK"),
        langs=("zh-TW","en-US"),
        timeout=1.2,
        max_langs=2,
    ):
        if (cand.get("symbol") or "").upper() == s:
            nm = (cand.get("name") or "").strip()
            if nm:
                return nm
    m = re.match(r"^(\d{4})\.(TW|TWO)$", s)
    if m:
        code_only = m.group(1)
        for cand in yahoo_autoc_all(
            code_only,
            regions=("TW",),
            langs=("zh-TW","en-US"),
            timeout=1.2,
            max_langs=2,
        ):
            if (cand.get("symbol") or "").upper() == s:
                nm = (cand.get("name") or "").strip()
                if nm:
                    return nm
    return None

def format_stock_reply(data: Optional[dict]) -> str:
    if not data:
        return "抱歉，無法取得該股票的日線資料。"
    name = data.get("name") or _display_name_for_symbol(data["symbol"])
    head = f"{data['symbol']}（{name}）" if name else data['symbol']
    vol = data["volume"] if data["volume"] is not None else "—"
    return (
        f"{head} 於 {data['date']}：開盤 {data['open']:.2f}、最高 {data['high']:.2f}、"
        f"最低 {data['low']:.2f}、收盤 {data['close']:.2f}，成交量 {vol} 股。"
    )

def fetch_yahoo_stock_news(symbol: Optional[str] = None, max_results: int = 5, company_kw: Optional[str] = None) -> List[dict]:
    def _req(url: str) -> Optional[BeautifulSoup]:
        try:
            resp = _req_get(
                url,
                headers={"User-Agent":"Mozilla/5.0","Accept-Language":"zh-TW,zh;q=0.9,en;q=0.8"},
                timeout=3.0,
            )
            return BeautifulSoup(resp.text, "html.parser") if resp else None
        except Exception:
            return None

    def _norm(href: str) -> str:
        return href if (href or "").startswith("http") else ("https://tw.stock.yahoo.com" + (href or ""))

    def _is_article(href: str, title: str) -> bool:
        href = _norm(href or "")
        if not href or not title: return False
        if "/news/" not in href: return False
        if re.search(r"/news($|/)$", href): return False
        if re.search(r"/(mail|weather|quote)($|/)", href): return False
        return len(title.strip()) >= 6

    def _extract_news(soup: Optional[BeautifulSoup]) -> List[dict]:
        if soup is None: return []
        news, seen = [], set()
        selectors = [
            'ul.My\\(0\\).P\\(0\\).Wow\\(bw\\).Ov\\(h\\) li.js-stream-content h3 a',
            'li.js-stream-content h3 a',
            'section a[href*="/news/"]',
            'a[href*="/news/"]',
        ]
        for css in selectors:
            for a in soup.select(css):
                title = a.get_text(strip=True)
                href = a.get("href", "")
                if not _is_article(href, title): continue
                key = (_norm(href), title)
                if key in seen: continue
                seen.add(key)
                news.append({"title": title, "link": _norm(href), "source": "Yahoo奇摩"})
                if len(news) >= max_results: return news
            if news: return news
        return news

    def _filter_by_kw(items: List[dict], kws: List[str]) -> List[dict]:
        if not kws: return items
        out = []
        for it in items:
            t = (it.get("title") or "").lower()
            if any(kw for kw in kws if kw and kw.lower() in t):
                out.append(it)
        return out or items

    if symbol:
        url = f"https://tw.stock.yahoo.com/quote/{symbol}/news"
        items = _extract_news(_req(url))
        if items:
            return items
        kws = [company_kw or "", symbol.split(".")[0]]
        items = _extract_news(_req("https://tw.stock.yahoo.com/news"))
        return _filter_by_kw(items, kws)[:max_results]

    items = _extract_news(_req("https://tw.stock.yahoo.com/news"))
    if company_kw:
        items = _filter_by_kw(items, [company_kw])
    return items[:max_results]

def format_news_markdown(news: List[dict]) -> str:
    def fmt_one(n: dict) -> str:
        title = n.get("title") or ""
        link = n.get("link") or "#"
        src = n.get("source") or "Yahoo奇摩"
        t = n.get("time")
        suffix = f" — {src}" + (f"｜{t}" if t else "")
        return f"- [{title}]({link}){suffix}"
    return "\n".join(fmt_one(x) for x in (news or [])) or "Yahoo 暫無相關新聞。"

# ===== 股票 / 新聞 Markdown 區塊的共用組裝器 =====
def _prepare_stock_md(
    user_q: str,
    intents: set,
    companies: list,
    *,
    symbol_cache: Optional[Dict[str, Optional[str]]] = None,
    quote_cache: Optional[Dict[str, Optional[List[dict]]]] = None,
    chart_cache: Optional[Dict[str, Optional[dict]]] = None,
) -> str:
    if "stock" not in intents:
        return ""

    targets = (companies or [])[:]
    if not targets:
        rx = re.findall(r"[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?|\b\d{4}\b", user_q)
        targets = rx
    if not targets:
        # 讓 guess_companies 先把候選詞丟出來，後面再用 resolve_symbol() 做最終判斷
        targets = guess_companies_from_text(user_q, limit=3)
    # 先做嚴格清洗，避免把疑問詞/描述語當成公司
    cleaned_targets = []
    seen_ct = set()
    for t in targets:
        ct = clean_company_token(t, original_text=user_q)
        if ct and ct not in seen_ct:
            cleaned_targets.append(ct)
            seen_ct.add(ct)
    targets = cleaned_targets
    if not targets:
        return "（未能辨識有效股票代號，請提供代號，例如：2330.TW、TSLA、AAPL）"

    # 目標公司去重（保序）
    seen_targets = []
    for t in targets:
        if t not in seen_targets:
            seen_targets.append(t)

    symbols = []
    cache = symbol_cache if symbol_cache is not None else {}
    for comp in seen_targets[:6]:
        cache_key = (comp or "").strip().casefold()
        sym = cache.get(cache_key) if cache_key else None
        if sym is None and cache_key in cache:
            continue  # 已知失敗，不重試
        if sym is None:
            sym = resolve_symbol(comp, fast_mode=True, autoc_timeout=1.0, autoc_max_langs=2)
        if not sym:
            sym = resolve_symbol(comp, fast_mode=False, autoc_timeout=3.0, autoc_max_langs=3)
        if symbol_cache is not None and cache_key:
            symbol_cache[cache_key] = sym
        if sym:
            _remember_resolved_symbol(comp, sym)
            symbols = [sym]  # 只保留第一個解析成功的代號
            break

    # 符號去重（保序；此時最多 1 個）
    uniq_syms = []
    seen_syms = set()
    for s in symbols:
        if s and s not in seen_syms:
            uniq_syms.append(s); seen_syms.add(s)
    symbols = uniq_syms

    # 若已包含台股 2330.TW，就移除重複的美股 ADR TSM（避免不必要的第二家公司）
    if "2330.TW" in symbols and "TSM" in symbols:
        symbols = [s for s in symbols if s != "TSM"]

    # ✅ 沒有任何有效代號 → 明確告知使用者
    if not symbols:
        return "（未能辨識有效股票代號，請提供代號，例如：2330.TW、TSLA、AAPL）"

    # 有指定日期 → 查該日「日線」
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", user_q)
    if m:
        date = m.group(1)
        lines = [format_stock_reply(fetch_stock_price_on_date(sym, date)) for sym in symbols]
    else:
        # 沒指定日期 → 先抓「即時價」，抓不到再退回最近收盤
        sym0 = symbols[0]
        cached_qts = None
        if quote_cache is not None:
            cached_qts = quote_cache.get(sym0)
        if cached_qts is None:
            cached_qts = fetch_realtime_quote_batch(symbols, timeout=2.0)
            if quote_cache is not None:
                quote_cache[sym0] = cached_qts
        qts = cached_qts
        if qts:
            lines = [format_realtime_quotes(qts)]
        else:
            lines = []
            for sym in symbols:
                cached_chart = None
                if chart_cache is not None:
                    cached_chart = chart_cache.get(sym)
                if cached_chart is None:
                    cached_chart = fetch_latest_close_via_chart(sym, lookback_days=7, timeout=4.0)
                    if chart_cache is not None:
                        chart_cache[sym] = cached_chart
                data = cached_chart
                lines.append(format_stock_reply(data))

    # 若仍無任何可用資訊，回覆統一訊息（避免誤導）
    if not lines or not any((l or "").strip() for l in lines):
        return "（抱歉，查無可用股價資料。請確認代號或稍後再試。）"

    return "\n".join(lines).strip()

def _prepare_news_md(
    intents: set,
    companies: list,
    user_q: str = "",
    *,
    symbol_cache: Optional[Dict[str, Optional[str]]] = None,
    news_cache: Optional[Dict[Tuple[Optional[str], Optional[str]], List[dict]]] = None,
) -> str:
    if "news" not in intents:
        return ""
    targets = (companies or [])[:]
    if not targets:
        targets = guess_companies_from_text(user_q, limit=1)
    sym = None
    cache = symbol_cache if symbol_cache is not None else {}
    if targets:
        tok = targets[0]
        cache_key = (tok or "").strip().casefold()
        sym = cache.get(cache_key) if cache_key else None
        if sym is None and cache_key in cache:
            sym = None
        else:
            if sym is None:
                sym = resolve_symbol(tok, fast_mode=True, autoc_timeout=1.0, autoc_max_langs=2)
            if not sym:
                sym = resolve_symbol(tok, fast_mode=False, autoc_timeout=3.0, autoc_max_langs=3)
            if symbol_cache is not None and cache_key:
                symbol_cache[cache_key] = sym
        if sym:
            _remember_resolved_symbol(tok, sym)
    kw_for_filter = (targets[0] if targets and re.search(r"[\u4e00-\u9fff]", targets[0]) else None)
    cache_key = (sym, (kw_for_filter or "").casefold() if kw_for_filter else None)
    cached_news = None
    if news_cache is not None:
        cached_news = news_cache.get(cache_key)
    if cached_news is None:
        cached_news = fetch_yahoo_stock_news(sym, max_results=5, company_kw=kw_for_filter)
        if news_cache is not None:
            news_cache[cache_key] = cached_news
    news = cached_news
    return format_news_markdown(news)


# ========== 中文→英文翻譯 + 英文多問句 ==========

@lru_cache(maxsize=1024)
def zh2en_query(q: str) -> str:
    """
    將中文金融/產業問句翻成英文，保留：股票代號、數字、日期格式不變。
    僅輸出英文一句，無標點裝飾、無多餘說明。
    """
    if not q or not is_zh(q):
        return q or ""
    prompt = (
        "Translate the following Chinese financial/industry query into precise English. "
        "Preserve stock tickers (e.g., 2330.TW, TSLA, AAPL), numbers, and dates as-is. "
        "Return ONLY the English sentence, no quotes or extra text.\n\n"
        f"Query: {q}"
    )
    try:
        txt = classifier_llm.invoke(prompt).content.strip()
        # 極簡清洗：移除包起來的引號
        txt = txt.strip().strip("「」\"'")
        return txt
    except Exception:
        return q  # 失敗時直接回原文，後續會有備援

# =========================
# 檢索（穩定去重 / 先過濾後加權 / 查詢去噪 / 回退節流）
# =========================
import hashlib
import math
from functools import lru_cache

def _stable_fp(text: str, md: dict) -> str:
    """跨程序穩定的內容指紋，避免 Python 內建 hash 的隨機化。"""
    basis = (
        (text or "")[:1024]
        + "|"
        + str(md.get("doc_id") or md.get("doc_url") or md.get("url") or md.get("source") or md.get("filename") or "")
        + "|"
        + str(md.get("page") or extract_page_number(md) or -1)
    ).encode("utf-8", "ignore")
    return hashlib.sha1(basis).hexdigest()

def _rank_decay(i: int) -> float:
    """回退時的名次分數：平滑、與 k 無關。"""
    return 1.0 / (1.0 + i)

def _dedup_norm_queries(qs: list[str]) -> list[str]:
    """查詢去噪去重：去空白、全半形/大小寫標準化、刪重。"""
    seen, out = set(), []
    for q in qs:
        if not q:
            continue
        t = str(q).strip()
        if not t:
            continue
        norm = t.casefold()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(t)
    return out

def _precompute_company_needles(
    companies: List[str] | None,
    user_q: str,
    fetch_info: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
) -> set[str]:
    """公司過濾用的 needles 預先算一次，避免在 hit 迴圈反覆計算。"""
    nds = set()
    for tok in (companies or []):
        if fetch_info is not None:
            info = fetch_info(tok)
            if not info:
                continue
            nds.update(info.get("needles") or ())
            continue
        tok_clean = clean_company_token(tok, original_text=user_q)
        if not tok_clean:
            continue
        _sym, needles = canonicalize_company_needles(tok_clean)
        needles = set(needles)
        needles.add(tok_clean.lower())
        nds.update(needles)
    return nds

def _pass_company(md: dict, text: str, needles: set[str]) -> bool:
    return True if not needles else doc_mentions_any_company(md, text or "", needles)

def _year_weight(md: dict, text: str, targets: set[int], strict: bool) -> float | None:
    """年份權重：盡量少用硬砍；strict 時才丟掉不相鄰的年份。"""
    y = detect_doc_report_year_from_md_or_text(md, text or "")
    if y in targets:
        return 1.35
    if (y is not None) and any(abs(y - t) == 1 for t in targets):
        return 0.85
    return (0.0 if strict else 0.55)

def _apply_bucket_weight(score: float, md: dict, type_scales: dict[str, float] | None, epsilon: float, prefilter: bool) -> float | None:
    if type_scales is None:
        return score
    w = float(type_scales.get(get_bucket(md), 1.0))
    if prefilter and w <= 0.0:
        return None
    return score * max(0.0, (epsilon + w))

def _search_with_scores(vs, q: str, k: int):
    # 1) 優先使用帶分數 API
    try:
        return vs.similarity_search_with_relevance_scores(q, k=k)
    except Exception:
        pass
    # 2) 回退：無分數時給名次衰減
    try:
        docs = vs.similarity_search(q, k=k)
        return [(d, _rank_decay(i)) for i, d in enumerate(docs)]
    except Exception:
        return []

@lru_cache(maxsize=512)
def expand_aliases_via_llm(user_q: str, max_terms: int = 12) -> dict:
    """
    讓 LLM 做術語判別與中英別名擴展。
    只回 JSON：{topic, zh_aliases, en_aliases, metrics}
    """
    prompt = f"""
    You are a finance term alias expander. Return ONLY valid JSON with keys:
    {{
    "topic": <canonical English term or null>,
    "zh_aliases": [ ... up to {max_terms} ... ],
    "en_aliases": [ ... up to {max_terms} ... ],
    "metrics": [ ... up to 6 ... ]
    }}
    Rules:
    - Include widely used aliases, abbreviations, and the official report/page names if any.
    - Prefer high-precision terms used in reputable sources (e.g., US government statistical releases).
    - Do NOT invent unknown terms. Keep items concise (<= 5 words).
    - Keep language of zh_aliases in Traditional Chinese; en_aliases in English.
    - Examples:
    - Input: "可以給我最新的非農數據嗎？"
        Output concept includes: topic ~ "nonfarm payrolls",
        zh_aliases contains ["非農", "非農就業", "就業報告"],
        en_aliases contains ["nonfarm payrolls", "NFP", "Employment Situation", "jobs report", "BLS"],
        metrics may contain ["unemployment rate", "average hourly earnings"].

    Input: {user_q}
    """
    try:
        raw = classifier_llm.invoke(prompt).content.strip()
        obj = extract_json_from_text(raw) or {}
        def _clean_list(xs):
            out, seen = [], set()
            for s in xs or []:
                if isinstance(s, str):
                    t = s.strip().strip("「」\"'()[]")
                    if 1 <= len(t) <= 40 and t.lower() not in seen:
                        out.append(t); seen.add(t.lower())
            return out[:max_terms]
        return {
            "topic": (obj.get("topic") if isinstance(obj.get("topic"), str) else None),
            "zh_aliases": _clean_list(obj.get("zh_aliases")),
            "en_aliases": _clean_list(obj.get("en_aliases")),
            "metrics": _clean_list(obj.get("metrics")),
        }
    except Exception:
        return {"topic": None, "zh_aliases": [], "en_aliases": [], "metrics": []}

def _merge_cap(a: list[str], b: list[str], cap: int) -> list[str]:
    seen, out = set(), []
    for s in [*a, *b]:
        if s and s not in seen:
            out.append(s); seen.add(s)
        if len(out) >= cap: break
    return out

def is_earnings_call_query(text: str) -> bool:
    t = (text or "").lower()
    keys = ["法說", "法說會", "電話會議", "earnings call", "conference call", "prepared remarks", "transcript"]
    return any(k in t for k in keys)

def similarity_search_vectors(
    user_q: str,
    k: int = 12,
    type_scales: Dict[str, float] | None = None,
    prefilter: bool = True,
    epsilon: float = 0.0,
    *,
    intents: set | None = None,
    year_targets: set[int] | None = None,
    strict_year: bool = True,
    companies: List[str] | None = None,
    lap: Optional[Callable[[str], None]] = None,   # ← 新增：外部可傳入計時器（例如 handle_question 的 _lap）
):
    """
    依語言把 query 分流，並強化「公司＋法說會/財報」種子。
    回傳：(docs, scores, zh_queries_used, en_queries_used)
    """
    import re, math

    def _lap(name: str):
        if lap:
            try:
                lap(name)
            except Exception:
                pass

    _lap("retrieval:start")

    # --- 中文庫 queries (base) ---
    zh_seed = (user_q or "").strip()
    zh_queries = [zh_seed] if zh_seed else []

    # --- 英文庫 queries (base) ---
    if is_zh(user_q):
        base_en = (zh2en_query(user_q) or "").strip()
        en_queries = [base_en] if base_en else []
    else:
        en_seed = zh_seed
        en_queries = [en_seed] if en_seed else []
    _lap("retrieval:mk_base_queries")

    # 先準備 pin 容器（之後會以較大 cap 合併，避免被裁掉）
    zh_pins: List[str] = []
    en_pins: List[str] = []

    # === 財報意圖：簡單補強關鍵詞 ===
    if intents and ("financial_report" in intents):
        yrs = sorted(year_targets) if year_targets else []
        ticker_candidate = None
        for c in (companies or []):
            c_clean = (c or "").strip()
            if not c_clean:
                continue
            if re.fullmatch(r"[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?", c_clean):
                ticker_candidate = c_clean.upper()
                break
        if ticker_candidate:
            zh_pins += [
                f"{ticker_candidate} 財報",
                f"{ticker_candidate} 季報",
                f"{ticker_candidate} 年報",
            ]
            en_pins += [
                f"{ticker_candidate} financial report",
                f"{ticker_candidate} earnings report",
                f"{ticker_candidate} filing",
            ]
        metrics_keywords = ["revenue", "eps", "gross margin", "guidance"]
        for m in metrics_keywords:
            if ticker_candidate:
                en_pins.append(f"{ticker_candidate} {m}")
            zh_pins.append(f"{m} 財報")
        for y in yrs:
            zh_pins += [f"{y} 年報", f"FY{y} 財報", f"{y} 年 10-K"]
            en_pins += [f"FY{y} report", f"{y} annual report", f"{y} 10-K"]

    company_info_cache: Dict[str, Dict[str, Any]] = {}
    company_cache_lock = threading.Lock()
    COMPANY_RESOLVE_BUDGET = 3.0  # 秒，整體公司解析的時間上限
    company_resolve_start = time.perf_counter()
    needs_precise_symbols = bool((intents or set()) & {"stock", "financial_report", "news"})

    def _make_company_info(token: str, sym: Optional[str], needles: Iterable[str], source: str) -> Dict[str, Any]:
        base = (token or "").strip()
        needles_set = set(n.lower() for n in needles if n)
        if base:
            needles_set.add(base.lower())
        if sym:
            needles_set.add(sym.lower())
            digits = re.sub(r"\D", "", sym)
            if digits:
                needles_set.add(digits)
        return {
            "token": base,
            "sym": sym,
            "needles": frozenset(n for n in needles_set if n),
            "source": source,
        }

    def _heuristic_company_info(tok: str, tok_clean: Optional[str]) -> Optional[Dict[str, Any]]:
        basis = (tok_clean or tok or "").strip()
        if not basis:
            return None
        up = basis.upper()

        # 1) 最近解析快取
        recent = _lookup_recent_symbol(basis) or (
            _lookup_recent_symbol(tok_clean) if tok_clean and tok_clean != basis else None
        )
        if recent:
            return _make_company_info(basis, recent, [basis, recent], source="recent")

        # 2) 符號快取命中 → 直接回傳
        hit = _symcache_get(basis) or (_symcache_get(tok_clean) if tok_clean and tok_clean != basis else None)
        if hit:
            return _make_company_info(basis, hit, [basis, hit], source="cache")

        # 3) 台股公開名單先行匹配（若快取已就緒）
        tw_ready = _is_tw_cache_fresh()
        if not tw_ready:
            _warm_tw_lists_async()
        by_name = _TW_CACHE.get("by_name") if tw_ready else {}
        by_code = _TW_CACHE.get("by_code") if tw_ready else {}
        if by_name and by_code:
            norm_basis = _normalize_name(basis)
            tw_syms = list(by_name.get(norm_basis, []))
            if not tw_syms and tok_clean and tok_clean != basis:
                tw_syms = list(by_name.get(_normalize_name(tok_clean), []))
            if tw_syms:
                sym_pick = tw_syms[0]
                needles = [basis, tok_clean or basis, sym_pick, sym_pick.split(".")[0]]
                return _make_company_info(basis, sym_pick, needles, source="twlist")
            sym_from_code = None
            if up in by_code:
                sym_from_code = up
            elif up + ".TW" in by_code:
                sym_from_code = up + ".TW"
            elif up + ".TWO" in by_code:
                sym_from_code = up + ".TWO"
            if sym_from_code:
                needles = [basis, tok_clean or basis, sym_from_code, sym_from_code.split(".")[0]]
                return _make_company_info(basis, sym_from_code, needles, source="twlist")

        # 4) 直接判斷是否為 ticker / 代碼，避免立即打外部 API
        if re.fullmatch(r"[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?", up):
            return _make_company_info(basis, up, [up], source="fast")
        if re.fullmatch(r"\d{4}(?:\.[A-Za-z]{2,4})?", up):
            if "." in up:
                return _make_company_info(basis, up, [up, up.split(".")[0]], source="fast")
            tw_sym = up + ".TW"
            needles = [tw_sym, up + ".TWO", up]
            return _make_company_info(basis, tw_sym, needles, source="fast")

        # 5) fallback：至少保留原字串作為 needle
        return _make_company_info(basis, None, [basis], source="fast")

    def _canonicalize_company_info(basis: str, seed: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        sym, needles = canonicalize_company_needles(
            basis,
            fast_mode=True,
            autoc_timeout=1.0,
            autoc_max_langs=2,
        )
        merged_needles: Set[str] = set(needles or [])
        if seed:
            merged_needles.update(seed.get("needles") or [])
        merged_needles.add(basis.lower())
        sym_final = sym or ((seed or {}).get("sym"))
        if sym_final:
            _remember_resolved_symbol(basis, sym_final)
        return _make_company_info((seed or {}).get("token") or basis, sym_final, merged_needles, source="slow")

    def _get_company_info(raw_token: str, allow_slow: bool = True) -> Optional[Dict[str, Any]]:
        tok = (raw_token or "").strip()
        if not tok:
            return None
        tok_clean = clean_company_token(tok, original_text=user_q)
        key = tok_clean or tok
        with company_cache_lock:
            cached = company_info_cache.get(key)
        if cached:
            return cached

        info_fast = _heuristic_company_info(tok, tok_clean)
        if info_fast:
            with company_cache_lock:
                company_info_cache[key] = info_fast
        else:
            info_fast = _make_company_info(tok_clean or tok, None, [tok_clean or tok], source="fast")
            with company_cache_lock:
                company_info_cache[key] = info_fast

        if info_fast.get("sym") or not allow_slow:
            return info_fast

        if (time.perf_counter() - company_resolve_start) > COMPANY_RESOLVE_BUDGET:
            return info_fast

        basis = (tok_clean or tok or "").strip()
        if not basis:
            return info_fast
        info_slow = _canonicalize_company_info(basis, info_fast)
        if info_slow:
            with company_cache_lock:
                company_info_cache[key] = info_slow
            return info_slow
        return info_fast

    company_tokens = [c for c in (companies or []) if str(c).strip()]
    allow_slow_companies = needs_precise_symbols
    company_infos_by_token: Dict[str, Optional[Dict[str, Any]]] = {}
    if company_tokens:
        _warm_tw_lists_async()
        if len(company_tokens) > 1:
            max_workers = min(4, len(company_tokens))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_get_company_info, tok, allow_slow_companies): (idx, tok)
                    for idx, tok in enumerate(company_tokens)
                }
                interim: Dict[int, Optional[Dict[str, Any]]] = {}
                for fut in as_completed(futures):
                    idx, tok = futures[fut]
                    try:
                        interim[idx] = fut.result()
                    except Exception:
                        interim[idx] = None
                for idx, tok in enumerate(company_tokens):
                    company_infos_by_token[tok] = interim.get(idx)
        else:
            for tok in company_tokens:
                company_infos_by_token.setdefault(
                    tok,
                    _get_company_info(tok, allow_slow=allow_slow_companies),
                )

    # ====== 法說會題：強化查詢（公司名＋代碼） ======
    ec = is_earnings_call_query(user_q)
    if ec:
        targets = [c for c in (companies or [])[:1] if str(c).strip()]
        for tok in targets:
            info = company_infos_by_token.get(tok)
            if info is None:
                info = _get_company_info(tok, allow_slow=allow_slow_companies)
                company_infos_by_token[tok] = info
            sym = (info or {}).get("sym") or ""
            if sym:
                _remember_resolved_symbol(tok, sym)
            num = re.sub(r"\D", "", sym) if sym else ""  # 例如 3416
            # 中文 pins
            zh_pins += [
                f"{tok} 法說會 重點",
                f"{tok} 法說 逐字稿 摘要",
                f"{tok} 法說會 摘要",
                f"{tok} 法說 Q&A 重點",
            ]
            # 英文 pins
            en_pins += [
                f"{tok} earnings call transcript highlights",
                f"{tok} earnings call prepared remarks summary",
                f"{tok} conference call Q&A key points",
                f"{tok} earnings transcript summary",
            ]
            if num:
                zh_pins += [f"{num} 法說會 摘要", f"{num} 法說會 重點"]
                en_pins += [f"{num} earnings call transcript"]

    # === 依公司加強（把公司 alias/ticker 直接當查詢詞） ===
    comp_needles: Set[str] = set()
    resolved_syms: List[str] = []
    for tok in company_tokens:
        info = company_infos_by_token.get(tok)
        if info is None and allow_slow_companies:
            info = _get_company_info(tok, allow_slow=allow_slow_companies)
            company_infos_by_token[tok] = info
        if not info:
            continue
        comp_needles.update(info.get("needles") or ())
        sym = info.get("sym")
        if sym:
            resolved_syms.append(sym)
            _remember_resolved_symbol(tok, sym)

    if resolved_syms:
        resolved_syms = list(dict.fromkeys(resolved_syms))

    if resolved_syms or comp_needles:
        boost_terms = list(sorted(comp_needles))[:6]
        # 這些也視為 pins（避免被截斷）
        zh_pins = boost_terms + zh_pins
        # 英文 pins 只留含英數的
        en_pins = list(resolved_syms) + [t for t in boost_terms if re.search(r"[A-Za-z0-9]", t)] + en_pins

    # === 年份 pins（財報題） ===
    if intents and ("financial_report" in intents) and year_targets:
        ylist = sorted(year_targets)
        for y in ylist:
            zh_pins += [f"{y} 年報", f"{y} 年度財報", f"FY{y} 財報", f"{y} 年 EPS 毛利率 營收"]
            en_pins += [f"{y} annual report", f"FY{y} earnings", f"FY{y} results", f"{y} 10-K"]

    # === 先合併 pins，再合併 alias；cap 放寬但會在最後去噪去重 ===
    BIG_CAP = max(12, k + 6)
    zh_queries = _merge_cap(zh_pins, zh_queries, cap=BIG_CAP)
    en_queries = _merge_cap(en_pins, en_queries, cap=BIG_CAP)
    _lap("retrieval:merged_pins_base")

    # === 別名／同義詞擴展（補 recall，不壓前面的 pins） ===
    _lap("retrieval:before_alias_expand")
    aliases = expand_aliases_via_llm(user_q, max_terms=12)
    _lap("retrieval:after_alias_expand")

    # 若有年份明確意圖，就關閉「最新/latest」模板，避免偏置
    def _mk_zh(seed: str) -> list[str]:
        if year_targets:
            return [f"{seed} 公布", f"{seed} 重點"]
        return [f"最新{seed}", f"{seed} 最新一期", f"{seed} 公布", f"{seed} 重點"]

    def _mk_en(seed: str) -> list[str]:
        if year_targets:
            return [f"{seed} report", f"{seed} highlights", f"{seed} update"]
        return [f"latest {seed}", f"{seed} latest report", f"most recent {seed} highlights", f"{seed} update"]

    zh_alias_qs: List[str] = []
    for z in aliases["zh_aliases"]:
        zh_alias_qs += _mk_zh(z)
    for m in aliases["metrics"]:
        zh_alias_qs += _mk_zh(m)

    en_alias_qs: List[str] = []
    for e in aliases["en_aliases"]:
        en_alias_qs += _mk_en(e)
    for m in aliases["metrics"]:
        en_alias_qs += _mk_en(m)

    zh_queries = _merge_cap(zh_queries, zh_alias_qs, cap=max(BIG_CAP, k + 12))
    en_queries = _merge_cap(en_queries, en_alias_qs, cap=max(BIG_CAP, k + 12))

    # 查詢去噪去重（非常重要，避免發太多冗餘 RPC）
    zh_queries = _dedup_norm_queries(zh_queries)
    en_queries = _dedup_norm_queries(en_queries)
    _lap("retrieval:queries_ready")

    # —— 分數標準化工具 —— 
    def _zscore_norm(scores: list[float]) -> list[float]:
        if not scores:
            return scores
        n = len(scores)
        mu = sum(scores) / n
        var = sum((s - mu)**2 for s in scores) / max(1, (n - 1))
        if var <= 1e-12:
            return [0.0] * n
        std = var ** 0.5
        # 將 z 分數壓回 0~1（sigmoid），避免負值難以直觀混分
        return [1.0 / (1.0 + math.exp(- (s - mu) / std)) for s in scores]

    # —— 收集 hits（保留 query/順位以便除錯或做多樣性） ——
    def _collect_hits(vs, queries: list[str], k: int):
        """
        回傳 [(doc, raw_score, q_index, rank_in_q), ...]
        raw_score 可能是 relevance 分數，或回退名次分數
        """
        out = []
        for qi, q in enumerate(queries):
            hits = _search_with_scores(vs, q, k)
            for ri, (d, sc) in enumerate(hits):
                out.append((d, float(sc), qi, ri))
        return out

    # ===== 檢索（加入中/英庫內部標準化）=====
    pool, seen = [], set()
    company_needles = _precompute_company_needles(companies, user_q, fetch_info=_get_company_info)

    # 1) 先各自收集 hits
    _lap("retrieval:zh_search_start")
    zh_hits = _collect_hits(vectorstore_zh, zh_queries, k)  # [(d, raw_s, qi, ri), ...]
    _lap("retrieval:zh_search_done")

    _lap("retrieval:en_search_start")
    en_hits = _collect_hits(vectorstore_en, en_queries, k)
    _lap("retrieval:en_search_done")

    # 2) 各庫內部標準化，避免標度不一致
    _lap("retrieval:score_norm_start")
    zh_scores_norm = _zscore_norm([h[1] for h in zh_hits])
    en_scores_norm = _zscore_norm([h[1] for h in en_hits])
    _lap("retrieval:score_norm_done")

    # 3) 封裝一個將（命中文件, 標準化分）推入 pool 的步驟（先過濾、後加權）
    def _push_after_filter(d, base_score: float):
        md = d.metadata or {}
        did  = md.get("doc_id") or md.get("doc_url") or md.get("url") or md.get("source") or md.get("filename") or ""
        page = extract_page_number(md) or -1
        fp   = _stable_fp(d.page_content or "", md)
        key  = (did, page, fp)
        if key in seen:
            return

        # 公司過濾（含 transcript 意圖）
        COMPANY_FILTER_ON = bool(company_needles) and (intents and ({"financial_report", "stock", "transcript"} & intents))
        if COMPANY_FILTER_ON and not _pass_company(md, d.page_content or "", company_needles):
            return

        score = float(base_score)

        # 年份權重（財報題）
        if intents and ("financial_report" in (intents or set())) and year_targets:
            yw = _year_weight(md, d.page_content or "", year_targets, strict_year)
            if yw == 0.0:
                return
            score *= yw

        # 類別倍率 / 預過濾
        score2 = _apply_bucket_weight(score, md, type_scales, epsilon, prefilter)
        if score2 is None:
            return

        seen.add(key)
        pool.append((d, float(score2)))

    # 4) 將中/英庫標準化後的分數套過濾與加權邏輯推入 pool
    _lap("retrieval:pooling_start")
    for (d, _raw, _qi, _ri), s in zip(zh_hits, zh_scores_norm):
        _push_after_filter(d, s)
    for (d, _raw, _qi, _ri), s in zip(en_hits, en_scores_norm):
        _push_after_filter(d, s)
    _lap("retrieval:pooling_done")

    # 5) 若完全召回為空：metadata 回退（限額）
    if not pool and ec and companies:
        _lap("retrieval:metadata_fallback_start")
        try:
            needles = {c.lower() for c in companies}
            more, cap, cnt = [], 50, 0
            for d in getattr(vectorstore_zh, "docstore", {})._dict.values():
                md = d.metadata or {}
                if get_bucket(md) != "transcript":
                    continue
                hay = " ".join([
                    md.get("title", ""),
                    md.get("filename", ""),
                    md.get("doc_id", ""),
                    md.get("document_title", ""),
                ]).lower()
                if any(n in hay for n in needles):
                    more.append((d, 0.51))
                    cnt += 1
                    if cnt >= cap:
                        break
            pool.extend(more)
        except Exception:
            pass
        _lap("retrieval:metadata_fallback_done")

    # 6) 排序 & 回傳
    if not pool:
        _lap("retrieval:done_empty")
        return ([], [], zh_queries, en_queries)

    pool.sort(key=lambda x: x[1], reverse=True)
    docs, scores = zip(*pool)
    _lap("retrieval:done_ok")
    return (list(docs), list(scores), zh_queries, en_queries)

# =========================
# LLM Rerank
# =========================
rerank_prompt = PromptTemplate.from_template(
"""
你是金融/產業檢索的「段落排序器」。請根據【問題】評估下列【段落】對回答問題的「語意相關性」，並輸出**由高到低**的排序。

【評分準則（由強到弱，加分）】
1) 精準匹配：包含問題中的「公司/代號/別名/產品線」或其常見縮寫、Ticker（例如：2330.TW、TSMC、台積電）。
2) 期間匹配：問題若提到年份/季度/月份/「最新/最近/近況」，段落若出現相同期間或更接近的期間，應優先。
3) 指標匹配：問題若涉及財報/營收(Revenue/Net sales)/毛利率(Gross margin)/EPS/Guidance/Outlook/Segment/Region，優先含這些關鍵詞與**數字**的段落。
4) 文檔類型偏好：若出現「法說/earnings call/conference call/transcript」，優先逐字稿/準備稿/Q&A；若是財報問題，優先 10-K/10-Q/20-F/6-K 等原始檔。
5) 具體性：包含明確數字、表格敘述、結論性句子的段落 > 僅概念性描述。

【扣分/排除】
- 法務/免責聲明、目錄、導航、封面/頁尾、重複 boilerplate。
- 與問題公司或主題無關者。
- 只有行銷口號、無任何指標或事實。

【輸出格式（嚴格）】
- 只輸出**一行**「逗號分隔的整數排序」，涵蓋全部段落且每個整數出現一次，例如：`2,1,4,3`
- 不要任何文字說明、不要空白行。

問題：
{question}

段落：
{contexts}
"""
)

def llm_rerank(question: str, docs: List[Any], cos_scores: List[float], candidate_cap: int = 8, clip_chars: int = 900) -> List[float]:
    if not docs: return []
    order = np.argsort(np.asarray(cos_scores, dtype=float))[::-1]
    keep = [int(i) for i in order[:min(candidate_cap, len(docs))]]

    def clip(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else (s[:n] + " …")

    contexts = "\n\n".join(
        f"段落 {j+1}: {clip(docs[i].page_content, clip_chars)}" for j, i in enumerate(keep)
    )
    try:
        resp = rerank_llm.invoke(rerank_prompt.format(question=question, contexts=contexts)).content.strip()
        idxs = [int(x)-1 for x in re.findall(r"\d+", resp)]
        chosen_global = [keep[i] for i in idxs if 0 <= i < len(keep)]
        scores = [0.0] * len(docs)
        for rank, g in enumerate(chosen_global):
            scores[g] = len(docs) - rank
        return scores
    except Exception:
        return [0.0]*len(docs)

# =========================
# 偏好倍率應用（唯一權重來源）
# =========================
def hybrid_sort(docs: List[Any], cos_scores: List[float], llm_scores: List[float],
                top_k: int, type_scales: Dict[str, float],
                recency_boost: bool = False, recency_half_life: int = 45) -> Tuple[List[Any], List[float], str]:
    if not docs: return [], [], "no docs"

    def norm(arr_in: np.ndarray) -> np.ndarray:
        a = np.asarray(arr_in, dtype=float)
        a = np.atleast_1d(a)
        if a.size == 0:
            return np.zeros(0, dtype=float)
        return (a - a.min())/(a.max()-a.min()) if a.max()>a.min() else np.zeros_like(a, dtype=float)

    cos = norm(np.asarray(cos_scores, dtype=float))
    rer = norm(np.asarray(llm_scores, dtype=float))
    final = 0.2 * cos + 0.8 * rer

    # 🔥 新增：啟用「最新」時的時間加權（溫和地影響排序）
    if recency_boost:
        r = np.array([_recency_weight(d.metadata or {}, half_life_days=recency_half_life) for d in docs], dtype=float)
        final = final * (0.7 + 0.3 * r)  # 30% 權重給新鮮度，可再調

    order = np.argsort(final)[::-1]
    kept = order[:min(top_k, len(docs))]

    dbg_rows = []
    for i in kept:
        bucket = get_bucket(docs[i].metadata or {})
        mult = type_scales.get(bucket, 1.0)
        dbg_rows.append(
            f"{i+1} | cos:{float(cos[i]):.3f} | llm:{float(rer[i]):.3f} | type:{bucket}:{mult:.2f} | final:{float(final[i]):.3f}"
        )
    dbg = "idx | cos | llm | 類別×倍率 | final\n" + "\n".join(dbg_rows)

    return [docs[i] for i in kept], [float(final[i]) for i in kept], dbg

def _minmax(arr: list[float]) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0: return np.zeros(0, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a, dtype=float)

def bge_rank_and_pick(
    question: str,
    docs: List[Any],
    cos_scores: List[float],
    *,
    top_k: int = 5,
    candidate_cap: int = 30,      # ← 預設也放寬，讓 BGE 有發揮空間
    clip_chars: int = 900,
    batch_size: int = 32,
    type_scales: Dict[str, float] | None = None,
    recency_boost: bool = False,
    recency_half_life: int = 45,
) -> Tuple[List[Any], List[float], str]:

    if (not _HAS_BGE_RERANK) or (bge_reranker is None) or (not docs):
        # 後援：無 BGE 就退回 LLM rerank + hybrid_sort
        llm_scores = llm_rerank(question, docs, cos_scores, candidate_cap=min(8, max(5, top_k + 2)), clip_chars=700)
        picked_docs, final_scores, dbg = hybrid_sort(
            docs, cos_scores, llm_scores, top_k=top_k,
            type_scales=type_scales or {}, recency_boost=recency_boost,
            recency_half_life=recency_half_life
        )
        return picked_docs, final_scores, "[FALLBACK] hybrid_sort"

    # -------- 1) 先用 cosine 做「召回」：取前 N 個候選 --------
    order = np.argsort(np.asarray(cos_scores, dtype=float))[::-1]
    keep_idx = [int(i) for i in order[: min(candidate_cap, len(docs))]]
    if not keep_idx:
        return [], [], "[BGE] no candidates"

    # -------- 2) 跑 BGE 交叉編碼分數 --------
    pairs: List[List[str]] = []
    for i in keep_idx:
        txt = (docs[i].page_content or "")
        if len(txt) > clip_chars:
            txt = txt[:clip_chars] + " …"
        pairs.append([question, txt])

    bge_scores: List[float] = []
    for s in range(0, len(pairs), batch_size):
        part = pairs[s:s + batch_size]
        try:
            sc = bge_reranker.compute_score(part, normalize=True)  # 返還通常在 0~1 的相似度
        except Exception:
            sc = [0.0] * len(part)
        if isinstance(sc, (float, int)):
            sc = [float(sc)]
        bge_scores.extend([float(x) for x in sc])

    # 與候選索引對齊
    assert len(bge_scores) == len(keep_idx), "reranker length mismatch"

    # -------- 3) 批內正規化，讓 BGE 真正主導 --------
    # A) min-max：穩定、直觀（預設）
    bge_arr  = np.asarray(bge_scores, dtype=float)
    bge_main = _minmax(bge_arr)

    # 極小 tie-breaker 權重（需要更純就把下面三個設成 0）
    W_BGE, W_COS, W_TYPE, W_REC = 1.00, 0.02, 0.02, 0.02

    final = W_BGE * bge_main

    # cosine 只拿同批 min-max 值來做極小平手裁決
    cos_kept = _minmax([cos_scores[i] for i in keep_idx])
    final = final + W_COS * cos_kept

    if type_scales:
        srcw = np.array([float(type_scales.get(get_bucket(docs[i].metadata or {}), 1.0)) for i in keep_idx], dtype=float)
        final = final + W_TYPE * _minmax(srcw)

    if recency_boost:
        rec = np.array([_recency_weight(docs[i].metadata or {}, half_life_days=recency_half_life) for i in keep_idx])
        final = final + W_REC * _minmax(rec)

    # -------- 4) 依 final 排序，挑前 K --------
    order2 = np.argsort(final)[::-1][: min(top_k, len(keep_idx))]
    chosen = [keep_idx[i] for i in order2]
    picked = [docs[i] for i in chosen]
    scores = [float(final[i]) for i in order2]

    # Debug：同時顯示 bge_raw（原值）與 bge（批內 0~1）
    dbg_rows = []
    for rank, i_keep in enumerate(order2, start=1):
        gi = keep_idx[i_keep]
        bucket = get_bucket(docs[gi].metadata or {})
        dbg_rows.append(
            f"{rank}. idx:{gi+1} | bge_raw:{bge_scores[i_keep]:.3f} | bge:{bge_main[i_keep]:.3f} | cos:{cos_kept[i_keep]:.3f} | type:{bucket} | final:{final[i_keep]:.3f}"
        )
    dbg = "[RANKER] bge-reranker-large (BGE-dominant)\n" + "\n".join(dbg_rows)
    return picked, scores, dbg

# 簡易財報數字偵測（用於 Debug 與微調排序）
_NUM_PAT = re.compile(r"\d[\d,\.]*")
_REV_KWS = re.compile(r"(revenue|net\s+sales|sales|net\s+revenue|total\s+net\s+sales|turnover|營收|收入)", re.I)
_EPS_KWS = re.compile(r"(EPS|earnings\s+per\s+share|每股盈餘)", re.I)
_GM_KWS  = re.compile(r"(gross\s+margin|毛利率)", re.I)

def _metric_flags(text: str) -> dict:
    t = text or ""
    return {
        "has_num": bool(_NUM_PAT.search(t)),
        "revenue": bool(_REV_KWS.search(t)),
        "eps": bool(_EPS_KWS.search(t)),
        "gm": bool(_GM_KWS.search(t)),
    }

# =========================
# 生成答案
# =========================
answer_prompt = PromptTemplate.from_template(
"""
你是一位專業的產業與財經分析助理。根據下方 context，請用**繁體中文**條列整理主要結論。
規則：
1) 每一點**結尾**標註來源標籤與日期，格式【S#｜YYYY-MM-DD】（若未知日期請寫「未知日期」）
2) 不要編造 context 沒有的內容；可合併重複資訊
3) 如含英文，請翻為繁體中文再整合
4) 只輸出條列，不要多餘說明；如無法在 context 中找到具體數據或資訊，避免使用空泛句式（例如：「缺乏完整的財務報表細節，無法進一步說明」），改以「請參考下方參考來源的原始文件」提示使用者。

---------
{context}
---------
問題：{question}
"""
)

beginner_answer_prompt = PromptTemplate.from_template(
"""
【語言規則】請全程使用繁體中文；如 context 含英文，請先翻為繁體中文再整合。
你是一位財經科普講解員。先輸出「🔰 名詞小辭典」，列出 3–7 個此回答會用到的專有名詞，
以「- 詞：10–25 字白話解釋」格式。
接著輸出「🧠 重點回答」條列，並遵守：
1) 每點**結尾**標註【S#｜YYYY-MM-DD】（未知日期寫「未知日期」）
2) 只使用 context 內容，不要臆測；數字盡量保留原值
3) 語氣淺白、句子精短；若 context 未提供具體數字或資訊，請提示「請參考下方參考來源的原始文件」，不要使用空泛句式
---------
{context}
---------
問題：{question}
"""
)

expert_answer_prompt = PromptTemplate.from_template(
"""
【語言規則】請全程使用繁體中文；如 context 含英文，請先翻為繁體中文再整合。
你是一位產業與財經分析師。請以專業、嚴謹、邏輯清晰的語氣回答，條列呈現，每點包含：
- 結論：一句話
- 依據：引用 context 的關鍵數字／句子（可概述）
- 影響／推論：在 context 支持範圍內
規則：
1) 每點**結尾**標註【S#｜YYYY-MM-DD】（未知日期寫「未知日期」）
2) 不要編造 context 沒有的資訊；若無法在 context 中找到具體數據或資訊，請提示「請參考下方參考來源的原始文件」，避免空泛句式  
---------
{context}
---------
問題：{question}
"""
)

financial_report_compact_prompt = PromptTemplate.from_template(
    """
【語言規則】全程使用**繁體中文**；如 context 含英文，先翻再整合。

你是一位產業財報分析師。請用一段話專業總結該公司財報，涵蓋：
- 營收 (Revenue / **Net sales**)
- 毛利率 (Gross Margin %)
- EPS (Earnings per Share)
- 業務/地區營收占比 (Breakdown by Business/Region)
- 管理層展望（若有）(Company Outlook)

【嚴格規則】
A) **來源約束**：僅使用下方 context；不得臆測。凡是數字必須能在 context 找到「原句」或能由 context 的數字**計算**得到。
B) **年份題一律顯示計算/佐證**：若使用者問題含四位數年份（例如 2024），對於每一個指標都要在句末以括號呈現：
   - 若原文就有該數字 → `（出處：引用「Net sales/Revenue」所在句；標示 S# 與頁碼）`
   - 若需要彙總或換算 → `（公式 → 代入數值與單位/幣別 → 逐步運算 → 最終結果；四捨五入規則）`
   - 只要 context 沒有足夠數字或原句，請改寫為：`未於引用段落查得（請參考來源）`，**不要輸出猜測的數字**。
C) **計算格式**（若有計算）：
   - 公式 → 代入數值（含單位/幣別）→ 逐步運算 → 最終結果（標示四捨五入規則與小數位數）。
   - 例：`FY 營收 = Q1 + Q2 + Q3 + Q4 = 95.0B + 81.2B + 85.0B + 122.1B = 383.3B（四捨五入至 1 位小數）`
D) 每點**結尾**務必加來源標註【S#｜YYYY-MM-DD】（未知寫「未知日期」；如同一點用到多個來源可連續標註）。
E) 百分比保留 1–2 位；金額與單位務必標示（USD、NTD、百萬/十億等）。
F) 若同一表含多期間數值，**預設取最接近文件日期的最新期**；無法判斷期別時，請標示「最新期」。

---------
{context}
---------
問題：{question}
"""
)



def build_cited_context(docs: List[Any]) -> tuple[str, dict]:
    """
    依 doc_id 指派 S1, S2 ...，並把【S#｜Title】【YYYY-MM-DD】直接寫進每個段落的抬頭。
    另外：彙整每個 S 的頁碼清單（src_map[lab]['pages'] = [int,...]）
    回傳：
    - ctx：餵 LLM 的完整 context（含 S# 抬頭、可含頁碼）
    - src_map：{"S1": {"title":..., "date":..., "url":..., "doc_id":..., "pages":[...]}, ...}
    """
    label_for: dict[Any, str] = {}
    src_map: dict[str, dict] = {}
    pages_map: dict[str, Set[int]] = {}

    for d in docs:
        md = d.metadata or {}
        did = md.get("doc_id") or (md.get("source"), md.get("title"))
        if did not in label_for:
            lab = f"S{len(label_for)+1}"
            label_for[did] = lab

            title = (md.get("title") or "無標題").strip() or "無標題"
            date  = get_doc_date_str(md)
            url   = md.get("doc_url") or md.get("url") or md.get("filename")
            src_map[lab] = {"title": title, "date": date, "url": url, "doc_id": did, "pages": [], "snippets": [], "chunks": []}
            pages_map[lab] = set()

    blocks: list[str] = []
    for d in docs:
        md = d.metadata or {}
        did = md.get("doc_id") or (md.get("source"), md.get("title"))
        lab = label_for[did]

        title = (md.get("title") or "無標題").strip() or "無標題"
        date  = get_doc_date_str(md)
        page  = extract_page_number(md)
        if page:
            pages_map[lab].add(page)

        # 抬頭：帶頁碼（單段時可以顯示單一頁）
        head = f"【{lab}｜{title}"
        if page:
            head += f"｜p. {page}"
        head += f"】【{date}】"

        txt   = d.page_content or ""
        blocks.append(f"{head}\n{txt}")
        # 保存完整片段於來源清單
        try:
            full_txt = d.page_content or ""
            src_map[lab]["chunks"].append({
                "page": int(page) if page else None,
                "text": full_txt,
            })
        except Exception:
            pass

        # 收集片段摘要供來源清單顯示（每來源最多 3 段）
        try:
            if len(src_map.get(lab, {}).get("snippets", [])) < 3:
                sn = re.sub(r"\s+", " ", (txt or "").strip())
                try:
                    sn = _clip_text(sn, 280)
                except Exception:
                    sn = sn[:280] + (" …" if len(sn) > 280 else "")
                if page:
                    sn = f"p. {page}｜" + sn
                src_map[lab]["snippets"].append(sn)
            # 也保存完整片段，供參考來源完整輸出
            src_map[lab]["chunks"].append({
                "page": int(page) if page else None,
                "text": txt,
            })
        except Exception:
            pass

    # 將彙整好的頁碼塞回 src_map
    for lab, s in pages_map.items():
        if lab in src_map:
            src_map[lab]["pages"] = sorted(s)

    ctx = "\n\n---\n\n".join(blocks)
    return ctx, src_map


def build_cited_context_smart(
    docs: List[Any],
    *,
    max_docs: int = LLM_CTX_MAX_DOCS,
    per_doc_chars: int = LLM_CTX_PER_DOC_CHAR,
    total_chars_soft: int = LLM_CTX_TOTAL_CHAR_SOFT,
) -> tuple[str, dict]:
    """
    受限於 LLM 請求大小時，採用剪裁版 context：
    - 只取前 max_docs 段
    - 每段最多 per_doc_chars 字元
    - 總長度超過 total_chars_soft 時再做二次壓縮
    """
    picked = list(docs[:max_docs])

    # 準備 label 與 src_map（與 build_cited_context 一致）
    label_for: dict[Any, str] = {}
    src_map: dict[str, dict] = {}
    pages_map: dict[str, Set[int]] = {}

    for d in picked:
        md = d.metadata or {}
        did = md.get("doc_id") or (md.get("source"), md.get("title"))
        if did not in label_for:
            lab = f"S{len(label_for)+1}"
            label_for[did] = lab
            title = (md.get("title") or "無標題").strip() or "無標題"
            date  = get_doc_date_str(md)
            url   = md.get("doc_url") or md.get("url") or md.get("filename")
            src_map[lab] = {"title": title, "date": date, "url": url, "doc_id": did, "pages": [], "snippets": [], "chunks": []}
            pages_map[lab] = set()

    blocks: list[str] = []
    for d in picked:
        md = d.metadata or {}
        did = md.get("doc_id") or (md.get("source"), md.get("title"))
        lab = label_for[did]
        title = (md.get("title") or "無標題").strip() or "無標題"
        date  = get_doc_date_str(md)
        page  = extract_page_number(md)
        if page:
            pages_map[lab].add(page)
        head = f"【{lab}｜{title}"
        if page:
            head += f"｜p. {page}"
        head += f"】【{date}】"
        txt = (d.page_content or "")
        if len(txt) > per_doc_chars:
            txt = txt[:per_doc_chars] + " …"
        blocks.append(f"{head}\n{txt}")
        # 保存完整片段於來源清單
        try:
            full_txt = d.page_content or ""
            src_map[lab]["chunks"].append({
                "page": int(page) if page else None,
                "text": full_txt,
            })
        except Exception:
            pass

    # 二次壓縮：總長度仍過長 → 均勻再裁短
    sep = "\n\n---\n\n"
    ctx = sep.join(blocks)
    if len(ctx) > total_chars_soft and blocks:
        ratio = total_chars_soft / max(1, len(ctx))
        new_blocks = []
        for b in blocks:
            limit = max(300, int(len(b) * ratio))
            new_blocks.append(b[:limit] + (" …" if len(b) > limit else ""))
        ctx = sep.join(new_blocks)

    for lab, s in pages_map.items():
        if lab in src_map:
            src_map[lab]["pages"] = sorted(s)

    return ctx, src_map


def annotate_bullet_sources(answer_text: str, src_map: dict) -> str:
    """
    將每條包含【S#｜...】的條列行，於行尾追加對應來源連結，例如：— 來源: [S1](url)、[S2](url)
    僅在該行已包含 S 標籤時追加；若來源無 URL，僅顯示標籤。
    """
    if not answer_text or not isinstance(src_map, dict):
        return answer_text

    lines = answer_text.splitlines()
    out = []
    for line in lines:
        labs = re.findall(r"【(S\d+)｜", line)
        if labs:
            uniq = []
            seen = set()
            for l in labs:
                if l not in seen:
                    seen.add(l); uniq.append(l)
            links = []
            for lab in uniq:
                node = src_map.get(lab) or {}
                url = node.get("url")
                if url:
                    links.append(f"[{lab}]({url})")
                else:
                    links.append(lab)
            if links and "來源:" not in line:
                line = f"{line}  — 來源: " + "、".join(links)
        out.append(line)
    return "\n".join(out)

def _build_metric_label_index(src_map: dict) -> Dict[str, List[str]]:
    """從 src_map.chunks 建立指標→S 標籤的索引，用於缺數據時提示參考來源。
    回傳如 {"revenue":["S1","S3"], "eps":[...], "gm":[...]}，依 S 編號排序且去重。
    """
    metric_to_labels: Dict[str, List[str]] = {"revenue": [], "eps": [], "gm": []}
    if not isinstance(src_map, dict):
        return metric_to_labels
    # 依 S1,S2...順序掃描，確保提示順序穩定
    for lab in sorted(src_map.keys(), key=lambda k: int(k[1:]) if isinstance(k, str) and k[1:].isdigit() else 9999):
        node = src_map.get(lab) or {}
        chunks = node.get("chunks") or []
        for ch in chunks:
            txt = (ch.get("text") or "")
            flags = _metric_flags(txt)
            if flags.get("revenue") and lab not in metric_to_labels["revenue"]:
                metric_to_labels["revenue"].append(lab)
            if flags.get("eps") and lab not in metric_to_labels["eps"]:
                metric_to_labels["eps"].append(lab)
            if flags.get("gm") and lab not in metric_to_labels["gm"]:
                metric_to_labels["gm"].append(lab)
    return metric_to_labels


def suggest_sources_for_missing_metrics(answer_text: str, src_map: dict, max_labels: int = 3) -> str:
    """當答案行包含『未於引用段落查得』『未提供』等語句時，根據行內關鍵詞（營收/EPS/毛利率）
    在行尾追加『可先查看：S#、S#』提示，幫助使用者直接定位來源。
    不重複添加，且若無匹配來源則不動作。
    """
    if not answer_text:
        return answer_text

    metric_index = _build_metric_label_index(src_map)
    if not any(metric_index.values()):
        return answer_text

    # 行內關鍵詞規則
    pat_rev = re.compile(r"(營收|收入|revenue|sales)", re.I)
    pat_eps = re.compile(r"(EPS|每股盈餘|earnings\s+per\s+share)", re.I)
    pat_gm  = re.compile(r"(毛利|毛利率|gross\s+margin)", re.I)
    pat_missing = re.compile(r"(未於引用段落查得|未提供|未找到|查無|找不到)")

    out_lines = []
    for line in (answer_text or "").splitlines():
        lstrip = line.strip()
        if pat_missing.search(lstrip) and ("可先查看" not in lstrip) and ("參考：" not in lstrip):
            labels: List[str] = []
            if pat_rev.search(lstrip):
                labels = metric_index.get("revenue", [])
            elif pat_eps.search(lstrip):
                labels = metric_index.get("eps", [])
            elif pat_gm.search(lstrip):
                labels = metric_index.get("gm", [])
            # 若行內未指明是哪個指標，就綜合給一組最常見來源
            if not labels:
                for key in ("revenue","eps","gm"):
                    if metric_index.get(key):
                        labels = metric_index[key]
                        break
            labels = labels[:max_labels]
            if labels:
                hint = "（可先查看：" + "、".join(labels) + "）"
                line = line + " " + hint
        out_lines.append(line)
    return "\n".join(out_lines)


# ====== 財報數據簡易擷取（規則式補救） ======
_SENT_SPLIT = re.compile(r"(?<=[。！？!?.\n])\s+")
_NUM_TOKEN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s?%|\s?(?:億|百萬|千萬|萬|billion|million|thousand))?", re.I)


def _doc_id_key(md: Dict[str, Any]) -> Any:
    return (md or {}).get("doc_id") or ((md or {}).get("source"), (md or {}).get("title"))


def _build_label_lookup(src_map: dict) -> Dict[Any, str]:
    lut: Dict[Any, str] = {}
    for lab, node in (src_map or {}).items():
        did = (node or {}).get("doc_id")
        if did is not None:
            lut[did] = lab
    return lut

def _first_number_in_sentence(sent: str) -> Optional[str]:
    m = _NUM_TOKEN.search(sent or "")
    return m.group(0) if m else None

def _extract_metric_from_text(text: str, kind: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    pats = {
        "revenue": re.compile(r"(revenue|net\s+sales|營收|收入)", re.I),
        "eps": re.compile(r"(EPS|earnings\s+per\s+share|每股盈餘)", re.I),
        "gm": re.compile(r"(gross\s+margin|毛利率)", re.I),
    }
    pat = pats.get(kind)
    if not pat:
        return None
    # 以句子為單位尋找關鍵字並擷取第一個數字
    for sent in _SENT_SPLIT.split(t):
        if pat.search(sent):
            num = _first_number_in_sentence(sent)
            if num:
                # 回傳「數值（片段）」
                snip = sent.strip()
                snip = snip if len(snip) <= 120 else (snip[:118] + "…")
                return f"{num}｜{snip}"
    return None

def extract_financial_metrics_from_docs(picked_docs: List[Any], src_map: dict, limit_per_metric: int = 1) -> List[str]:
    """從已選段落中規則式擷取營收/EPS/毛利率，回傳適合直接作為條列的字串陣列。
    例如：- 營收：$383.29B｜…【S1｜YYYY-MM-DD】
    """
    if not picked_docs:
        return []
    label_lut = _build_label_lookup(src_map)
    results: List[str] = []
    counts = {"revenue": 0, "eps": 0, "gm": 0}
    for d in picked_docs:
        md = d.metadata or {}
        did = _doc_id_key(md)
        lab = label_lut.get(did)
        date = get_doc_date_str(md)
        if not lab:
            continue
        for kind, title in (("revenue","營收"),("eps","EPS"),("gm","毛利率")):
            if counts[kind] >= limit_per_metric:
                continue
            hit = _extract_metric_from_text(d.page_content or "", kind)
            if hit:
                results.append(f"- {title}：{hit}【{lab}｜{date}】")
                counts[kind] += 1
    return results

def render_sources_md(src_map: dict) -> str:
    """把來源對照表渲染成 Markdown 清單（含可點連結、頁碼範圍）。"""
    if not src_map:
        return ""
    lines = []
    for lab in sorted(src_map.keys(), key=lambda k: int(k[1:])):  # 依 S1, S2... 排序
        node  = src_map[lab]
        title = node["title"]
        date  = node["date"]
        pages = node.get("pages") or []
        pg_str = _format_pages_suffix(pages)  # → " ｜pp. 5–7, 12" 或 ""

        lines.append(f"- [{lab}] {title}｜{date}{pg_str}")
        chunks = node.get("chunks") or []
        if chunks:
            for ch in chunks:
                p = ch.get("page")
                pfx = f"p. {p}｜" if p else ""
                text = (ch.get("text") or "").strip()
                first = text.splitlines()[0] if "\n" in text else text
                lines.append(f"  - 片段：{pfx}" + first)
                for extra in text.splitlines()[1:]:
                    lines.append(f"    > {extra}")
        else:
            for sn in (node.get("snippets") or []):
                lines.append(f"  - 片段：{sn}")
    return "**參考來源**\n" + "\n".join(lines)

def is_summary_query(text: str) -> bool:
    """
    判斷是否為『財報總結型』問題
    """
    # 關鍵詞觸發：總結、重點、摘要、overview、overall
    if re.search(r"(總結|重點|摘要|overview|overall)", text):
        return True
    # 問句如果同時提到多個財報指標（營收+EPS、營收+毛利率），也偏向總結
    metrics = ["營收","毛利","EPS","獲利","收入","profit","earnings"]
    hit = sum(1 for m in metrics if m in text)
    return hit >= 3

def synthesize_answer(question: str, picked_docs: List[Any], mode: str = "初學者模式", intents: Optional[List[str]] = None, custom_prompt: str = "") -> str:
    if not picked_docs:
        return "目前找不到可直接回答的內容。", "", {}

    # 使用剪裁版 context，避免過長導致 LLM 413 或延遲
    ctx, src_map = build_cited_context_smart(picked_docs)

    # 優先：財報專用（精簡條列版）
    intents = intents or []
    use_custom = bool(custom_prompt and ("客製" in (mode or "")))
    if use_custom:
        cp = (custom_prompt or "").strip()
        # 客製化：若包含占位符則 format，否則把 context 與 question 接在指令後
        if ("{context}" in cp) or ("{question}" in cp):
            prompt_text = cp.format(context=ctx, question=question)
        else:
            prompt_text = f"{cp}\n---------\n{ctx}\n---------\n問題：{question}"
        body = llm.invoke(prompt_text).content.strip()
    else:
        if "financial_report" in intents:
            tmpl = financial_report_compact_prompt
            body = llm.invoke(tmpl.format(context=ctx, question=question)).content.strip()
        else:
            if "專家" in (mode or ""):
                tmpl = expert_answer_prompt
            elif "初學" in (mode or ""):
                tmpl = beginner_answer_prompt
            else:
                tmpl = answer_prompt
            body = llm.invoke(tmpl.format(context=ctx, question=question)).content.strip()

    # 於每一條列行尾追加來源連結（依 S 標籤）
    body = annotate_bullet_sources(body, src_map)
    # 若有「未於引用段落查得／未提供」等提示，補上「可先查看：S#」建議
    body = suggest_sources_for_missing_metrics(body, src_map, max_labels=3)

    # 財報題：若答案幾乎沒有數字，嘗試從片段中規則式補充關鍵數據
    if ("financial_report" in intents):
        has_digit = bool(re.search(r"\d", body))
        if not has_digit:
            extras = extract_financial_metrics_from_docs(picked_docs, src_map, limit_per_metric=1)
            if extras:
                body = body + "\n\n📊 補充：自動擷取到的關鍵數據\n" + "\n".join(extras)
    return body, ctx, src_map

# =========================
# 延伸問題
# =========================
followup_prompt = PromptTemplate.from_template(
"""
你是檢索式問答系統的「延伸問題產生器」。請根據下列資訊，產出 {n_min}~{n_max} 個**可由現有檢索內容回答**、且**使用者原題未直接詢問**的延伸問題。

嚴格規則：
1) 來源限制：問題必須能僅依「回答摘要」中的資訊（含其 S# 標註所對應內容）回答，避免需要外部知識或最新時點資料。
2) 不重複：不得改寫原題或重複其語意；必須探索「未被提及或未被詳述」的面向。
3) 可回答性：每個問題都應聚焦在摘要中已出現的名詞、指標、期間、產品線、地區、風險、展望/指引等可被佐證的元素。
4) 語言與格式：使用繁體中文；每題單一句，結尾加「？」；避免含糊字眼（如「最新」、「最近」）除非摘要已含相應期間。

輸出格式（嚴格）：只輸出一行合法 JSON：{{"followups":["問題1","問題2", ...]}}

【使用者原問題】{question}
【回答摘要（含 S# 提示）】{answer_snippets}
"""
)

def make_followups(question: str, answer_text: str, n_min=3, n_max=5) -> List[str]:
    """
    更健壯的延伸問題產生器：
    1) 先用 LLM（嚴格 JSON 指令）嘗試；失敗再重試一次（更嚴苛的提示）
    2) 若仍不足 n_min，偵測「查股價」型問題改走規則式 fallback
    3) 最終清洗去重並補標點
    """
    def _norm_items(items: Any) -> List[str]:
        out, seen = [], set()
        for s in (items or []):
            if not isinstance(s, str):
                continue
            t = s.strip().strip("・-—*•").strip()
            if not t:
                continue
            # 只留合理長度的單句
            if 6 <= len(t) <= 120 and t not in seen:
                out.append(t)
                seen.add(t)
        return out

    # ---- A) 先壓縮回答，避免超長影響 LLM ----
    ans = (answer_text or "").strip()
    if len(ans) > 600:
        ans = ans[:600] + " …"

    # ---- B) 第一次 LLM 嘗試（原提示）----
    try:
        raw = classifier_llm.invoke(
            followup_prompt.format(
                question=question,
                answer_snippets=ans or "（無）",
                n_min=n_min,
                n_max=n_max
            )
        ).content.strip()
    except Exception:
        raw = ""

    obj = extract_json_from_text(raw) or {}
    items = obj.get("followups", []) if isinstance(obj, dict) else []
    out = _norm_items(items)

    # ---- C) 第二次 LLM 嘗試（更嚴格提示 + 明確格式要求）----
    if len(out) < n_min:
        example = '{{"followups":["..."]}}'  # 純文字樣例（雙大括號輸出花括號）
        strict_prompt = (
            '只輸出**一行**合法 JSON（UTF-8；鍵名固定為 "followups"；值為字串陣列），不得包含任何前後綴或多餘換行。\n'
            '請產出 {n_min}~{n_max} 題延伸問題，且必須同時滿足：\n'
            '1) 問題可由下方「回答摘要」與其 S# 所代表的內容回答，不需外部知識；\n'
            '2) 不重複使用者原題，且避免僅是同義改寫；\n'
            '3) 以摘要中已出現的名詞/指標/期間/產品線/地區/風險/展望等為焦點；\n'
            '4) 使用繁體中文、單一句、以「？」結尾；避免含糊期間（如「最新/最近」）除非摘要已明示。\n\n'
            f"範例：{example}\n\n"
            f"【使用者原問題】{question}\n"
            f"【回答摘要（含 S# 提示）】{ans or '（無）'}\n"
            f"【數量要求】{n_min}~{n_max} 題"
        )
        try:
            raw2 = classifier_llm.invoke(strict_prompt).content.strip()
        except Exception:
            raw2 = ""
        obj2 = extract_json_from_text(raw2) or {}
        items2 = obj2.get("followups", []) if isinstance(obj2, dict) else []
        out.extend(_norm_items(items2))

    # ---- D) 最終清洗：補問號、去重、裁切到 n_max ----
    final, seen = [], set()
    for t in out:
        tt = t.strip()
        if not (tt.endswith("？") or tt.endswith("?")):
            tt += "？" if is_zh(question) else "?"
        if tt not in seen:
            final.append(tt)
            seen.add(tt)
        if len(final) >= n_max:
            break

    return final


# =========================
# 意圖判斷 + 公司抽取（關鍵字＋LLM 聯集 & 正規化）
# =========================
# 1) 關鍵字意圖
def keyword_intents(text: str) -> set:
    intents = set()
    if contains_any(text, STOCK_KWS): intents.add("stock")
    if contains_any(text, NEWS_KWS): intents.add("news")
    if contains_any(text, FINREP_KWS): intents.add("financial_report")
    if re.search(r"(是什麼|什麼是|介紹|概述|科普|入門|定義|原理|overview|what is)", text or "", re.I):
        intents.add("concept")
    return intents

# 2) LLM 多意圖＋公司抽取（單行 JSON）
multi_intent_template = PromptTemplate.from_template(
r"""
你是精準的「多重意圖與公司抽取器」，只輸出**一行合法 JSON**（UTF-8）：
{{"intents":[...], "companies":[...]}}

【任務】
- 僅在**明確**要求「特定上市公司/代號的即時或歷史**股價資訊**」時，才把 "stock" 放入 intents。
- 只要問到財報/三大表/營收/EPS/毛利率/展望/guidance/segment，就把 "financial_report" 放入 intents。
- 問「新聞/更新/報導」時，可加入 "news"。

【公司抽取（極度嚴格）】
- "companies" 僅允許：
  1) **單一公司中文名**（例如：台積電、微軟、美光、輝達）
  2) **公司英文名**（例如：TSMC、Microsoft、Micron、NVIDIA、Apple）
  3) **股票代號**（例如：2330、2330.TW、AAPL、MSFT）
- 不得包含任何助詞（的/之…）、空格句子、標點尾綴或整段話。
- 每個項目長度 2~20 字元，且不包含空白。
- 無法確定就輸出空陣列 []。

【排除規則】
- 宏觀名詞（OPEC、Fed、CPI、原油、黃金…）若未問個股或財報，intents 皆留空。
- 組織/商品/產業名稱不算公司。

【嚴格格式要求】
- 僅輸出一行 JSON，鍵名固定為 "intents" 與 "companies"，不得包含其他欄位或說明文字。
- 例（正確）：
  {{"intents":["financial_report"], "companies":["微軟","MSFT"]}}
- 例（錯誤，會被判定為無效）：{{"intents":["financial_report"], "companies":["可以給我 微軟 的財報重點嗎"]}}

輸入：{text}
"""
)


# === 新增：年份判斷小工具 ===
_YEAR_PAT = re.compile(r"\b(19\d{2}|20\d{2})\b")

def looks_like_year(token: str, text: str) -> bool:
    if not re.fullmatch(r"\d{4}", token):
        return False
    y = int(token)
    if 1900 <= y <= 2099:
        # 就算沒上下文，四碼多半是年份：先判 True
        return True
    return False

# 3) 公司清單正規化
def normalize_companies(companies_raw: Any, *, original_text: str = "") -> List[str]:
    if not companies_raw:
        return []
    out: List[str] = []

    def _extract_one(x: Any) -> Optional[str]:
        if isinstance(x, str):
            return clean_company_token(x, original_text=original_text)
        if isinstance(x, dict):
            for k in ("symbol", "ticker", "code", "name", "company", "company_name", "title"):
                v = x.get(k)
                if isinstance(v, str):
                    t = clean_company_token(v, original_text=original_text)
                    if t:
                        return t
        return None

    for item in (companies_raw or []):
        v = _extract_one(item)
        if v:
            out.append(v)

    # 去重
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:3]

# 4) 中文句型與中文片段輔助（保留你原本的強韌備援）
_CJK_HINTS = ("股價","報價","新聞","快訊","目標價","財報","法說","法說會","營收","公告","重訊","走勢")
_CJK_STOP = {"股價","報價","新聞","快訊","目標價","財報","法說","營收","公告","重訊",
             "公司","股份","有限","控股","集團","ETF","基金","指數"}
_CJK_STOP |= {"股票","哪支","哪檔","哪家","哪間","哪個","哪一支","哪一檔","什麼","是什麼"}
_CJK_STOP |= {"是什麼", "是什麼嗎", "什麼是", "介紹", "概述", "簡介", "定義", "原理", "請問", "可以", "是否", "嗎", "呢"}

def guess_companies_from_text(text: str, limit: int = 3) -> List[str]:
    text = (text or "").strip()

    cand: List[str] = []

    # 允許公司名與關鍵詞之間有空白；且若公司名前面有年份，優先取公司名本身
    # 例如：「2025 微軟 的財報」「微軟 財報」「台積電的財報」
    hints = "|".join(_CJK_HINTS)
    pat = rf"(?:^|[\s（(,，])(?:19\d{{2}}|20\d{{2}})?\s*([\u4e00-\u9fffA-Za-z0-9\.\-]{{2,18}})\s*(?:的)?(?:{hints})"
    m = re.search(pat, text)
    if m:
        name = m.group(1)
        # 移除誤抓到的前導四碼年份（極少數會貼在一起）
        name = re.sub(r"^(?:19\d{2}|20\d{2})", "", name).strip()
        if name:
            cand.append(name)

    # 原本的補強
    cand += re.findall(r"\b\d{4}\b", text)
    cand += re.findall(r"\b[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?\b", text)

    # 中文片段備援（去掉 stop 詞）
    chunks = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    chunks = [c for c in chunks if c not in _CJK_STOP]
    head = cand[0] if cand and re.search(r"[\u4e00-\u9fff]", cand[0]) else None
    ordered = ([head] if head else []) + [c for c in chunks if c and c != head]

    out: List[str] = []
    seen: set = set()
    for token in cand + ordered:
        if not token or token in seen:
            continue
        seen.add(token)
        if re.fullmatch(r"\d{4}", token):
            # 年份不要
            continue
        if re.fullmatch(r"[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?", token):
            out.append(token)
        else:
            if 2 <= len(token) <= 18:
                out.append(token)
        if len(out) >= limit:
            break

    uniq, seen2 = [], set()
    for x in out:
        if x not in seen2:
            seen2.add(x); uniq.append(x)
    return uniq[:limit]

# 5) 最終對外：意圖＋公司（關鍵字 ∪ LLM；公司正規化＋備援）
TICKER_RE = re.compile(
    r"(\$[A-Za-z]{1,5}\b|\b[A-Z]{1,5}\.[A-Z]{2,4}\b|\b\d{4}\.[A-Za-z]{2,4}\b)"
)
PRICE_RE = re.compile(
    r"(股[\s\u200B-\u200D\uFEFF]*價|報[\s\u200B-\u200D\uFEFF]*價|現[\s\u200B-\u200D\uFEFF]*價|"
    r"收[\s\u200B-\u200D\uFEFF]*盤|開[\s\u200B-\u200D\uFEFF]*盤|目標[\s\u200B-\u200D\uFEFF]*價|"
    r"price|quote|target\s*price)",
    re.I
)
def classify_intents_and_companies(text: str) -> Tuple[List[str], List[str]]:
    kw_ints = keyword_intents(text)

    llm_ints, llm_companies = set(), []
    try:
        resp = classifier_llm.invoke(multi_intent_template.format(text=text)).content.strip()
        info = extract_json_from_text(resp) or {}
        llm_ints = set(info.get("intents", []) or [])
        llm_companies = normalize_companies(info.get("companies", []), original_text=text)
    except Exception:
        pass

    final_intents = set()
    if "stock" in kw_ints:
        final_intents.add("stock")
    else:
        if ("stock" in llm_ints) and TICKER_RE.search(text):
            final_intents.add("stock")
        companies_probe = llm_companies or re.findall(r"\b[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?\b", text) or re.findall(r"\b\d{4}\b", text)
        if "stock" not in final_intents and (companies_probe or guess_companies_from_text(text)) and PRICE_RE.search(text):
            final_intents.add("stock")

    if ("financial_report" in kw_ints) or ("financial_report" in llm_ints):
        final_intents.add("financial_report")

    # 新聞意圖：關鍵字或 LLM 任一命中即加入
    if ("news" in kw_ints) or ("news" in llm_ints):
        final_intents.add("news")

    # ✅ 這裡改成「總是併入」規則抽取的結果
    companies: List[str] = []
    companies += llm_companies
    companies += re.findall(r"\b[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?\b", text)
    companies += re.findall(r"\b\d{4}\b", text)
    companies += guess_companies_from_text(text, limit=3)   # << 新增：永遠 union

    cleaned = []
    seen = set()
    for c in (llm_companies +
              re.findall(r"\b[A-Za-z]{1,6}(?:\.[A-Za-z]{2,4})?\b", text) +
              re.findall(r"\b\d{4}\b", text) +
              guess_companies_from_text(text, limit=3)):
        cc = clean_company_token(c, original_text=text)
        if not cc:
            continue
        if re.fullmatch(r"\d{4}", cc) and looks_like_year(cc, text):
            continue  # 2024 這種年份踢掉
        key = _normalize_name(re.sub(r"(?:的|之)?(?:股票代號|股票|個股|本股|股|代號|代碼)+$", "", cc))
        if key not in seen:
            seen.add(key)
            cleaned.append(cc if cc else key)

    if re.search(r"(是什麼|什麼是|介紹|概述|科普|入門|定義|原理|overview|what\s+is)", text, re.I):
        final_intents.add("concept")
    # 額外：涉及「產業/行業/供應鏈/生態系/價值鏈」視為產業概念
    if re.search(r"(產業|行業|供應鏈|生態系|價值鏈|市場)", text):
        final_intents.add("industry")

    return list(final_intents), cleaned[:3]

# =========================
# 月份偵測
# =========================
def chinese_month_to_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if s in CHI_NUM:
        v = CHI_NUM[s]
        return v if 1 <= v <= 12 else None
    if s.startswith("十") and len(s) == 2 and s[1] in CHI_NUM:
        v = 10 + CHI_NUM[s[1]]; return v if 1 <= v <= 12 else None
    if s.endswith("十") and len(s) == 2 and s[0] in CHI_NUM:
        v = CHI_NUM[s[0]] + 10; return v if 1 <= v <= 12 else None
    return None

def extract_date_range_from_text(text: str) -> Optional[Tuple[datetime, datetime]]:
    t = (text or "").strip()
    m_zh = re.search(r"(\d{4})\s*年\s*([0-9]{1,2}|[一二三四五六七八九十〇○Ｏ]{1,3})\s*月", t)
    if m_zh:
        y = int(m_zh.group(1)); raw = m_zh.group(2)
        mo = int(raw) if raw.isdigit() else chinese_month_to_int(raw)
        if mo and 1<=mo<=12:
            start = datetime(y, mo, 1)
            return start, start + relativedelta(months=1)
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", t)
    if m:
        y, mo, d = map(int, m.groups())
        start = datetime(y, mo, d); return start, start + timedelta(days=1)
    m2 = re.search(r"(\d{4})[/-](\d{1,2})", t)
    if m2:
        y, mo = map(int, m2.groups())
        start = datetime(y, mo, 1); return start, start + relativedelta(months=1)
    if not re.search(r"(去年|今年|明年)", t):
        m3 = re.search(r"(^|[^年])([一二三四五六七八九十〇○Ｏ]{1,3}|[0-9]{1,2})\s*月", t)
        if m3:
            raw = m3.group(2)
            mo = int(raw) if raw.isdigit() else chinese_month_to_int(raw)
            if mo and 1<=mo<=12:
                y = datetime.now().year
                start = datetime(y, mo, 1); return start, start + relativedelta(months=1)
    return None

def resolve_relative_span(text: str, now: Optional[datetime] = None) -> Optional[Tuple[datetime, datetime]]:
    if not text: return None
    now = now or datetime.now()
    t = re.sub(r"\s+", "", text)
    m3 = re.search(r"(今年|去年|明年)\s*(\d{1,2}|[一二三四五六七八九十〇○Ｏ]{1,3})\s*月", t)
    if m3:
        base = now.year if m3.group(1)=="今年" else (now.year-1 if m3.group(1)=="去年" else now.year+1)
        raw = m3.group(2); mo = int(raw) if raw.isdigit() else chinese_month_to_int(raw)
        if mo and 1<=mo<=12:
            start = datetime(base, mo, 1); return start, start + relativedelta(months=1)
    if "本月" in t or "這個月" in t:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, start + relativedelta(months=1)
    if "上月" in t or "上個月" in t:
        start = (now.replace(day=1) - relativedelta(months=1))
        return start, start + relativedelta(months=1)
    if "下月" in t or "下個月" in t:
        start = (now.replace(day=1) + relativedelta(months=1))
        return start, start + relativedelta(months=1)
    return None

def is_whole_month_query(user_text: str) -> Tuple[bool, Optional[Tuple[datetime, datetime]]]:
    rel = resolve_relative_span(user_text)
    absr = extract_date_range_from_text(user_text)
    rng = rel or absr
    if not rng: return False, None
    start, end = rng
    is_full = (start.day==1) and ((end-start).days in (28,29,30,31))
    return (True, (start, end)) if is_full else (False, None)


# =========================
# 整月掃描與摘要
# =========================
def scan_docs_by_month(year: int, month: int) -> List[Any]:
    pools: List[Any] = []
    for vs in [vectorstore_zh, vectorstore_en]:
        store = getattr(vs, "docstore", None)
        if not store or not hasattr(store, "_dict"): continue
        for d in store._dict.values():
            md = d.metadata or {}
            dt = get_doc_date_dt(md)
            if dt and dt.year==year and dt.month==month:
                pools.append(d)
    pools.sort(key=lambda x: get_doc_date_dt(x.metadata or {}) or datetime.min, reverse=True)
    return pools

def pick_representative_chunks_per_doc(chunks: List[Any], k: int = 1) -> List[Any]:
    prefer_kw = ("重點","摘要","結論","要點","takeaways")
    scored = []
    for ch in chunks:
        txt = (ch.page_content or "")
        score = sum(1 for kw in prefer_kw if kw.lower() in txt.lower())*3
        L = len(txt)
        score += 2 if 300<=L<=1200 else 0
        try:
            m = re.search(r"(\d+)$", (ch.metadata or {}).get("chunk_id", ""))
            cid = int(m.group(1)) if m else 9999
        except Exception:
            cid = 9999
        score -= cid*0.001
        scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max(1,int(k))]]

month_digest_prompt = PromptTemplate.from_template(
    """
你是一位專業的財經與產業分析顧問。以下是「某月份內的多篇文章代表段落」，
請**逐篇**整理 1–2 句重點（不可跨文混寫），每點**末尾附來源標籤與日期**（格式：【S#｜YYYY-MM-DD】），**繁體中文**輸出。
---------
{context}
---------
問題：{question}
"""
)

month_digest_prompt_beg = PromptTemplate.from_template(
    """
你是一位財經科普講解員。先輸出「🔰 名詞小辭典」：從下列代表段落挑選 3–7 個專有名詞，
以「- 詞：10–25 字白話解釋」定義。
接著輸出「🗓️ 月份總結」——**逐篇**整理 1–2 句重點（不可跨文混寫），每點末尾附來源標籤與日期（格式：【S#｜YYYY-MM-DD】），以繁體中文輸出。
---------
{context}
---------
問題：{question}
"""
)

month_digest_prompt_pro = PromptTemplate.from_template(
    """
你是一位專業的財經與產業分析顧問。以下是「某月份內的多篇文章代表段落」，
請**逐篇**整理 1–2 句重點（不可跨文混寫），每點**末尾附來源標籤與日期**（格式：【S#｜YYYY-MM-DD】），**繁體中文**輸出。
（若可能，強調結論與依據的對應關係）
---------
{context}
---------
問題：{question}
"""
)


def _clip_text(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + " …")

def _batch_by_char_limit(blocks: list[str], limit: int) -> list[list[str]]:
    batches, cur, cur_len = [], [], 0
    sep = "\n\n---\n\n"
    for b in blocks:
        bb = b if len(b) <= limit else b[: int(limit * 0.9)]
        extra = (len(sep) if cur else 0) + len(bb)
        if cur and (cur_len + extra > limit):
            batches.append(cur); cur, cur_len = [], 0
        cur.append(bb); cur_len += (len(sep) if cur_len > 0 else 0) + len(bb)
    if cur:
        batches.append(cur)
    return batches

def _renumber_bullets(text: str) -> str:
    out_lines, idx = [], 1
    for line in (text or "").splitlines():
        l = line.strip()
        if re.match(r"^\s*\d+\.\s+", l):
            l = re.sub(r"^\s*\d+\.\s+", "", l)
            out_lines.append(f"{idx}. {l}")
            idx += 1
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def run_month_digest(
    user_input: str,
    start: datetime,
    end: datetime,
    *,
    per_doc_k: int = 1,
    max_docs: int = 12,
    candidates: Optional[List[Any]] = None,
    mode: str = "初學者模式",
    custom_prompt: str = "",
) -> Tuple[str, str]:
    """
    產生「整月摘要」。回傳 (answer_md, sources_md)
    - answer_md：最終回答（含 🔰名詞小辭典 / 🗓️月份總結 取決於 mode）
    - sources_md：對應來源清單（S1/S2…，含頁碼區間）
    """
    CLIP_PER_BLOCK = 600          # 每個代表段落的最大字元數（避免 prompt 過長）
    BATCH_CHAR_LIMIT = 3600       # 每一批送 LLM 的上限（粗估，可視模型調整）
    SEP = "\n\n---\n\n"

    # 1) 取得本月候選 chunks（若外部沒傳 candidates 就掃整月）
    if candidates is None:
        y, m = start.year, start.month
        month_chunks = scan_docs_by_month(y, m)
        if not month_chunks:
            return "僅能部分回答：該月份內查無文章。", ""
        used_chunks = month_chunks
    else:
        used_chunks = candidates or []
        if not used_chunks:
            y, m = start.year, start.month
            month_chunks = scan_docs_by_month(y, m)
            if not month_chunks:
                return "僅能部分回答：該月份內查無文章。", ""
            used_chunks = month_chunks

    # 2) 依文件分組
    groups: Dict[Any, List[Any]] = {}
    for ch in used_chunks:
        md = ch.metadata or {}
        key = md.get("doc_id") or (md.get("source"), md.get("title"))
        groups.setdefault(key, []).append(ch)

    def _safe_dt(md):
        dt = get_doc_date_dt(md or {})
        # 若無時區，補 UTC，確保可排序
        if dt and not dt.tzinfo:
            dt = dt.replace(tzinfo=tz.tzutc())
        return dt or datetime.min.replace(tzinfo=tz.tzutc())

    # 3) 取最新的前 max_docs 份文件（每份文件內再挑 per_doc_k 個代表段落）
    #    以「該份文件第一個 chunk 的日期」近→遠排序（粗略但實用）
    doc_keys_sorted = sorted(
        groups.keys(),
        key=lambda k: _safe_dt((groups[k][0].metadata or {})),
        reverse=True,
    )[:max_docs]

    # 4) 為每份文件挑代表段落，並建立「帶 S 標籤、日期、頁碼」的 context blocks
    picked_chunks: List[Any] = []
    for doc_key in doc_keys_sorted:
        reps = pick_representative_chunks_per_doc(groups.get(doc_key) or [], k=max(1, int(per_doc_k)))
        picked_chunks.extend(reps)

    if not picked_chunks:
        return "僅能部分回答：該月份內查無可用段落。", ""

    # 建立「可被 LLM 引用」的 context（含 S#｜Title｜p. 等頭）
    ctx_full, src_map = build_cited_context(picked_chunks)

    # 將 context 依長度切批，避免超過模型長度
    blocks = [b.strip() for b in ctx_full.split(SEP) if b.strip()]
    # clip 過長 block
    blocks = [_clip_text(b, CLIP_PER_BLOCK) for b in blocks]
    batches = _batch_by_char_limit(blocks, limit=BATCH_CHAR_LIMIT)

    # 5) 依 mode 選用 prompt（客製化模式使用自訂 prompt）
    use_custom = bool(custom_prompt and ("客製" in (mode or "")))
    if not use_custom:
        if "專家" in (mode or ""):
            tmpl = month_digest_prompt_pro
        elif "初學" in (mode or ""):
            tmpl = month_digest_prompt_beg
        else:
            tmpl = month_digest_prompt  # 一般模式：只有月份總結（無名詞小辭典）

    # 6) 分批丟給 LLM，再把結果接起來
    answers: List[str] = []
    for bs in batches:
        part_ctx = SEP.join(bs)
        try:
            if use_custom:
                # 客製化：若包含占位符則 format，否則把 context 與 question 接在指令後
                cp = (custom_prompt or "").strip()
                if ("{context}" in cp) or ("{question}" in cp):
                    prompt_text = cp.format(context=part_ctx, question=user_input)
                else:
                    prompt_text = f"{cp}\n---------\n{part_ctx}\n---------\n問題：{user_input}"
                resp = llm.invoke(prompt_text).content.strip()
            else:
                resp = llm.invoke(tmpl.format(context=part_ctx, question=user_input)).content.strip()
        except Exception as e:
            resp = f"（本批摘要失敗：{e}）"
        answers.append(resp)

    final_answer = "\n\n".join(answers)
    final_answer = _renumber_bullets(final_answer).strip()

    # 7) 渲染來源清單（含頁碼範圍）並在每條列末尾補來源連結
    final_answer = annotate_bullet_sources(final_answer, src_map)
    sources_md = render_sources_md(src_map)
    return final_answer, sources_md

# =========================
# 主流程（含月份模式 & 類別偏好 + 新 Yahoo）
# =========================
def handle_question(user_q: str, top_n: int, top_k: int,
                    blog_scale: float, pdf_scale: float, research_scale: float, summary_scale: float,
                    filing_scale: float,
                    mode: str, gen_model_choice: str,
                    custom_prompt: str = ""):
    # 計時工具（毫秒）
    t0 = time.perf_counter()
    t_prev = t0
    times: dict[str, float] = {}
    def _lap(name: str):
        nonlocal t_prev
        now = time.perf_counter()
        times[name] = now - t_prev
        t_prev = now
    def _fmt_times(order: list[str]) -> str:
        parts = []
        for k in order:
            if k in times:
                parts.append(f"{k}:{int(times[k]*1000)}ms")
        parts.append(f"total:{int((time.perf_counter()-t0)*1000)}ms")
        return "[TIMES] " + " | ".join(parts)

    # 類別倍率顯示工具（固定順序、避免 KeyError）
    def _fmt_type_scales(ts: dict) -> str:
        order = ("blog", "pdf", "research", "transcript", "filing", "other")
        return ", ".join(f"{key}:{ts.get(key, 1.0):.2f}" for key in order)

    user_q = (user_q or "").strip()
    if not user_q:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "—", None

    effective_gen = set_gen_llm(gen_model_choice)
    _lap("set_llm")

    # === 類別倍率 ===
    type_scales = {
        "blog":       float(blog_scale),
        "pdf":        float(pdf_scale),
        "research":   float(research_scale),
        "transcript": float(summary_scale),
        "filing":     float(filing_scale),
        "other":      1.0,
    }

    # === 問題解析 ===
    is_month, month_rng = is_whole_month_query(user_q)
    intents, companies = classify_intents_and_companies(user_q)
    intents_set = set(intents)
    requested_years = extract_report_year_targets(user_q) if ("financial_report" in intents) else set()
    wants_latest = bool(re.search(r"(最新|最近|近況|未來)", user_q))
    _lap("parse")

    if ("concept" in intents_set) or ("industry" in intents_set):
        type_scales = {**type_scales}
        type_scales["blog"] = max(type_scales.get("blog", 1.0), 1.5)
        type_scales["research"] = max(type_scales.get("research", 1.0), 1.0)
        type_scales["transcript"] = min(type_scales.get("transcript", 1.0), 0.6)
        type_scales["filing"] = min(type_scales.get("filing", 1.0), 0.3)

    # 若是財報且有指定公司，但使用者把 filing 拉到 0，避免 prefilter 清空
    if ("financial_report" in intents_set) and companies and type_scales.get("filing", 0.0) <= 0.0:
        type_scales["filing"] = 0.1

    # 準備：若包含股價/新聞意圖，先產生可前置的區塊
    lead_parts = []
    stock_md_pref = ""
    news_md_pref = ""
    symbol_cache_local: Dict[str, Optional[str]] = {}
    quote_cache_local: Dict[str, Optional[List[dict]]] = {}
    chart_cache_local: Dict[str, Optional[dict]] = {}
    news_cache_local: Dict[Tuple[Optional[str], Optional[str]], List[dict]] = {}
    need_stock = "stock" in intents_set
    need_news = "news" in intents_set
    if need_stock or need_news:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: Dict[str, Any] = {}
            if need_stock:
                futures["stock"] = pool.submit(
                    _prepare_stock_md,
                    user_q,
                    intents_set,
                    companies,
                    symbol_cache=symbol_cache_local,
                    quote_cache=quote_cache_local,
                    chart_cache=chart_cache_local,
                )
            if need_news:
                futures["news"] = pool.submit(
                    _prepare_news_md,
                    intents_set,
                    companies,
                    user_q,
                    symbol_cache=symbol_cache_local,
                    news_cache=news_cache_local,
                )
        if need_stock:
            stock_md_pref = futures["stock"].result()
        if need_news:
            news_md_pref = futures["news"].result()
    if not (need_stock or need_news):
        stock_md_pref = _prepare_stock_md(
            user_q,
            intents_set,
            companies,
            symbol_cache=symbol_cache_local,
            quote_cache=quote_cache_local,
            chart_cache=chart_cache_local,
        )
        news_md_pref = _prepare_news_md(
            intents_set,
            companies,
            user_q,
            symbol_cache=symbol_cache_local,
            news_cache=news_cache_local,
        )
    if stock_md_pref:
        lead_parts.append("📈 **股票查詢**\n" + stock_md_pref)
    if news_md_pref:
        lead_parts.append("📰 **即時新聞**\n" + news_md_pref)
    _lap("yahoo_blocks")

    # === Yahoo-only 短路 ===
    if (("stock" in intents) or ("news" in intents)) and lead_parts:
        answer_md = "\n\n".join(lead_parts).strip()
        queries_md = "（此問題被判定為『股價/新聞』，已直接使用 Yahoo 資訊）"
        debug_txt = "[SHORT-CIRCUIT] yahoo_only\n" + _fmt_times(["set_llm", "parse", "yahoo_blocks"])
        dd_choices = [FOLLOWUP_PLACEHOLDER]
        meta = {
            "type": "yahoo_only",
            "question": user_q,
            "intents": list(intents),
            "companies": companies,
            "picked": 0,
        }
        return (
            answer_md,
            "",
            queries_md,
            debug_txt,
            gr.update(choices=dd_choices, value=dd_choices[0]),
            "—",
            meta,
        )

    # === (A) 月份模式 ===
    if is_month and month_rng:
        start, end = month_rng

        # 檢索 + 詳細打點
        docs, cos_scores, zh_qs, en_qs = similarity_search_vectors(
            user_q,
            k=max(12, top_n),
            type_scales=type_scales,
            prefilter=True,
            intents=intents_set,
            year_targets=requested_years,
            strict_year=True,
            companies=companies,
            lap=_lap,  # ← 傳入計時 callback
        )
        _lap("retrieval")  # 外層收尾

        # 該月過濾
        month_docs, cos_masked = [], []
        for d, sc in zip(docs, cos_scores):
            dt = get_doc_date_dt(d.metadata or {})
            if dt:
                dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt
                if start <= dt_naive < end:
                    month_docs.append(d)
                    cos_masked.append(sc)
        _lap("month_filter")

        if month_docs:
            # 產出月份總結
            answer_md, sources_md = run_month_digest(
                user_input=user_q,
                start=start, end=end,
                per_doc_k=1, max_docs=min(12, max(6, top_n)),
                candidates=month_docs,
                mode=mode or "一般模式",
                custom_prompt=custom_prompt,
            )
            if lead_parts:
                answer_md = ("\n\n".join(lead_parts + [answer_md])).strip()

            # 查詢顯示
            def _fmt_queries(zh, en):
                z = "\n".join(f"- {q}" for q in (zh or [])[:8]) or "（無）"
                e = "\n".join(f"- {q}" for q in (en or [])[:8]) or "（無）"
                return f"**中文查詢**\n{z}\n\n**英文查詢**\n{e}"
            queries_md = _fmt_queries(zh_qs, en_qs)
            _lap("fmt_queries")

            # Debug
            metric_counts = {"has_num": 0, "revenue": 0, "eps": 0, "gm": 0}
            for d in month_docs:
                f = _metric_flags(d.page_content or "")
                for k in metric_counts:
                    metric_counts[k] += int(bool(f.get(k)))

            debug_txt = (
                f"[MODE] month_digest {start.strftime('%Y-%m')} ~ {end.strftime('%Y-%m-%d')}\n"
                f"[INTENTS] {', '.join(intents) if intents else '—'} | [COMPANIES] {', '.join(companies) if companies else '—'}\n"
                "[TYPES] " + _fmt_type_scales(type_scales) + "\n"
                f"[CANDIDATES] {len(month_docs)} | num:{metric_counts['has_num']} rev:{metric_counts['revenue']} eps:{metric_counts['eps']} gm:{metric_counts['gm']}\n"
                + _fmt_times([
                    "set_llm", "parse", "yahoo_blocks",
                    # 內部檢索節點
                    "retrieval:start",
                    "retrieval:mk_base_queries",
                    "retrieval:merged_pins_base",
                    "retrieval:before_alias_expand", "retrieval:after_alias_expand",
                    "retrieval:queries_ready",
                    "retrieval:zh_search_start", "retrieval:zh_search_done",
                    "retrieval:en_search_start", "retrieval:en_search_done",
                    "retrieval:score_norm_start", "retrieval:score_norm_done",
                    "retrieval:pooling_start", "retrieval:pooling_done",
                    "retrieval:metadata_fallback_start", "retrieval:metadata_fallback_done",
                    "retrieval:done_ok", "retrieval:done_empty",
                    # 外層
                    "retrieval", "month_filter", "fmt_queries"
                ])
            )

            followups = make_followups(user_q, answer_md, n_min=3, n_max=5)
            dd_choices = [FOLLOWUP_PLACEHOLDER] + followups
            month_state_msg = f"🗓️ 已整理 {start.strftime('%Y-%m')}：{len(month_docs)} 份文件"
            meta = {
                "type": "month_digest",
                "question": user_q,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "intents": intents,
                "companies": companies,
            }
            return (
                answer_md,
                sources_md or "（本月來源整理）",
                queries_md,
                debug_txt,
                gr.update(choices=dd_choices, value=dd_choices[0]),
                month_state_msg,
                meta,
            )
        else:
            # 月份無候選 → fallback
            parts = []
            stock_md = _prepare_stock_md(
                user_q,
                intents_set,
                companies,
                symbol_cache=symbol_cache_local,
                quote_cache=quote_cache_local,
                chart_cache=chart_cache_local,
            )
            if stock_md:
                parts.append("📈 **股票查詢**\n" + stock_md)
            news_md = _prepare_news_md(
                intents_set,
                companies,
                user_q,
                symbol_cache=symbol_cache_local,
                news_cache=news_cache_local,
            )
            if news_md:
                parts.append("📰 **即時新聞**\n" + news_md)
            parts.append(f"該月份（{start.strftime('%Y-%m')}）未找到**與此主題**相關的文章段落。")
            final_out = "\n\n".join(parts)

            ctx_md = "（該月沒有可引用的段落，若需要仍可點選延伸問題產生該月整體概覽）"
            debug_txt = (
                "[DEBUG] month_pending | " + start.strftime("%Y-%m") +
                f"\n[INTENTS] {', '.join(intents) if intents else '—'}"
                f" | [COMPANIES] {', '.join(companies) if companies else '—'}"
                "\n[TYPES] " + _fmt_type_scales(type_scales) +
                "\n" + _fmt_times([
                    "set_llm", "parse", "yahoo_blocks",
                    "retrieval:start",
                    "retrieval:mk_base_queries",
                    "retrieval:merged_pins_base",
                    "retrieval:before_alias_expand", "retrieval:after_alias_expand",
                    "retrieval:queries_ready",
                    "retrieval:zh_search_start", "retrieval:zh_search_done",
                    "retrieval:en_search_start", "retrieval:en_search_done",
                    "retrieval:score_norm_start", "retrieval:score_norm_done",
                    "retrieval:pooling_start", "retrieval:pooling_done",
                    "retrieval:metadata_fallback_start", "retrieval:metadata_fallback_done",
                    "retrieval:done_ok", "retrieval:done_empty",
                    "retrieval"
                ])
            )
            followups = [SPECIAL_OVERVIEW_LABEL] + make_followups(user_q, final_out, n_min=3, n_max=5)
            dd_choices = [FOLLOWUP_PLACEHOLDER] + (followups or [])
            queries_md = "（月份模式下不展示 multi-queries）"
            month_state_msg = f"🗓️ {start.strftime('%Y-%m')}：未找到主題段落，可點「{SPECIAL_OVERVIEW_LABEL}」產出整月總結"
            meta = {
                "type": "month_pending",
                "question": user_q,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "intents": intents,
                "companies": companies,
            }
            return (
                final_out,
                ctx_md,
                queries_md,
                debug_txt,
                gr.update(choices=dd_choices, value=dd_choices[0]),
                month_state_msg,
                meta,
            )

    # === (B) 一般模式（RAG） ===
    docs, cos_scores, zh_qs, en_qs = similarity_search_vectors(
        user_q,
        k=max(12, top_n),
        type_scales=type_scales,
        prefilter=True,
        intents=intents_set,
        year_targets=requested_years,
        strict_year=True,
        companies=companies,
        lap=_lap,  # ← 傳入計時 callback
    )
    _lap("retrieval")

    if not docs:
        # 沒有候選：給股票/新聞 fallback
        parts = []
        stock_md = _prepare_stock_md(
            user_q,
            intents_set,
            companies,
            symbol_cache=symbol_cache_local,
            quote_cache=quote_cache_local,
            chart_cache=chart_cache_local,
        )
        if stock_md:
            parts.append("📈 **股票查詢**\n" + stock_md)
        news_md = _prepare_news_md(
            intents_set,
            companies,
            user_q,
            symbol_cache=symbol_cache_local,
            news_cache=news_cache_local,
        )
        if news_md:
            parts.append("📰 **即時新聞**\n" + news_md)
        parts.append("目前找不到可直接回答的內容。可嘗試：放寬條件、改寫關鍵字、或提高類別倍率。")
        answer_md = "\n\n".join(parts)

        zh_view = "\n".join(f"- {q}" for q in (zh_qs or [])[:8]) or "（無）"
        en_view = "\n".join(f"- {q}" for q in (en_qs or [])[:8]) or "（無）"
        queries_md = f"**中文查詢**\n{zh_view}\n\n**英文查詢**\n{en_view}"
        _lap("fmt_queries")

        debug_txt = (
            "[DEBUG] no_docs\n"
            f"[INTENTS] {', '.join(intents) if intents else '—'} | [COMPANIES] {', '.join(companies) if companies else '—'}\n"
            "[TYPES] " + _fmt_type_scales(type_scales) + "\n"
            + _fmt_times([
                "set_llm", "parse", "yahoo_blocks",
                "retrieval:start",
                "retrieval:mk_base_queries",
                "retrieval:merged_pins_base",
                "retrieval:before_alias_expand", "retrieval:after_alias_expand",
                "retrieval:queries_ready",
                "retrieval:zh_search_start", "retrieval:zh_search_done",
                "retrieval:en_search_start", "retrieval:en_search_done",
                "retrieval:score_norm_start", "retrieval:score_norm_done",
                "retrieval:pooling_start", "retrieval:pooling_done",
                "retrieval:metadata_fallback_start", "retrieval:metadata_fallback_done",
                "retrieval:done_ok", "retrieval:done_empty",
                "retrieval", "fmt_queries"
            ])
        )
        followups = make_followups(user_q, answer_md, n_min=3, n_max=5)
        dd_choices = [FOLLOWUP_PLACEHOLDER] + followups
        meta = {
            "type": "no_docs",
            "question": user_q,
            "intents": intents,
            "companies": companies,
        }
        return (
            answer_md,
            "（無 context 可顯示）",
            queries_md,
            debug_txt,
            gr.update(choices=dd_choices, value=dd_choices[0]),
            "—",
            meta,
        )

    picked_docs, final_scores, dbg = bge_rank_and_pick(
        user_q, docs, cos_scores,
        top_k=top_k,
        candidate_cap=max(36, top_n+12),   # 可按索引規模微調
        clip_chars=900,                     # 依原設定
        batch_size=32,
        type_scales=type_scales,
        recency_boost=wants_latest,
        recency_half_life=45,
    )
    _lap("rerank")

    # 產生答案
    answer_md, ctx_full, src_map = synthesize_answer(
        question=user_q,
        picked_docs=picked_docs,
        mode=mode or "一般模式",
        intents=intents,
        custom_prompt=custom_prompt,
    )
    _lap("answer")

    if lead_parts:
        answer_md = ("\n\n".join(lead_parts + [answer_md])).strip()
    sources_md = render_sources_md(src_map)

    # 查詢顯示
    zh_view = "\n".join(f"- {q}" for q in (zh_qs or [])[:8]) or "（無）"
    en_view = "\n".join(f"- {q}" for q in (en_qs or [])[:8]) or "（無）"
    queries_md = f"**中文查詢**\n{zh_view}\n\n**英文查詢**\n{en_view}"
    _lap("fmt_queries")

    debug_txt = (
        "[MODE] normal_rag (BGE reranker)\n"
        f"[INTENTS] {', '.join(intents) if intents else '—'} | [COMPANIES] {', '.join(companies) if companies else '—'} | wants_latest={wants_latest}\n"
        "[TYPES] " + _fmt_type_scales(type_scales) + "\n"
        f"{dbg}\n" +
        _fmt_times([
            "set_llm", "parse", "yahoo_blocks",
            # 內部檢索節點
            "retrieval:start",
            "retrieval:mk_base_queries",
            "retrieval:merged_pins_base",
            "retrieval:before_alias_expand", "retrieval:after_alias_expand",
            "retrieval:queries_ready",
            "retrieval:zh_search_start", "retrieval:zh_search_done",
            "retrieval:en_search_start", "retrieval:en_search_done",
            "retrieval:score_norm_start", "retrieval:score_norm_done",
            "retrieval:pooling_start", "retrieval:pooling_done",
            "retrieval:metadata_fallback_start", "retrieval:metadata_fallback_done",
            "retrieval:done_ok", "retrieval:done_empty",
            # 外層
            "retrieval", "rerank", "answer", "fmt_queries"
        ])
    )

    followups = make_followups(user_q, answer_md, n_min=3, n_max=5)
    dd_choices = [FOLLOWUP_PLACEHOLDER] + followups

    meta = {
        "type": "normal_rag",
        "question": user_q,
        "intents": intents,
        "companies": companies,
        "picked": len(picked_docs),
    }
    return (
        answer_md,
        sources_md or ctx_full or "（來源/Context）",
        queries_md,
        debug_txt,
        gr.update(choices=dd_choices, value=dd_choices[0]),
        "—",
        meta,
    )

def on_followup_change(v, n, k, sb, sp, sr, st, sf, mode, model_name, meta, custom_prompt=""):
    DEFAULT_OPT = FOLLOWUP_PLACEHOLDER

    # 可能是字串就嘗試 parse
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = None

    # 如果沒選或選了預設，什麼都不做
    if not v or v == DEFAULT_OPT:
        return (
            gr.update(),   # question（不變）
            gr.update(),   # answer
            gr.update(),   # context
            gr.update(),   # queries_md
            gr.update(),   # debug
            gr.update(),   # followup_dd（不變）
            gr.update(),   # month_state
            meta           # month_meta_state 保持原值
        )

    # 特殊：點了「整月概覽」→ 直接跑月份總結
    if v == SPECIAL_OVERVIEW_LABEL and isinstance(meta, dict) and meta.get("type") == "month_pending":
        try:
            start = dtparse(meta["start"])
            end   = dtparse(meta["end"])
        except Exception:
            # meta 壞掉就只把選項填回輸入框
            return (
                gr.update(value=v), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), meta
            )

        # 生成整月摘要（使用選定模式）
        try:
            set_gen_llm(model_name)  # ← 可選：讓月份摘要也用當前的生成模型
        except Exception:
            pass
        month_question = f"{start.strftime('%Y-%m')} 月份總結"
        ans_text, ctx_full = run_month_digest(
            month_question, start, end, per_doc_k=1, max_docs=12, candidates=None, mode=mode, custom_prompt=custom_prompt
        )

        # 產生新的延伸問題清單
        new_followups = make_followups(month_question, ans_text, n_min=3, n_max=5)
        dd_choices = [FOLLOWUP_PLACEHOLDER] + new_followups

        return (
            gr.update(value=month_question),           # question：把月份總結題目填回去
            ans_text,                                  # answer
            ctx_full,                                  # context（月份摘要的 context）
            "（月份模式不展示 Multi-queries）",          # queries_md
            "[DEBUG] month_digest generated",          # debug
            gr.update(choices=dd_choices, value=dd_choices[0]),  # followup_dd
            f"🗓️ 已生成整月：{start.strftime('%Y-%m')}",   # month_state
            None                                       # month_meta_state 用不到了，清掉
        )

    # 一般情況：把選到的延伸問題回填到輸入框，留給使用者按 Submit
    return (
        gr.update(value=v),  # question
        gr.update(),         # answer
        gr.update(),         # context
        gr.update(),         # queries_md
        gr.update(),         # debug
        gr.update(),         # followup_dd
        gr.update(),         # month_state
        meta                 # month_meta_state 保留
    )

# =========================
# Gradio 介面
# =========================
with gr.Blocks(title="RAG (zh/en) + LLM rerank + Yahoo 新聞\當日股價 + 部落格月份重點整理") as demo:
    gr.Markdown("## 🔎 RAG ＋🔁 BGE Rerank＋📈/📰 Yahoo＋🗓️ 月份自動偵測\n- **類別偏好**（Blog / PDF / Research / Transcripts_summary）")

    # 自訂回答指令（在「客製化模式」下生效；其他模式可留空）
    custom_prompt_tb = gr.Textbox(
        label="自訂回答指令（客製化模式使用）",
        placeholder="範例：請用繁體中文、專業且清楚的語氣回答；先列出 3–5 點重點（每點含具體數字/日期），之後用一段話統整影響，最後給出 2 個可行建議。",
        lines=6
    )

    with gr.Row():
        question = gr.Textbox(label="請輸入問題 / Ask a question",
                              placeholder="例：2024 年 11 月部落格重點？ 或 OPEC 今年減產影響？ 或 台積電股價？",
                              lines=2, autofocus=True)

    with gr.Row():
        top_n = gr.Slider(10, 20, value=15, step=1, label="檢索 Top N")
        top_k = gr.Slider(1, 10, value=5, step=1, label="最終 Top K")

    mode_sel = gr.Radio(
        choices=["初學者模式", "專家模式", "客製化模式"],
        value="初學者模式",
        label="回答模式（單選）"
    )

    # 回答模型選擇（由使用者決定本次回合使用的模型）
    gen_model_dd = gr.Dropdown(
        label="生成模型（只影響最終回答）",
        choices=GEN_MODEL_WHITELIST,
        value=GEN_MODEL_DEFAULT
    )

    with gr.Row():
        blog_scale       = gr.Slider(0, 2, value=1.0, step=0.1, label="Blog 類別偏好 (0–2)")
        pdf_scale        = gr.Slider(0, 2, value=1.0, step=0.1, label="PDF 類別偏好 (0–2)")
        research_scale   = gr.Slider(0, 2, value=1.0, step=0.1, label="Research 類別偏好 (0–2)")
        summary_scale    = gr.Slider(0, 2, value=1.0, step=0.1, label="Transcript Summary 類別偏好 (0–2)")
        filing_scale     = gr.Slider(0, 2, value=1.0, step=0.1, label="財報 類別偏好 (0–2)") 
    
    submit_btn = gr.Button("Submit", variant="primary")

    followup_dd = gr.Dropdown(
        label="💡 延伸問題（點選自動填回）",
        choices=[FOLLOWUP_PLACEHOLDER],
        value=FOLLOWUP_PLACEHOLDER,
        interactive=True,
    )

    answer = gr.Markdown(label="🧠 回答 / Answer")
    with gr.Row():
        context = gr.Markdown(label="📚 引用 Context（含類別與倍率）")
        queries_md = gr.Markdown(label="🔁 查詢紀錄（Multi-queries / 月份模式則隱藏）")
    debug = gr.Textbox(label="🛠 Debug（分數表 / 月份狀態）", lines=12)
    month_state = gr.Markdown("—")

    month_meta_state = gr.State(value=None)

    def _scales_all_zero(sb, sp, sr, st, sf) -> bool:
        try:
            return all(float(x) <= 0 for x in (sb, sp, sr, st, sf))
        except Exception:
            return False

    def wrapped(q, n, k, sb, sp, sr, st, sf, mode, model_name, custom_prompt):
        if _scales_all_zero(sb, sp, sr, st, sf):
            gr.Warning("請至少開啟一種來源（Blog / PDF / Research / Transcript / Filing）。全部為 0 無法檢索。")
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None)

        if not (q or "").strip():
            gr.Info("請先輸入問題")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), None

        final_out, ctx_md, qmd, dbg, dd, month_msg, meta = handle_question(
            q, n, k, sb, sp, sr, st, sf, mode, model_name, custom_prompt
        )
        return final_out, ctx_md, qmd, dbg, dd, month_msg, meta


    submit_btn.click(
        fn=wrapped,
        inputs=[question, top_n, top_k, blog_scale, pdf_scale, research_scale, summary_scale, filing_scale, mode_sel, gen_model_dd, custom_prompt_tb],
        outputs=[answer, context, queries_md, debug, followup_dd, month_state, month_meta_state],
    )
    question.submit(
        fn=wrapped,
        inputs=[question, top_n, top_k, blog_scale, pdf_scale, research_scale, summary_scale, filing_scale, mode_sel, gen_model_dd, custom_prompt_tb],
        outputs=[answer, context, queries_md, debug, followup_dd, month_state, month_meta_state],
    )

    # 事件綁定需在 Blocks 內容中；用 wrapper 延後引用最終定義的處理函式
    def _on_followup_wrapper(v, n, k, sb, sp, sr, st, sf, mode, model_name, meta, custom_prompt):
        return on_followup_change(v, n, k, sb, sp, sr, st, sf, mode, model_name, meta, custom_prompt)

    followup_dd.change(
        fn=_on_followup_wrapper,
        inputs=[followup_dd, top_n, top_k, blog_scale, pdf_scale, research_scale, summary_scale, filing_scale, mode_sel, gen_model_dd, month_meta_state, custom_prompt_tb],
        outputs=[question, answer, context, queries_md, debug, followup_dd, month_state, month_meta_state],
    )

if __name__ == "__main__":
    if os.getenv("PURGE_SYMBOL_CACHE", "0") == "1":
        try:
            _SYM_KV["map"].clear()
            _symcache_flush()
            print("[symbol-cache] purged")
        except Exception as e:
            print(f"[symbol-cache] purge failed: {e}")
    _load_tw_lists(force=True)
    init_internal_llms()
    set_gen_llm(GEN_MODEL_DEFAULT)
    demo.queue()
    demo.launch()
    '''
    用來查看每個分類有多少個 chunk
    def _count_by_bucket(vs):
        d = getattr(vs, "docstore", None)
        if not d or not hasattr(d, "_dict"): return {}
        from collections import Counter
        c = Counter()
        for x in d._dict.values():
            c[get_bucket(x.metadata or {})] += 1
        return c
    print("ZH buckets:", _count_by_bucket(vectorstore_zh))
    print("EN buckets:", _count_by_bucket(vectorstore_en))
    '''
