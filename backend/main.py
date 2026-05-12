"""
Resume NER v7 — Production API (FastAPI)
Base model: dslim/bert-base-NER + LoRA (r=32, α=64)
Test F1: 0.8368 | Val F1: 0.8324 | Skills F1: 0.7273

BONUS: Confidence-based routing with secure review queue + full audit trail
- High confidence  (≥ HIGH_CONF=0.75) → auto-accepted  → S3: approved/
- Medium confidence (≥ INF_CONF=0.60) → accepted with warning flag (no S3 storage)
- Low confidence   (< INF_CONF=0.60)  → routed to human review queue → S3: pending/

Review actions:
  approve  → pending/ moved to approved/
  correct  → pending/ moved to corrected/ with new corrected txt
"""

import os
import re
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List

from core.config import settings
from core.logging_config import setup_logging
from middleware.security import SecurityMiddleware
from services.s3_service import S3Service

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
from collections import Counter

# ─────────────────────────── Logging ────────────────────────────
logger = setup_logging()

# ─────────────────────────── Config ──────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "backend/models")
BASE_MODEL  = "dslim/bert-base-NER"
MAX_LEN     = 256
INF_CONF    = 0.60   # below this → human review queue
HIGH_CONF   = 0.75   # threshold for auto-accept vs flagged
TTA_RUNS    = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────── Label Schema ────────────────────────
ENTITIES = [
    "College Name", "Companies worked at", "Degree", "Designation",
    "Email Address", "Graduation Year", "Location", "Name",
    "Skills", "Years of Experience",
]
ALL_LABELS = ["O"] + [f"{p}-{e}" for e in ENTITIES for p in ("B", "I")]
label2id   = {l: i for i, l in enumerate(ALL_LABELS)}
id2label   = {i: l for i, l in enumerate(ALL_LABELS)}
N_LABELS   = len(ALL_LABELS)

# ─────────────────────────── Post-processing constants ───────────
NOISE_RE    = re.compile(r"^[\s\-|,.:;()\[\]{}\'\"\u2014\u2013]+|[\s\-|,.:;()\[\]{}\'\"\u2014\u2013]+$")
_NOISE_TOK  = {":", ";", "-", ".", "|", "/", "(", ")", "+", "*", "#", "@", "&", "\u2014", "\u2013"}
EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
YOE_NOISE   = re.compile(r"[%MmKkBb+]")
PHONE_RE    = re.compile(r"(?:\+?\d[\d\s\-]{7,}\d)")
GPA_RE      = re.compile(r"\b(gpa|cgpa|grade)\b", re.IGNORECASE)

SECTION_HEADERS = {
    "education", "experience", "experiences", "skills", "summary",
    "objective", "projects", "certifications", "languages", "awards",
    "publications", "references", "contact", "profile", "about",
    "technical skills", "soft skills", "work experience",
    "professional experience", "academic background",
}

DESIGNATION_LIST = [
    "Software Engineer", "Data Scientist", "Backend Developer",
    "Full Stack Developer", "ML Engineer", "DevOps Engineer",
    "Frontend Developer", "Data Analyst", "Cloud Architect",
    "NLP Engineer", "Computer Vision Engineer", "AI Researcher",
    "Site Reliability Engineer", "Data Engineer", "Product Manager",
    "Business Analyst", "Systems Analyst", "Database Administrator",
    "Security Engineer", "Machine Learning Engineer",
    "Data Analyst and Machine Learning Engineer",
    "Software Developer", "Web Developer", "Mobile Developer",
    "QA Engineer", "Test Engineer", "Platform Engineer",
]

ALL_SKILLS_FLAT = [
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#",
    "Go", "Rust", "Swift", "Kotlin", "R", "MATLAB", "Scala", "Ruby",
    "PHP", "Perl", "Shell", "Bash", "PowerShell", "Dart", "Lua",
    "React", "Vue.js", "Angular", "Next.js", "Svelte", "HTML", "CSS",
    "Sass", "Tailwind CSS", "Bootstrap", "Material UI", "Redux",
    "GraphQL", "REST API", "WebSockets", "jQuery", "Webpack", "Vite",
    "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot",
    "Laravel", "Ruby on Rails", "ASP.NET", "NestJS", "Microservices",
    "gRPC", "OAuth", "JWT", "PostgreSQL", "MySQL", "SQLite", "Oracle",
    "SQL Server", "MongoDB", "Redis", "Cassandra", "Elasticsearch",
    "DynamoDB", "Firebase", "Neo4j", "Snowflake", "BigQuery", "Redshift",
    "MariaDB", "SQL", "AWS", "Azure", "Google Cloud", "GCP", "Docker",
    "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions",
    "Helm", "Prometheus", "Grafana", "Nginx", "Linux", "Serverless",
    "Lambda", "EC2", "S3", "RDS", "TensorFlow", "PyTorch", "Scikit-learn",
    "Keras", "Hugging Face", "XGBoost", "LightGBM", "CatBoost",
    "Machine Learning", "Deep Learning", "Neural Networks",
    "Natural Language Processing", "NLP", "Computer Vision",
    "Reinforcement Learning", "Generative AI", "Feature Engineering",
    "Model Evaluation", "Pandas", "NumPy", "SciPy", "Matplotlib",
    "Seaborn", "Plotly", "Tableau", "Power BI", "Looker", "Jupyter",
    "Data Analysis", "Data Visualization", "Statistical Analysis",
    "A/B Testing", "ETL", "Data Pipeline", "Apache Spark", "Data Cleaning",
    "Data Transformation", "Data Modeling", "EDA", "DAX", "Power Query",
    "Android", "iOS", "React Native", "Flutter", "SwiftUI",
    "Git", "GitHub", "GitLab", "Agile", "Scrum", "JIRA", "TDD",
    "Unit Testing", "CI/CD", "System Design",
    "Excel", "Word", "PowerPoint", "Figma", "Postman", "Swagger",
]

# ─────────────────────────── Global model holders ────────────────
_model     = None
_tokenizer = None

# ─────────────── In-memory Review Queue & Audit Log ──────────────
# In production replace with a DB (PostgreSQL / DynamoDB / etc.)
_review_queue: list[dict] = []
_audit_log:    list[dict] = []


def _audit(event: str, request_id: str, **kwargs):
    entry = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "event":      event,
        **kwargs,
    }
    _audit_log.append(entry)
    logger.info(f"AUDIT | {event} | req={request_id} | {kwargs}")


def load_model():
    global _model, _tokenizer
    logger.info(f"Loading tokenizer from {BASE_MODEL} ...")
    _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    logger.info(f"Loading base model {BASE_MODEL} ...")
    base = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=N_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    logger.info("Applying LoRA adapter ...")
    _model = PeftModel.from_pretrained(base, MODEL_DIR)
    _model.to(DEVICE)
    _model.eval()
    logger.info(f"Model ready on {DEVICE} ✅")


# ─────────────────────────── Lifespan ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down.")


# ─────────────────────────── App ─────────────────────────────────
app = FastAPI(
    title="Resume NER API",
    description="Extract structured entities from resumes using BERT + LoRA (v7) — with confidence routing & audit trail",
    version="7.2.0",
    lifespan=lifespan,
)

app.add_middleware(SecurityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────── Schemas ─────────────────────────────
class TextRequest(BaseModel):
    text: str
    conf: Optional[float] = INF_CONF
    tta:  Optional[int]   = TTA_RUNS

class ParsedResume(BaseModel):
    name:              list[str] = []
    designation:       list[str] = []
    email_address:     list[str] = []
    location:          list[str] = []
    companies:         list[str] = []
    degree:            list[str] = []
    college_name:      list[str] = []
    graduation_year:   list[str] = []
    years_experience:  list[str] = []
    skills:            list[str] = []
    links:             list[str] = []
    raw:               dict      = {}

class InferenceResponse(BaseModel):
    entities:        ParsedResume
    latency_ms:      float
    model_ver:       str   = "v7-LoRA"
    request_id:      str   = ""
    confidence:      float = 0.0
    routing:         str   = "auto"   # "auto" | "flagged" | "review"
    review_queue_id: Optional[str] = None
    s3_keys:         Optional[dict] = None   # pdf/txt keys stored in S3

class ReviewQueueItem(BaseModel):
    queue_id:    str
    request_id:  str
    queued_at:   str
    confidence:  float
    entities:    ParsedResume
    status:      str          # "pending" | "approved" | "corrected"
    reviewer:    Optional[str] = None
    reviewed_at: Optional[str] = None
    correction:  Optional[dict] = None
    # S3 tracking
    cv_id:       Optional[str] = None
    filename_stem: Optional[str] = None
    s3_keys:     Optional[dict] = None

class ReviewDecision(BaseModel):
    action:     str            # "approve" | "correct"
    reviewer:   str
    correction: Optional[dict] = None

class AuditEntry(BaseModel):
    timestamp:   str
    request_id:  str
    event:       str


# ─────────────────────────── Inference helpers ───────────────────
def tta_predict(text: str, conf: float = INF_CONF, tta: int = TTA_RUNS) -> list[dict]:
    words = re.findall(r"\S+", text)
    if not words:
        return []

    acc = np.zeros((len(words), N_LABELS), dtype=np.float64)

    for run in range(tta + 1):
        if run > 0:
            _model.train()
        else:
            _model.eval()

        for i in range(0, len(words), 128):
            chunk = words[i : i + 128]
            enc = _tokenizer(
                chunk,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding=False,
            )
            with torch.no_grad():
                logits = _model(**{k: v.to(DEVICE) for k, v in enc.items()}).logits[0]
            probs = torch.softmax(logits, -1).cpu().numpy()
            seen: set = set()
            for ti, wid in enumerate(enc.word_ids(0)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                gw = i + wid
                if gw < len(words):
                    acc[gw] += probs[ti]

    acc /= tta + 1
    _model.eval()

    return [
        {
            "word":  words[i],
            "label": id2label[int(np.argmax(acc[i]))],
            "score": float(np.max(acc[i])),
        }
        for i in range(len(words))
    ]


def compute_mean_confidence(preds: list[dict]) -> float:
    entity_scores = [p["score"] for p in preds if not p["label"].startswith("O")]
    if not entity_scores:
        return 0.0
    return float(np.mean(entity_scores))


def _is_email(s):  return bool(EMAIL_RE.fullmatch(s.strip()))
def _is_phone(s):  return bool(PHONE_RE.fullmatch(s.strip()))
def _is_url(s):    return any(s.startswith(p) for p in ["http", "www.", "linkedin", "github"])
def _is_section_header(s): return s.lower().strip().rstrip(":") in SECTION_HEADERS

def _is_valid_yoe(s):
    s = s.strip()
    if YOE_NOISE.search(s): return False
    m = re.match(r"^\(?(\d{1,2})", s)
    if not m: return False
    return 1 <= int(m.group(1)) <= 50

def _deduplicate_locations(locs):
    sorted_locs = sorted(set(locs), key=len, reverse=True)
    kept = []
    for loc in sorted_locs:
        if not any(loc.lower() in k.lower() for k in kept):
            kept.append(loc)
    return kept

def _split_comma_skills(skill_str):
    parts = [p.strip().strip("-").strip() for p in re.split(r"[,\u060c]", skill_str)]
    return [p for p in parts if len(p) >= 2]

def _pick_best_designation(candidates, skills):
    filtered = [c for c in candidates if not _is_section_header(c)]
    if not filtered: return []
    if len(filtered) == 1: return [filtered[0]]
    skill_words = {s.lower() for s in skills}

    def score(d):
        known = any(d.lower() == k.lower() for k in DESIGNATION_LIST)
        overlap = len(set(d.lower().split()) & skill_words)
        return int(known) * 10 + overlap * 2 - len(d.split())

    return [max(filtered, key=score)]

def _pick_best_degree(candidates):
    filtered = [
        c for c in candidates
        if not _is_section_header(c)
        and not _is_email(c)
        and not _is_phone(c)
        and not GPA_RE.search(c)
        and not re.search(r"\d{4}.*\d{4}", c)
        and len(c.split()) >= 2
    ]
    return [max(filtered, key=len)] if filtered else []

def _skills_fallback(text, found_skills):
    text_lower  = text.lower()
    found_lower = {s.lower() for s in found_skills}
    extra = []
    for sk in sorted(ALL_SKILLS_FLAT, key=len, reverse=True):
        sk_lower = sk.lower()
        if sk_lower in found_lower: continue
        if re.search(r"\b" + re.escape(sk_lower) + r"\b", text_lower):
            extra.append(sk)
            found_lower.add(sk_lower)
    return extra


def group_entities(preds: list[dict], text: str, conf: float = INF_CONF) -> dict:
    ents: dict = {}
    cur_type, cur_words = None, []

    def flush():
        if cur_type and cur_words:
            v = NOISE_RE.sub("", " ".join(cur_words)).strip()
            if len(v) >= 2:
                ents.setdefault(cur_type, []).append(v)

    for wp in preds:
        lbl, sc, w = wp["label"], wp["score"], wp["word"]
        if w in _NOISE_TOK:
            flush(); cur_type, cur_words = None, []; continue
        if lbl.startswith("B-") and sc >= conf:
            flush(); cur_type, cur_words = lbl[2:], [w]
        elif lbl.startswith("I-") and cur_type == lbl[2:] and sc >= conf * 0.75:
            cur_words.append(w)
        else:
            flush(); cur_type, cur_words = None, []
    flush()

    emails = [e for e in EMAIL_RE.findall(text) if not _is_phone(e)]
    if emails:
        ents["Email Address"] = list(dict.fromkeys(emails))

    comp_key = next((k for k in ents if "compan" in k.lower() or "employer" in k.lower()), None)
    if comp_key:
        ents[comp_key] = [
            c for c in ents[comp_key]
            if not _is_email(c) and "@" not in c and not _is_url(c)
            and not re.fullmatch(r"[a-zA-Z0-9._+\-]+", c)
        ]
        if not ents[comp_key]:
            del ents[comp_key]

    if "Name" in ents:
        ents["Name"] = [n for n in ents["Name"] if not _is_email(n) and not _is_phone(n) and not _is_url(n)]
        if ents["Name"]:
            names_s = sorted(set(ents["Name"]), key=len)
            kept = []
            for nm in names_s:
                if not any(nm.lower() in k.lower() for k in kept):
                    kept.append(nm)
            ents["Name"] = [kept[0]] if kept else []
        if not ents["Name"]:
            del ents["Name"]

    if "Location" in ents:
        ents["Location"] = _deduplicate_locations(ents["Location"])

    yoe_key = next((k for k in ents if "year" in k.lower() or "experience" in k.lower()), None)
    if yoe_key:
        ents[yoe_key] = [y for y in ents[yoe_key] if _is_valid_yoe(y)]
        if not ents[yoe_key]:
            del ents[yoe_key]

    raw_skills = ents.get("Skills", [])
    expanded = []
    for sk in raw_skills:
        if "," in sk:
            expanded.extend(_split_comma_skills(sk))
        else:
            expanded.append(sk.strip("-").strip())
    expanded = [
        s for s in expanded
        if len(s) >= 2 and not _is_section_header(s)
        and s.lower() not in {"skills", "technical", "soft", "tools"}
    ]
    extra = _skills_fallback(text, expanded)
    all_skills = list(dict.fromkeys(expanded + extra))
    if all_skills:
        ents["Skills"] = all_skills
    elif "Skills" in ents:
        del ents["Skills"]

    if "Designation" in ents:
        ents["Designation"] = _pick_best_designation(ents["Designation"], ents.get("Skills", []))
        if not ents["Designation"]:
            del ents["Designation"]

    if "Degree" in ents:
        ents["Degree"] = _pick_best_degree(ents["Degree"])
        if not ents["Degree"]:
            del ents["Degree"]

    return ents


def ents_to_schema(ents: dict) -> ParsedResume:
    return ParsedResume(
        name             = ents.get("Name", []),
        designation      = ents.get("Designation", []),
        email_address    = ents.get("Email Address", []),
        location         = ents.get("Location", []),
        companies        = ents.get("Companies worked at", []),
        degree           = ents.get("Degree", []),
        college_name     = ents.get("College Name", []),
        graduation_year  = ents.get("Graduation Year", []),
        years_experience = ents.get("Years of Experience", []),
        skills           = ents.get("Skills", []),
        links            = ents.get("Links", []),
        raw              = ents,
    )


def _entities_to_txt(parsed: ParsedResume) -> str:
    """Convert parsed entities to a human-readable text file."""
    lines = []
    fields = [
        ("Name",            parsed.name),
        ("Designation",     parsed.designation),
        ("Email",           parsed.email_address),
        ("Location",        parsed.location),
        ("Companies",       parsed.companies),
        ("Degree",          parsed.degree),
        ("College",         parsed.college_name),
        ("Graduation Year", parsed.graduation_year),
        ("Experience",      parsed.years_experience),
        ("Skills",          parsed.skills),
        ("Links",           parsed.links),
    ]
    for label, values in fields:
        if values:
            lines.append(f"{label}: {', '.join(values)}")
    return "\n".join(lines)


# ─────────────── Confidence Routing Logic ─────────────────────────
def route_response(
    request_id: str,
    parsed: ParsedResume,
    confidence: float,
    source: str = "text",
    pdf_bytes: bytes = b"",
    filename: str = "",
    extracted_text: str = "",
) -> InferenceResponse:
    s3 = S3Service()
    cv_id = str(uuid.uuid4())[:8]
    s3_keys = None

    if confidence >= HIGH_CONF:
        routing = "auto"
        review_queue_id = None
        _audit("AUTO_ACCEPTED", request_id, confidence=round(confidence, 4), source=source)

        if pdf_bytes and filename:
            txt_content = _entities_to_txt(parsed)
            s3_keys = s3.save_approved(pdf_bytes, filename, txt_content, cv_id)

    elif confidence >= INF_CONF:
        routing = "flagged"
        review_queue_id = None
        _audit("FLAGGED_LOW_CONF", request_id, confidence=round(confidence, 4), source=source)

        if pdf_bytes and filename:
            txt_content = _entities_to_txt(parsed)
            s3_keys = s3.save_flagged(pdf_bytes, filename, txt_content, cv_id)

    else:
        routing = "review"
        review_queue_id = str(uuid.uuid4())
        stem = filename.rsplit(".", 1)[0] if "." in filename else filename or cv_id

        pending_s3 = {}
        if pdf_bytes and filename:
            txt_content = _entities_to_txt(parsed)
            pending_s3 = s3.save_pending(pdf_bytes, filename, txt_content, cv_id)

        _review_queue.append({
            "queue_id":     review_queue_id,
            "request_id":   request_id,
            "queued_at":    datetime.now(timezone.utc).isoformat(),
            "confidence":   round(confidence, 4),
            "entities":     parsed.model_dump(),
            "status":       "pending",
            "reviewer":     None,
            "reviewed_at":  None,
            "correction":   None,
            "cv_id":        cv_id,
            "filename_stem": stem,
            "s3_keys":      pending_s3,
        })
        s3_keys = pending_s3

        _audit(
            "SENT_TO_REVIEW",
            request_id,
            confidence=round(confidence, 4),
            queue_id=review_queue_id,
            source=source,
        )
        logger.warning(
            f"Low-confidence result (conf={confidence:.3f}) → review queue [{review_queue_id}]"
        )

    return InferenceResponse(
        entities        = parsed,
        latency_ms      = 0.0,
        request_id      = request_id,
        confidence      = round(confidence, 4),
        routing         = routing,
        review_queue_id = review_queue_id,
        s3_keys         = s3_keys,
    )


# ─────────────────────────── Routes ──────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model":         "resume-ner-v7-lora",
        "device":        str(DEVICE),
        "labels":        N_LABELS,
        "queue_pending": sum(1 for q in _review_queue if q["status"] == "pending"),
    }


@app.get("/labels")
def get_labels():
    return {"entities": ENTITIES, "all_labels": ALL_LABELS}


@app.post("/parse/text", response_model=InferenceResponse)
def parse_text(req: TextRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")
    if len(req.text) > 50_000:
        raise HTTPException(status_code=413, detail="text too long (max 50k chars)")

    request_id = str(uuid.uuid4())
    _audit("REQUEST_RECEIVED", request_id, source="text", text_len=len(req.text))

    t0    = time.perf_counter()
    preds = tta_predict(req.text, conf=req.conf, tta=req.tta)
    ents  = group_entities(preds, req.text, conf=req.conf)
    ms    = (time.perf_counter() - t0) * 1000

    confidence = compute_mean_confidence(preds)
    parsed     = ents_to_schema(ents)
    response   = route_response(request_id, parsed, confidence, source="text")
    response.latency_ms = round(ms, 1)

    _audit("REQUEST_COMPLETED", request_id, latency_ms=round(ms, 1), routing=response.routing)
    return response


@app.post("/parse/pdf", response_model=InferenceResponse)
async def parse_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    request_id = str(uuid.uuid4())
    _audit("REQUEST_RECEIVED", request_id, source="pdf", filename=file.filename)

    try:
        from pypdf import PdfReader
        import io
        raw    = await file.read()
        reader = PdfReader(io.BytesIO(raw))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        _audit("PDF_PARSE_ERROR", request_id, error=str(e))
        raise HTTPException(status_code=422, detail=f"Could not read PDF: {e}")

    if not text:
        _audit("PDF_EMPTY_TEXT", request_id)
        raise HTTPException(status_code=422, detail="No extractable text in PDF (scanned?)")

    t0    = time.perf_counter()
    preds = tta_predict(text)
    ents  = group_entities(preds, text)
    ms    = (time.perf_counter() - t0) * 1000

    confidence = compute_mean_confidence(preds)
    parsed     = ents_to_schema(ents)
    response   = route_response(
        request_id, parsed, confidence,
        source="pdf",
        pdf_bytes=raw,
        filename=file.filename,
        extracted_text=text,
    )
    response.latency_ms = round(ms, 1)

    _audit("REQUEST_COMPLETED", request_id, latency_ms=round(ms, 1), routing=response.routing)
    return response


# ─────────────── Review Queue Routes ─────────────────────────────

@app.get("/review/queue", response_model=List[ReviewQueueItem])
def get_review_queue(status: Optional[str] = None):
    """
    List review queue items.
    Default shows only pending. Pass ?status=all for everything,
    or ?status=approved / ?status=corrected for those.
    """
    if status == "all":
        items = _review_queue
    elif status in ("approved", "corrected"):
        items = [q for q in _review_queue if q["status"] == status]
    else:
        # default: pending only
        items = [q for q in _review_queue if q["status"] == "pending"]
    return [ReviewQueueItem(**q) for q in items]


@app.get("/review/queue/{queue_id}", response_model=ReviewQueueItem)
def get_review_item(queue_id: str):
    item = next((q for q in _review_queue if q["queue_id"] == queue_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    return ReviewQueueItem(**item)


@app.post("/review/approve/{queue_id}", response_model=ReviewQueueItem)
def approve_review_item(queue_id: str, reviewer: str):
    """
    Approve a pending CV:
    - Moves files from pending/ → approved/ in S3
    - Marks status as 'approved'
    """
    item = next((q for q in _review_queue if q["queue_id"] == queue_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    if item["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Item already {item['status']}")

    # S3: move pending → approved
    s3 = S3Service()
    cv_id = item.get("cv_id", "")
    stem  = item.get("filename_stem", "")
    s3_result = {}
    if cv_id and stem:
        s3_result = s3.approve_pending(cv_id, stem)

    now = datetime.now(timezone.utc).isoformat()
    item["status"]      = "approved"
    item["reviewer"]    = reviewer
    item["reviewed_at"] = now
    item["s3_keys"]     = s3_result

    _audit("REVIEW_DECISION", item["request_id"], queue_id=queue_id, action="approve", reviewer=reviewer)
    return ReviewQueueItem(**item)


@app.post("/review/correct/{queue_id}", response_model=ReviewQueueItem)
def correct_review_item(queue_id: str, reviewer: str, correction: dict):
    """
    Correct a pending CV:
    - Writes corrected txt and moves PDF from pending/ → corrected/ in S3
    - Marks status as 'corrected'
    """
    item = next((q for q in _review_queue if q["queue_id"] == queue_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    if item["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"Item already {item['status']}")

    # Build corrected text from the correction dict
    corrected_lines = []
    for key, values in correction.items():
        if values:
            label = key.replace("_", " ").title()
            corrected_lines.append(f"{label}: {', '.join(values) if isinstance(values, list) else values}")
    corrected_text = "\n".join(corrected_lines)

    s3 = S3Service()
    cv_id = item.get("cv_id", "")
    stem  = item.get("filename_stem", "")
    s3_result = {}
    if cv_id and stem:
        s3_result = s3.save_corrected(cv_id, stem, corrected_text)

    now = datetime.now(timezone.utc).isoformat()
    item["status"]      = "corrected"
    item["reviewer"]    = reviewer
    item["reviewed_at"] = now
    item["correction"]  = correction
    item["s3_keys"]     = s3_result

    _audit("REVIEW_DECISION", item["request_id"], queue_id=queue_id, action="correct", reviewer=reviewer)
    return ReviewQueueItem(**item)


# Keep old /decide endpoint for backward compatibility
@app.post("/review/queue/{queue_id}/decide", response_model=ReviewQueueItem)
def decide_review_item(queue_id: str, decision: ReviewDecision):
    if decision.action == "approve":
        return approve_review_item(queue_id, decision.reviewer)
    elif decision.action == "correct":
        if not decision.correction:
            raise HTTPException(status_code=400, detail="correction payload required")
        return correct_review_item(queue_id, decision.reviewer, decision.correction)
    else:
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'correct'")


# ─────────────── Audit Trail Routes ──────────────────────────────

@app.get("/audit/log", response_model=List[AuditEntry])
def get_audit_log(request_id: Optional[str] = None, limit: int = 100):
    limit  = min(limit, 1000)
    entries = _audit_log
    if request_id:
        entries = [e for e in entries if e["request_id"] == request_id]
    return [AuditEntry(**e) for e in entries[-limit:]]


@app.get("/audit/stats")
def audit_stats():
    routing_counts: Counter = Counter()
    for e in _audit_log:
        if e["event"] == "REQUEST_COMPLETED":
            routing_counts[e.get("routing", "unknown")] += 1

    return {
        "total_requests":  routing_counts.total(),
        "routing": {
            "auto":    routing_counts["auto"],
            "flagged": routing_counts["flagged"],
            "review":  routing_counts["review"],
        },
        "queue": {
            "total":     len(_review_queue),
            "pending":   sum(1 for q in _review_queue if q["status"] == "pending"),
            "approved":  sum(1 for q in _review_queue if q["status"] == "approved"),
            "corrected": sum(1 for q in _review_queue if q["status"] == "corrected"),
        },
    }