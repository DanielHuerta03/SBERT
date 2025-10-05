from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import unicodedata, re, threading

app = FastAPI()

# --------- Carga en segundo plano al iniciar ---------
model = None
model_ready = False
MODEL_NAME = "distiluse-base-multilingual-cased-v2"  # usa MiniLM si quieres menor RAM

def _load_model():
    global model, model_ready
    try:
        model = SentenceTransformer(MODEL_NAME)
        model_ready = True
    except Exception as e:
        # si falla, queda en False y lo verás en /health
        print("ERROR cargando modelo:", e, flush=True)

@app.on_event("startup")
def startup():
    # dispara la carga SIN bloquear que el servidor escuche el puerto
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()

# ------------------- EQUIVALENCIAS -------------------
equivalencias_carreras = {
    "ingenieria de sistemas": ["ingenieria de software", "ingenieria informatica", "ciencia de la computacion"],
    "ingenieria de software": ["ingenieria de sistemas", "ingenieria informatica"],
    "ingenieria informatica": ["ingenieria de sistemas", "ingenieria de software"],
    "economia": ["economia y finanzas", "administracion y economia"],
    "contabilidad": ["contaduria publica", "contabilidad y finanzas"],
    "derecho": ["ciencias juridicas", "ciencias politicas y derecho"]
}

# ------------------- MODELOS Pydantic -------------------
class EmbeddingRequest(BaseModel):
    text: str

class SimilarityExplainRequest(BaseModel):
    puesto: dict
    candidato: dict

# ------------------- UTILS -------------------
def _norm(s: str) -> str:
    if not s: return ""
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s

def limpiar_texto(texto: str) -> str:
    texto = _norm(texto)
    irrelevantes = ["bachiller en", "titulo en", "grado en", "afines", "universidad", "licenciado en"]
    for irr in irrelevantes:
        texto = texto.replace(irr, "")
    return texto.strip()

_SPLIT_RE = re.compile(r"[,\n;/•\-–]| y | or | o ")
def _split(s: str): return _SPLIT_RE.split(s)

def dividir_palabras_clave(texto: str) -> set[str]:
    texto = _norm(texto)
    if not texto:
        return set()
    tokens = [t.strip() for t in _split(texto) if t.strip()]
    return {t for t in tokens if len(t) > 1}

def safe_encode(texto: str):
    if not model_ready or model is None:
        raise HTTPException(status_code=503, detail="Modelo cargando… intenta en unos segundos.")
    texto = (texto or "").strip()
    return model.encode(texto, convert_to_tensor=True)

# --- Parseo de años/meses de experiencia ---
RE_YEARS = re.compile(r"(\d+(?:[\.,]\d+)?)(?=\s*\+?\s*(?:anos|años|year|years|yrs))")
RE_MONTHS = re.compile(r"(\d+)(?=\s*(?:mes|meses|months))")
RE_Y_AND_M = re.compile(r"(\d+)\s*(?:ano|años|anos|year|years|yrs)\s*(?:y|and)\s*(\d+)\s*(?:mes|meses|months)")

def extract_years(text: str) -> float | None:
    t = _norm(text)
    m = RE_Y_AND_M.search(t)
    if m:
        y = float(m.group(1).replace(",", "."))
        mo = float(m.group(2))
        return y + (mo / 12.0)
    m = RE_MONTHS.search(t)
    months_val = float(m.group(1)) if m else None
    m = RE_YEARS.search(t)
    if m:
        return float(m.group(1).replace(",", "."))
    if months_val is not None:
        return months_val / 12.0
    return None

# ------------------- CORE -------------------
def calcular_similitud(puesto: dict, candidato: dict, equivalencias: dict,
                       refuerzo_min=90, alpha=0.60, beta_exp=0.60):
    resultados, explicaciones = {}, {}
    claves = ["experiencia", "educacion", "habilidades_tecnicas", "habilidades_blandas", "certificaciones"]
    claves_presentes = [k for k in claves if k in puesto]

    for key in claves_presentes:
        txt_p = puesto.get(key, "") or ""
        txt_c = candidato.get(key, "") or ""
        score_base = float(util.cos_sim(safe_encode(txt_p), safe_encode(txt_c)).item() * 100.0)

        if key == "educacion":
            puesto_clean = limpiar_texto(txt_p)
            cand_clean   = limpiar_texto(txt_c)
            equivalencia_detectada = False
            carreras_puesto = [c.strip() for c in puesto_clean.replace(" o ", ",").split(",") if c.strip()]
            for carrera_puesto in carreras_puesto:
                for carrera, eqs in equivalencias.items():
                    if carrera in carrera_puesto:
                        for eq in eqs:
                            if eq in cand_clean:
                                equivalencia_detectada = True
                                score_final = max(score_base, float(refuerzo_min))
                                explicaciones[key] = (
                                    f"Educación: similitud SBERT {round(score_base,2)}%. "
                                    f"Se ajusta a {round(score_final,2)}% por equivalencia detectada."
                                )
                                break
                    if equivalencia_detectada: break
                if equivalencia_detectada: break
            if not equivalencia_detectada:
                score_final = score_base
                explicaciones[key] = f"Educación: similitud SBERT {round(score_base,2)}%. No hay equivalencias."
        elif key in ("habilidades_tecnicas", "habilidades_blandas", "certificaciones"):
            set_req = dividir_palabras_clave(txt_p); set_cv = dividir_palabras_clave(txt_c)
            inter   = sorted(set_req & set_cv)
            faltan  = sorted(set_req - set_cv)
            extras  = sorted(set_cv - set_req)
            coverage = (len(inter) / len(set_req)) if set_req else 1.0
            score_cov = coverage * 100.0
            score_final = (1 - alpha) * score_base + alpha * score_cov
            etiqueta = "Habilidades técnicas" if key=="habilidades_tecnicas" else "Habilidades blandas" if key=="habilidades_blandas" else "Certificaciones"
            partes = [f"{etiqueta}: SBERT {round(score_base,2)}%, cobertura {round(score_cov,2)}%."]
            if inter:  partes.append(f"Coincidencias: {', '.join(inter)}.")
            if faltan: partes.append(f"Faltan: {', '.join(faltan)}.")
            if extras: partes.append(f"Extras: {', '.join(extras)}.")
            explicaciones[key] = " ".join(partes)
        elif key == "experiencia":
            req_years = extract_years(txt_p)
            cand_years = extract_years(txt_c)
            if req_years is not None and req_years > 0 and cand_years is not None and cand_years >= 0:
                coverage = min(1.0, cand_years / req_years)
                score_cov = coverage * 100.0
                score_final = (1 - beta_exp) * score_base + beta_exp * score_cov
                explicaciones[key] = (f"Experiencia: SBERT {round(score_base,2)}%. "
                                      f"Req ~{round(req_years,2)} vs cand ~{round(cand_years,2)} "
                                      f"(cobertura {round(score_cov,2)}%).")
            else:
                score_final = score_base
                explicaciones[key] = f"Experiencia: SBERT {round(score_base,2)}%."
        else:
            score_final = score_base
            explicaciones[key] = f"{key.capitalize()}: SBERT {round(score_base,2)}%."
        resultados[key] = round(score_final, 2)

    resultados["TOTAL"] = round(sum(resultados[k] for k in claves_presentes)/len(claves_presentes), 2) if claves_presentes else 0.0
    return resultados, explicaciones

# ------------------- ENDPOINTS -------------------
@app.get("/health")
def health():
    return {"listening": True, "model_ready": model_ready}

@app.post("/embedding")
def generar_embedding(req: EmbeddingRequest):
    emb = safe_encode(req.text).tolist()
    return {"embedding": emb}

@app.post("/similaridad_explicable")
def similaridad_explicable(req: SimilarityExplainRequest):
    resultados, explicaciones = calcular_similitud(req.puesto, req.candidato, equivalencias_carreras)
    return {"resultados": resultados, "explicaciones": explicaciones}
