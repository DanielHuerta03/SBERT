from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import unicodedata
import re

app = FastAPI()

# ------------------- MODELO SBERT -------------------
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

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

def dividir_palabras_clave(texto: str) -> set[str]:
    """
    Convierte una lista libre (separada por comas / saltos / 'y') a un set normalizado.
    """
    texto = _norm(texto)
    if not texto:
        return set()
    tokens = [t.strip() for t in _split(texto) if t.strip()]
    return {t for t in tokens if len(t) > 1}

def _split(s: str):
    return _SPLIT_RE.split(s)

def safe_encode(texto: str):
    texto = (texto or "").strip()
    return model.encode(texto, convert_to_tensor=True)

# --- Parseo de años/meses de experiencia ---
# soporta: "2 años", "2+ años", "1 año y 6 meses", "18 meses", "3 yrs", "3 years"
RE_YEARS = re.compile(r"(\d+(?:[\.,]\d+)?)(?=\s*\+?\s*(?:anos|años|year|years|yrs))")
RE_MONTHS = re.compile(r"(\d+)(?=\s*(?:mes|meses|months))")
RE_Y_AND_M = re.compile(r"(\d+)\s*(?:ano|años|anos|year|years|yrs)\s*(?:y|and)\s*(\d+)\s*(?:mes|meses|months)")

def extract_years(text: str) -> float | None:
    """
    Devuelve años como float. Si no se encuentra, None.
    Reglas:
      - "X años" -> X
      - "X años y Y meses" -> X + Y/12
      - "Z meses" -> Z/12
      - "2+ años" -> 2
    """
    t = _norm(text)

    # "X años y Y meses"
    m = RE_Y_AND_M.search(t)
    if m:
        y = float(m.group(1).replace(",", "."))
        mo = float(m.group(2))
        return y + (mo / 12.0)

    # "X meses"
    m = RE_MONTHS.search(t)
    months_val = None
    if m:
        months_val = float(m.group(1))

    # "X años" (acepta decimales y "+")
    m = RE_YEARS.search(t)
    if m:
        years = float(m.group(1).replace(",", "."))
        return years

    if months_val is not None:
        return months_val / 12.0

    # patrones tipo "3+ años" también los capta RE_YEARS (por el lookahead)
    return None

# ------------------- CORE -------------------
def calcular_similitud(puesto: dict, candidato: dict, equivalencias: dict,
                       refuerzo_min=90, alpha=0.60, beta_exp=0.60):
    """
    Devuelve:
      - resultados: % por atributo (0..100)
      - explicaciones: textos por atributo

    alpha: peso de cobertura para skills/certs (vs SBERT).
    beta_exp: peso de cobertura de años para experiencia (vs SBERT).
    """
    resultados = {}
    explicaciones = {}

    claves = ["experiencia", "educacion", "habilidades_tecnicas", "habilidades_blandas", "certificaciones"]
    claves_presentes = [k for k in claves if k in puesto]

    for key in claves_presentes:
        txt_p = puesto.get(key, "") or ""
        txt_c = candidato.get(key, "") or ""

        emb_puesto = safe_encode(txt_p)
        emb_cand   = safe_encode(txt_c)
        score_base = float(util.cos_sim(emb_puesto, emb_cand).item() * 100.0)

        # --- EDUCACIÓN con equivalencias ---
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
                                    f"Se ajusta a {round(score_final,2)}% porque '{eq}' se considera equivalente a '{carrera}'."
                                )
                                break
                    if equivalencia_detectada:
                        break
                if equivalencia_detectada:
                    break

            if not equivalencia_detectada:
                score_final = score_base
                explicaciones[key] = f"Educación: similitud SBERT {round(score_base,2)}%. No se detectaron equivalencias específicas."

        # --- HABILIDADES / CERTIFICACIONES (cobertura asimétrica + SBERT) ---
        elif key in ("habilidades_tecnicas", "habilidades_blandas", "certificaciones"):
            set_req = dividir_palabras_clave(txt_p)     # lo que pide el puesto
            set_cv  = dividir_palabras_clave(txt_c)     # lo que trae el candidato
            inter   = sorted(set_req.intersection(set_cv))
            faltan  = sorted(set_req.difference(set_cv))
            extras  = sorted(set_cv.difference(set_req))

            coverage = (len(inter) / len(set_req)) if len(set_req) > 0 else 1.0
            score_cov = coverage * 100.0
            score_final = (1 - alpha) * score_base + alpha * score_cov

            etiqueta = "Habilidades técnicas" if key == "habilidades_tecnicas" \
                       else "Habilidades blandas" if key == "habilidades_blandas" \
                       else "Certificaciones"

            parts = [
                f"{etiqueta}: similitud SBERT {round(score_base,2)}%, cobertura de requeridos {round(score_cov,2)}%."
            ]
            if inter:
                parts.append(f"Coincidencias: {', '.join(inter)}.")
            if faltan:
                parts.append(f"Faltan: {', '.join(faltan)}.")
            if extras:
                parts.append(f"Extras (no penaliza): {', '.join(extras)}.")
            explicaciones[key] = " ".join(parts)

        # --- EXPERIENCIA (años + SBERT) ---
        elif key == "experiencia":
            req_years = extract_years(txt_p)
            cand_years = extract_years(txt_c)

            if req_years is not None and req_years > 0 and cand_years is not None and cand_years >= 0:
                coverage = min(1.0, cand_years / req_years)     # >= req no penaliza
                score_cov = coverage * 100.0
                score_final = (1 - beta_exp) * score_base + beta_exp * score_cov

                # explicación
                msg = (
                    f"Experiencia: similitud SBERT {round(score_base,2)}%. "
                    f"Se detectó requisito de ~{round(req_years,2)} años y el candidato tiene ~{round(cand_years,2)} años "
                    f"(cobertura {round(score_cov,2)}%). "
                    f"El puntaje combina área (SBERT) y cobertura de años (β={beta_exp})."
                )
                if cand_years > req_years:
                    msg += " Excedente de años no penaliza."
                explicaciones[key] = msg
            else:
                # si no se pudieron extraer años, usa SBERT puro
                score_final = score_base
                extra = []
                if req_years is None: extra.append("no se detectaron años en el requisito")
                if cand_years is None: extra.append("no se detectaron años en el CV")
                nota = f" ({'; '.join(extra)})" if extra else ""
                explicaciones[key] = f"Experiencia: similitud SBERT {round(score_base,2)}%{nota}."

        else:
            score_final = score_base
            explicaciones[key] = f"{key.capitalize()}: similitud SBERT {round(score_base,2)}%."

        resultados[key] = round(score_final, 2)

    # Promedio total sólo de lo evaluado
    if claves_presentes:
        resultados["TOTAL"] = round(sum(resultados[k] for k in claves_presentes) / len(claves_presentes), 2)
    else:
        resultados["TOTAL"] = 0.0

    return resultados, explicaciones

# ------------------- ENDPOINTS -------------------
@app.post("/embedding")
def generar_embedding(req: EmbeddingRequest):
    emb = model.encode(req.text).tolist()
    return {"embedding": emb}

@app.post("/similaridad_explicable")
def similaridad_explicable(req: SimilarityExplainRequest):
    resultados, explicaciones = calcular_similitud(req.puesto, req.candidato, equivalencias_carreras)
    return {"resultados": resultados, "explicaciones": explicaciones}
