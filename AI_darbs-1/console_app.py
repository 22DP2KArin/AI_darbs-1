#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
console_app.py
Patstāvīgais darbs — "Mākslīgā intelekta izmantošana programmatūrā ar API"
Autors: Jūs (lai ieliktu GitHub PR)
Apraksts (latviski): Šī konsole-programma:
 - noliec no teksta faila (.txt) kopsavilkumu, izmantojot Hugging Face modeli
 - ģenerē atslēgvārdus (noteikts skaits) un viktorīnas jautājumus ar 4 atbilžu variantiem, izmantojot OpenAI
 - API atslēgas tiek ielādētas no .env (NEIETVER .env repozitorijā)
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional

# Ielādē .env mainīgos
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Hugging Face un OpenAI bibliotēkas
try:
    from huggingface_hub import InferenceApi
except Exception:
    InferenceApi = None

try:
    import openai
except Exception:
    openai = None

# --- Konfigurācija (maināma pēc vajadzības) ---
HF_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # populārs kopsavilkuma modelis
# Ja gribat citu: "facebook/bart-large-cnn" vai citu HF inferenču modeli
OPENAI_MODEL = "gpt-3.5-turbo"  # drošs variants mācību darbam

# --- Palīdzības un kļūdu ziņojumi latviski ---
MSG_ENV_MISSING = "Kļūda: nav ielādētas API atslēgas. Pārbaudiet .env failu un mainīgos HUGGINGFACE_API_KEY / OPENAI_API_KEY."
MSG_HF_MISSING = "Kļūda: huggingface_hub nav instalēts. Instalējiet to: pip install huggingface-hub"
MSG_OPENAI_MISSING = "Kļūda: openai pakotne nav instalēta. Instalējiet to: pip install openai"
MSG_DOTENV_MISSING = "Brīdinājums: python-dotenv nav pieejams; vides mainīgos varat iestatīt arī cita veidā."

# --- Funkcijas ---
def load_env():
    """Ielādē .env failu (ja pieejams)."""
    if load_dotenv is None:
        print(MSG_DOTENV_MISSING)
    else:
        load_dotenv()
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    return hf_key, openai_key

def read_text_file(path: str) -> str:
    """Nolasa tekstu no .txt faila."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Teksta fails nav atrasts: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def summarize_with_hf(text: str, hf_token: str, model: str = HF_SUMMARIZATION_MODEL, max_length: int = 200) -> str:
    """Veic kopsavilkumu, izmantojot huggingface_hub InferenceApi."""
    if InferenceApi is None:
        raise RuntimeError(MSG_HF_MISSING)
    try:
        # Izveido InferenceApi instanci
        client = InferenceApi(repo_id=model, token=hf_token)
        # Parametri modelim
        params = {"max_length": max_length}
        result = client(inputs=text, parameters=params)
        # Rezultāts var būt string vai dict atkarībā no modeļa/versijas
        if isinstance(result, dict):
            # bieži atslēga 'generated_text'
            return result.get("generated_text") or result.get("summary_text") or str(result)
        elif isinstance(result, str):
            return result
        elif isinstance(result, list) and len(result) > 0:
            # saraksta pirmais elements bieži satur 'generated_text'
            first = result[0]
            if isinstance(first, dict):
                return first.get("generated_text") or str(first)
            return str(first)
        else:
            return str(result)
    except Exception as e:
        raise RuntimeError(f"Hugging Face API kļūda: {e}")

def generate_keywords_openai(text: str, openai_key: str, num_keywords: int = 10) -> List[str]:
    """Ģenerē atslēgvārdus, izmantojot OpenAI (chat completion)."""
    if openai is None:
        raise RuntimeError(MSG_OPENAI_MISSING)
    openai.api_key = openai_key
    prompt = (
        f"Izveido {num_keywords} atslēgvārdus (komas atdalīti) no dotā teksta. "
        "Atgriez tikai skaitā prasītos atslēgvārdus, īsi un kodveidā.\n\n"
        f"Teksts:\n{text[:4000]}\n\n"
        "Rezultāts:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"Jūs esat asistents, kas izvelk svarīgākos atslēgvārdus."},
                      {"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        out = resp["choices"][0]["message"]["content"].strip()
        # Mēģinām sadalīt atslēgvārdus pēc komatiem vai jaunām rindām
        # Un atgriezt tieši prasīto skaitu (ja modelis atgriezīs vairāk, samazinām)
        for sep in [",", "\n", ";"]:
            if sep in out:
                parts = [p.strip() for p in out.split(sep) if p.strip()]
                break
        else:
            parts = [p.strip() for p in out.split() if p.strip()]
        parts = parts[:num_keywords]
        return parts
    except Exception as e:
        raise RuntimeError(f"OpenAI atslēgvārdu ģenerēšanas kļūda: {e}")

def generate_quiz_openai(text: str, openai_key: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    """Ģenerē viktorīnas jautājumus ar 4 atbilžu variantiem un pareizo atbildi norādītu.
       Atgriež sarakstu ar elementiem: {'q': str, 'options': [..], 'answer': index}"""
    if openai is None:
        raise RuntimeError(MSG_OPENAI_MISSING)
    openai.api_key = openai_key
    prompt = (
        f"Ģenerē {num_questions} viktorīnas jautājumus par zemāk esošo tekstu. "
        "Katram jautājumam jābūt 4 atbilžu variantiem (A, B, C, D). Atzīmē pareizo atbildi ar taustiņu (piem., 'Answer: B').\n\n"
        f"Teksts:\n{text[:6000]}\n\n"
        "Formāts:\n1) Jautājums?\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: X\n\nSākam:"
    )
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":"Jūs esat ekspertassistents, kurš izveido vairākizvēles jautājumus un norāda pareizo atbildi."},
                      {"role":"user","content":prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        out = resp["choices"][0]["message"]["content"].strip()
        # Parsē izvadāto tekstu uz struktūru
        questions: List[Dict[str, Any]] = []
        # Vienkāršs parser: sadala pa jautājumiem pēc rindstarpām, meklē A/B/C/D un Answer
        blocks = out.split("\n\n")
        current = {}
        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            if not lines:
                continue
            # pirma rinda var būt "1) Jautājums?"
            qline = lines[0]
            if qline[0].isdigit() or qline.startswith(("Q", "1", "2")):
                # sākt jaunu jautājumu
                current = {"q": qline, "options": [], "answer": None}
                for l in lines[1:]:
                    if l.startswith(("A)", "A)","A ")):
                        current["options"].append(l.split(")",1)[1].strip())
                    elif l.startswith(("B)", "B)","B ")):
                        current["options"].append(l.split(")",1)[1].strip())
                    elif l.startswith(("C)", "C)","C ")):
                        current["options"].append(l.split(")",1)[1].strip())
                    elif l.startswith(("D)", "D)","D ")):
                        current["options"].append(l.split(")",1)[1].strip())
                    elif l.lower().startswith("answer"):
                        # Answer: B
                        parts = l.split(":")
                        if len(parts) > 1:
                            key = parts[1].strip().upper()
                            idx = {"A":0,"B":1,"C":2,"D":3}.get(key, None)
                            current["answer"] = idx
                # Ja viss ok, pievieno
                if current.get("q") and current.get("options"):
                    # aizpilda, ja trūkst opciju, pievieno tukšas vietas
                    while len(current["options"]) < 4:
                        current["options"].append("---")
                    questions.append(current)
            else:
                # mēģinām saprast, ja formāts nedaudz atšķiras
                # samezglošana: ja lines satur A) B) C) D) un Answer
                opts = []
                ans = None
                for l in lines:
                    if l.startswith(("A)","A ")):
                        opts.append(l.split(")",1)[1].strip())
                    elif l.startswith(("B)","B ")):
                        opts.append(l.split(")",1)[1].strip())
                    elif l.startswith(("C)","C ")):
                        opts.append(l.split(")",1)[1].strip())
                    elif l.startswith(("D)","D ")):
                        opts.append(l.split(")",1)[1].strip())
                    elif l.lower().startswith("answer"):
                        parts = l.split(":")
                        if len(parts) > 1:
                            key = parts[1].strip().upper()
                            ans = {"A":0,"B":1,"C":2,"D":3}.get(key, None)
                if opts:
                    questions.append({"q": lines[0], "options": opts, "answer": ans})
        # Ja parsēšana izgāžas un nav jautājumu, atgriež vienu fallback jautājumu
        if not questions:
            raise RuntimeError("Neizdevās parsēt jautājumus no OpenAI atbildes.")
        return questions[:num_questions]
    except Exception as e:
        raise RuntimeError(f"OpenAI viktorīnas ģenerēšanas kļūda: {e}")

def save_results(out_dir: str, summary: str, keywords: List[str], quiz: List[Dict[str, Any]]):
    """Saglabā rezultātus failos mapē out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")
    kw_path = os.path.join(out_dir, "keywords.txt")
    quiz_path = os.path.join(out_dir, "quiz.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(kw_path, "w", encoding="utf-8") as f:
        f.write(", ".join(keywords))
    with open(quiz_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(quiz, start=1):
            f.write(f"{i}) {q.get('q','')}\n")
            for idx, opt in enumerate(q.get("options",[])):
                letter = chr(ord("A") + idx)
                f.write(f"   {letter}) {opt}\n")
            ans_idx = q.get("answer")
            ans_letter = chr(ord("A") + ans_idx) if isinstance(ans_idx, int) and 0 <= ans_idx < 4 else "?"
            f.write(f"Answer: {ans_letter}\n\n")
    return summary_path, kw_path, quiz_path

# --- CLI un izpilde ---
def main():
    parser = argparse.ArgumentParser(description="Patstāvīgais darbs: kopsavilkums, atslēgvārdi un viktorīna (latviski).")
    parser.add_argument("input", help="Ievaddokuments (.txt)")
    parser.add_argument("--keywords", "-k", type=int, default=8, help="Cik atslēgvārdus ģenerēt (noklusējums 8)")
    parser.add_argument("--questions", "-q", type=int, default=5, help="Cik viktorīnas jautājumus ģenerēt (noklusējums 5)")
    parser.add_argument("--out", "-o", default="out_results", help="Mape rezultātiem")
    parser.add_argument("--max-summary-length", type=int, default=200, help="Maksimālais kopsavilkuma garums (tokeni/aptuveni vārdi)")
    args = parser.parse_args()

    try:
        hf_key, openai_key = load_env()
        if not hf_key or not openai_key:
            print(MSG_ENV_MISSING)
            sys.exit(1)

        text = read_text_file(args.input)
        print("Teksta nolasīšana pabeigta. Garums:", len(text), "rakstzīmes")

        print("Veidoju kopsavilkumu ar Hugging Face...")
        summary = summarize_with_hf(text, hf_key, max_length=args.max_summary_length)
        print("Kopsavilkums gatavs. Garums:", len(summary), "rakstzīmes")

        print(f"Ģenerēju {args.keywords} atslēgvārdus ar OpenAI...")
        keywords = generate_keywords_openai(text, openai_key, num_keywords=args.keywords)
        print("Atslēgvārdi:", ", ".join(keywords))

        print(f"Ģenerēju {args.questions} viktorīnas jautājumus ar OpenAI...")
        quiz = generate_quiz_openai(text, openai_key, num_questions=args.questions)
        print(f"Viktorīna ģenerēta: {len(quiz)} jautājumi")

        summary_path, kw_path, quiz_path = save_results(args.out, summary, keywords, quiz)
        print("Rezultāti saglabāti mapē:", args.out)
        print("Faili:", summary_path, kw_path, quiz_path)
        print("\nDarbs pabeigts veiksmīgi.")

    except Exception as e:
        print("Kļūda izpildes laikā:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
