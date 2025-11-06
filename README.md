# AI_darbs-1

# Mākslīgā intelekta izmantošana programmatūrā ar API

### Patstāvīgais darbs — “AI_darbs-1”

## Apraksts
Šī Python konsole-programma demonstrē mākslīgā intelekta (AI) izmantošanu, apstrādājot teksta datus ar **Hugging Face** un **OpenAI** API.  
Tā spēj:

1.  **Apkopot (summarize)** tekstu no `.txt` faila, izmantojot Hugging Face modeli.  
2.  **Ģenerēt atslēgvārdus**, izmantojot OpenAI valodas modeli.  
3.  **Izveidot viktorīnas jautājumus** ar četriem atbilžu variantiem, izmantojot OpenAI.

Rezultāti tiek automātiski saglabāti mapē `out_results` kā trīs faili:

- `summary.txt` — teksta kopsavilkums;  
- `keywords.txt` — atslēgvārdu saraksts;  
- `quiz.txt` — ģenerētie viktorīnas jautājumi ar pareizajām atbildēm.


## Tehnoloģijas

- **Python 3.10+**
- **Hugging Face Hub API** (`huggingface-hub`)
- **OpenAI API** (`openai`)
- **dotenv** bibliotēka drošai API atslēgu glabāšanai


## Instalācija

1. Klonē repozitoriju:
   ```bash
   git clone https://github.com/<lietotājvārds>/AI_darbs-1.git
   cd AI_darbs-1
