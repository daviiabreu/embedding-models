#!/usr/bin/env python3
"""
Lista modelos disponíveis na API Google GenAI usando a chave em .env ou variável de ambiente.
"""

import os
import sys

from dotenv import load_dotenv


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Erro: defina GOOGLE_API_KEY no ambiente ou em .env", file=sys.stderr)
        sys.exit(1)

    try:
        from google import genai
    except Exception as exc:  # pragma: no cover
        print(f"Falha ao importar google.genai: {exc}", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    try:
        models = list(client.models.list())
    except Exception as exc:
        print(f"Erro ao listar modelos: {exc}", file=sys.stderr)
        sys.exit(1)

    if not models:
        print("Nenhum modelo retornado.")
        return

    print("Modelos disponíveis:")
    for m in models:
        print(f"- {m.name}")


if __name__ == "__main__":
    main()
