#!/usr/bin/env python3
"""
Test script for voice-unfriendly text cleanup.
Verifies that URLs and markdown are properly removed before TTS.
"""

import sys

sys.path.insert(0, "/Users/davi/github/inteli-projetos/embedding-models/backend")

from tts.tts_service import TTSService


def test_text_cleaning():
    """Test the clean_text_for_speech function."""

    tts = TTSService()

    test_cases = [
        {
            "name": "URL in text",
            "input": "Recomendo explorar a página de graduação no site oficial para obter informações detalhadas sobre cada curso: https://www.inteli.edu.br/graduacao/.",
            "expected_contains": ["página de graduação", "nosso site"],
            "expected_not_contains": ["https://", "www.inteli.edu.br"],
        },
        {
            "name": "Markdown bold",
            "input": "**O que eu fiz ontem** para ajudar o time a atingir o objetivo?",
            "expected_contains": ["O que eu fiz ontem"],
            "expected_not_contains": ["**"],
        },
        {
            "name": "Mixed markdown",
            "input": "Temos **três cursos principais**: *Ciência da Computação*, __Engenharia__, e _Design_",
            "expected_contains": [
                "três cursos principais",
                "Ciência da Computação",
                "Engenharia",
                "Design",
            ],
            "expected_not_contains": ["**", "*", "__", "_"],
        },
        {
            "name": "Markdown headers",
            "input": "## Cursos Disponíveis\n\nTemos vários cursos.",
            "expected_contains": ["Cursos Disponíveis", "Temos vários cursos"],
            "expected_not_contains": ["##"],
        },
        {
            "name": "Markdown lists",
            "input": "Nossos cursos:\n- Ciência da Computação\n- Engenharia\n- Design",
            "expected_contains": [
                "Nossos cursos",
                "Ciência da Computação",
                "Engenharia",
                "Design",
            ],
            "expected_not_contains": ["\n-"],
        },
        {
            "name": "Inline code",
            "input": "O comando `pip install` é usado para instalar pacotes.",
            "expected_contains": ["pip install", "instalar pacotes"],
            "expected_not_contains": ["`"],
        },
        {
            "name": "Complex formatting",
            "input": "**Importante**: Acesse https://www.inteli.edu.br para mais informações.\n\n## Detalhes\n- Item 1\n- Item 2",
            "expected_contains": [
                "Importante",
                "Acesse",
                "mais informações",
                "Detalhes",
                "Item 1",
                "Item 2",
            ],
            "expected_not_contains": ["**", "https://", "##", "\n-"],
        },
    ]

    print("=" * 60)
    print("Testing Voice-Unfriendly Text Cleanup")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 60)
        print(f"Input:  {test['input'][:80]}...")

        cleaned = tts.clean_text_for_speech(test["input"])
        print(f"Output: {cleaned[:80]}...")

        # Check expected content
        success = True
        for expected in test["expected_contains"]:
            if expected not in cleaned:
                print(f"  ❌ FAIL: Expected to contain '{expected}'")
                success = False

        for not_expected in test["expected_not_contains"]:
            if not_expected in cleaned:
                print(f"  ❌ FAIL: Should NOT contain '{not_expected}'")
                success = False

        if success:
            print("  ✅ PASS")
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = test_text_cleaning()
    sys.exit(0 if success else 1)
