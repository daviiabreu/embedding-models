#!/usr/bin/env python3
"""
Standalone test for voice-unfriendly text cleanup.
Tests the regex patterns without requiring TTS dependencies.
"""

import re


def clean_text_for_speech(text: str) -> str:
    """
    Clean text before converting to speech.
    Removes markdown, URLs, and formatting that sounds bad when spoken.
    """
    if not text:
        return text

    # Remove markdown headers (## Title -> Title)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic markdown (**text**, __text__, *text*, _text_)
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

    # Remove strikethrough (~~text~~)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Remove inline code (`code`)
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove standalone URLs (they sound terrible when spoken)
    text = re.sub(r"https?://[^\s]+", "nosso site", text)

    # Remove markdown list markers (-, *, +, 1., 2., etc)
    text = re.sub(r"^[\s]*[-\*\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove multiple newlines (replace with single space)
    text = re.sub(r"\n\n+", " ", text)
    text = re.sub(r"\n", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Clean up common artifacts
    text = text.replace("```", "")
    text = text.replace("---", "")

    return text.strip()


def test_text_cleaning():
    """Test the clean_text_for_speech function."""

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
            "expected_not_contains": ["**", "**", "__", "__"],
        },
        {
            "name": "Markdown headers",
            "input": "## Cursos Disponíveis\n\nTemos vários cursos.",
            "expected_contains": ["Cursos Disponíveis", "Temos vários cursos"],
            "expected_not_contains": ["##", "\n\n"],
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
            "expected_not_contains": ["- "],
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
                "nosso site",
                "mais informações",
                "Detalhes",
                "Item 1",
                "Item 2",
            ],
            "expected_not_contains": ["**", "https://", "##"],
        },
        {
            "name": "Real user complaint - URL",
            "input": "Recomendo explorar a página de graduação no site oficial para obter informações detalhadas sobre cada curso: https://www.inteli.edu.br/graduacao/.",
            "expected": "Recomendo explorar a página de graduação no site oficial para obter informações detalhadas sobre cada curso: nosso site.",
        },
        {
            "name": "Real user complaint - Bold",
            "input": "**O que eu fiz ontem** para ajudar o time a atingir o objetivo?\n2. **O que eu farei hoje** para aj",
            "expected_contains": ["O que eu fiz ontem", "O que eu farei hoje"],
            "expected_not_contains": ["**"],
        },
    ]

    print("=" * 80)
    print("Testing Voice-Unfriendly Text Cleanup")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 80}")
        print(f"Input:\n  {test['input']}")

        cleaned = clean_text_for_speech(test["input"])
        print(f"\nOutput:\n  {cleaned}")

        # Check expected exact output if provided
        if "expected" in test:
            if cleaned == test["expected"]:
                print("\n  ✅ PASS - Output matches expected exactly")
                passed += 1
                continue
            else:
                print(f"\n  ❌ FAIL - Expected:\n  {test['expected']}")
                failed += 1
                continue

        # Check expected content
        success = True
        if "expected_contains" in test:
            for expected in test["expected_contains"]:
                if expected in cleaned:
                    print(f"  ✅ Contains '{expected}'")
                else:
                    print(f"  ❌ FAIL: Expected to contain '{expected}'")
                    success = False

        if "expected_not_contains" in test:
            for not_expected in test["expected_not_contains"]:
                if not_expected not in cleaned:
                    print(f"  ✅ Does NOT contain '{not_expected}'")
                else:
                    print(f"  ❌ FAIL: Should NOT contain '{not_expected}'")
                    success = False

        if success:
            print("\n  ✅ PASS")
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = test_text_cleaning()
    sys.exit(0 if success else 1)
