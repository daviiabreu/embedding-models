"""
Processador de HTML para Web Scraping do Inteli
Extrai conteúdo estruturado mantendo hierarquia semântica
Remove ruído (navegação, scripts, footers, etc)
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup, NavigableString, Tag


class InteliHTMLProcessor:
    """
    Processador especializado para HTMLs do site do Inteli.
    Mantém hierarquia semântica enquanto remove ruído.
    """

    # Tags semânticas que queremos preservar
    SEMANTIC_TAGS = ["article", "section", "main", "div"]

    # Tags de conteúdo
    CONTENT_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"]

    # Tags/IDs/Classes para REMOVER (ruído de navegação)
    NOISE_PATTERNS = {
        "ids": [
            "header",
            "nav",
            "navigation",
            "menu",
            "sidebar",
            "footer",
            "comments",
            "social",
            "share",
            "related",
            "cookie",
            "banner",
            "ads",
            "advertisement",
        ],
        "classes": [
            "nav",
            "menu",
            "header",
            "footer",
            "sidebar",
            "share",
            "social",
            "comment",
            "related",
            "cookie",
            "banner",
            "breadcrumb",
            "pagination",
            "author-box",
            "meta",
            "tags",
            "categories",
            "newsletter",
            "subscribe",
        ],
        "tags": [
            "nav",
            "aside",
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "button",
            "input",
        ],
    }

    # Padrões de texto para remover
    TEXT_NOISE_PATTERNS = [
        r"skip to content",
        r"acompanhe seu processo",
        r"quem somos",
        r"fundadores",
        r"campus",
        r"docentes",
        r"programa de bolsas",
        r"blog",
        r"contato",
        r"cursos",
        r"graduação",
        r"ensino básico",
        r"educação executiva",
        r"seja um parceiro",
        r"doadores",
        r"redes sociais",
        r"acesse já",
        r"localização",
        r"copyright",
        r"todos os direitos reservados",
        r"compartilhe:?",
        r"veja também:?",
    ]

    def __init__(self):
        self.noise_pattern = re.compile(
            "|".join(self.TEXT_NOISE_PATTERNS), re.IGNORECASE
        )

    def is_noise_element(self, element: Tag) -> bool:
        """Detecta se um elemento é ruído (navegação, menu, etc)."""
        if not isinstance(element, Tag):
            return False

        # Proteção contra elementos com attrs=None
        if not hasattr(element, "attrs") or element.attrs is None:
            return False

        # Remove por tag
        if element.name in self.NOISE_PATTERNS["tags"]:
            return True

        # Remove por ID (com proteção)
        try:
            elem_id = element.get("id", "").lower() if element.get("id") else ""
        except (AttributeError, TypeError):
            elem_id = ""

        if elem_id and any(noise in elem_id for noise in self.NOISE_PATTERNS["ids"]):
            return True

        # Remove por classe (com proteção)
        try:
            elem_classes = (
                " ".join(element.get("class", [])).lower()
                if element.get("class")
                else ""
            )
        except (AttributeError, TypeError):
            elem_classes = ""

        if elem_classes and any(
            noise in elem_classes for noise in self.NOISE_PATTERNS["classes"]
        ):
            return True

        # Remove elementos vazios ou muito pequenos
        try:
            text = element.get_text(strip=True)
        except (AttributeError, TypeError):
            return True

        if len(text) < 10:
            return True

        # Remove por padrão de texto (menu items)
        if len(text) < 50 and self.noise_pattern.search(text):
            return True

        return False

    def extract_hierarchy(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extrai conteúdo mantendo hierarquia semântica.
        Retorna lista de elementos com contexto hierárquico.
        """
        elements = []

        # Usa o body direto (mais seguro para HTMLs complexos)
        main_content = soup.find("body")

        if not main_content:
            print("⚠️ Nenhum <body> encontrado no HTML")
            return elements

        print("✅ Processando <body> completo")

        # Remove elementos de ruído (com tratamento de erro)
        try:
            tags_to_remove = []
            all_tags = list(main_content.find_all())
            print(f"📊 Total de tags encontradas no body: {len(all_tags)}")

            for tag in all_tags:
                try:
                    if self.is_noise_element(tag):
                        tags_to_remove.append(tag)
                except Exception:
                    # Se der erro ao verificar, ignora o elemento
                    continue

            print(f"🗑️ Tags marcadas para remoção (ruído): {len(tags_to_remove)}")

            # Remove os tags identificados
            for tag in tags_to_remove:
                try:
                    tag.decompose()
                except Exception:
                    continue
        except Exception as e:
            # Se falhar completamente, continua sem remover ruído
            print(f"⚠️ Aviso: Erro ao remover ruído, continuando: {e}")

        # Tracking de contexto hierárquico
        hierarchy_stack = []
        current_section = "Introdução"

        # Contadores para debug
        processed_count = 0
        extracted_count = 0

        # Extrai elementos estruturados
        for element in main_content.descendants:
            try:
                processed_count += 1

                if isinstance(element, NavigableString):
                    continue

                if not isinstance(element, Tag):
                    continue

                tag_name = element.name
                if not tag_name:
                    continue

                # TÍTULOS (Hierarquia)
                if tag_name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    try:
                        text = element.get_text(strip=True)
                    except (AttributeError, TypeError):
                        continue

                    if len(text) < 5:
                        continue

                    # Detecta nível hierárquico
                    level = int(tag_name[1])  # h1 -> 1, h2 -> 2, etc

                    # Atualiza stack de hierarquia
                    while hierarchy_stack and hierarchy_stack[-1]["level"] >= level:
                        hierarchy_stack.pop()

                    hierarchy_stack.append({"level": level, "title": text})

                    current_section = text

                    elements.append(
                        {
                            "text": text,
                            "type": "Title",
                            "hierarchy_level": f"level_{level}",
                            "context_section": current_section,
                            "parent_sections": [
                                h["title"] for h in hierarchy_stack[:-1]
                            ],
                        }
                    )

                # PARÁGRAFOS
                elif tag_name == "p":
                    try:
                        text = element.get_text(strip=True)
                    except (AttributeError, TypeError):
                        continue

                    if len(text) < 20:  # Ignora parágrafos muito curtos
                        continue

                    # Verifica se não é ruído
                    if self.noise_pattern.search(text):
                        continue

                    elements.append(
                        {
                            "text": text,
                            "type": "NarrativeText",
                            "hierarchy_level": "body",
                            "context_section": current_section,
                            "parent_sections": [h["title"] for h in hierarchy_stack],
                        }
                    )

                # LISTAS
                elif tag_name == "li":
                    try:
                        text = element.get_text(strip=True)
                    except (AttributeError, TypeError):
                        continue

                    if len(text) < 10:
                        continue

                    elements.append(
                        {
                            "text": text,
                            "type": "ListItem",
                            "hierarchy_level": "list_item",
                            "context_section": current_section,
                            "parent_sections": [h["title"] for h in hierarchy_stack],
                        }
                    )

                # BLOCKQUOTES
                elif tag_name == "blockquote":
                    try:
                        text = element.get_text(strip=True)
                    except (AttributeError, TypeError):
                        continue

                    if len(text) < 15:
                        continue

                    elements.append(
                        {
                            "text": text,
                            "type": "Quote",
                            "hierarchy_level": "body",
                            "context_section": current_section,
                            "parent_sections": [h["title"] for h in hierarchy_stack],
                        }
                    )

            except Exception:
                # Se der qualquer erro ao processar um elemento, pula ele
                continue

        print(f"📊 Elementos processados: {processed_count}")
        print(f"✅ Elementos extraídos: {len(elements)}")

        # Aplicar consolidação semântica
        if elements:
            print("\n🔗 Aplicando consolidação semântica...")
            elements = self.group_by_semantic_context(elements)

        return elements

    def group_by_semantic_context(
        self, elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Agrupa elementos por contexto semântico para reduzir fragmentação.

        Estratégias de agrupamento:
        1. Listas: Agrupa list_items consecutivos da mesma seção
        2. Professores: Agrupa nome + descrição (se aplicável)
        3. Conteúdo relacionado: Agrupa elementos da mesma seção semântica

        Args:
            elements: Lista de elementos extraídos do HTML

        Returns:
            Lista de elementos consolidados
        """
        grouped = []
        i = 0

        while i < len(elements):
            current = elements[i]
            current_text = current.get("text", "")
            current_type = current.get("type", "")
            current_section = current.get("context_section", "")

            # ========================================
            # 1. CONSOLIDAR LISTAS CONSECUTIVAS
            # ========================================
            if current_type == "ListItem":
                # Coletar itens de lista consecutivos da mesma seção
                list_items = [current_text]
                j = i + 1

                while j < len(elements):
                    next_elem = elements[j]

                    if (
                        next_elem.get("type") == "ListItem"
                        and next_elem.get("context_section") == current_section
                    ):
                        list_items.append(next_elem.get("text", ""))
                        j += 1
                    else:
                        break

                # Se consolidou mais de 1 item, criar bloco de lista
                if len(list_items) > 1:
                    consolidated_list = {
                        "text": "\n".join(f"• {item}" for item in list_items),
                        "type": "list_block",
                        "hierarchy_level": "list",
                        "context_section": current_section,
                        "parent_sections": current.get("parent_sections", []),
                        "metadata": {
                            **current.get("metadata", {}),
                            "consolidated": True,
                            "list_items_count": len(list_items),
                        },
                    }

                    grouped.append(consolidated_list)
                    i = j
                    continue

            # ========================================
            # 2. CONSOLIDAR PROFESSORES/ENTIDADES
            # ========================================
            elif current_type == "NarrativeText" and i + 1 < len(elements):
                next_elem = elements[i + 1]
                next_text = next_elem.get("text", "")
                next_section = next_elem.get("context_section", "")
                next_type = next_elem.get("type", "")

                # Se ambos são da mesma seção e parecem ser nome + descrição
                if (
                    next_section == current_section
                    and next_type == "NarrativeText"
                    and len(current_text) < 100  # Nome curto
                    and len(next_text) > 50
                ):  # Descrição mais longa
                    # Mesclar nome + descrição
                    consolidated_entity = {
                        "text": f"{current_text}\n{next_text}",
                        "type": "NarrativeText",
                        "hierarchy_level": current.get("hierarchy_level", "body"),
                        "context_section": current_section,
                        "parent_sections": current.get("parent_sections", []),
                        "metadata": {
                            **current.get("metadata", {}),
                            "consolidated": True,
                            "entity_type": "profile",
                        },
                    }

                    grouped.append(consolidated_entity)
                    i += 2  # Pular ambos
                    continue

            # ========================================
            # 3. ELEMENTO NÃO CONSOLIDADO
            # ========================================
            # Adicionar normalmente
            grouped.append(current)
            i += 1

        # Estatísticas
        reduction = len(elements) - len(grouped)
        if reduction > 0:
            print("   ✅ Consolidação aplicada:")
            print(f"      • Elementos antes: {len(elements)}")
            print(f"      • Elementos depois: {len(grouped)}")
            print(
                f"      • Redução: {reduction} ({reduction / len(elements) * 100:.1f}%)"
            )

        return grouped

    def process_html_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Processa um arquivo HTML e retorna elementos estruturados.
        """
        try:
            html_content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback para outras encodings
            try:
                html_content = file_path.read_text(encoding="latin-1")
            except:
                print(f"⚠️ Erro ao ler {file_path.name}")
                return []

        # Parse com BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Extrai elementos com hierarquia
        elements = self.extract_hierarchy(soup)

        # Adiciona metadados do arquivo
        for element in elements:
            element["metadata"] = {
                "source": file_path.name,
                "file_type": "html",
                "category": self._infer_category(file_path.name),
            }

        return elements

    def _infer_category(self, filename: str) -> str:
        """Infere categoria baseado no nome do arquivo."""
        fname = filename.lower()

        if "processo-seletivo" in fname or "edital" in fname:
            return "processo_seletivo"
        elif "graduacao" in fname or "curso" in fname:
            return "graduacao"
        elif "mercado-tech" in fname or "tecnologia" in fname:
            return "mercado_tech"
        elif "campus" in fname or "estrutura" in fname:
            return "campus"
        elif "faq" in fname or "pergunta" in fname:
            return "faq"
        elif "bolsa" in fname or "financiamento" in fname:
            return "bolsas"

        return "geral"

    def batch_process_directory(
        self, dir_path: Path
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processa todos os HTMLs de um diretório.
        Retorna dicionário com {filename: elements}.
        """
        results = {}

        html_files = list(dir_path.glob("*.html"))
        print(f"📂 Encontrados {len(html_files)} arquivos HTML")

        for html_file in html_files:
            print(f"⚙️ Processando: {html_file.name}")
            elements = self.process_html_file(html_file)

            if elements:
                results[html_file.name] = elements
                print(f"   ✅ {len(elements)} elementos extraídos")
            else:
                print("   ⚠️ Nenhum elemento extraído")

        return results


# ================= FUNÇÕES AUXILIARES =================


def clean_extracted_text(text: str) -> str:
    """Limpa texto extraído do HTML."""
    if not text:
        return ""

    # Remove espaços múltiplos
    text = re.sub(r"\s+", " ", text)

    # Remove caracteres de controle
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove URLs isoladas
    text = re.sub(r"https?://\S+", "", text)

    # Normaliza aspas e travessões
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("–", "-").replace("—", "-")

    return text.strip()


def convert_html_to_pipeline_format(
    elements: List[Dict[str, Any]], source_file: str
) -> List[Dict[str, Any]]:
    """
    Converte elementos extraídos do HTML para formato do pipeline.
    """
    pipeline_elements = []

    for idx, element in enumerate(elements):
        text = clean_extracted_text(element["text"])

        if len(text) < 10:
            continue

        pipeline_elements.append(
            {
                "text": text,
                "type": element["type"],
                "metadata": {
                    "source": source_file,
                    "page_number": 1,
                    "element_type": element["type"],
                    "hierarchy_level": element["hierarchy_level"],
                    "context_section": element["context_section"],
                    "context_header": element.get("parent_sections", [])[-1]
                    if element.get("parent_sections")
                    else "",
                    "category": element["metadata"]["category"],
                    "is_header": element["type"] == "Title",
                },
                "element_id": f"html_{idx}",
            }
        )

    return pipeline_elements


# ================= EXEMPLO DE USO =================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Uso: python html_processor.py <caminho_html_ou_diretorio>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    processor = InteliHTMLProcessor()

    if input_path.is_file():
        # Processa arquivo único
        print(f"📄 Processando arquivo: {input_path.name}\n")
        elements = processor.process_html_file(input_path)

        print(f"\n✅ Total de elementos extraídos: {len(elements)}")
        print("\n📊 Distribuição por tipo:")
        types_count = {}
        for el in elements:
            t = el["type"]
            types_count[t] = types_count.get(t, 0) + 1

        for t, count in sorted(types_count.items()):
            print(f"   {t}: {count}")

        # Exibe primeiros 3 elementos
        print("\n📝 Primeiros 3 elementos:")
        for el in elements[:3]:
            print(f"\n[{el['type']}] (Nível: {el['hierarchy_level']})")
            print(f"Seção: {el['context_section']}")
            print(f"Texto: {el['text'][:100]}...")

    elif input_path.is_dir():
        # Processa diretório
        print(f"📂 Processando diretório: {input_path}\n")
        results = processor.batch_process_directory(input_path)

        print(f"\n✅ Total de arquivos processados: {len(results)}")

        total_elements = sum(len(els) for els in results.values())
        print(f"✅ Total de elementos extraídos: {total_elements}")

        # Estatísticas
        print("\n📊 Top 10 arquivos com mais elementos:")
        sorted_files = sorted(results.items(), key=lambda x: len(x[1]), reverse=True)
        for fname, els in sorted_files[:10]:
            print(f"   {fname}: {len(els)} elementos")

    else:
        print(f"❌ Erro: {input_path} não é um arquivo ou diretório válido")
        sys.exit(1)
