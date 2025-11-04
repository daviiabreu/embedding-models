from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from sentence_transformers import SentenceTransformer

import json
import re

file_path = "documents"
base_file_name = "Edital-Processo-Seletivo-Inteli_-Graduacao-2026_AJUSTADO"

def clean_text(text):
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR/parsing issues
    text = text.replace('\ufb01', 'fi')  # Fix ligatures
    text = text.replace('\ue009', 'tt')  # Fix special characters
    
    # Remove page numbers and footers if they appear in content
    text = re.sub(r'^\d+$', '', text)  # Remove standalone page numbers
    
    # Normalize bullet points
    text = re.sub(r'[•◦▪▫]', '•', text)
    
    return text.strip()

def determine_hierarchy_level(element):
    """Determine document hierarchy level based on element type and content"""
    element_type = element.get('type')
    text = element.get('text', '')
    
    if element_type == 'Title':
        # Detect numbered sections (1., 1.1, 1.1.1, etc.)
        if re.match(r'^\d+\.', text):
            level = len(text.split('.')[0]) 
            return f"level_{level}"
        return "title_main"
    elif element_type == 'ListItem':
        return "list_item"
    else:
        return "body"

def extract_section_info(element):
    """Extract section information from element"""
    text = element.get('text', '')
    element_type = element.get('type')
    
    # If it's a numbered section title
    if element_type == 'Title' and re.match(r'^\d+\.', text):
        return text.split('.')[0] + "." + text.split('.')[1].strip() if '.' in text else text
    
    # If it's a ListItem that looks like a section
    if element_type == 'ListItem' and re.match(r'^\d+\.', text):
        return text
    
    return "general"

def get_chunk_metadata(chunk_elements):
    """Get metadata for a chunk based on its elements"""
    if not chunk_elements:
        return {}
    
    # For now, return basic metadata
    # In a more complex implementation, you could analyze all elements in the chunk
    return {
        'chunk_size': len(' '.join(chunk_elements)),
        'element_count': len(chunk_elements),
        'chunk_type': 'mixed'
    }

def extract_enhanced_metadata(element):
    """Extract and enrich metadata from each element"""
    metadata = element.get('metadata', {})
    
    enhanced_metadata = {
        'element_id': element.get('element_id'),
        'element_type': element.get('type'),
        'page_number': metadata.get('page_number'),
        'parent_id': metadata.get('parent_id'),
        'text_length': len(element.get('text', '')),
        'is_header': element.get('type') in ['Title'],
        'is_list_item': element.get('type') == 'ListItem',
        'is_table_content': element.get('type') in ['Table', 'TableRow'],
        'hierarchy_level': determine_hierarchy_level(element),
        'section': extract_section_info(element)
    }
    
    return enhanced_metadata

def preprocess_elements(elements):
    """Main preprocessing function"""
    processed_elements = []
    
    for element in elements:
        text = element.get('text', '').strip()
        if not text or len(text) < 10:  # Skip very short elements
            continue
            
        # Skip footers and page numbers
        if element.get('type') == 'Footer':
            continue
            
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue
            
        processed_element = {
            'text': cleaned_text,
            'metadata': extract_enhanced_metadata(element),
            'original_element': element
        }
        
        processed_elements.append(processed_element)
    
    return processed_elements

def create_contextual_chunks(processed_elements, max_tokens=400):
    """Create chunks with section context"""
    chunks = []
    current_section = "Introduction"
    current_subsection = ""
    
    for element in processed_elements:
        text = element['text']
        element_type = element['metadata']['element_type']
        
        # Update section tracking
        if element_type == 'Title' and any(char.isdigit() for char in text[:5]):
            current_section = text
            current_subsection = ""
        elif element_type == 'Title':
            current_subsection = text
        
        # Create chunk with context
        chunk_metadata = {
            **element['metadata'],
            'section': current_section,
            'subsection': current_subsection,
            'document_type': 'admission_notice',
            'language': 'portuguese'
        }
        
        chunks.append({
            'text': text,
            'metadata': chunk_metadata
        })
    
    return chunks

def optimize_chunks(chunks, target_size=300):
    """Optimize chunk sizes while preserving semantic meaning"""
    optimized = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i]
        current_text = current_chunk['text']
        current_metadata = current_chunk['metadata']
        
        # Try to combine with next chunks if they're in the same section
        # and the combined size doesn't exceed limits
        j = i + 1
        while (j < len(chunks) and 
               len(current_text) < target_size and
               chunks[j]['metadata']['section'] == current_metadata['section']):
            
            combined_text = current_text + " " + chunks[j]['text']
            if len(combined_text) <= target_size * 1.5:  # Allow some flexibility
                current_text = combined_text
                j += 1
            else:
                break
        
        optimized.append({
            'text': current_text,
            'metadata': current_metadata
        })
        
        i = j
    
    return optimized

def create_semantic_chunks(elements, max_chunk_size=512):
    """Create chunks that respect document structure - Alternative chunking method"""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for element in elements:
        element_text = element.get('text', '').strip()
        if not element_text:
            continue
            
        element_type = element.get('type')
        
        # Start new chunk for major sections
        if element_type in ['Title'] and current_chunk:
            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'metadata': get_chunk_metadata(current_chunk)
                })
            current_chunk = [element_text]
            current_size = len(element_text)
        else:
            # Check if adding this element would exceed size limit
            if current_size + len(element_text) > max_chunk_size and current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'metadata': get_chunk_metadata(current_chunk)
                })
                current_chunk = [element_text]
                current_size = len(element_text)
            else:
                current_chunk.append(element_text)
                current_size += len(element_text)
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk),
            'metadata': get_chunk_metadata(current_chunk)
        })
    
    return chunks

def preprocess_for_embedding(json_file_path):
    """Complete preprocessing pipeline"""
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    
    print(f"Loaded {len(elements)} elements from JSON")
    
    # Step 1: Basic preprocessing
    processed_elements = preprocess_elements(elements)
    print(f"Preprocessed {len(processed_elements)} elements")
    
    # Step 2: Create contextual chunks
    chunks = create_contextual_chunks(processed_elements)
    print(f"Created {len(chunks)} initial chunks")
    
    # Step 3: Combine related chunks if needed
    optimized_chunks = optimize_chunks(chunks)
    print(f"Optimized to {len(optimized_chunks)} chunks")
    
    # Step 4: Prepare for embedding
    embedding_ready_chunks = []
    for i, chunk in enumerate(optimized_chunks):
        embedding_ready_chunks.append({
            'id': f"chunk_{i}",
            'content': chunk['text'],
            'metadata': chunk['metadata']
        })
    
    return embedding_ready_chunks

def chuncks_embedding(elemtents):
    pass

def main():
    try:
        # Step 1: Extract elements (only if JSON doesn't exist)
        json_output_path = f"{file_path}/{base_file_name}-output.json"
        
        # Check if JSON already exists
        try:
            with open(json_output_path, 'r') as f:
                pass
            print("JSON file already exists, skipping PDF extraction")
        except FileNotFoundError:
            print("Extracting elements from PDF...")
            elements = partition_pdf(filename=f"{file_path}/{base_file_name}.pdf")
            elements_to_json(elements=elements, filename=json_output_path)
            print("PDF extraction completed")
        
        # Step 2: Preprocess for embedding
        print("Starting preprocessing pipeline...")
        embedding_chunks = preprocess_for_embedding(json_output_path)
        
        # Step 3: Save preprocessed chunks
        chunks_output_path = f"{file_path}/{base_file_name}-chunks.json"
        with open(chunks_output_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Created {len(embedding_chunks)} chunks ready for embedding")
        print(f"✅ Saved chunks to: {chunks_output_path}")
        
        # Step 4: Preview chunks
        print("\n--- CHUNK PREVIEW ---")
        for i, chunk in enumerate(embedding_chunks[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {chunk['id']}")
            print(f"Content: {chunk['content'][:200]}...")
            print(f"Content Length: {len(chunk['content'])}")
            print(f"Section: {chunk['metadata'].get('section', 'N/A')}")
            print(f"Element Type: {chunk['metadata'].get('element_type', 'N/A')}")
            print(f"Page: {chunk['metadata'].get('page_number', 'N/A')}")
        
        # Step 5: Show statistics
        print(f"\n--- STATISTICS ---")
        total_chars = sum(len(chunk['content']) for chunk in embedding_chunks)
        avg_chunk_size = total_chars / len(embedding_chunks)
        print(f"Total chunks: {len(embedding_chunks)}")
        print(f"Total characters: {total_chars}")
        print(f"Average chunk size: {avg_chunk_size:.1f} characters")
        
        # Show section distribution
        sections = {}
        for chunk in embedding_chunks:
            section = chunk['metadata'].get('section', 'Unknown')
            sections[section] = sections.get(section, 0) + 1
        
        print(f"Section distribution:")
        for section, count in sorted(sections.items()):
            print(f"  {section}: {count} chunks")
            
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    # convert the document to markdown
import pymupdf4llm
md_text = pymupdf4llm.to_markdown("input.pdf")

# Write the text to some file in UTF8-encoding
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())