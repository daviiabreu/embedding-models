#!/usr/bin/env python3
"""Create indexes on Qdrant collection for metadata filtering."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# Load environment from root .env
load_dotenv(".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "inteli-documents-embeddings")

print("=" * 60)
print("CREATING QDRANT INDEXES FOR METADATA FILTERING")
print("=" * 60)
print(f"\nCollection: {COLLECTION}")
print(f"URL: {QDRANT_URL}")

# Connect to Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Fields to index for filtering
fields_to_index = [
    ("metadata.project", PayloadSchemaType.KEYWORD),
    ("metadata.academic_year", PayloadSchemaType.KEYWORD),
    ("metadata.section", PayloadSchemaType.KEYWORD),
    ("metadata.subsection", PayloadSchemaType.KEYWORD),
    ("metadata.category", PayloadSchemaType.KEYWORD),
    ("metadata.source", PayloadSchemaType.KEYWORD),
    ("metadata.professor_name", PayloadSchemaType.KEYWORD),  # For professor searches
    ("metadata.course", PayloadSchemaType.KEYWORD),  # For course-specific queries
]

print("\nCreating indexes...")
for field_name, field_type in fields_to_index:
    try:
        print(f"\n  Creating index: {field_name} ({field_type})")
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field_name,
            field_schema=field_type,
        )
        print(f"  ✅ Index created successfully")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  ℹ️  Index already exists (skipping)")
        else:
            print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("Checking collection info...")
print("=" * 60)

try:
    collection_info = client.get_collection(collection_name=COLLECTION)
    print(f"\nVectors config: {collection_info.config.params.vectors}")
    print(f"Points count: {collection_info.points_count}")
    print(f"\nPayload schema:")
    if collection_info.payload_schema:
        for field, schema in collection_info.payload_schema.items():
            print(f"  - {field}: {schema}")
    else:
        print("  (No payload schema defined)")
except Exception as e:
    print(f"Error getting collection info: {e}")

print("\n" + "=" * 60)
print("INDEXES CREATED SUCCESSFULLY")
print("=" * 60)
print("\nYou can now use metadata filters like:")
print('  {"project": "Projeto 7"}')
print('  {"academic_year": "2º ano"}')
print('  {"category": "web_scraping"}')
print('  {"professor_name": "Bryan Kano"}')
print('  {"course": "adm-tech"}')
