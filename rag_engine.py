
from pathlib import Path

import json
import re

import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Small free embedding model.
# This is NOT the generative LLM.
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text or ""

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=900,
    overlap=150
):

    text = clean_text(text)

    if not text:

        return []


    chunks = []

    start = 0


    while start < len(text):

        end = min(
            len(text),
            start + chunk_size
        )


        chunk = (

            text[
                start:end
            ]
            .strip()
        )


        if chunk:

            chunks.append(
                chunk
            )


        if end >= len(text):

            break


        start = max(
            0,
            end - overlap
        )


    return chunks


# ============================================================
# LOAD OFFICIAL FORENSIC CORPUS
# ============================================================

def load_corpus(
    knowledge_dir="knowledge"
):

    knowledge_dir = Path(
        knowledge_dir
    )


    manifest_path = (

        knowledge_dir
        /
        "manifest.json"
    )


    manifest = json.loads(

        manifest_path
        .read_text(
            encoding="utf-8"
        )
    )


    documents = []


    for source in manifest["sources"]:

        path = (

            knowledge_dir
            /
            source["file"]
        )


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        if (
            path.suffix.lower()
            ==
            ".pdf"
        ):

            reader = PdfReader(
                str(path)
            )


            for page_number, page in enumerate(

                reader.pages,

                start=1
            ):

                text = clean_text(

                    page.extract_text()
                )


                chunks = chunk_text(
                    text
                )


                for chunk_number, chunk in enumerate(

                    chunks,

                    start=1
                ):

                    documents.append({

                        "text":
                            chunk,

                        "source_id":
                            source["id"],

                        "title":
                            source["title"],

                        "authority":
                            source["authority"],

                        "url":
                            source["url"],

                        "sha256":
                            source.get(
                                "sha256",
                                ""
                            ),

                        "page":
                            page_number,

                        "chunk":
                            chunk_number
                    })


        # ----------------------------------------------------
        # WEB SNAPSHOT TEXT
        # ----------------------------------------------------

        elif (
            path.suffix.lower()
            ==
            ".txt"
        ):

            text = path.read_text(

                encoding="utf-8",

                errors="ignore"
            )


            chunks = chunk_text(
                text
            )


            for chunk_number, chunk in enumerate(

                chunks,

                start=1
            ):

                documents.append({

                    "text":
                        chunk,

                    "source_id":
                        source["id"],

                    "title":
                        source["title"],

                    "authority":
                        source["authority"],

                    "url":
                        source["url"],

                    "sha256":
                        source.get(
                            "sha256",
                            ""
                        ),

                    "page":
                        None,

                    "chunk":
                        chunk_number
                })


    if not documents:

        raise RuntimeError(

            "No forensic knowledge "
            "could be extracted."
        )


    return documents


# ============================================================
# FORENSIC RAG
# ============================================================

class ForensicRAG:

    def __init__(
        self,
        knowledge_dir="knowledge"
    ):

        # Load chunks
        self.documents = load_corpus(
            knowledge_dir
        )


        # Load embedding model
        self.encoder = (

            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )


        texts = [

            document["text"]

            for document
            in self.documents
        ]


        # Convert chunks to vectors
        embeddings = (

            self.encoder.encode(

                texts,

                convert_to_numpy=True,

                show_progress_bar=False
            )
        )


        embeddings = np.asarray(

            embeddings,

            dtype="float32"
        )


        # Normalize for cosine similarity
        faiss.normalize_L2(
            embeddings
        )


        # Create vector index
        self.index = faiss.IndexFlatIP(

            embeddings.shape[1]
        )


        self.index.add(
            embeddings
        )


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def search(
        self,
        query,
        k=7
    ):

        query_embedding = (

            self.encoder.encode(

                [query],

                convert_to_numpy=True,

                show_progress_bar=False
            )
        )


        query_embedding = np.asarray(

            query_embedding,

            dtype="float32"
        )


        faiss.normalize_L2(
            query_embedding
        )


        scores, indices = (

            self.index.search(

                query_embedding,

                min(
                    k,
                    len(
                        self.documents
                    )
                )
            )
        )


        results = []


        for score, index in zip(

            scores[0],

            indices[0]
        ):

            if index < 0:

                continue


            document = dict(

                self.documents[
                    int(index)
                ]
            )


            document["score"] = float(
                score
            )


            results.append(
                document
            )


        return results


# ============================================================
# CREATE CITABLE CONTEXT FOR LLM
# ============================================================

def make_context(results):

    context_blocks = []


    for number, result in enumerate(

        results,

        start=1
    ):

        page_text = ""


        if result.get("page"):

            page_text = (

                f", page "
                f"{result['page']}"
            )


        block = f"""
[S{number}]
SOURCE:
{result["title"]}{page_text}

AUTHORITY:
{result["authority"]}

OFFICIAL URL:
{result["url"]}

EXCERPT:
{result["text"]}
"""


        context_blocks.append(
            block.strip()
        )


    return "\n\n".join(
        context_blocks
    )
