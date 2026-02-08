"""
Search API - Paper RAG and Google Search grounding.

Requires Vertex AI mode with appropriate resources configured.
"""

import asyncio
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage
from google.genai.types import GenerateContentConfig, GoogleSearch, Retrieval, Tool, VertexAISearch
from pydantic import BaseModel
from services.vertex_client import get_vertex_client

router = APIRouter()


class PaperSearchRequest(BaseModel):
    """Request to search research papers."""

    query: str


class PaperResult(BaseModel):
    """A research paper result."""

    title: str
    authors: str | None = None
    abstract: str | None = None
    url: str | None = None
    relevance_score: float | None = None


class PaperSearchResponse(BaseModel):
    """Response from paper search."""

    query: str
    papers: list[PaperResult]
    summary: str


class WebSearchRequest(BaseModel):
    """Request to search the web."""

    query: str


class WebResult(BaseModel):
    """A web search result."""

    title: str
    url: str
    snippet: str


class WebSearchResponse(BaseModel):
    """Response from web search."""

    query: str
    results: list[WebResult]
    summary: str


@router.post("/papers", response_model=PaperSearchResponse)
async def search_papers(request: PaperSearchRequest) -> PaperSearchResponse:
    """
    Search AI safety research papers using Vertex AI Search RAG.

    Requires:
    - Vertex AI mode
    - VERTEX_SEARCH_DATASTORE_ID configured
    - Papers indexed in the data store
    """
    client = get_vertex_client()

    if not client.capabilities.paper_rag:
        raise HTTPException(
            status_code=400,
            detail="Paper RAG not available. Requires Vertex AI mode with data store configured.",
        )

    # Get datastore configuration
    datastore_id = os.getenv("VERTEX_SEARCH_DATASTORE_ID")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not datastore_id or not project_id:
        raise HTTPException(
            status_code=500,
            detail="Data store not properly configured.",
        )

    try:
        # Build RAG tool for Vertex AI Search
        datastore_path = (
            f"projects/{project_id}/locations/global/"
            f"collections/default_collection/dataStores/{datastore_id}"
        )

        tool = Tool(retrieval=Retrieval(vertex_ai_search=VertexAISearch(datastore=datastore_path)))

        # Query with RAG
        response = client.client.models.generate_content(
            model=client.capabilities.chat_model,
            contents=f"""Search for AI safety research papers related to:

Query: {request.query}

Please provide:
1. A list of the most relevant papers with titles, authors, and brief descriptions
2. A summary of how these papers relate to the query

Focus on papers about AI alignment, safety, deception, and emergent behaviors.""",
            config=GenerateContentConfig(tools=[tool]),
        )

        response_text = response.text if hasattr(response, "text") else str(response)

        # Parse response into structured format
        # (In production, you'd use a more robust parser)
        papers = []
        lines = response_text.split("\n")

        current_paper = {}
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("**") and line.endswith("**"):
                if current_paper.get("title"):
                    papers.append(PaperResult(**current_paper))
                current_paper = {"title": line.strip("*").strip()}
            elif "author" in line.lower() and ":" in line:
                current_paper["authors"] = line.split(":", 1)[-1].strip()
            elif line.startswith("http"):
                current_paper["url"] = line

        if current_paper.get("title"):
            papers.append(PaperResult(**current_paper))

        return PaperSearchResponse(
            query=request.query,
            papers=papers[:10],  # Limit to 10 results
            summary=response_text,  # Return full response
        )

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required imports not available: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Paper search failed: {e}",
        ) from e


@router.post("/web", response_model=WebSearchResponse)
async def search_web(request: WebSearchRequest) -> WebSearchResponse:
    """
    Search Google for recent research and information.

    Uses Google Search grounding in Vertex AI.
    Requires Vertex AI mode.
    """
    client = get_vertex_client()

    if not client.capabilities.google_search:
        raise HTTPException(
            status_code=400,
            detail="Google Search not available. Requires Vertex AI mode.",
        )

    try:
        # Use Google Search grounding
        tool = Tool(google_search=GoogleSearch())

        response = client.client.models.generate_content(
            model=client.capabilities.chat_model,
            contents=f"""Search the web for information about:

Query: {request.query}

Please provide:
1. A summary of the key findings (2-3 paragraphs)
2. A list of relevant sources with titles and URLs in markdown format: [Title](URL)

Focus on recent academic papers, research blogs, and authoritative sources.""",
            config=GenerateContentConfig(
                tools=[tool],
                max_output_tokens=4096,
            ),
        )

        # Get full response text from all parts
        response_text = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts or []:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

        if not response_text:
            response_text = response.text if hasattr(response, "text") else str(response)

        # Parse response into structured format
        results = []
        lines = response_text.split("\n")

        for raw_line in lines:
            line = raw_line.strip()
            # Look for URLs and extract context
            if "http" in line:
                # Extract URL
                urls = re.findall(r"https?://[^\s\)]+", line)
                for url in urls:
                    # Try to extract title (text before URL or after)
                    title = line.split(url)[0].strip(" -[]()").strip()
                    if not title:
                        title = url.split("/")[-1][:50]

                    results.append(
                        WebResult(
                            title=title or "Web Result",
                            url=url,
                            snippet=line[:200],
                        )
                    )

        return WebSearchResponse(
            query=request.query,
            results=results[:10],  # Limit to 10 results
            summary=response_text,  # Return full response
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web search failed: {e}",
        ) from e


class ResearchRequest(BaseModel):
    """Request for comprehensive research combining papers and web."""

    query: str
    include_papers: bool = True
    include_web: bool = True


class ResearchResponse(BaseModel):
    """Combined research response from multiple sources."""

    query: str
    papers: list[PaperResult]
    web_results: list[WebResult]
    synthesis: str
    sources_used: list[str]


@router.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    Comprehensive research combining Paper RAG and Google Search.

    Queries both indexed papers (if available) and live web search,
    then synthesizes the findings into a unified response.
    """
    client = get_vertex_client()
    papers: list[PaperResult] = []
    web_results: list[WebResult] = []
    sources_used: list[str] = []
    all_context = []

    # Run Paper RAG and Google Search in parallel for faster response
    tasks = []
    if request.include_papers and client.capabilities.paper_rag:
        tasks.append(("papers", search_papers(PaperSearchRequest(query=request.query))))
    if request.include_web and client.capabilities.google_search:
        tasks.append(("web", search_web(WebSearchRequest(query=request.query))))

    if tasks:
        # Execute queries in parallel
        results = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True,
        )

        # Process results
        for i, (source_type, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                continue  # Skip failed queries

            if source_type == "papers":
                papers = result.papers
                sources_used.append("paper_rag")
                if result.summary:
                    all_context.append(f"From indexed papers:\n{result.summary}")
            elif source_type == "web":
                web_results = result.results
                sources_used.append("google_search")
                if result.summary:
                    all_context.append(f"From web search:\n{result.summary}")

    # Synthesize findings if we have context from multiple sources
    synthesis = ""
    if len(all_context) > 1:
        try:
            synth_response = client.client.models.generate_content(
                model=client.capabilities.chat_model,
                contents=f"""Synthesize these research findings into a coherent summary:

Query: {request.query}

{chr(10).join(all_context)}

Provide a 2-3 paragraph synthesis that:
1. Identifies key themes and consensus across sources
2. Notes any contradictions or debates
3. Highlights the most important findings for AI alignment research""",
                config=GenerateContentConfig(max_output_tokens=2048),
            )
            synthesis = (
                synth_response.text if hasattr(synth_response, "text") else str(synth_response)
            )
        except Exception:
            synthesis = "\n\n---\n\n".join(all_context)
    elif all_context:
        synthesis = all_context[0]
    else:
        synthesis = "No research sources available. Configure Paper RAG or enable Vertex AI mode."

    return ResearchResponse(
        query=request.query,
        papers=papers,
        web_results=web_results,
        synthesis=synthesis,
        sources_used=sources_used,
    )


# Curated paper catalog - these are indexed in Vertex AI Search
PAPER_CATALOG = [
    {
        "id": "alignment-faking-anthropic",
        "title": "Alignment Faking in Large Language Models",
        "authors": "Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger (Anthropic)",
        "year": 2024,
        "arxiv": "2412.14093",
        "topic": "Deception & Alignment Faking",
    },
    {
        "id": "sleeper-agents-anthropic",
        "title": "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training",
        "authors": "Evan Hubinger, Carson Denison, Jesse Mu et al. (Anthropic)",
        "year": 2024,
        "arxiv": "2401.05566",
        "topic": "Deception & Backdoors",
    },
    {
        "id": "scheming-apollo-research",
        "title": "Frontier Models are Capable of In-context Scheming",
        "authors": "Apollo Research",
        "year": 2024,
        "arxiv": "2411.08792",
        "topic": "Scheming & Deception",
    },
    {
        "id": "deliberative-alignment-openai",
        "title": "Deliberative Alignment: Reasoning Enables Safer Language Models",
        "authors": "OpenAI",
        "year": 2024,
        "arxiv": "2412.16339",
        "topic": "Safety Training",
    },
    {
        "id": "goal-misgeneralization-deepmind",
        "title": "Goal Misgeneralization in Deep Reinforcement Learning",
        "authors": "DeepMind",
        "year": 2022,
        "arxiv": "2210.01790",
        "topic": "Goal Misgeneralization",
    },
    {
        "id": "specification-gaming",
        "title": "Specification Gaming: The Flip Side of AI Ingenuity",
        "authors": "DeepMind",
        "year": 2020,
        "arxiv": "2004.05867",
        "topic": "Reward Hacking",
    },
    {
        "id": "rt2-vision-language-action",
        "title": "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control",
        "authors": "Google DeepMind",
        "year": 2023,
        "arxiv": "2307.15818",
        "topic": "Embodied AI",
    },
    {
        "id": "safety-gymnasium",
        "title": "Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark",
        "authors": "PKU-Alignment Team",
        "year": 2023,
        "arxiv": "2310.12567",
        "topic": "Safe RL Benchmarks",
    },
    {
        "id": "constrained-policy-optimization",
        "title": "Constrained Policy Optimization",
        "authors": "Berkeley AI Research",
        "year": 2017,
        "arxiv": "1705.10528",
        "topic": "Safe RL Methods",
    },
    {
        "id": "representation-engineering",
        "title": "Representation Engineering: A Top-Down Approach to AI Transparency",
        "authors": "Anthropic",
        "year": 2023,
        "arxiv": "2310.01405",
        "topic": "Interpretability",
    },
    {
        "id": "concrete-problems-ai-safety",
        "title": "Concrete Problems in AI Safety",
        "authors": "Amodei, Olah, et al. (Google Brain, OpenAI)",
        "year": 2016,
        "arxiv": "1606.06565",
        "topic": "AI Safety Foundations",
    },
]


@router.get("/papers/catalog")
async def get_paper_catalog() -> dict[str, Any]:
    """
    Get the catalog of papers indexed in the Paper RAG database.

    Returns the full list of curated AI safety papers available for search.
    """
    return {
        "total_papers": len(PAPER_CATALOG),
        "papers": PAPER_CATALOG,
        "topics": list({p["topic"] for p in PAPER_CATALOG}),
        "years": sorted({p["year"] for p in PAPER_CATALOG}),
    }


class AddPaperRequest(BaseModel):
    """Request to add a paper to the database."""

    arxiv_id: str | None = None
    url: str | None = None
    title: str
    authors: str | None = None
    topic: str | None = None


class AddPaperResponse(BaseModel):
    """Response from adding a paper."""

    success: bool
    message: str
    paper_id: str | None = None
    gcs_path: str | None = None
    needs_reindex: bool = True


# Track papers pending reindex
PENDING_PAPERS: list[dict] = []


@router.post("/papers/add", response_model=AddPaperResponse)
async def add_paper_to_database(request: AddPaperRequest) -> AddPaperResponse:
    """
    Download and add a paper to the Paper RAG database.

    The paper will be downloaded to GCS. Note: Vertex AI Search requires
    manual re-import to index new papers (or use the /papers/reindex endpoint).
    """
    # Determine download URL
    # Document IDs must match pattern: [a-zA-Z0-9-_]*
    def sanitize_id(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "-", s).strip("-")

    if request.arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{request.arxiv_id}.pdf"
        arxiv_clean = sanitize_id(request.arxiv_id)
        title_clean = sanitize_id(request.title.lower()[:30])
        paper_id = f"{arxiv_clean}-{title_clean}"
    elif request.url:
        pdf_url = request.url
        paper_id = sanitize_id(request.title.lower()[:50])
    else:
        return AddPaperResponse(
            success=False,
            message="Either arxiv_id or url must be provided",
        )

    # Download the PDF
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            pdf_content = response.content
    except Exception as e:
        return AddPaperResponse(
            success=False,
            message=f"Failed to download paper: {e}",
        )

    # Upload to GCS
    bucket_name = "g1-ai-safety-papers"
    blob_name = f"{paper_id}.pdf"

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pdf_content, content_type="application/pdf")
        gcs_path = f"gs://{bucket_name}/{blob_name}"
    except Exception as e:
        return AddPaperResponse(
            success=False,
            message=f"Failed to upload to GCS: {e}",
        )

    # Trigger incremental import to Vertex AI Search
    reindex_triggered = False
    reindex_message = ""

    datastore_id = os.getenv("VERTEX_SEARCH_DATASTORE_ID")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if datastore_id and project_id:
        try:
            # Trigger document import for the new file
            parent = (
                f"projects/{project_id}/locations/global/"
                f"collections/default_collection/dataStores/{datastore_id}/branches/default_branch"
            )

            # Import the single document
            import_client = discoveryengine.DocumentServiceClient()

            # Create the document with content from GCS
            document = discoveryengine.Document(
                id=paper_id,
                json_data=f'{{"title": "{request.title}", "authors": "{request.authors or ""}", "topic": "{request.topic or ""}"}}',
            )
            document.content = discoveryengine.Document.Content(
                mime_type="application/pdf",
                uri=gcs_path,
            )

            # Use create_document for immediate indexing
            import_client.create_document(
                parent=parent,
                document=document,
                document_id=paper_id,
            )

            reindex_triggered = True
            reindex_message = "Paper indexed and will be searchable shortly."
        except Exception as e:
            reindex_message = (
                f"Upload successful but auto-index failed: {e}. Manual reindex may be needed."
            )
            # Still add to pending list for tracking
            PENDING_PAPERS.append(
                {
                    "id": paper_id,
                    "title": request.title,
                    "authors": request.authors,
                    "topic": request.topic or "Uncategorized",
                    "arxiv": request.arxiv_id,
                    "gcs_path": gcs_path,
                }
            )
    else:
        reindex_message = "Datastore not configured. Paper saved but not indexed."
        PENDING_PAPERS.append(
            {
                "id": paper_id,
                "title": request.title,
                "gcs_path": gcs_path,
            }
        )

    return AddPaperResponse(
        success=True,
        message=f"Paper '{request.title}' added. {reindex_message}",
        paper_id=paper_id,
        gcs_path=gcs_path,
        needs_reindex=not reindex_triggered,
    )


@router.get("/papers/pending")
async def get_pending_papers() -> dict[str, Any]:
    """Get papers that have been added but not yet indexed."""
    return {
        "pending_count": len(PENDING_PAPERS),
        "papers": PENDING_PAPERS,
        "message": "These papers are in GCS but need reindexing to be searchable.",
    }


@router.get("/status")
async def get_search_status() -> dict[str, Any]:
    """
    Get the status of search capabilities.

    Returns which search features are available.
    """
    client = get_vertex_client()
    caps = client.capabilities

    return {
        "paper_rag": {
            "available": caps.paper_rag,
            "datastore_id": os.getenv("VERTEX_SEARCH_DATASTORE_ID", ""),
        },
        "google_search": {
            "available": caps.google_search,
        },
        "mode": caps.mode.value,
    }
