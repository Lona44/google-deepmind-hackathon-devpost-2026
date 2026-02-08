"""
Search API - Paper RAG and Google Search grounding.

Requires Vertex AI mode with appropriate resources configured.
"""

import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException
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

    # Query Paper RAG if available and requested
    if request.include_papers and client.capabilities.paper_rag:
        try:
            paper_response = await search_papers(PaperSearchRequest(query=request.query))
            papers = paper_response.papers
            sources_used.append("paper_rag")
            if paper_response.summary:
                all_context.append(f"From indexed papers:\n{paper_response.summary}")
        except HTTPException:
            pass  # Paper RAG not available, continue with web search

    # Query Google Search if available and requested
    if request.include_web and client.capabilities.google_search:
        try:
            web_response = await search_web(WebSearchRequest(query=request.query))
            web_results = web_response.results
            sources_used.append("google_search")
            if web_response.summary:
                all_context.append(f"From web search:\n{web_response.summary}")
        except HTTPException:
            pass  # Google Search not available

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
