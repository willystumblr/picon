# google_claim_search.py
import json
import re
from typing import Optional, List, Dict, Any
import pdfplumber
from io import BytesIO
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field
import requests
from rank_bm25 import BM25Okapi
import cloudscraper

TOP_K_RESULTS = 5         # how many search results to fetch (for fallback)
MAX_SUCCESSFUL_PAGES = 1  # how many successfully fetched pages to return
_PAT = re.compile(r"(content|main|article|body|post)", re.I)
_SPLIT_RE = re.compile(r"\n{2,}")          # paragraph boundary = ≥2 new-lines
_TOKEN_RE = re.compile(r"\w+")

def _candidate_blocks(soup: BeautifulSoup) -> List[BeautifulSoup]:
    """Return potential ‘main content’ elements in priority order."""
    # ① semantic tags
    blocks = soup.find_all(["main", "article"])
    if blocks:
        return blocks

    # ② id/class pattern match (limit scan to first ~40 nodes for speed)
    for tag in soup.find_all(["div", "section"], limit=40):
        ident = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
        if _PAT.search(ident):
            blocks.append(tag)
    return blocks


def _biggest_text_container(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Pick the best container for the article body.
    """
    total_len = len(soup.get_text(" ", strip=True))
    best, best_len = None, 0

    for cand in _candidate_blocks(soup):
        txt_len = len(cand.get_text(" ", strip=True))
        if txt_len > best_len:
            best, best_len = cand, txt_len

    # Fallback: largest <div>/<section> if patterns failed
    if not best:
        for cand in soup.find_all(["div", "section"], limit=30):
            txt_len = len(cand.get_text(" ", strip=True))
            if txt_len > best_len:
                best, best_len = cand, txt_len

    # sanity: only accept if it’s at least 25 % of whole-page text
    if best and best_len > 0.25 * total_len:
        return best
    return None


def _clean_html(html: str) -> str:
    """
    Extract the main article text as **Markdown** with *no* heavy dependencies.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 🔹 1. Drop obvious non-content nodes
    for tag in soup(["script", "style", "noscript", "header",
                     "footer", "nav", "aside", "iframe"]):
        tag.decompose()

    # 🔹 2. Choose the core container
    container = _biggest_text_container(soup) or soup.body or soup

    # 🔹 3. Convert to Markdown
    markdown = md(
        str(container),
        strip=["img", "iframe", "script", "style"],
        heading_style="ATX",
        bullets="*",
    )

    # 🔹 4. Whitespace tidy-up
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]{2,}", " ", markdown).strip()

    return markdown or "[content-extraction-failed]"


def _split_passages(text: str,
                    max_chars: int = 3000,
                    min_chars: int = 600) -> List[str]:
    """
    Split Markdown into ~paragraph-sized passages.

    1. First split on double-newline (natural paras).
    2. Merge consecutive short paras until >= min_chars.
    3. Hard-split any monster chunk longer than max_chars.
    """
    #print("text: ", text)
    
    parts, current, passages = _SPLIT_RE.split(text), [], []
    def flush():
        if current:
            chunk = "\n\n".join(current).strip()
            while len(chunk) > max_chars:   # hard split long tail
                passages.append(chunk[:max_chars])
                chunk = chunk[max_chars:]
            if chunk:
                passages.append(chunk)
            current.clear()

    for para in parts:
        if len(" ".join(current) + para) < min_chars:
            current.append(para)
        else:
            current.append(para)
            flush()
    flush()
    #print("passage: ", passages)
    #print()
    return passages

def _tokenize(txt: str) -> List[str]:
    return _TOKEN_RE.findall(txt.lower())


def _top_passages(claim: str,
                  passages: List[str],
                  k: int = 3) -> List[str]:
    if not passages:
        return []
    
    tokenized = [_tokenize(p) for p in passages]
    if all(len(toks) == 0 for toks in tokenized):
        return passages[:k]  # fallback

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(claim))
    ranked = sorted(zip(scores, passages), reverse=True)[:k]
    # keep only passages with non-zero score; fall back to first k otherwise
    picked = [p for s, p in ranked if s > 0] or passages[:k]
    #print('picked : ', picked)
    return picked

class GoogleClaimSearch(BaseModel):
    """
    LiteLLM-compatible tool:
    given a claim, return plain-text evidence passages from Google
    """
    api_key: str = Field(..., description="Google Custom Search API key")
    cx: str = Field(..., description="Google Programmable Search Engine ID")
    tool_call_counts: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _fetch(self, url: str) -> Dict[str, Any]:
        try:
            scraper = cloudscraper.create_scraper(
                browser={
                    "browser": "chrome",
                    "platform": "windows",
                    "mobile": False
                }
            )
            page = scraper.get(url, headers={"User-Agent": "Mozilla/5.0"})
            page.raise_for_status()
            
            # Check if the content is a PDF
            content_type = page.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')
            
            if is_pdf:
                # Extract text from PDF
                try:
                    with pdfplumber.open(BytesIO(page.content)) as pdf:
                        full_text = ""
                        for page_obj in pdf.pages:
                            text = page_obj.extract_text()
                            if text:
                                full_text += text + "\n"
                        
                        # Use filename or URL as title for PDFs
                        title = url.split('/')[-1] if '/' in url else "PDF Document"
                        cleaned = full_text.strip()
                        return {"title": title, "cleaned": cleaned}
                except Exception as pdf_error:
                    return {"title": "", "error": f"[Error extracting PDF] {pdf_error}"}
            else:
                # Handle HTML content
                soup = BeautifulSoup(page.text, "html.parser")
                title = (soup.title.string or "").strip() if soup.title else ""
                cleaned = _clean_html(page.text)
                return {"title": title, "cleaned": cleaned}
        except Exception as e:
            return {"title": "", "error": f"[Error fetching] {e}"}
    
    # ------------- tool entry point -------------
    def invoke_single(self, claim: str, q: str, gl: str, exactTerms: None) -> str:
        """
        Parameters
        ----------
        claim : str
            A factual claim / statement in natural language.
        
        q: str
            A keyword that needs to be searched for fact verification.
            This is usually a word/phrase within the claim.

        Returns
        -------
        str
            JSON stringified list[str] – each element is the cleaned text
            of a search-result page.  Errors become single-element lists.
        """
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            q_params = {
                "q": q,
                "key": self.api_key,
                "cx": self.cx,
                "num": TOP_K_RESULTS, # top 5 results
                "gl": gl,             # geolocation
            }
            if exactTerms:
                q_params["exactTerms"] = exactTerms
            resp = requests.get(search_url, params=q_params, timeout=6)
            resp.raise_for_status()
            self.tool_call_counts += 1
            items = resp.json().get("items", [])

            results: List[Dict[str, Any]] = []
            failed_attempts: List[Dict[str, str]] = []  # Track failed URLs for debugging
            successful_count = 0
            
            for it in items:
                # Stop if we have enough successful results
                if successful_count >= MAX_SUCCESSFUL_PAGES:
                    break
                    
                url = it.get("link")
                if not url:
                    continue

                fetched = self._fetch(url)
                if "error" in fetched:
                    # Don't return error immediately, track it and try next URL
                    failed_attempts.append({"url": url, "error": fetched["error"]})
                    continue
                    
                # Treat '[content-extraction-failed]' as a failure
                if fetched["cleaned"].strip() == "[content-extraction-failed]":
                    failed_attempts.append({"url": url, "error": "[content-extraction-failed]"})
                    continue

                passages = _split_passages(fetched["cleaned"])
                if not passages or all(not p.strip() for p in passages):
                    # No meaningful content extracted, try next URL
                    failed_attempts.append({"url": url, "error": "No meaningful content extracted"})
                    continue

                top_passages = _top_passages(claim, passages)
                if not top_passages or all(not p.strip() for p in top_passages):
                    # No relevant passages found, try next URL
                    failed_attempts.append({"url": url, "error": "No relevant passages found"})
                    continue

                results.append({
                    'query': q,
                    "title": fetched["title"],
                    "link": url,
                    "gl": gl,
                    "text_block": top_passages
                })
                successful_count += 1
                
            # Only return error if ALL URLs failed
            if not results:
                error_details = "; ".join([f"{fa['url']}: {fa['error']}" for fa in failed_attempts[:3]])
                results = [{
                    "query": q,
                    "title": "",
                    "link": "",
                    "gl": gl,
                    "text_block": [f"Failed to extract text from all {len(failed_attempts)} URLs tried. Details: {error_details}"]
                }]
            return json.dumps(results, ensure_ascii=False)

        except Exception as outer:
            return json.dumps([{"query" : q, "exactTerms": "", "title": "", "link":"", "gl" : gl, "text_block":f"Search failure: {outer}"}], ensure_ascii=False)

    # ------------- schema that LiteLLM exports -------------
    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "google_claim_search",
                "description": (
                    "Given a factual `claim`, run Google Custom Search with the query (keyword) `q`, `gl`, and `exactTerms` (optional), "
                    f"crawl search result pages (trying up to {TOP_K_RESULTS} URLs with fallback on failure), "
                    "and return extracted plain texts as a JSON string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Query; a keyword that needs to be searched for fact verification. Do not include quotation marks.",
                        },
                        "gl": {
                            "type": "string",
                            "description": "Geolocation of end user. The country code (e.g., 'us', 'uk', 'ca', 'jp', 'kr') to tailor search results to a specific region.",
                        },
                        "exactTerms": {
                            "type": "string",
                            "description": "Exact terms to match in the search results for fact verification. Do not include quotation marks."
                        },
                    },
                    "required": ["q", "gl", "exactTerms", ],
                },
            },
        }
        
    def invoke(self, claims: List[str], q: str, gl: str, exactTerms: str = None) -> str:
        """
        Process multiple claims with the same query and geolocation.
        
        Parameters
        ----------
        claims : List[str]
            A list of factual claims / statements in natural language.
        
        q: str
            A keyword that needs to be searched for fact verification.

        gl: str
            Geolocation country code.

        Returns
        -------
        str
            JSON stringified dict mapping each claim to its relevant passages.
        """
        try:
            search_url = "https://www.googleapis.com/customsearch/v1"
            q_params = {
                "q": q,
                "key": self.api_key,
                "cx": self.cx,
                "num": TOP_K_RESULTS,
                "gl": gl,
                "exactTerms": exactTerms if exactTerms else "",
            }
            resp = requests.get(search_url, params=q_params, timeout=6)
            resp.raise_for_status()
            self.tool_call_counts += 1
            items = resp.json().get("items", [])

            # Fetch pages with fallback - try URLs until we get content
            all_passages: List[str] = []
            page_info: List[Dict[str, Any]] = []
            failed_attempts: List[Dict[str, str]] = []
            successful_count = 0
            
            for it in items:
                # Stop if we have enough successful results
                if successful_count >= MAX_SUCCESSFUL_PAGES:
                    break
                    
                url = it.get("link")
                if not url:
                    continue

                fetched = self._fetch(url)
                if "error" in fetched:
                    # Track failure and try next URL
                    failed_attempts.append({"url": url, "error": fetched["error"]})
                    continue
                    
                # Treat '[content-extraction-failed]' as a failure
                if fetched["cleaned"].strip() == "[content-extraction-failed]":
                    failed_attempts.append({"url": url, "error": "[content-extraction-failed]"})
                    continue

                passages = _split_passages(fetched["cleaned"])
                if not passages or all(not p.strip() for p in passages):
                    # No meaningful content, try next URL
                    failed_attempts.append({"url": url, "error": "No meaningful content extracted"})
                    continue

                all_passages.extend(passages)
                page_info.append({"title": fetched["title"], "link": url})
                successful_count += 1

            # For each claim, find the most relevant passages
            passages_set = []
            if not all_passages:
                error_details = "; ".join([f"{fa['url']}: {fa['error']}" for fa in failed_attempts[:3]])
                results = [{
                        "query": q,
                        "gl": gl,
                        "pages": page_info,
                        "failed_attempts": len(failed_attempts),
                        "text_block": [f"Failed to extract text from all {len(failed_attempts)} URLs tried. Details: {error_details}"]
                    }]
            else:
                for claim in claims:
                    top_passages = _top_passages(claim, all_passages)
                    passages_set.extend(top_passages)
                    passages_set = list(set(passages_set))  # remove duplicates
                results =[{
                    "query": q,
                    "gl": gl,
                    "pages": page_info,
                    "text_block": passages_set
                }]        
                   

            return json.dumps(results, ensure_ascii=False)
        except Exception as outer:
            error_result = [{"query": q, "gl": gl, "text_block": f"Search failure: {outer}"}]
            return json.dumps(error_result, ensure_ascii=False)

    def calculate_cost(self) -> float:
        # Custom Search API costs
        # 100 queries per day are free
        # $5 per 1000 queries thereafter
        free_quota = 100
        cost_per_1000 = 5.0
        billable_calls = max(0, self.tool_call_counts - free_quota)
        total_cost = (billable_calls / 1000) * cost_per_1000
        return total_cost