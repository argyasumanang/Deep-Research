import streamlit as st
import asyncio
import json
import markdown
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

from pydantic import BaseModel, Field
from together import AsyncTogether
from tavily import AsyncTavilyClient

# Set page config
st.set_page_config(page_title="Deep Research AI", page_icon="📚", layout="wide")

# Preset API keys (dapat diganti oleh pengguna)
DEFAULT_TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")
DEFAULT_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ==========================================
# Data Models
# ==========================================

class ResearchPlan(BaseModel):
    """
    Structured representation of a research plan with search queries.
    """
    queries: list[str] = Field(description="A list of search queries to thoroughly research the topic")

class SourceList(BaseModel):
    """
    Structured representation of filtered source indices.
    """
    sources: list[int] = Field(description="A list of source numbers from the search results")

@dataclass
class SearchResult:
    """
    Container for an individual search result with its metadata and content.
    """
    title: str
    link: str
    content: str
    filtered_raw_content: Optional[str] = None

    def __str__(self):
        """String representation with title, link and refined content."""
        return (
            (f"Title: {self.title}\n" f"Link: {self.link}\n" f"Refined Content: {self.filtered_raw_content}")
            if self.filtered_raw_content
            else (f"Title: {self.title}\n" f"Link: {self.link}\n" f"Raw Content: {self.content[:1000]}")
        )

    def short_str(self):
        """Abbreviated string representation with truncated raw content."""
        return f"Title: {self.title}\nLink: {self.link}\nRaw Content: {self.content[:1000]}"


@dataclass
class SearchResults:
    """
    Collection of search results with utilities for manipulation and display.
    """
    results: list[SearchResult]

    def __str__(self):
        """Detailed string representation of all search results with indices."""
        return "\n\n".join(f"[{i+1}] {str(result)}" for i, result in enumerate(self.results))

    def __add__(self, other):
        """Combine two SearchResults objects by concatenating their result lists."""
        return SearchResults(self.results + other.results)

    def short_str(self):
        """Abbreviated string representation of all search results with indices."""
        return "\n\n".join(f"[{i+1}] {result.short_str()}" for i, result in enumerate(self.results))

    def dedup(self):
        """
        Remove duplicate search results based on URL.
        Returns a new SearchResults object with unique entries.
        """
        def deduplicate_by_link(results):
            seen_links = set()
            unique_results = []

            for result in results:
                if result.link not in seen_links:
                    seen_links.add(result.link)
                    unique_results.append(result)

            return unique_results

        return SearchResults(deduplicate_by_link(self.results))

# ==========================================
# Research Pipeline Functions
# ==========================================

async def generate_initial_queries(topic: str, together_client: AsyncTogether, max_queries: int, planning_model: str, json_model: str, prompts: dict, status_placeholder) -> List[str]:
    """Step 1: Generate initial research queries based on the topic"""
    status_placeholder.write("Generating initial research queries...")
    queries = await generate_research_queries(topic, together_client, planning_model, json_model, prompts, status_placeholder)
    
    if max_queries > 0:
        queries = queries[:max_queries]
    
    status_placeholder.write(f"Generated {len(queries)} initial queries")
    
    if len(queries) == 0:
        status_placeholder.error("ERROR: No initial queries generated")
        return []
    
    return queries

async def generate_research_queries(topic: str, together_client: AsyncTogether, planning_model: str, json_model: str, prompts: dict, status_placeholder) -> list[str]:
    """Generate research queries for a given topic using LLM"""
    PLANNING_PROMPT = prompts["planning_prompt"]
    
    status_placeholder.write("Planning research approach...")
    planning_response = await together_client.chat.completions.create(
        model=planning_model,
        messages=[
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user", "content": f"Research Topic: {topic}"}
        ]
    )
    plan = planning_response.choices[0].message.content
    
    await asyncio.sleep(1.2)  # Avoid rate limiting
    
    status_placeholder.write("Extracting search queries...")
    SEARCH_PROMPT = prompts["plan_parsing_prompt"]
    
    json_response = await together_client.chat.completions.create(
        model=json_model,
        messages=[
            {"role": "system", "content": SEARCH_PROMPT},
            {"role": "user", "content": f"Plan to be parsed: {plan}"}
        ],
        response_format={"type": "json_object", "schema": ResearchPlan.model_json_schema()}
    )
    
    response_json = json_response.choices[0].message.content
    plan = json.loads(response_json)
    return plan["queries"]

async def tavily_search(query: str, tavily_client: AsyncTavilyClient, prompts: dict, together_client: AsyncTogether, summary_model: str, status_placeholder) -> SearchResults:
    """Perform a single Tavily search with rate-limited summarization"""
    status_placeholder.write(f"Searching for: {query}")
    
    response = await tavily_client.search(query, include_raw_content=True)
    result_count = len(response['results'])
    status_placeholder.write(f"Found {result_count} results for query: {query}")
    
    RAW_CONTENT_SUMMARIZER_PROMPT = prompts["raw_content_summarizer_prompt"]
    
    # Create a list of tasks for summarization and store corresponding result info
    summarization_tasks = []
    result_info = []
    for result in response["results"]:
        if result["raw_content"] is None or result["raw_content"].strip() == "":
            continue
        task = summarize_content(result["raw_content"], query, RAW_CONTENT_SUMMARIZER_PROMPT, together_client, summary_model, status_placeholder)
        summarization_tasks.append(task)
        result_info.append(result)
    
    # Execute tasks serially with a delay to avoid rate limits
    summarized_contents = []
    for i, task in enumerate(summarization_tasks):
        try:
            status_placeholder.write(f"Summarizing content {i+1}/{len(summarization_tasks)}...")
            summary = await task
            summarized_contents.append(summary)
        except Exception as e:
            status_placeholder.error(f"Error while summarizing content: {e}")
            summarized_contents.append("Summary not available")
        await asyncio.sleep(1.2)  # Delay to avoid exceeding rate limit (1 QPS)
    
    formatted_results = []
    for result, summarized_content in zip(result_info, summarized_contents):
        formatted_results.append(
            SearchResult(
                title=result["title"],
                link=result["url"],
                content=result["raw_content"],
                filtered_raw_content=summarized_content,
            )
        )
    return SearchResults(formatted_results)

async def summarize_content(raw_content: str, query: str, prompt: str, together_client: AsyncTogether, summary_model: str, status_placeholder) -> str:
    """Summarize content asynchronously using the LLM with error handling"""
    try:
        summarize_response = await together_client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"<Raw Content>{raw_content}</Raw Content>\n\n<Research Topic>{query}</Research Topic>"}
            ]
        )
        return summarize_response.choices[0].message.content
    except Exception as e:
        status_placeholder.error(f"Error in summarize_content: {e}")
        return "Summary not available"

async def perform_search(queries: List[str], tavily_client: AsyncTavilyClient, prompts: dict, together_client: AsyncTogether, summary_model: str, status_placeholder) -> SearchResults:
    """Execute searches for all queries and process results"""
    tasks = [tavily_search(query, tavily_client, prompts, together_client, summary_model, status_placeholder) for query in queries]
    results_list = await asyncio.gather(*tasks)
    
    combined_results = SearchResults([])
    for results in results_list:
        combined_results = combined_results + results
    
    combined_results_dedup = combined_results.dedup()
    status_placeholder.write(f"Search complete, found {len(combined_results_dedup.results)} unique results")
    return combined_results_dedup

async def conduct_iterative_research(topic: str, initial_results: SearchResults, all_queries: List[str],
                                    budget: int, max_queries: int, tavily_client: AsyncTavilyClient, 
                                    together_client: AsyncTogether, planning_model: str, json_model: str, 
                                    summary_model: str, prompts: dict, status_placeholder) -> tuple[SearchResults, List[str]]:
    """
    Conduct iterative research within budget to refine results.
    """
    results = initial_results
    
    for iteration in range(1, budget + 1):
        status_placeholder.write(f"Starting research iteration {iteration}/{budget}...")
        
        # Evaluate if more research is needed
        status_placeholder.write("Evaluating research completeness...")
        additional_queries = await evaluate_research_completeness(
            topic, results, all_queries, together_client, planning_model, json_model, prompts, status_placeholder
        )
        
        # Exit if research is complete
        if not additional_queries:
            status_placeholder.write("Research is complete - no additional queries needed.")
            break
        
        # Limit the number of queries if needed
        if max_queries > 0:
            additional_queries = additional_queries[:max_queries]
        
        status_placeholder.write(f"Generated {len(additional_queries)} additional queries for iteration {iteration}")
        
        # Expand research with new queries
        status_placeholder.write("Searching for additional information...")
        new_results = await perform_search(
            additional_queries,
            tavily_client,
            prompts,
            together_client,
            summary_model,
            status_placeholder
        )
        
        results = results + new_results
        all_queries.extend(additional_queries)
        
    return results, all_queries

async def evaluate_research_completeness(topic: str, results: SearchResults, queries: List[str],
                                     together_client: AsyncTogether, planning_model: str, json_model: str, 
                                     prompts: dict, status_placeholder) -> list[str]:
    """
    Evaluate if the current search results are sufficient or if more research is needed.
    """
    # Format the search results for the LLM
    formatted_results = str(results)
    
    EVALUATION_PROMPT = prompts["evaluation_prompt"]
    
    status_placeholder.write("Analyzing search results for information gaps...")
    evaluation_response = await together_client.chat.completions.create(
        model=planning_model,
        messages=[
            {"role": "system", "content": EVALUATION_PROMPT},
            {"role": "user", "content": (
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Search Queries Used>{queries}</Search Queries Used>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            )}
        ]
    )
    evaluation = evaluation_response.choices[0].message.content
    
    await asyncio.sleep(1.2)  # Avoid rate limiting
    
    EVALUATION_PARSING_PROMPT = prompts["evaluation_parsing_prompt"]
    
    status_placeholder.write("Extracting follow-up queries...")
    json_response = await together_client.chat.completions.create(
        model=json_model,
        messages=[
            {"role": "system", "content": EVALUATION_PARSING_PROMPT},
            {"role": "user", "content": f"Evaluation to be parsed: {evaluation}"}
        ],
        response_format={"type": "json_object", "schema": ResearchPlan.model_json_schema()}
    )
    
    response_json = json_response.choices[0].message.content
    evaluation = json.loads(response_json)
    return evaluation["queries"]

async def filter_results(topic: str, results: SearchResults, together_client: AsyncTogether, 
                      json_model: str, max_sources: int, prompts: dict, status_placeholder) -> tuple[SearchResults, List[int]]:
    """
    Filter and rank search results based on relevance to the research topic.
    """
    # Format the search results for the LLM, without the raw content
    formatted_results = results.short_str()
    
    FILTER_PROMPT = prompts["filter_prompt"]
    SOURCE_PARSING_PROMPT = prompts['source_parsing_prompt']
    
    status_placeholder.write("Evaluating search results for relevance and quality...")
    llm_filter_response = await together_client.chat.completions.create(
        model=json_model,
        messages=[
            {"role": "system", "content": FILTER_PROMPT},
            {"role": "user", "content": (
                f"<Research Topic>{topic}</Research Topic>\n\n"
                f"<Current Search Results>{formatted_results}</Current Search Results>"
            )}
        ]
    )
    
    llm_filter_response_content = llm_filter_response.choices[0].message.content
    
    await asyncio.sleep(1.2)  # Avoid rate limiting
    
    status_placeholder.write("Parsing source rankings...")
    json_response = await together_client.chat.completions.create(
        model=json_model,
        messages=[
            {"role": "system", "content": SOURCE_PARSING_PROMPT},
            {"role": "user", "content": f"<FILTER_RESPONSE>{llm_filter_response_content}</FILTER_RESPONSE>"}
        ],
        response_format={"type": "json_object", "schema": SourceList.model_json_schema()}
    )
    
    response_json = json_response.choices[0].message.content
    evaluation = json.loads(response_json)
    sources = evaluation["sources"]
    
    status_placeholder.write(f"Found {len(sources)} relevant sources")
    
    if max_sources > 0:
        sources = sources[:max_sources]
        status_placeholder.write(f"Using top {len(sources)} sources for report generation")
    
    # Filter the results based on the source list
    filtered_results = [results.results[i] for i in sources if i < len(results.results)]
    
    return SearchResults(filtered_results), sources

async def process_search_results(topic: str, results: SearchResults, together_client: AsyncTogether, 
                              json_model: str, max_sources: int, prompts: dict, status_placeholder) -> SearchResults:
    """Process search results by deduplicating and filtering"""
    # Deduplicate results
    results = results.dedup()
    status_placeholder.write(f"Removed duplicate results, kept {len(results.results)} unique sources")
    
    # Filter results
    filtered_results, sources = await filter_results(topic, results, together_client, json_model, max_sources, prompts, status_placeholder)
    status_placeholder.write(f"Selected {len(filtered_results.results)} high-quality sources for report generation")
    
    return filtered_results

def remove_thinking_tags_from_answer(answer: str) -> str:
    """Remove content within <think> tags"""
    while "<think>" in answer and "</think>" in answer:
        start = answer.find("<think>")
        end = answer.find("</think>") + len("</think>")
        answer = answer[:start] + answer[end:]
    return answer

async def generate_research_answer(topic: str, results: SearchResults, together_client: AsyncTogether, 
                                answer_model: str, prompts: dict, max_tokens: int, 
                                status_placeholder, remove_thinking_tags: bool = True) -> str:
    """
    Generate a comprehensive answer to the research topic based on the search results.
    """
    formatted_results = str(results)
    
    ANSWER_PROMPT = prompts["answer_prompt"]
    
    status_placeholder.write("Generating comprehensive research report...")
    answer_response = await together_client.chat.completions.create(
        model=answer_model,
        messages=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"Research Topic: {topic}\n\nSearch Results:\n{formatted_results}"}
        ],
        max_tokens=max_tokens
    )
    
    answer = answer_response.choices[0].message.content
    
    # Remove <think> tokens for reasoning models
    if remove_thinking_tags:
        answer = remove_thinking_tags_from_answer(answer)
    
    # Handle potential error cases
    if answer is None or not isinstance(answer, str):
        status_placeholder.error("ERROR: No answer generated")
        return "No answer generated"
    
    status_placeholder.write("Research report generated successfully!")
    return answer.strip()

async def run_research_pipeline(research_topic, together_api_key, tavily_api_key, 
                               max_queries, budget, max_sources, max_tokens,
                               status_placeholder, progress_bar):
    """Run the complete research pipeline and return the final report"""
    
    # Initialize API clients
    together_client = AsyncTogether(api_key=together_api_key)
    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    
    # Define models
    planning_model = "Qwen/Qwen2.5-72B-Instruct-Turbo"  # Used for research planning and evaluation
    json_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # Used for structured data parsing
    summary_model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Used for web content summarization
    answer_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"  # Used for final answer synthesis
    
    # Define prompt templates
    prompts = {
        # Planning: Generates initial research queries
        "planning_prompt": """You are a strategic research planner with expertise in breaking down complex
                         questions into logical search steps. Generate focused, specific, and self-contained queries that
                         will yield relevant information for the research topic.""",

        # Plan Parsing: Extracts structured data from planning output
        "plan_parsing_prompt": """Extract search queries that should be executed.""",

        # Content Processing: Identifies relevant information from search results
        "raw_content_summarizer_prompt": """Extract and synthesize only the information relevant to the research
                                       topic from this content. Preserve specific data, terminology, and
                                       context while removing irrelevant information.""",

        # Completeness Evaluation: Determines if more research is needed
        "evaluation_prompt": """Analyze these search results against the original research goal. Identify
                          specific information gaps and generate targeted follow-up queries to fill
                          those gaps. If no significant gaps exist, indicate that research is complete.""",

        # Evaluation Parsing: Extracts structured data from evaluation output
        "evaluation_parsing_prompt": """Extract follow-up search queries from the evaluation. If no follow-up queries are needed, return an empty list.""",

        # Source Filtering: Selects most relevant sources
        "filter_prompt": """Evaluate each search result for relevance, accuracy, and information value
                       related to the research topic. At the end, you need to provide a list of
                       source numbers with the rank of relevance. Remove the irrelevant ones.""",
                       
        # Source Filtering: Selects most relevant sources
        "source_parsing_prompt": """Extract the source list that should be included.""",

        # Answer Generation: Creates final research report
        "answer_prompt": """Create a comprehensive, publication-quality markdown research report based exclusively
                       on the provided sources. The report should include: title, introduction, analysis (multiple sections with insights titles)
                       and conclusions, references. Use proper citations (source with link; using \n\n \[Ref. No.\] to improve format),
                       organize information logically, and synthesize insights across sources. Include all relevant details while
                       maintaining readability and coherence. In each section, You MUST write in plain
                       paragraghs and NEVER describe the content following bullet points or key points (1,2,3,4... or point X: ...)
                       to improve the report readability."""
    }
    
    try:
        # Step 1: Generate initial queries
        progress_bar.progress(10, text="Generating research queries...")
        initial_queries = await generate_initial_queries(
            topic=research_topic,
            together_client=together_client,
            max_queries=max_queries,
            planning_model=planning_model,
            json_model=json_model,
            prompts=prompts,
            status_placeholder=status_placeholder
        )
        
        if not initial_queries:
            return "Failed to generate research queries. Please try again."
        
        # Step 2: Perform initial search
        progress_bar.progress(25, text="Searching for information...")
        initial_results = await perform_search(
            queries=initial_queries, 
            tavily_client=tavily_client,
            prompts=prompts,
            together_client=together_client,
            summary_model=summary_model,
            status_placeholder=status_placeholder
        )
        
        # Step 3: Conduct iterative research
        progress_bar.progress(40, text="Refining research...")
        results, all_queries = await conduct_iterative_research(
            topic=research_topic,
            initial_results=initial_results,
            all_queries=initial_queries,
            budget=budget,
            max_queries=max_queries,
            tavily_client=tavily_client,
            together_client=together_client,
            planning_model=planning_model,
            json_model=json_model,
            summary_model=summary_model,
            prompts=prompts,
            status_placeholder=status_placeholder
        )
        
        # Step 4: Process and filter search results
        progress_bar.progress(70, text="Processing and filtering results...")
        processed_results = await process_search_results(
            topic=research_topic,
            results=results,
            together_client=together_client,
            json_model=json_model,
            max_sources=max_sources,
            prompts=prompts,
            status_placeholder=status_placeholder
        )
        
        # Step 5: Generate final research report
        progress_bar.progress(85, text="Generating research report...")
        research_report = await generate_research_answer(
            topic=research_topic,
            results=processed_results,
            together_client=together_client,
            answer_model=answer_model,
            prompts=prompts,
            max_tokens=max_tokens,
            status_placeholder=status_placeholder
        )
        
        progress_bar.progress(100, text="Research complete!")
        return research_report
        
    except Exception as e:
        status_placeholder.error(f"Error in research pipeline: {str(e)}")
        return f"An error occurred during research: {str(e)}"

# ==========================================
# Streamlit UI
# ==========================================

def main():
    st.title("🔍 AI Riset Mendalam") # Diubah ke Bahasa Indonesia
    st.markdown("""
    Aplikasi ini menggunakan AI untuk menghasilkan laporan riset komprehensif tentang topik apa pun.
    Masukkan topik riset Anda di bawah, dan dapatkan laporan terperinci berbasis bukti.
    """) # Diubah ke Bahasa Indonesia
    
    with st.sidebar:
        # Pastikan variabel API key tetap didefinisikan (mengambil dari env var)
        together_api_key = DEFAULT_TOGETHER_API_KEY
        tavily_api_key = DEFAULT_TAVILY_API_KEY

        st.header("Pengaturan Proses Riset") # Diubah ke Bahasa Indonesia
        st.caption("Tips: Gunakan pengaturan default jika baru pertama kali mencoba.") # Ditambahkan catatan

        max_queries = st.slider(
            "AI Akan Cari Berapa Kali Sekaligus?",
            min_value=1, 
            max_value=5, 
            value=2,
            help="Setiap putaran riset, AI akan melakukan beberapa pencarian berbeda untuk memperkaya informasi. Semakin banyak, hasil riset bisa jadi lebih luas — tapi juga lebih lama." # Diubah ke Bahasa Indonesia
        )
        
        budget = st.slider(
            "Jumlah Iterasi Riset", # Diubah ke Bahasa Indonesia
            min_value=1, 
            max_value=5, 
            value=2,
            help="Berapa kali proses riset diulang untuk menyempurnakan hasil." # Diubah ke Bahasa Indonesia
        )
        
        max_sources = st.slider(
            "Jumlah Maksimum Sumber", # Diubah ke Bahasa Indonesia
            min_value=5, 
            max_value=20, 
            value=10,
            help="Batas jumlah sumber yang digunakan dalam laporan akhir." # Diubah ke Bahasa Indonesia
        )
        
        max_tokens = st.slider(
            "Panjang Laporan (Token)", # Diubah ke Bahasa Indonesia
            min_value=2048, 
            max_value=16384, 
            value=8192,
            help="Semakin besar nilainya, semakin panjang dan detail laporan yang dihasilkan." # Diubah ke Bahasa Indonesia
        )
    
    research_topic = st.text_area("Topik Riset", placeholder="Contoh: Dampak kecerdasan buatan dalam pendidikan (sumber-sumbernya harus berupa jurnal)", height=100, help="Jika Anda ingin laporan hanya mengambil referensi dari jurnal akademik, tambahkan kalimat seperti: 'sumber-sumbernya harus berupa jurnal' di akhir topik Anda." )  # Diubah ke Bahasa Indonesia
    
    # Tombol dinonaktifkan jika API keys tidak ada ATAU topik riset kosong
    start_button_disabled = not (together_api_key and tavily_api_key and research_topic)
    
    if st.button("Mulai Riset", type="primary", disabled=start_button_disabled): # Diubah ke Bahasa Indonesia
        if not research_topic:
            st.error("Silakan masukkan topik riset") # Diubah ke Bahasa Indonesia
            return

        # Cek apakah API keys berhasil diambil dari environment variables
        if not together_api_key:
             st.error("Kunci API Together AI tidak ditemukan di environment variables.") # Diubah ke Bahasa Indonesia
             return
        if not tavily_api_key:
             st.error("Kunci API Tavily tidak ditemukan di environment variables.") # Diubah ke Bahasa Indonesia
             return

        # Tampilkan informasi progres
        progress_container = st.container()
        progress_bar = progress_container.progress(0, text="Menginisialisasi pipeline riset...") # Diubah ke Bahasa Indonesia
        status_placeholder = st.empty()
        
        # Jalankan pipeline riset
        with st.spinner("Sedang melakukan riset..."): # Diubah ke Bahasa Indonesia
            try:
                # Setup event loop for asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                report = loop.run_until_complete(run_research_pipeline(
                    research_topic=research_topic,
                    together_api_key=together_api_key,
                    tavily_api_key=tavily_api_key,
                    max_queries=max_queries,
                    budget=budget,
                    max_sources=max_sources,
                    max_tokens=max_tokens,
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar
                ))
                
                # Tampilkan laporan
                st.subheader("Laporan Riset") # Diubah ke Bahasa Indonesia
                st.markdown(report, unsafe_allow_html=True)
                
                # Tambahkan tombol unduh untuk laporan
                try:
                    report_html = markdown.markdown(report)
                    st.download_button(
                        label="Unduh Laporan sebagai HTML", # Diubah ke Bahasa Indonesia
                        data=report_html,
                        file_name="laporan_riset.html", # Diubah ke Bahasa Indonesia
                        mime="text/html"
                    )
                except Exception as md_e:
                     st.error(f"Gagal mengonversi laporan ke HTML: {str(md_e)}") # Diubah ke Bahasa Indonesia

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}") # Diubah ke Bahasa Indonesia

if __name__ == "__main__":
    main()