from fpdf import FPDF
from datetime import datetime
from fpdf.enums import XPos, YPos
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import tempfile
import os

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        # Use the recommended property instead of deprecated set_doc_option
        self.core_fonts_encoding = 'latin-1'
        self.section_color = (0, 71, 133)
        self.body_font_size = 11
        self.code_font_size = 9
        self.line_height = 7
        self.margin = 18

    def header(self):
        self.set_fill_color(0, 51, 102)
        self.rect(0, 0, 210, 22, 'F')
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(255, 255, 255)
        self.set_y(7)
        self.cell(0, 8, "Ramayana Fact Checker: Algorithm Report", align='C', border=0)
        self.ln(10)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', 'I', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align='C')

    def section_title(self, title):
        # Ensure enough space before a new section, especially after a page break
        if self.get_y() < 30:
            self.set_y(30)
        else:
            self.ln(10)
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(*self.section_color)
        self.cell(0, 10, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_line_width(0.7)
        self.set_draw_color(*self.section_color)
        x = self.get_x()
        y = self.get_y()
        self.line(x, y, x + 60, y)
        self.ln(8)  # Increased space after underline

    def sub_title(self, text):
        # Ensure enough space after a page break or section heading
        if self.get_y() < 38:
            self.set_y(38)
        else:
            self.ln(8)
        self.set_font('Helvetica', 'B', self.body_font_size + 1)
        self.set_text_color(0, 51, 102)
        self.cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def body_text(self, text):
        # Add extra space if too close to the top (e.g., after a page break)
        if self.get_y() < 46:
            self.set_y(46)
        else:
            self.ln(4)
        self.set_font('Helvetica', '', self.body_font_size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, self.line_height, text, align='J')
        self.ln(4)

    def code_block(self, code):
        # Use Pygments to generate HTML with syntax highlighting
        formatter = HtmlFormatter(style="colorful", noclasses=True)
        highlighted_code = highlight(code, PythonLexer(), formatter)

        # Write HTML to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp_html:
            tmp_html.write(f"<div>{highlighted_code}</div>")
            tmp_html_path = tmp_html.name

        # Convert HTML to image using external tool (e.g., wkhtmltoimage or imgkit)
        # For portability, we use imgkit if available, else fallback to plain code block
        try:
            import imgkit
            img_path = tmp_html_path.replace(".html", ".png")
            imgkit.from_file(tmp_html_path, img_path, options={"format": "png", "width": 700, "disable-smart-width": ""})
            self.ln(2)
            self.image(img_path, w=170)
            self.ln(5)
            os.remove(img_path)
        except Exception:
            # Fallback: plain code block (no color)
            self.set_font('Courier', '', self.code_font_size)
            self.set_text_color(230, 230, 230)
            self.set_fill_color(30, 30, 30)
            self.ln(2)
            self.multi_cell(0, 5.5, code, border=0, fill=True)
            self.ln(5)
        finally:
            os.remove(tmp_html_path)

    def list_item(self, text):
        self.set_font('Helvetica', '', self.body_font_size)
        self.set_text_color(50, 50, 50)
        self.cell(self.margin, self.line_height, '', border=0)
        # Use ASCII dash instead of Unicode bullet for compatibility
        self.multi_cell(0, self.line_height, f"- {text}", align='L')
        self.ln(1)

    def key_value(self, key, value):
        self.set_font('Helvetica', 'B', self.body_font_size)
        self.set_text_color(*self.section_color)
        self.cell(40, self.line_height, key, border=0)
        self.set_font('Helvetica', '', self.body_font_size)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, self.line_height, value, border=0)
        self.ln(1)

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=pdf.margin)
pdf.add_page()
pdf.alias_nb_pages()

# Title Page
pdf.set_y(40)
pdf.set_font('Helvetica', 'B', 26)
pdf.set_text_color(0, 51, 102)
pdf.cell(0, 15, "Ramayana Fact Checker", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.set_font('Helvetica', '', 15)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, "Algorithm Implementation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.ln(5)
pdf.set_font('Helvetica', 'I', 11)
pdf.set_text_color(120, 120, 120)


# Add codebase info and pipeline summary on the first page
pdf.set_font('Helvetica', 'B', 12)
pdf.set_text_color(0, 71, 133)
pdf.cell(0, 8, "Codebase", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(30, 30, 30)
pdf.cell(0, 7, "ollama_fact_checker.py, model_report.py", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.ln(4)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(30, 30, 30)

# Executive Summary
pdf.section_title("Executive Summary")
pdf.body_text(
    "The Ramayana Fact Checker implements a Retrieval-Augmented Generation (RAG) pipeline using Ollama's local Llama 3.8B model for offline verification of claims about Valmiki's Ramayana. The system achieves reliable fact-checking through keyword-based retrieval, structured prompt engineering, and comprehensive evaluation tools."
)

# Algorithm Pipeline
pdf.section_title("Algorithm Pipeline")
pdf.sub_title("Core Processing Flow")
pdf.set_font('Courier', '', pdf.body_font_size)
pdf.set_fill_color(240, 240, 240)
pdf.set_text_color(0, 0, 0)
pdf.multi_cell(0, 8, "Input Claim -> NLTK Keywords -> LLM Query Generation -> Chunk Retrieval -> Context Assembly -> Ollama LLM -> Response Parsing -> JSON Output", border=0, fill=True, align='C')
pdf.ln(8)


pdf.sub_title("1. Keyword Extraction (NLTK)")
pdf.body_text(
    "The core pipeline is implemented in three main code components:\n"
    "1. Keyword Extraction (NLTK)\n"
    "2. Query Generation & Chunk Retrieval\n"
    "3. Prompt Engineering & LLM Processing\n"
    "\n"
    "\n"
    "These three steps cover the main logic of the fact-checking pipeline. "
    "Other supporting code (e.g., response parsing, evaluation, error handling) is not shown here for brevity."
)   
pdf.code_block(
"""def get_keywords_from_claim(self, claim: str) -> List[str]:
    words = re.findall(r'\\b\\w+\\b', claim.lower())
    keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
    freq = nltk.FreqDist(keywords)
    return [kw for kw, _ in freq.most_common(10)]"""
)

pdf.sub_title("2. Query Generation & Chunk Retrieval")
pdf.code_block(
"""def retrieve_relevant_chunks(self, claim: str, chunks: List[Dict]) -> List[Dict]:
    keywords = self.get_keywords_from_claim(claim)
    queries = self.generate_search_queries(claim)
    combined_keywords = set(keywords)
    for q in queries:
        combined_keywords.update(self.get_keywords_from_claim(q))
    scored = []
    for chunk_data in chunks:
        score = sum(1 for kw in combined_keywords if kw in chunk_data["text"].lower())
        if score > 0:
            scored.append({"score": score, **chunk_data})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:5]"""
)

pdf.sub_title("3. Prompt Engineering & LLM Processing")
pdf.code_block(
"""def create_fact_checking_prompt(self, claim: str) -> str:
    chunks = self.retrieve_relevant_chunks(claim, self.all_chunks)
    context = "\\n\\n---\\n\\n".join([c["text"] for c in chunks])
    return f'''Fact-check: "{claim}"
CONTEXT:
{context}
RULES:
1. EXACT MATCHING: Details must match context.
2. COMPONENT ANALYSIS: Break into WHO/WHAT/WHEN etc.
3. NO ASSUMPTIONS: Verify explicit facts only.
FORMAT:
VERDICT: [TRUE/FALSE/INSUFFICIENT_DATA]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [Supporting text]
EXPLANATION: [Reasoning]'''"""
)

# Technical Configuration
pdf.add_page()
pdf.section_title("Technical Configuration")
pdf.sub_title("Key Parameters")
pdf.set_font('Helvetica', 'B', pdf.body_font_size)
pdf.set_fill_color(220, 230, 240)
pdf.set_text_color(0,0,0)
pdf.cell(60, 8, "Parameter", border=1, fill=True)
pdf.cell(0, 8, "Value", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', pdf.body_font_size)
pdf.set_text_color(50,50,50)
params = [
    ("CHUNK_SIZE", "1500 characters"),
    ("CHUNK_OVERLAP", "200 characters"),
    ("MAX_RETRIEVED_CHUNKS", "5 chunks"),
    ("TOP_N_KEYWORDS", "10 keywords"),
    ("Ollama Model", "ramayana-fact-checker:latest"),
    ("LLM Temperature", "0.3 (for consistency)"),
    ("LLM Top-p", "0.9 (for quality)"),
    ("API Timeout", "180 seconds"),
    ("Source Texts", "6 Ramayana books (All Kandas)")
]
for param, value in params:
    pdf.cell(60, 7, param, border='LR', align='L')
    # Use a fixed width for value cell to avoid "Not enough horizontal space" error
    pdf.cell(120, 7, value, border='R', align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(180,0,'',border='T')
pdf.ln(8)

pdf.sub_title("Performance Metrics")
pdf.set_font('Helvetica', 'B', pdf.body_font_size)
pdf.set_fill_color(220, 230, 240)
pdf.set_text_color(0,0,0)
pdf.cell(60, 8, "Metric", border=1, fill=True)
pdf.cell(0, 8, "Value", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_font('Helvetica', '', pdf.body_font_size)
pdf.set_text_color(50,50,50)
perf_metrics = [
    ("Avg. Processing Time", "~20-35 seconds per claim"),
    ("Model Memory Usage", "8-16GB (Llama 3.8B)"),
    ("Text Corpus Size", "6 books, approx. 1500 chunks"),
    ("Keyword Extraction", "<100ms (NLTK)"),
    ("Chunk Retrieval", "~50ms (keyword matching)")
]
for metric, value in perf_metrics:
    pdf.cell(60, 7, metric, border='LR', align='L')
    pdf.cell(120, 7, value, border='R', align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.cell(180,0,'',border='T')
pdf.ln(8)

# Evaluation System
pdf.add_page()
pdf.section_title("Evaluation System")
pdf.sub_title("model_report.py Framework")
pdf.code_block(
"""# Evaluation Process Snippet
for item in evaluation_data:
    claim = item.get("statement")
    expected = item.get("label")
    result = fact_checker.fact_check(claim)
    log = {"claim": claim, "expected": expected, **result}
    # ... (logging and accuracy calculation) ...
accuracy = (correct / total) * 100"""
)

pdf.sub_title("Debug Tools")
debug_tools = [
    "debug_specific_claims(): Analyze problematic cases with full retrieval visibility.",
    "search_pushpaka_origin(): Targeted keyword searches for specific elements.",
    "Chunk Analysis: Detailed scoring and source tracking for retrieved text segments."
]
for tool in debug_tools:
    pdf.list_item(tool)

# Critical Verification Rules
pdf.add_page()
pdf.section_title("Critical Verification Rules")
rules = [
    "Exact Matching: All claim details must precisely match the provided context; no assumptions allowed.",
    "Component Analysis: Complex claims are broken down into core components (WHO, WHAT, WHEN, WHERE, HOW, WHY).",
    "Timing Verification: Specific temporal assertions must be explicitly verifiable from the context.",
    "Character Precision: Similar characters and their roles must be distinguished accurately."
]
for rule in rules:
    pdf.list_item(rule)

# Error Handling & Reliability
pdf.section_title("Error Handling & Reliability")
reliability = [
    "Connection Testing: Automatic verification of Ollama server connectivity.",
    "Model Validation: Checks for the availability of the specified model.",
    "Graceful Degradation: Provides fallback facts if source text files are unavailable.",
    "NLTK Dependency Management: Automatically downloads missing NLTK resources (e.g., stopwords).",
    "Comprehensive Logging: INFO level logging provides detailed insights into processing steps."
]
for item in reliability:
    pdf.list_item(item)

# Future Enhancements
pdf.section_title("Future Enhancements")
enhancements = [
    "Vector Embeddings: Implement semantic search using sentence transformers for improved retrieval relevance.",
    "Model Fine-tuning: Further train Llama models specifically on the Ramayana corpus for enhanced domain expertise.",
    "Multi-source Validation: Cross-reference information from multiple Ramayana translations and commentaries.",
    "Performance Optimization: Introduce caching mechanisms and batch processing for faster response times.",
    "Web Interface: Develop a user-friendly web interface for interactive fact-checking."
]
for enhancement in enhancements:
    pdf.list_item(enhancement)

try:
    pdf_file_name = "Ramayana_Fact_Checker_Report_Professional.pdf"
    pdf.output(pdf_file_name)
    print(f"PDF generated successfully: {pdf_file_name}")
except Exception as e:
    print(f"Error generating PDF: {e}")