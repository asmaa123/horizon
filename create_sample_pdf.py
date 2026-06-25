"""Create a sample PDF for testing the RAG system"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def create_sample_pdf():
    doc = SimpleDocTemplate("data/raw/knowledge_base.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title = Paragraph("Sample Knowledge Base", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))
    
    # Add content
    content = """
    <b>Introduction</b>
    This is a sample document for testing the RAG system. It contains various sections of text that can be used for question answering.
    
    <b>Pricing Information</b>
    The Cairo package costs $99 per month for the basic plan and $199 per month for the premium plan. Annual subscriptions receive a 20% discount.
    
    <b>Cancellation Policy</b>
    You can cancel your subscription at any time. Cancellations take effect at the end of the current billing period. No refunds are provided for partial months.
    
    <b>Features</b>
    The system includes PDF processing, intelligent chunking, vector search, hybrid retrieval, smart reranking, and LLM integration for generating grounded answers.
    
    <b>Technical Specifications</b>
    The system uses FAISS for vector search, sentence-transformers for embeddings, and supports both OpenAI and Google Gemini for LLM capabilities.
    
    <b>Support</b>
    Customer support is available 24/7 via email at support@example.com or through the online chat system.
    """
    
    para = Paragraph(content, styles['Normal'])
    story.append(para)
    
    doc.build(story)
    print("Sample PDF created at data/raw/knowledge_base.pdf")

if __name__ == "__main__":
    create_sample_pdf()
