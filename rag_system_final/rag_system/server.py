import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import RAGSystem
from utils.config import RAGConfig

# Global RAG Instance
rag = None

class RAGHTTPRequestHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        # Parse path and query parameters
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        if path == "/" or path == "/index.html":
            self.serve_static_file(PROJECT_ROOT / "web" / "index.html", "text/html")
        elif path == "/api/query":
            self.handle_query_api(query_params)
        else:
            self.send_error(404, "File Not Found")

    def serve_static_file(self, filepath, content_type):
        try:
            if not filepath.exists():
                self.send_error(404, "File Not Found")
                return
                
            with open(filepath, "rb") as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def handle_query_api(self, query_params):
        if 'q' not in query_params or not query_params['q'][0]:
            self.send_json_response({"error": "Missing 'q' query parameter"}, 400)
            return
            
        question = query_params['q'][0]
        
        try:
            # Run retrieval
            retrieval_result = rag.retriever.retrieve(
                question=question,
                k=rag.config.top_k,
                use_reranking=True
            )
            
            # Generate answer
            answer = rag.llm.generate(
                question=question,
                context_chunks=retrieval_result['chunks']
            )
            
            # Build API response including chunk content
            response_data = {
                "question": question,
                "answer": answer,
                "retrieval_time": retrieval_result['time'],
                "sources": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "page": chunk.page_num,
                        "section": chunk.section_title,
                        "score": score,
                        "content": chunk.content
                    }
                    for chunk, score in zip(
                        retrieval_result['chunks'],
                        retrieval_result['scores']
                    )
                ]
            }
            
            self.send_json_response(response_data, 200)
            
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)

    def send_json_response(self, data, status_code):
        try:
            json_str = json.dumps(data)
            json_bytes = json_str.encode("utf-8")
            
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(json_bytes))
            self.end_headers()
            self.wfile.write(json_bytes)
        except Exception as e:
            print(f"Error sending JSON response: {e}")

def run_server(port=8000):
    global rag
    
    # Configure and build the RAG system
    print("Initializing RAG system with local embeddings. Please wait...")
    config = RAGConfig(
        pdf_path="data/raw/Horizon_Tours_Complete_Knowledge_Base_2025.pdf",
        use_mock_llm=True,
        embedder_type="local"
    )
    rag = RAGSystem(config)
    
    # Start HTTP server
    server_address = ('', port)
    httpd = HTTPServer(server_address, RAGHTTPRequestHandler)
    print(f"\n==================================================")
    print(f"Server is running at: http://localhost:{port}")
    print(f"==================================================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
