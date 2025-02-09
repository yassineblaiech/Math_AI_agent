from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # Updated import
from langchain_community.embeddings import HuggingFaceEmbeddings      # Updated import
from langchain_community.vectorstores import FAISS                    # Updated import
from sympy.parsing.latex import parse_latex
from sympy import latex
from typing import List, Dict
import re
import os

class MathFormulaExtractor:
    @staticmethod
    def extract_latex(text: str) -> List[str]:
        """Extract LaTeX formulas from text."""
        # Match both inline ($...$) and display ($$...$$) math
        pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'
        matches = re.finditer(pattern, text)
        return [m.group(1) or m.group(2) for m in matches]
    
    @staticmethod
    def validate_formula(latex_str: str) -> bool:
        """Validate if LaTeX formula is parseable."""
        try:
            expr = parse_latex(latex_str)
            return True
        except:
            return False

class MathDocumentPreprocessor:
    def __init__(self):
        self.formula_extractor = MathFormulaExtractor()
    
    def preprocess_text(self, text: str) -> Dict:
        """Preprocess text to extract and validate math content."""
        formulas = self.formula_extractor.extract_latex(text)
        valid_formulas = [f for f in formulas if self.formula_extractor.validate_formula(f)]
        
        # Add metadata about mathematical content
        metadata = {
            "formula_count": len(valid_formulas),
            "has_equations": len(valid_formulas) > 0,
            "formulas": valid_formulas
        }
        
        return {
            "text": text,
            "metadata": metadata
        }

class MathCorpusProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"}
        )
        self.preprocessor = MathDocumentPreprocessor()
        
    def create_math_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create a text splitter optimized for mathematical content."""
        math_separators = [
            "\n\nTheorem", "\n\nLemma", "\n\nProposition", "\n\nCorollary",
            "\n\nProof", "\n\nDefinition", "\n\nExample", "\n\n", "\n", " "
        ]
        
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=math_separators,
            length_function=len
        )
    
    def process_document(self, file_path: str) -> List[Dict]:
        """Process a single document with math-specific handling."""
        if file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            return []
            
        documents = loader.load()
        processed_docs = []
        
        for doc in documents:
            processed = self.preprocessor.preprocess_text(doc.page_content)
            doc.page_content = processed["text"]
            doc.metadata.update(processed["metadata"])
            processed_docs.append(doc)
            
        return processed_docs
    
    def process_documents(self):
        """Process all documents in the data directory."""
        text_splitter = self.create_math_splitter()
        all_documents = []
        
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            processed_docs = self.process_document(file_path)
            all_documents.extend(processed_docs)
            
        texts = text_splitter.split_documents(all_documents)
        
        # Create vector store with enhanced math metadata
        return FAISS.from_documents(
            texts,
            self.embeddings,
            metadata_keys=["formula_count", "has_equations", "formulas"]
        )