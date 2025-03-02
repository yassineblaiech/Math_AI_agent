from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader  # Updated import
from langchain_huggingface import HuggingFaceEmbeddings  # Update this import
from langchain_community.vectorstores import FAISS                    # Updated import
from sympy.parsing.latex import parse_latex
from sympy import latex
from typing import List, Dict, Any, Optional
import re
import os
import logging
from langchain.schema import Document

logging.getLogger('faiss').setLevel(logging.ERROR)  # Suppress FAISS GPU warning

logger = logging.getLogger(__name__)

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
        except Exception as e:
            logger.debug(f"Failed to parse LaTeX formula: {latex_str}. Error: {str(e)}")
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
    def __init__(self, data_dir: str, hf_api_key: str):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            logger.info(f"Creating directory: {data_dir}")
            os.makedirs(data_dir)
        
        # Modified embeddings initialization to include the token
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={
                "device": "cpu",  # Force CPU usage
                "token": hf_api_key  # Add the token here
            },
            encode_kwargs={"device": "cpu"}   # Force CPU usage for encoding
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
    

    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document with math-specific handling."""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return []
                
            try:
                documents = loader.load()
            except Exception as e:
                logger.error(f"Failed to load document {file_path}: {str(e)}")
                return []
                
            processed_docs = []
            for doc in documents:
                try:
                    processed = self.preprocessor.preprocess_text(doc.page_content)
                    doc.page_content = processed["text"]
                    doc.metadata.update(processed["metadata"])
                    processed_docs.append(doc)
                except Exception as e:
                    logger.error(f"Failed to process document content: {str(e)}")
                    continue
            
            text_splitter = self.create_math_splitter()
            split_docs = text_splitter.split_documents(processed_docs)
            return split_docs
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
        
    def process_documents(self) -> Optional[FAISS]:
        """Process all documents in the data directory."""
        all_documents = []
        
        try:
            files = os.listdir(self.data_dir)
            if not files:
                logger.warning(f"No documents found in {self.data_dir}")
                return None
                
            for file in files:
                file_path = os.path.join(self.data_dir, file)
                documents = self.process_document(file_path)
                all_documents.extend(documents)
                
            if not all_documents:
                logger.warning("No documents were successfully processed")
                return None
                
            # Remove metadata_keys parameter as it's not supported
            return FAISS.from_documents(
                all_documents,
                self.embeddings
            )
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise