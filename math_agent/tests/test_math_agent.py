import pytest
from math_agent.rag_agent import MathRAGAgent
from math_agent.retriever import MathRetriever
from math_agent.data_processing import MathCorpusProcessor
from math_agent.validation import MathValidator

@pytest.fixture
def math_agent():
    """Create a MathRAGAgent instance for testing."""
    processor = MathCorpusProcessor("test_data")
    vector_store = processor.process_documents()
    retriever = MathRetriever(vector_store)
    return MathRAGAgent(retriever)

@pytest.fixture
def validator():
    """Create a MathValidator instance for testing."""
    return MathValidator()

class TestMathRAGAgent:
    def test_basic_math_question(self, math_agent):
        """Test basic math question answering."""
        question = "What is the quadratic formula?"
        result = math_agent.answer_question(question)
        
        assert "answer" in result
        assert result["math_metadata"]["has_equations"]
        assert result["math_metadata"]["total_formulas"] > 0

    def test_answer_structure(self, math_agent, validator):
        """Test if answer follows required structure."""
        question = "Explain integration by parts."
        result = math_agent.answer_question(question)
        
        structure_check = validator.check_answer_structure(result["answer"])
        assert all(structure_check.values()), "Missing required sections"

    def test_latex_validation(self, math_agent, validator):
        """Test LaTeX formula validation in answers."""
        question = "What is the chain rule?"
        result = math_agent.answer_question(question)
        
        latex_check = validator.validate_latex_syntax(result["answer"])
        assert latex_check["is_valid"], "Invalid LaTeX syntax found"
        assert latex_check["valid_count"] > 0, "No LaTeX formulas found"

    @pytest.mark.parametrize("search_type", ["hybrid", "basic"])
    def test_search_methods(self, math_agent, search_type):
        """Test different search methods."""
        question = "What is a derivative?"
        result = math_agent.answer_question(
            question,
            use_hybrid_search=(search_type == "hybrid")
        )
        assert result["answer"], "No answer returned"
        assert len(result["source_documents"]) > 0, "No sources found"

    def test_complex_query(self, math_agent, validator):
        """Test handling of complex mathematical queries."""
        question = "Prove the fundamental theorem of calculus"
        result = math_agent.answer_question(
            question,
            use_hybrid_search=True,
            use_compression=True
        )
        
        # Validate answer structure
        structure_check = validator.check_answer_structure(result["answer"])
        assert structure_check["Proof"], "Proof section missing"
        
        # Validate LaTeX content
        latex_check = validator.validate_latex_syntax(result["answer"])
        assert latex_check["is_valid"], "Invalid LaTeX in proof"