from sympy.parsing.latex import parse_latex  # Updated import
from sympy import simplify, solve, Eq
from typing import Dict, List, Optional
import re

class MathValidator:
    @staticmethod
    def validate_latex_syntax(text: str) -> Dict[str, any]:
        """Validate LaTeX syntax in the text."""
        pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'
        formulas = re.finditer(pattern, text)
        valid_formulas = []
        invalid_formulas = []
        
        for match in formulas:
            formula = match.group(1) or match.group(2)
            try:
                parse_latex(formula)
                valid_formulas.append(formula)
            except:
                invalid_formulas.append(formula)
                
        return {
            "is_valid": len(invalid_formulas) == 0,
            "valid_count": len(valid_formulas),
            "invalid_count": len(invalid_formulas),
            "invalid_formulas": invalid_formulas
        }
    
    @staticmethod
    def validate_equation_solution(equation: str, solution: str) -> bool:
        """Validate if a solution satisfies an equation."""
        try:
            eq = parse_latex(equation)
            sol = parse_latex(solution)
            result = solve(eq - sol)
            return len(result) > 0
        except:
            return False

    @staticmethod
    def check_answer_structure(answer: str) -> Dict[str, bool]:
        """Check if the answer follows the required structure."""
        required_sections = [
            "Definition/Overview",
            "Detailed Explanation",
            "Key Formulas",
            "Example",
            "References"
        ]
        
        return {
            section: section.lower() in answer.lower()
            for section in required_sections
        }