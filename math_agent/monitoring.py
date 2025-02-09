import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('math_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('math_agent')

# Prometheus metrics
MATH_QUESTIONS_TOTAL = Counter(
    'math_questions_total', 
    'Total number of math questions processed'
)
QUERY_PROCESSING_TIME = Histogram(
    'query_processing_time_seconds',
    'Time spent processing queries',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)
FORMULA_COUNT = Counter(
    'formula_count_total',
    'Total number of LaTeX formulas processed'
)

class MathAgentMonitor:
    def __init__(self, metrics_port: int = 8000):
        self.start_time = datetime.now()
        try:
            start_http_server(metrics_port)
            logger.info(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def log_query(self, question: str, result: Dict[str, Any]) -> None:
        """Log query details and update metrics."""
        try:
            MATH_QUESTIONS_TOTAL.inc()
            formula_count = result.get("math_metadata", {}).get("total_formulas", 0)
            FORMULA_COUNT.inc(formula_count)

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "has_equations": result.get("math_metadata", {}).get("has_equations", False),
                "formula_count": formula_count,
                "source_count": len(result.get("source_documents", []))
            }

            logger.info(f"Query processed: {json.dumps(log_entry)}")
        except Exception as e:
            logger.error(f"Error logging query: {e}")

    def log_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Log errors with context."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        logger.error(f"Error occurred: {json.dumps(error_entry)}")

    @QUERY_PROCESSING_TIME.time()
    def time_query_processing(self, func, *args, **kwargs):
        """Decorator to measure query processing time."""
        return func(*args, **kwargs)