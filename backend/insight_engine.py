"""
Intelligent Business Insight Engine for DataSense AI.
Orchestrates Layer 1 (Statistical Analytics), Layer 2 (Business Rules),
and Layer 3 (AI / Rule-based Interpretation).
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from backend.business_rules import BusinessRuleEngine
from backend.ai_interpreter import AIInterpreter


class InsightEngine:
    """End-to-end insight generation engine."""

    def __init__(self, ai_interpreter: Optional[AIInterpreter] = None):
        self.ai_interpreter = ai_interpreter or AIInterpreter()

    def generate_all_insights(
        self,
        df: pd.DataFrame,
        column_roles: Dict[str, Dict[str, Any]],
        domain: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Orchestrate rule evaluation and produce enriched 5-part business insights.
        """
        if df.empty or len(df.columns) == 0:
            return []

        # Layer 1 & 2: Evaluate deterministic business rules
        raw_findings = BusinessRuleEngine.evaluate_rules(df, column_roles, domain=domain)
        if not raw_findings:
            return []

        # Layer 3: Enrich with business implications and strategic recommendations
        enriched_insights = []
        for finding in raw_findings:
            insight = self.ai_interpreter.interpret_finding(finding)
            enriched_insights.append(insight)

        return enriched_insights

