"""
AI Interpretation Engine (Layer 3) for DataSense AI.
Default provider: Google Gemini (gemini-2.5-flash-lite).
Also supports Ollama (local), OpenAI, Groq, and Anthropic.
Falls back gracefully to the deterministic rule engine when no key is present.
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from utils.constants import (
    PROVIDER_OLLAMA, PROVIDER_OPENAI, PROVIDER_GROQ,
    PROVIDER_GEMINI, PROVIDER_ANTHROPIC
)

# Default Gemini model — change via LLM_MODEL env var or Settings UI
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class AIInterpreter:
    """Interprets business analytical findings using an LLM or deterministic fallback."""

    def __init__(
        self,
        provider: str = PROVIDER_GEMINI,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", PROVIDER_GEMINI)).lower()

        # Model defaults per provider
        _default_models = {
            PROVIDER_GEMINI: GEMINI_DEFAULT_MODEL,
            PROVIDER_OPENAI: "gpt-4o-mini",
            PROVIDER_GROQ: "llama-3.1-70b-versatile",
            PROVIDER_ANTHROPIC: "claude-3-haiku-20240307",
            PROVIDER_OLLAMA: "llama3:8b",
        }
        self.model = model or os.getenv("LLM_MODEL", _default_models.get(self.provider, GEMINI_DEFAULT_MODEL))

        # API key resolution — Gemini uses GEMINI_API_KEY
        self.api_key = api_key or os.getenv(
            f"{self.provider.upper()}_API_KEY",
            os.getenv("LLM_API_KEY", "")
        )
        self.base_url = base_url or os.getenv(
            "LLM_BASE_URL",
            "http://localhost:11434" if self.provider == PROVIDER_OLLAMA else ""
        )

    # ------------------------------------------------------------------
    # Availability Check
    # ------------------------------------------------------------------
    def is_ai_available(self) -> bool:
        """Return True when the configured provider can accept requests."""
        if self.provider == PROVIDER_OLLAMA:
            try:
                r = requests.get(f"{self.base_url.rstrip('/')}/api/tags", timeout=1.5)
                return r.status_code == 200
            except Exception:
                return False
        # For all cloud providers (including Gemini) a non-empty key is enough
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------
    def interpret_finding(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a 5-part executive business insight via LLM or deterministic fallback:
          1. Observation   – What happened?
          2. Implication   – What does it mean?
          3. Recommendation – What action should management consider?
          4. Category       – Risk / Opportunity / General
          5. Supporting Metrics
        """
        if self.is_ai_available():
            try:
                result = self._call_llm_for_interpretation(finding)
                if result:
                    return result
            except Exception:
                pass  # Graceful fallback

        return self._generate_rule_based_interpretation(finding)

    # ------------------------------------------------------------------
    # LLM Dispatch Router
    # ------------------------------------------------------------------
    def _call_llm_for_interpretation(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route to the correct provider's API method."""
        if self.provider == PROVIDER_GEMINI:
            return self._call_gemini(finding)
        if self.provider == PROVIDER_OLLAMA:
            return self._call_ollama(finding)
        if self.provider in [PROVIDER_OPENAI, PROVIDER_GROQ]:
            return self._call_openai_compat(finding)
        return None

    # ------------------------------------------------------------------
    # Shared Prompt Builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(finding: Dict[str, Any]) -> str:
        return (
            "You are a Senior Business Strategy Consultant and Data Analyst.\n"
            "Analyze the following measured analytical finding from a business dataset:\n\n"
            f"Title: {finding.get('title')}\n"
            f"Observed Fact: {finding.get('observation')}\n"
            f"Underlying Evidence: {finding.get('evidence')}\n"
            f"Category: {finding.get('category')}\n\n"
            "Explain this finding in concise, executive language.\n"
            "Respond ONLY with a valid JSON object matching this exact schema:\n"
            "{\n"
            '  "implication": "What this finding means for business operations or economics (2 sentences max)",\n'
            '  "recommendation": "Concrete, actionable recommendation for management (2 sentences max)"\n'
            "}\n"
            "DO NOT invent numbers. Rely strictly on the provided facts."
        )

    @staticmethod
    def _package_result(finding: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": finding.get("title"),
            "category": finding.get("category"),
            "observation": finding.get("observation"),
            "implication": parsed.get("implication"),
            "recommendation": parsed.get("recommendation"),
            "evidence": finding.get("evidence"),
            "metrics": finding.get("metrics"),
        }

    # ------------------------------------------------------------------
    # Google Gemini  (REST API — no SDK dependency required at runtime)
    # ------------------------------------------------------------------
    def _call_gemini(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Calls the Gemini generateContent REST endpoint directly.
        Uses JSON response MIME type to get structured output reliably.
        """
        prompt = self._build_prompt(finding)
        url = f"{GEMINI_API_BASE}/{self.model}:generateContent?key={self.api_key}"

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 512,
            }
        }

        try:
            res = requests.post(url, json=payload, timeout=20)
            if res.status_code != 200:
                return None

            data = res.json()
            # Extract text from Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                return None
            raw_text = parts[0].get("text", "")
            parsed = json.loads(raw_text)
            return self._package_result(finding, parsed)

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ollama (local)
    # ------------------------------------------------------------------
    def _call_ollama(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = self._build_prompt(finding)
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
            if res.status_code == 200:
                raw_txt = res.json().get("response", "{}")
                parsed = json.loads(raw_txt)
                return self._package_result(finding, parsed)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # OpenAI-compatible (OpenAI + Groq)
    # ------------------------------------------------------------------
    def _call_openai_compat(self, finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = self._build_prompt(finding)
        url = (
            "https://api.openai.com/v1/chat/completions"
            if self.provider == PROVIDER_OPENAI
            else "https://api.groq.com/openai/v1/chat/completions"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional business analytics AI. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return self._package_result(finding, parsed)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Deterministic Rule-Based Fallback (no LLM required)
    # ------------------------------------------------------------------
    def _generate_rule_based_interpretation(self, finding: Dict[str, Any]) -> Dict[str, Any]:
        """Factual, structured interpretation produced without any LLM."""
        rule_id  = finding.get("rule_id", "")
        obs      = finding.get("observation", "")
        evidence = finding.get("evidence", "")
        metrics  = finding.get("metrics", {})
        category = finding.get("category", "general")

        if rule_id == "top_category_dominance":
            top_cat    = metrics.get("top_category", "the leading segment")
            bottom_cat = metrics.get("bottom_category", "the lowest segment")
            implication = (
                f"Revenue is heavily anchored on '{top_cat}', representing the core engine of cash flow. "
                f"Conversely, '{bottom_cat}' demonstrates low traction or weak customer adoption in the current mix."
            )
            recommendation = (
                f"Protect market share in '{top_cat}' with dedicated inventory allocation and customer retention programs. "
                f"Conduct a pricing and product-market fit audit for '{bottom_cat}' to decide whether to re-position or discontinue."
            )

        elif rule_id == "pareto_concentration_risk":
            implication = (
                "A significant majority of top-line revenue is generated by a narrow slice of offerings. "
                "Any macroeconomic disruption or competitive pressure on these key drivers poses severe business risk."
            )
            recommendation = (
                "Develop cross-selling strategies and incentivize diversification into mid-tier offerings "
                "to reduce single-point dependency and build portfolio resilience."
            )

        elif rule_id == "negative_margin_leakage":
            segment = metrics.get("segment", "the identified segment")
            implication = (
                f"'{segment}' is actively eroding overall operating profits. Although it generates sales, "
                "direct costs or aggressive discounting are exceeding realized revenue."
            )
            recommendation = (
                f"Immediately review the cost of goods sold (COGS) and discount schedules for '{segment}'. "
                "Cap maximum allowed discounts and renegotiate supplier terms to restore positive gross margin."
            )

        elif rule_id == "high_margin_champion":
            segment = metrics.get("segment", "the top-margin segment")
            implication = (
                f"'{segment}' yields superior capital efficiency and profitability per dollar of sales, "
                "representing high customer willingness-to-pay and disciplined cost structure."
            )
            recommendation = (
                f"Increase marketing spend and sales resource allocation towards '{segment}'. "
                "Analyze its winning characteristics (packaging, target persona, pricing tier) and replicate them across other offerings."
            )

        elif rule_id == "regional_performance_variance":
            top_reg    = metrics.get("top_region", "the top region")
            bottom_reg = metrics.get("bottom_region", "the trailing region")
            implication = (
                f"Significant geographic performance disparity. '{top_reg}' shows established channel effectiveness, "
                f"while '{bottom_reg}' indicates either untapped expansion potential or ineffective local distribution."
            )
            recommendation = (
                f"Document sales playbooks and partner incentives from '{top_reg}' and benchmark against '{bottom_reg}'. "
                "Evaluate local customer demand before allocating further expansion capital."
            )

        elif rule_id == "timeline_growth_velocity":
            growth_pct = metrics.get("growth_rate_pct", 0)
            if growth_pct > 0:
                implication   = "Business momentum is accelerating across recent intervals, indicating expanding demand or favorable seasonality."
                recommendation = "Ensure working capital and supply chain throughput can support continued demand without fulfillment delays."
            else:
                implication   = "Top-line volume has slowed in recent intervals, indicating possible market saturation, churn, or reduced conversion."
                recommendation = "Investigate conversion drop-offs, conduct customer feedback reviews, and introduce targeted promotional campaigns."

        elif rule_id == "hr_overtime_attrition_risk":
            implication   = "Chronic overtime workload is a statistically significant driver of employee departures and burnout."
            recommendation = "Conduct immediate workload balancing, review headcount capacity in high-overtime teams, and implement retention bonuses."

        else:
            implication   = "This metric highlights an observable business operational pattern that warrants executive monitoring."
            recommendation = "Continue monitoring historical variance and establish alert thresholds to prevent sudden performance degradation."

        return {
            "title":          finding.get("title", "Business Operational Insight"),
            "category":       category,
            "observation":    obs,
            "implication":    implication,
            "recommendation": recommendation,
            "evidence":       evidence,
            "metrics":        metrics,
        }
