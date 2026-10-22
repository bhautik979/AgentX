# ============================================================================
# SPECIALIZED TASK HANDLERS - V4
# 3 Main Tasks: QA, Budget Planner, Risk Profiler
# ============================================================================

import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from live_data_providers import DataAggregator
from output_generators import PDFGenerator, EmailSender
import os

logger = logging.getLogger(__name__)

class TaskHandler:
    """Base class for all task handlers"""
    
    def __init__(self, llm_handler, data_aggregator: DataAggregator):
        self.llm = llm_handler
        self.data_aggregator = data_aggregator

    def _load_chat_history(self, user_id: str) -> str:
        """
        Load chat history for user and return as formatted context string.
        Returns formatted string ready to inject into prompts.
        """
        safe_user = user_id or "default_user"
        CHAT_HISTORY_DIR = "./chat_history"
        path = os.path.join(CHAT_HISTORY_DIR, f"{safe_user}.json")
        
        history_context = ""
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                
                # Last 5 Q&A pairs for reasonable context
                relevant_history = history[-5:]
                
                if relevant_history:
                    history_context = "PREVIOUS CONVERSATION CONTEXT:\n"
                    history_context += "=" * 60 + "\n"
                    for i, entry in enumerate(relevant_history, 1):
                        history_context += f"\nQ{i}: {entry.get('question', '')}\n"
                        history_context += f"A{i}: {entry.get('answer', '')[:500]}...\n"
                        history_context += "-" * 40 + "\n"
                    history_context += "=" * 60 + "\n\n"
        except Exception as e:
            logger.error("Failed to load chat history: %s", e)
        
        return history_context

class FinancialQAHandler(TaskHandler):
    """
    Handler for simple financial question answering
    Uses live data and web search
    """

    def __init__(self, llm_handler, data_aggregator: DataAggregator):
        super().__init__(llm_handler, data_aggregator)

        # NEW INSTRUCTION TEMPLATE
        self.instruction_template = """
You are a certified financial analyst. Your job is to produce a concise,
accurate, and risk-aware answer based strictly on the context provided.

RULES:
- Use live data (stocks/commodities) only if provided.
- Use news context only for insights, not predictions.
- Provide a direct answer first, then reasoning.
- Add 2–3 actionable insights.
- NEVER guarantee returns.
- Do NOT hallucinate data.
- Final answer must be clean text, no JSON.
"""

    async def handle_query(self, query: str, rag_context: str,user_id: str = "default_user") -> Dict:
        """Handle financial Q&A with live market + news context"""

        # Extract entities
        stocks = self._extract_stock_symbols(query)
        commodities = self._extract_commodities(query)

        live_data = {}

        # Fetch live data
        for stock in stocks:
            stock_data = await self.data_aggregator.get_stock_data(stock)
            if stock_data:
                live_data[f"stock_{stock}"] = stock_data

        for commodity in commodities:
            commodity_data = await self.data_aggregator.get_commodity_data(commodity)
            if commodity_data:
                live_data[f"commodity_{commodity}"] = commodity_data

        search_results = await self.data_aggregator.search_financial_news(query)

        # Build unified prompt with template
        prompt =f"""You are a professional financial advisor providing personalized guidance on finance.

        CHAT HISTORY:
        {self._load_chat_history(user_id)}

        CONTEXT FROM PINECONE HYBRID RETRIEVAL:
        {rag_context[:1000] if rag_context else 'No context available'}

        REQUIREMENTS:
        1. Be accurate and specific
        2. Include reasoning
        3. Mention relevant financial principles
        4. **CRITICAL:** If the CONTEXT provided is irrelevant to the USER QUESTION, you MUST ignore the context and answer based only on your general financial knowledge, without stating that the provided context was not useful for the current question.
        5. Include appropriate disclaimers if needed
        6. Be structured and easy to follow
        7 **OUTPUT STRUCTURE:** If User has asked about sending email, do not say that as AI I can not send email. Just provide the JSON output. with email body and context that user has asked for.
        if User has asked for PDF generation, do not say that as AI I can not generate PDF. Just provide the Markdown output with PDF content that user has asked for.

        USER QUESTION:
        {query}
        ---
        FINAL RESPONSE:
        
        If User has asked about sending email or generate PDF, do not say that as AI I can not send email. 
            - do not draft any email content. Just provide the Markdown output with PDF content that user has asked for.
            - do not mention like this:
                Subject: Your Personalized Monthly Budget Plan
                Hi Karan,
                Here is the detailed monthly budget plan you requested, designed to help
                you manage your 50000 income effectively and achieve your financial goals.

            -ONLY draft PDF content that user has asked for like this.
                Monthly Budget Plan: 50000
                Essentials: 20000 (Rationale: Covers......
        If none of the above, provide the final answer in the requested Markdown format.
        (Return ONLY the Markdown object)
        ---

        ANSWER:"""

        answer = await self.llm.generate(prompt)

        return {
            "answer": answer,
            "query": query,
            "live_data": live_data,
            "search_results": search_results,
            "task_type": "financial_qa",
            "timestamp": datetime.now().isoformat()
        }

    # ------------------------------------------------------------------------------------
    # RESTORED HELPER METHODS
    # ------------------------------------------------------------------------------------

    def _extract_stock_symbols(self, query: str) -> List[str]:
        """Extract stock ticker symbols."""
        query_upper = query.upper()
        common_stocks = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'FB', 'NFLX',
            'INFY', 'TCS', 'BAJAJ', 'RELIANCE', 'HDFC'
        ]
        return [sym for sym in common_stocks if sym in query_upper]

    def _extract_commodities(self, query: str) -> List[str]:
        """Extract commodity references."""
        q = query.lower()
        mapping = {
            "oil": "OIL",
            "gold": "GOLD",
            "silver": "SILVER",
            "copper": "COPPER"
        }
        return [v for k, v in mapping.items() if k in q]

    def _extract_keywords(self, query: str) -> List[str]:
        stop = {"the", "is", "at", "which", "on", "a", "an", "and", "or"}
        words = [w for w in query.lower().split() if w not in stop]
        return words[:5]  # top 5 keywords

    def _build_qa_context(self, live_data: Dict, search_results: List, keywords: List) -> str:
        """Build readable context block."""
        ctx = "Context Information:\n"

        if live_data:
            ctx += "\nCurrent Market Data:\n"
            for k, v in live_data.items():
                ctx += f"- {k}: {v}\n"

        if search_results:
            ctx += "\nRecent News:\n"
            for r in search_results[:3]:
                ctx += f"- {r.get('title', 'N/A')}\n"

        return ctx


class BudgetPlannerHandler(TaskHandler):
    """
    Handler for personalized budget planning
    """

    def __init__(self, llm_handler, data_aggregator: DataAggregator):
        super().__init__(llm_handler, data_aggregator)

    async def collect_budget_questionnaire(self) -> Dict:
        return {
            "monthly_income": 0,
            "expenses": {},
            "financial_goals": [],
            "saving_target": 0,
            "location": "",
            "family_size": 1,
            "existing_debts": []
        }

    async def generate_budget_plan(self, query: str, rag_context: str,user_id: str = "default_user") -> Dict:
        """Generate personalized budget plan"""

        prompt = f"""
        You are a certified, meticulous personal finance advisor.

        CHAT HISTORY:
        {self._load_chat_history(user_id)}

        Your MANDATORY Task:
        Generate a complete, strictly valid budget plan in the final requested Markdown format.

        CONTEXT (Use for guidance, but prioritize Rules):
        {rag_context[:1000] if rag_context else 'No context available'}

        USER QUESTION/INPUT:
        {query}

        ---
        STRICT OUTPUT GENERATION RULES:
        ---
        1.  **MANDATORY CATEGORIES:** You MUST include ALL five of the following categories in the final output structure:
            * Essentials
            * Savings
            * Discretionary
            * Debt Repayment
            * Insurance/Safety

        2.  **NUMERICAL CONSTRAINTS:**
            * The **Savings** allocation MUST be greater than or equal to 20% of the total income (Savings >= 20% of Income).
            * **Essentials** MUST explicitly include allocations for: housing, food, and utilities.

        3.  **RATIONALE:** You MUST provide a short, specific rationale *immediately* following the allocated amount for each of the five mandatory categories.

        4.  **FORMATTING/EXCLUSION:**
            * The ONLY output allowed is the final budget plan in the requested Markdown format.
            * You MUST NOT include any introductory sentences, conversational text, closing remarks, or commentary outside of the final structured output.

        5. **OUTPUT STRUCTURE:** If User has asked about sending email, do not say that as AI I can not send email. Just provide the JSON output. with email body and context that user has asked for.
        if User has asked for PDF generation, do not say that as AI I can not generate PDF. Just provide the Markdown output with PDF content that user has asked for.
        ---
        MANDATORY OUTPUT FORMAT TEMPLATE:
        (Use such structure for the final answer as given below and add more details as much as possible, add table if needed)
        ---
        ## 💰 Monthly Budget Plan: [Insert Income Amount Here]

        * **Essentials:** [Numeric Value] (Rationale: [Insert Rationale Here])
            * Housing: [Numeric Value]
            * Food: [Numeric Value]
            * Utilities: [Numeric Value]
        * **Savings:** [Numeric Value] (Rationale: [Insert Rationale Here])
        * **Debt Repayment:** [Numeric Value] (Rationale: [Insert Rationale Here])
        * **Insurance/Safety:** [Numeric Value] (Rationale: [Insert Rationale Here])
        * **Discretionary:** [Numeric Value] (Rationale: [Insert Rationale Here])
        ---
        **Total Allocated:** [Sum of all main category values]

        ---
        FINAL RESPONSE:
        If User has asked about sending email or generate PDF, do not say that as AI I can not send email. 
            - do not draft any email content. Just provide the Markdown output with PDF content that user has asked for.
            - do not mention like this:
                Subject: Your Personalized Monthly Budget Plan
                Hi Karan,
                Here is the detailed monthly budget plan you requested, designed to help
                you manage your 50000 income effectively and achieve your financial goals.

            -ONLY draft PDF content that user has asked for like this.
                Monthly Budget Plan: 50000
                Essentials: 20000 (Rationale: Covers......
        If none of the above, provide the final budget plan in the requested Markdown format.
        
        (Return ONLY the Markdown object)
        ---

        """

        raw = await self.llm.generate(prompt)

        return {
            "answer": raw,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "task_type": "budget_planning"
        }
class RiskProfilerHandler(TaskHandler):
    """
    Handler for investment risk profiling
    """

    def __init__(self, llm_handler, data_aggregator: DataAggregator):
        super().__init__(llm_handler, data_aggregator)

    async def collect_risk_questionnaire(self) -> Dict:
        # Defaults if user skips
        defaults = {
            "age": None,
            "investment_horizon": "",
            "investment_amount": None,
            "income": None,
            "current_savings": None,
            "risk_experience": "",
            "loss_tolerance": "",
            "investment_goals": [],
            "dependents": None,
            "existing_debts": None
        }

        # Questions with correct keys
        questions = {
            "age": "What is your age?",
            "investment_horizon": "What is your investment horizon? (short / medium / long)",
            "investment_amount": "How much do you plan to invest (approx)?",
            "income": "What is your monthly income (approx)?",
            "current_savings": "How much do you currently have saved?",
            "risk_experience": "What is your investing experience? (beginner / intermediate / advanced)",
            "loss_tolerance": "How much loss can you tolerate in the short term? (low / medium / high)",
            "investment_goals": "What are your primary investment goals? (growth / income / capital protection)",
            "dependents": "How many financial dependents do you have?",
            "existing_debts": "Do you have any existing debts? (approx amount)"
        }

        answers = {}

        for key, question in questions.items():
            user_answer = input(question + " ")

            # If user skips → leave empty / None
            if not user_answer:
                answers[key] = defaults[key]
                continue

            # Handle numeric fields
            if key in ["age", "investment_amount", "income", "current_savings", "dependents", "existing_debts"]:
                try:
                    answers[key] = int(user_answer)
                except:
                    answers[key] = defaults[key]   # fallback to None
            elif key == "investment_goals":
                answers[key] = [g.strip().lower() for g in user_answer.split(",")]
            else:
                answers[key] = user_answer.strip().lower()

        return answers

    async def calculate_risk_profile(self, query, questionnaire: Dict) -> Dict:
        """Calculate investment risk profile"""

        prompt = f"""
        You are a certified SEBI-compliant, meticulous investment advisor. Your SOLE task is to **CALCULATE** the risk profile and return a single, STRICT, complete Markdown object. **DO NOT perform an analysis or commentary on missing information.**

        CHAT HISTORY:
        {self._load_chat_history("default_user")}

        
        Markdown MUST contain the following fields:
        - risk_level: MUST be one of "Conservative" | "Moderate" | "Aggressive"
        - risk_score: MUST be an integer between 0 and 100
        - asset_allocation: MUST contain {{Stocks:%, Bonds:%, Cash:%, RealEstate:%, Alternative:%}}. ALL percentages MUST sum to exactly 100%.
        - rebalancing_schedule: MUST suggest a practical schedule (e.g., "Annually", "Semi-Annually").
        - warnings: MUST include 1-2 safety notes (e.g., "Returns are not guaranteed," "Subject to inflation risk").
        - brief_reason: MUST be a concise, 2-line explanation of the assessment, *based only on the data provided*.

        STRICT RULES:
        1.  **NO EXTERNAL TEXT:** You MUST NOT output any introductory text, closing remarks, or explanation outside of the final, complete Markdown object.
        2.  **IMMEDIATE CALCULATION:** Process the questionnaire data immediately and jump straight to the Markdown output. Do not analyze missing fields.
        3.  **DATA INTEGRITY:** Ensure all numerical rules (100% allocation sum) are met.
        4. **OUTPUT STRUCTURE:** If User has asked about sending email, do not say that as AI I can not send email. Just provide the dict output with email_body and attachment_context that user has asked for.
        if User has asked for PDF generation, do not say that as AI I can not generate PDF. Just provide the Markdown output with PDF content that user has asked for.

        USER QUESTIONNAIRE (Use this data for calculation):
        {query}

        QUESTIONNAIRE DATA:
        {json.dumps(questionnaire, indent=2)}

        ---
        FINAL RESPONSE:

        If User has asked about sending email or generate PDF, do not say that as AI I can not send email. 
            - do not draft any email content. Just provide the Markdown output with PDF content that user has asked for.
            - do not mention like this:
                Subject: Your Personalized Monthly Budget Plan
                Hi Karan,
                Here is the detailed monthly budget plan you requested, designed to help
                you manage your 50000 income effectively and achieve your financial goals.

            - ONLY draft PDF content that user has asked for like this.
                Monthly Budget Plan: 50000
                Essentials: 20000 (Rationale: Covers......

        If none of the above, provide the risk assessment in the requested Markdown format.
        (Return ONLY the Markdown object)

        """

        raw = await self.llm.generate(prompt)

        # Parsing can be simple because your fallback logic already exists.
        assessment = self._parse_risk_assessment(raw, questionnaire, query)

        return assessment

    def _parse_risk_assessment(self, response: str, questionnaire: Dict, query: str) -> Dict:
        """Existing calculation logic kept for safety"""

        age = questionnaire.get("age", 40)
        horizon = questionnaire.get("investment_horizon", "medium")
        loss = questionnaire.get("loss_tolerance", "medium")

        score = 50

        if age < 30: score += 15
        elif age > 60: score -= 15

        score += {"short": -10, "medium": 0, "long": 10}.get(horizon, 0)
        score += {"low": -15, "medium": 0, "high": 15}.get(loss, 0)

        score = max(0, min(100, score))

        if score < 30:
            level = "Conservative"
            alloc = {"Bonds": 60, "Cash": 25, "Dividend Stocks": 10, "Alternative": 5}
        elif score < 60:
            level = "Moderate"
            alloc = {"Stocks": 40, "Bonds": 35, "Cash": 15, "Alternative": 10}
        else:
            level = "Aggressive"
            alloc = {"Stocks": 60, "Growth Stocks": 20, "Bonds": 10, "Alternative": 10}

        return {
            "answer": response,
            "query": query + json.dumps(questionnaire, indent=2),
            "risk_level": level,
            "risk_score": int(score),
            "asset_allocation": alloc,
            "recommendation": f"Based on your profile, a {level} strategy is suitable.",
            "rebalancing_schedule": "Quarterly",
            "timestamp": datetime.now().isoformat()
        }


__all__ = [
    'FinancialQAHandler',
    'BudgetPlannerHandler',
    'RiskProfilerHandler'
]
