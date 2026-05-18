import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class IntentType(str, Enum):
    """Types of user intents"""
    QA = "qa"
    BUDGET_PLAN = "budget_plan"
    SAVINGS_PLAN = "savings_plan"
    EMERGENCY_FUND = "emergency_fund"
    EXPENSE_TRACKING = "expense_tracking"
    GENERAL = "general"

class ConversationMessage:
    """Represents a single message in conversation"""

    def __init__(self, role: str, content: str, intent: Optional[IntentType] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = datetime.now()
        self.intent = intent

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent.value if self.intent else None
        }

class ConversationState:
    """Manages conversation state for a single user"""

    def __init__(self, user_id: str, max_history: int = 10):
        self.user_id = user_id
        self.messages: List[ConversationMessage] = []
        self.max_history = max_history

        # User financial profile
        self.user_profile = {
            "income": None,
            "monthly_expenses": {},
            "goals": [],
            "risk_profile": None
        }

        logger.info(f"✅ Created conversation state for user: {user_id}")

    def add_message(self, role: str, content: str, intent: Optional[IntentType] = None):
        """Add message to conversation"""
        message = ConversationMessage(role, content, intent)
        self.messages.append(message)

        # Keep only recent messages
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history:]
            logger.info(f"⚠️ Trimmed conversation history to last {self.max_history} messages")

        logger.info(f"📝 Added message from {role}")

    def get_context(self, max_turns: int = 5) -> str:
        """Get formatted recent conversation for LLM context"""
        recent = self.messages[-max_turns:]
        context_lines = []

        for msg in recent:
            context_lines.append(f"{msg.role.upper()}: {msg.content}")

        return "\n".join(context_lines)

    def extract_entities(self, text: str) -> Dict:
        """Extract financial entities from user text"""
        import re

        entities = {
            "income": None,
            "expenses": {},
            "goals": []
        }

        # Extract amounts in Rs format
        amount_pattern = r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        amounts = re.findall(amount_pattern, text, re.IGNORECASE)

        # Extract expense categories
        categories = ['rent', 'food', 'groceries', 'transport', 'utilities', 'insurance', 'travel']
        found_categories = {}

        for cat in categories:
            if cat.lower() in text.lower():
                found_categories[cat] = True

        # If income pattern found
        if 'income' in text.lower() or 'earn' in text.lower() or 'salary' in text.lower():
            if amounts:
                entities["income"] = int(amounts[0].replace(',', ''))

        # Extract goals
        goal_keywords = {
            'emergency': 'emergency_fund',
            'invest': 'investment',
            'retire': 'retirement',
            'travel': 'vacation',
            'house': 'home_purchase'
        }

        for keyword, goal in goal_keywords.items():
            if keyword in text.lower():
                if goal not in entities["goals"]:
                    entities["goals"].append(goal)

        logger.info(f"📊 Extracted entities: {entities}")
        return entities

    def get_profile_summary(self) -> str:
        """Get summary of user profile"""
        profile = self.user_profile

        summary = []
        if profile["income"]:
            summary.append(f"Monthly Income: Rs {profile['income']:,}")

        if profile["monthly_expenses"]:
            total_exp = sum(profile["monthly_expenses"].values())
            summary.append(f"Total Monthly Expenses: Rs {total_exp:,}")

        if profile["goals"]:
            summary.append(f"Goals: {', '.join(profile['goals'])}")

        return " | ".join(summary) if summary else "No profile data yet"

class FinancialAdvisorAgent:
    """Main agent that processes user queries"""

    def __init__(self, rag_pipeline, llm_handler, tools: Dict):
        """Initialize agent"""
        self.rag_pipeline = rag_pipeline
        self.llm_handler = llm_handler
        self.tools = tools
        self.conversations: Dict[str, ConversationState] = {}

        logger.info("✅ Financial Advisor Agent initialized")

    def _detect_intent(self, query: str) -> IntentType:
        """Detect user intent from query"""
        query_lower = query.lower()

        intent_keywords = {
            'budget': IntentType.BUDGET_PLAN,
            'expense': IntentType.EXPENSE_TRACKING,
            'emergency': IntentType.EMERGENCY_FUND,
            'save': IntentType.SAVINGS_PLAN,
            'saving': IntentType.SAVINGS_PLAN,
        }

        for keyword, intent in intent_keywords.items():
            if keyword in query_lower:
                logger.info(f"🎯 Detected intent: {intent.value}")
                return intent

        logger.info("🎯 Detected intent: general Q&A")
        return IntentType.QA

    def _handle_qa(self, query: str, conv: ConversationState) -> Dict:
        """Handle question-answering"""
        logger.info(f"📖 Handling Q&A: {query}")

        # Retrieve relevant chunks
        chunks_with_scores = self.rag_pipeline.retrieve_with_reranking(query, top_k=5)

        if not chunks_with_scores:
            return {
                "answer": "I couldn\'t find specific information on this topic. Could you rephrase your question?",
                "sources": [],
                "type": "qa",
                "confidence": 0.2
            }

        chunks = [chunk for chunk, _ in chunks_with_scores]
        context = self.rag_pipeline.format_context(chunks)

        # Generate answer with LLM
        system_prompt = """You are a Financial Literacy Expert specializing in budgeting and saving.

Guidelines:
- Answer based on the provided context
- Be practical and actionable
- Use Indian financial context
- Keep responses concise (2-3 paragraphs max)
- If info not in context, say so
"""

        user_prompt = f"""Context:
{context}

Question: {query}

Provide a clear, practical answer."""

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        answer = self.llm_handler.generate(full_prompt, max_tokens=300)

        return {
            "answer": answer,
            "sources": chunks,
            "type": "qa",
            "confidence": 0.8
        }

    def _handle_budget_plan(self, query: str, conv: ConversationState) -> Dict:
        """Handle budget planning request"""
        logger.info("💰 Handling budget plan request")

        profile = conv.user_profile

        if profile["income"] is None:
            return {
                "answer": "To create a budget plan, I need to know your monthly income. What\'s your monthly take-home income in rupees?",
                "type": "clarification",
                "confidence": 0.5
            }

        # Retrieve budget strategies
        chunks_with_scores = self.rag_pipeline.retrieve_with_reranking(
            "budgeting strategy, monthly budget, 50-30-20 rule",
            top_k=3
        )

        chunks = [chunk for chunk, _ in chunks_with_scores] if chunks_with_scores else []
        context = self.rag_pipeline.format_context(chunks)

        # Generate personalized budget advice
        prompt = f"""You are a financial advisor. Create a personalized budget recommendation.

User Profile:
- Income: Rs {profile['income']}/month
- Expenses: {profile['monthly_expenses']}

Reference Information:
{context}

Provide:
1. 50-30-20 rule breakdown
2. Recommended allocation for their situation
3. Tips to optimize spending
4. Action items

Keep it under 400 words."""

        advice = self.llm_handler.generate(prompt, max_tokens=400)

        return {
            "answer": advice,
            "type": "budget_plan",
            "income": profile["income"],
            "expenses": profile["monthly_expenses"],
            "confidence": 0.85
        }

    def _handle_emergency_fund(self, query: str, conv: ConversationState) -> Dict:
        """Handle emergency fund calculation"""
        logger.info("🆘 Handling emergency fund request")

        profile = conv.user_profile
        total_expenses = sum(profile["monthly_expenses"].values()) if profile["monthly_expenses"] else None

        if total_expenses is None:
            return {
                "answer": "To calculate your emergency fund, I need to know your monthly expenses. What are your typical monthly expenses?",
                "type": "clarification",
                "confidence": 0.5
            }

        emergency_fund_amount = total_expenses * 6
        monthly_target = emergency_fund_amount / 12

        # Retrieve emergency fund strategies
        chunks_with_scores = self.rag_pipeline.retrieve_with_reranking(
            "emergency fund, building savings, financial security",
            top_k=3
        )

        chunks = [chunk for chunk, _ in chunks_with_scores] if chunks_with_scores else []
        context = self.rag_pipeline.format_context(chunks)

        prompt = f"""You are a financial advisor. Provide emergency fund guidance.

Emergency Fund Target: Rs {emergency_fund_amount:,} (6 months of Rs {total_expenses:,} expenses)
Monthly Savings Target: Rs {monthly_target:,.0f}

Reference Information:
{context}

Provide:
1. Why this emergency fund is needed
2. Step-by-step building strategy
3. Where to keep the emergency fund
4. What counts as emergency
5. Do\'s and Don\'ts

Keep it under 400 words."""

        advice = self.llm_handler.generate(prompt, max_tokens=400)

        return {
            "answer": advice,
            "type": "emergency_fund",
            "calculation": {
                "monthly_expenses": total_expenses,
                "emergency_fund": emergency_fund_amount,
                "monthly_target": monthly_target
            },
            "confidence": 0.9
        }

    def process_query(self, user_id: str, query: str) -> Dict:
        """Main entry point for processing user queries"""
        logger.info(f"🔄 Processing query from {user_id}: {query[:50]}...")

        # Get or create conversation
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationState(user_id)

        conv = self.conversations[user_id]

        # Extract entities
        entities = conv.extract_entities(query)
        if entities["income"]:
            conv.user_profile["income"] = entities["income"]
        if entities["expenses"]:
            conv.user_profile["monthly_expenses"].update(entities["expenses"])
        if entities["goals"]:
            for goal in entities["goals"]:
                if goal not in conv.user_profile["goals"]:
                    conv.user_profile["goals"].append(goal)

        # Detect intent
        intent = self._detect_intent(query)

        # Route to appropriate handler
        if intent == IntentType.QA:
            response = self._handle_qa(query, conv)
        elif intent == IntentType.BUDGET_PLAN:
            response = self._handle_budget_plan(query, conv)
        elif intent == IntentType.EMERGENCY_FUND:
            response = self._handle_emergency_fund(query, conv)
        else:
            response = self._handle_qa(query, conv)

        # Add to conversation history
        conv.add_message("user", query, intent)
        conv.add_message("assistant", response["answer"], intent)

        response["user_profile"] = conv.user_profile
        response["intent"] = intent.value

        logger.info(f"✅ Response ready (confidence: {response.get('confidence', 0):.2f})")

        return response


if __name__ == "__main__":
    print("Agent module ready for import")
