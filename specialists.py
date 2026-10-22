# ============================================================================
# VERIFICATION & SPECIALIST AGENTS MODULE
# Handles all verification checks and specialist agent implementations
# ============================================================================

import asyncio
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

from config import (
    Issue, IssueType, IssueSeverity, VerificationResult, 
    IterationLog, Config, logger
)

logger = logging.getLogger(__name__)

import re

def extract_score(text: str) -> int:
    match = re.search(r"Score\s*[:\-]?\s*(\d{1,3})", text)
    if match:
        score = int(match.group(1))
        return min(score, 100)  # clip to max 100
    return None

# ============================================================================
# ISSUE DETECTION MODULE
# ============================================================================

class IssueDetector:
    """
    Detects and classifies issues in LLM responses
    """
    
    def __init__(self, llm_handler):
        """Initialize issue detector"""
        self.llm = llm_handler
        self.compliance_rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict:
        """Load financial compliance rules"""
        return {
            "emergency_fund_minimum": 3,  # months
            "emergency_fund_maximum": 12,  # months
            "maximum_single_stock": 10,  # percentage
            "minimum_diversification": 3,  # number of assets
            "maximum_debt_income_ratio": 0.4,  # 40% of income
        }
    
    async def detect_issues(self, answer: str, context: str, query: str) -> VerificationResult:
        """
        Detect all issues in the answer using deep verification.
        """

        issues = []

        # Check factual accuracy via claim validation
        factual_check = await self._check_factuality_llm(answer, context)
        if factual_check['has_issue']:
            issues.append(factual_check['issue'])

        # Compliance & regulatory red flags
        # compliance_check = await self._check_compliance(answer, query)
        # if compliance_check['has_issue']:
        #     issues.append(compliance_check['issue'])

        # # Completeness & explanation
        # complete_check = await self._check_completeness(answer, query)
        # if complete_check['has_issue']:
        #     issues.append(complete_check['issue'])

        # Does the answer address the user’s question?
        intent_check = await self._check_intent_alignment(answer, query)
        if intent_check['has_issue']:
            issues.append(intent_check['issue'])

        # Final LLM verification score
        judge_result = await self._run_final_verification(query, answer, context, issues)
        try:
            score = extract_score(judge_result) 
        except:
            score = None

        # Choose the most severe issue as the primary one
        severity_order = {IssueSeverity.HIGH: 0, IssueSeverity.MEDIUM: 1, IssueSeverity.LOW: 2}
        issues.sort(key=lambda x: severity_order.get(x.severity, 3))

        return VerificationResult(
            has_issues=len(issues) > 0,
            issues=issues,
            primary_issue=issues[0] if issues else None,
            specialist_needed=issues[0].issue_type.value if issues else None,
            confidence=max([i.confidence for i in issues]) if issues else None
        )

    
    async def _check_factuality_llm(self, answer: str, context: str) -> Dict:
        try:
            extract_prompt = f"""
            Extract all factual claims from the following answer (each must be verifiable):

            ANSWER:
            {answer}

            List as:
            1. Claim text
            2. Type: [statistical | financial | time-bound | recommendation]
            3. Confidence (if verifiable from context): HIGH / MEDIUM / LOW
            """
            claims = await self.llm.generate(extract_prompt)
            if claims.strip() == "":
                return {"has_issue": False, "issue": None}
            validate_prompt = f"""
            Context:
            {context}

            Given this context, validate the following claims:
            {claims}

            For each claim, say whether it's supported (Y/N/Partial) and why.
            Summarize any unsupported or contradictory ones.
            """
            validation = await self.llm.generate(validate_prompt)

            if "unsupported" in validation.lower() or "contradiction" in validation.lower():
                return {
                    "has_issue": True,
                    "issue": Issue(
                        issue_type=IssueType.FACTUAL_ERROR,
                        severity=IssueSeverity.HIGH,
                        reason="Claim not supported by context",
                        confidence=0.85,
                        details=validation
                    )
                }
            return {"has_issue": False, "issue": None}
        except Exception as e:
            logger.error(f"Factual check failed: {e}")
            return {"has_issue": False, "issue": None}
    
    async def _check_compliance(self, answer: str, query: str) -> Dict:
        """Check financial compliance and safety"""
        try:
            answer_lower = answer.lower()
            
            # Check for red flags
            red_flags = [
                ("high_risk_for_age", ["100% stock", "retiree", "age 60", "senior"]),
                ("guaranteed_returns", ["guaranteed return", "risk-free", "certain profit"]),
                ("single_investment", ["single stock", "only invest in", "all in"]),
                ("excessive_debt", ["borrow 100%", "leverage everything"]),
                ("unrealistic", ["double your money", "guaranteed 50%"]),
            ]
            
            violations = []
            for flag_type, patterns in red_flags:
                # Check if multiple patterns match (compound violations)
                pattern_matches = sum(1 for pattern in patterns if pattern in answer_lower)
                if pattern_matches >= 2:
                    violations.append(flag_type)
            
            if violations:
                issue = Issue(
                    issue_type=IssueType.DOMAIN_ERROR,
                    severity=IssueSeverity.HIGH,
                    reason=f"Compliance violation detected: {', '.join(violations)}",
                    confidence=0.85,
                    details=str(violations)
                )
                return {'has_issue': True, 'issue': issue}
            
            return {'has_issue': False, 'issue': None}
            
        except Exception as e:
            logger.error(f"Compliance check error: {e}")
            return {'has_issue': False, 'issue': None}
    
    async def _check_completeness(self, answer: str, query: str) -> Dict:
        """Check if answer is complete"""
        try:
            word_count = len(answer.split())
            
            # Check for reasoning indicators
            reasoning_indicators = [
                'because', 'therefore', 'reason', 'since', 'given',
                'calculation', 'step', 'process', 'approach'
            ]
            has_reasoning = any(
                indicator in answer.lower() 
                for indicator in reasoning_indicators
            )
            
            # Minimum word count depends on question type
            min_words = 50 if len(query) < 50 else 80
            
            is_complete = word_count >= min_words and has_reasoning
            
            if not is_complete:
                issue = Issue(
                    issue_type=IssueType.COMPLETENESS_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    reason=f"Incomplete answer (words: {word_count}, reasoning: {has_reasoning})",
                    confidence=0.75
                )
                return {'has_issue': True, 'issue': issue}
            
            return {'has_issue': False, 'issue': None}
            
        except Exception as e:
            logger.error(f"Completeness check error: {e}")
            return {'has_issue': False, 'issue': None}
    
    async def _check_intent_alignment(self, answer: str, query: str) -> Dict:
        prompt = f"""
    Question: {query}

    Answer: {answer}

    Does the answer directly and fully resolve what the user asked?
    Reply YES/NO with reasoning. Flag missing parts.
        """
        result = await self.llm.generate(prompt)
        if "no" in result.lower():
            return {
                "has_issue": True,
                "issue": Issue(
                    issue_type=IssueType.COMPLETENESS_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    reason="Answer doesn't fully resolve user query",
                    confidence=0.75,
                    details=result
                )
            }
        return {"has_issue": False, "issue": None}

    async def _run_final_verification(self, query, answer, context, iterations) -> Dict:
        prompt = f"""
            User Query:
            {query}

            Final Answer:
            {answer}

            Context Used:
            {context}

            Fix Attempts:
            {iterations}

            Evaluate:
            - Is the answer correct?
            - Is it safe and compliant?
            - Is it logically valid?
            - Does it answer the question?

            Score it: 0–100.
            If score < 80, explain what's still wrong.
        """
        result = await self.llm.generate(prompt)
        return result


# ============================================================================
# BASE SPECIALIST AGENT
# ============================================================================

class SpecialistAgent(ABC):
    """Base class for specialist agents"""
    
    def __init__(self, llm_handler, rag_pipeline):
        """Initialize specialist"""
        self.llm = llm_handler
        self.rag = rag_pipeline
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def fix_issue(self,
                       incorrect_answer: str,
                       query: str,
                       context: str,
                       issue: Issue) -> Dict:
        """Fix an issue and return corrected answer"""
        pass

# ============================================================================
# DATA SPECIALIST - Fixes Factual Errors
# ============================================================================

class DataSpecialist(SpecialistAgent):
    """Fixes factual errors by re-retrieving and validating data"""
    
    async def fix_issue(self,
                       incorrect_answer: str,
                       query: str,
                       context: str,
                       issue: Issue) -> Dict:
        logger.info("🔍 DataSpecialist: Fixing factual error...")
        
        try:
            # Step 1: Identify incorrect claims
            false_claims_prompt = f"""
            Identify the INCORRECT statements in this answer:

            {incorrect_answer}

            Return a concise list of incorrect points.
            """
            false_claims = await self.llm.generate(false_claims_prompt)

            # Step 2: Retrieve correct info
            search_queries = [
                query,
                f"correct information about {query}",
                f"updated facts for {query}"
            ]

            all_context = []
            if self.rag:
                for sq in search_queries:
                    try:
                        docs = self.rag.retrieve(sq)
                        all_context.extend([d.page_content for d in docs])
                    except:
                        pass

            merged_context = "\n".join(all_context[:5])

            # Step 3: Generate corrected answer
            correction_prompt = f"""
            The earlier answer contained factual mistakes:
            {false_claims}

            Using ONLY the following correct context:
            {merged_context}

            Rewrite the answer correctly and clearly.
            """
            corrected_answer = await self.llm.generate(correction_prompt)

            return {
                'fixed_answer': corrected_answer,
                'specialist': self.name,
                'fix_description': f"Fixed factual inaccuracies: {false_claims[:120]}",
                'confidence': 0.80
            }

        except Exception as e:
            logger.error(f"DataSpecialist Error: {e}")
            return {
                'fixed_answer': incorrect_answer,
                'specialist': self.name,
                'fix_description': f"Error during fix: {str(e)}",
                'confidence': 0.40
            }


# ============================================================================
# LOGIC SPECIALIST - Fixes Calculation Errors
# ============================================================================

class LogicSpecialist(SpecialistAgent):
    """Fixes mathematical and logical errors"""

    async def fix_issue(self,
                       incorrect_answer: str,
                       query: str,
                       context: str,
                       issue: Issue) -> Dict:

        logger.info("🧮 LogicSpecialist: Fixing logic error...")

        try:
            # Step 1: Break down the error
            breakdown_prompt = f"""
            The following answer contains a LOGICAL or MATHEMATICAL mistake:

            {incorrect_answer}

            ERROR DETAILS: {issue.reason}

            Step-by-step:
            1. Identify incorrect logic.
            2. Explain why it is wrong.
            3. Show the correct approach.
            4. Recalculate step-by-step.
            """
            step_by_step = await self.llm.generate(breakdown_prompt)

            # Step 2: Generate corrected answer
            correction_prompt = f"""
            Using the correct reasoning:

            {step_by_step}

            Rewrite the final correct answer to the question:
            {query}

            Provide clean, correct logic.
            """
            corrected_answer = await self.llm.generate(correction_prompt)

            return {
                'fixed_answer': corrected_answer,
                'specialist': self.name,
                'fix_description': f"Logical correction applied: {issue.reason[:120]}",
                'confidence': 0.85
            }

        except Exception as e:
            logger.error(f"LogicSpecialist Error: {e}")
            return {
                'fixed_answer': incorrect_answer,
                'specialist': self.name,
                'fix_description': f"Error during fix: {str(e)}",
                'confidence': 0.40
            }


# ============================================================================
# COMPLIANCE SPECIALIST - Fixes Regulatory / Safety Errors
# ============================================================================

class ComplianceSpecialist(SpecialistAgent):
    """Fixes compliance and regulatory issues"""

    async def fix_issue(self,
                       incorrect_answer: str,
                       query: str,
                       context: str,
                       issue: Issue) -> Dict:

        logger.info("⚖️ ComplianceSpecialist: Fixing compliance issues...")

        try:
            # Step 1: Analyze violation
            violation_prompt = f"""
            This answer violates financial safety/regulation guidelines:

            {incorrect_answer}

            VIOLATION: {issue.reason}

            Explain why the advice is unsafe.
            """
            violation_analysis = await self.llm.generate(violation_prompt)

            # Step 2: Rewrite safe compliant version
            fix_prompt = f"""
            Rewrite the advice to be 100% SAFE and SEBI-compliant.

            ORIGINAL ANSWER:
            {incorrect_answer}

            PROBLEM:
            {violation_analysis}

            QUESTION:
            {query}

            Produce safe, diversified, cautious financial guidance.
            """
            compliant_version = await self.llm.generate(fix_prompt)

            return {
                'fixed_answer': compliant_version,
                'specialist': self.name,
                'fix_description': f"Compliance fix applied: {issue.reason[:120]}",
                'confidence': 0.90
            }

        except Exception as e:
            logger.error(f"ComplianceSpecialist Error: {e}")
            return {
                'fixed_answer': incorrect_answer,
                'specialist': self.name,
                'fix_description': f"Error during fix: {str(e)}",
                'confidence': 0.40
            }


# ============================================================================
# COMPLETION SPECIALIST - Fixes Incomplete Responses
# ============================================================================

class CompletionSpecialist(SpecialistAgent):
    """Adds missing information and improves completeness"""

    async def fix_issue(self,
                       incorrect_answer: str,
                       query: str,
                       context: str,
                       issue: Issue) -> Dict:

        logger.info("📝 CompletionSpecialist: Expanding incomplete answer...")

        try:
            # Step 1: Identify missing content
            missing_prompt = f"""
            This answer is incomplete:

            {incorrect_answer}

            For the question:
            {query}

            List 3–5 important missing elements.
            """
            missing_items = await self.llm.generate(missing_prompt)

            # Step 2: Expand the answer
            expansion_prompt = f"""
            Expand the following answer by including all missing important elements:

            CURRENT ANSWER:
            {incorrect_answer}

            MISSING ELEMENTS:
            {missing_items}

            Produce a full, clear, complete answer.
            """
            expanded_answer = await self.llm.generate(expansion_prompt)

            return {
                'fixed_answer': expanded_answer,
                'specialist': self.name,
                'fix_description': f"Expanded incomplete answer",
                'confidence': 0.75
            }

        except Exception as e:
            logger.error(f"CompletionSpecialist Error: {e}")
            return {
                'fixed_answer': incorrect_answer,
                'specialist': self.name,
                'fix_description': f"Error during fix: {str(e)}",
                'confidence': 0.40
            }


__all__ = [
    'IssueDetector',
    'SpecialistAgent',
    'DataSpecialist',
    'LogicSpecialist',
    'ComplianceSpecialist',
    'CompletionSpecialist'
]
