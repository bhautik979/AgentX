# dragonllm_task_detector.py
# Updated: returns structured output with enhanced_query, tools_needed, apis_needed, actions
import logging
import json
from typing import Dict, Tuple, Optional, List, Any
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import TaskType, logger

logger = logging.getLogger(__name__)


class DragonLLMTaskDetector:
    """
    Enhanced DragonLLM Task Detector.
    New public method: detect_and_enhance(query) -> Dict:
      {
        "task_type": TaskType,
        "enhanced_query": str,
        "tools_needed": List[str],
        "apis_needed": List[str],
        "actions": List[str],
        "metadata": {...}
      }
    Fallbacks to keyword-matching if LLM unavailable or parsing fails.
    """

    def __init__(self, device: str = "cpu", use_pipeline: bool = True):
        self.device = device
        self.use_pipeline = use_pipeline
        self.model = None
        self.tokenizer = None
        self.pipe = None

        # minimal task definitions (keeps existing mapping)
        self.task_definitions = {
            TaskType.BUDGET_PLANNING: {
                "name": "Budget Planning",
                "keywords": ["budget", "expense", "allocation", "spending", "distribute", "percentage"]
            },
            TaskType.RISK_PROFILING: {
                "name": "Risk Profiling",
                "keywords": ["risk", "invest", "portfolio", "stock", "bond", "allocation", "aggressive", "conservative"]
            },
            TaskType.DEBT_MANAGEMENT: {
                "name": "Debt Management",
                "keywords": ["debt", "loan", "credit", "card", "repay", "interest", "emi", "payoff"]
            },
            TaskType.SAVINGS_OPTIMIZATION: {
                "name": "Savings Optimization",
                "keywords": ["save", "saving", "emergency", "fund", "goal", "accumulate", "corpus"]
            },
            TaskType.GENERAL_QA: {
                "name": "General Q&A",
                "keywords": ["what", "how", "why", "concept", "explain", "understand", "define"]
            }
        }

        self._initialize_model()

    def _initialize_model(self):
        """Load local/model weights — keep safe fallback if unavailable."""
        try:
            model_name = "instruction-pretrain/finance-Llama3-8B"
            cache_dir = "/ssd_scratch/sathya.marisetti"
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,cache_dir=cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                cache_dir=cache_dir
            )
            if self.use_pipeline:
                try:
                    self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
                    logger.info("DragonLLM loaded with pipeline")
                except Exception as e:
                    logger.warning("Pipeline init failed, will use direct model: %s", e)
                    self.model = model
                    self.tokenizer = tokenizer
            else:
                self.model = model
                self.tokenizer = tokenizer
        except Exception as e:
            logger.warning("DragonLLM initialization failed: %s. Detector will use fallback keyword matcher.", e)
            self.model = None
            self.tokenizer = None
            self.pipe = None

    # ------------------------------
    # Public enhanced detection API
    # ------------------------------
    def detect_and_enhance(self, query: str) -> Dict[str, Any]:
        """
        Returns structured detection + enhancement output.
        If LLM unavailable or parsing fails, returns fallback mapping with
        enhanced_query equal to original query.
        """
        # Default structure
        result = {
            "task_type": None,
            "enhanced_query": query,
            "tools_needed": [],
            "apis_needed": [],
            "actions": [],
            "metadata": {
                "method": "dragonllm_enhanced",
                "confidence": 0.0,
                "raw_response": None,
                "fallback": False
            }
        }

        # If model is not available, fallback
        if not (self.pipe or (self.model and self.tokenizer)):
            fallback = self._fallback_detection(query)
            result.update({
                "task_type": fallback["task_type"],
                "metadata": {"method": "fallback_keyword", "confidence": fallback.get("confidence", 0.5), "fallback": True}
            })
            return result

        # Build instruction to produce structured JSON
        prompt = self._build_enhanced_prompt(query)

        # Generate
        try:
            raw = self._generate_response(prompt)
            result["metadata"]["raw_response"] = raw[:2000] if raw else ""
            # Try parse JSON from raw response (LLM instructed to give strict JSON)
            parsed = self._try_parse_json(raw)
            if parsed:
                # Map parsed to result fields with safety checks
                task_name = parsed.get("task_type") or parsed.get("primary_task")
                task_type = self._map_task_name_to_type(task_name) or self._fallback_detection(query)["task_type"]
                result["task_type"] = task_type
                result["enhanced_query"] = parsed.get("enhanced_query", query).strip() or query
                result["tools_needed"] = parsed.get("tools_needed", []) or []
                result["apis_needed"] = parsed.get("apis_needed", []) or []
                result["actions"] = parsed.get("actions", []) or []
                # Confidence
                conf = parsed.get("confidence", None)
                try:
                    result["metadata"]["confidence"] = float(conf) / 100.0 if conf is not None else 0.8
                except:
                    result["metadata"]["confidence"] = 0.8
                result["metadata"]["method"] = "dragonllm_structured"
                result["metadata"]["raw_parsed"] = parsed
                return result
            else:
                logger.warning("Could not parse JSON from DragonLLM response — using fallback mapping")
                # fallback but try some lightweight extraction
                fallback = self._fallback_detection(query)
                result.update({
                    "task_type": fallback["task_type"],
                    "enhanced_query": query,
                    "tools_needed": [],
                    "apis_needed": [],
                    "actions": [],
                    "metadata": {"method": "fallback_after_parse_fail", "confidence": fallback.get("confidence", 0.5), "fallback": True}
                })
                return result
        except Exception as e:
            logger.error("DragonLLM detection error: %s", e)
            fallback = self._fallback_detection(query)
            result.update({
                "task_type": fallback["task_type"],
                "metadata": {"method": "fallback_on_exception", "confidence": fallback.get("confidence", 0.5), "fallback": True}
            })
            return result

    # ------------------------------
    # Prompt building and generation
    # ------------------------------
    def _build_enhanced_prompt(self, query: str) -> str:
        """
        Enhanced DragonLLM prompt that separates:
        - retrieval_query (for RAG)
        - output_instructions (for Gemini final formatting)
        - tool/action detection
        """

        prompt = f"""
        You are an advanced financial intent analyzer. Read the user query and output a STRICT JSON object.

        User Input:
        \"\"\"{query}\"\"\"

        You MUST return JSON with EXACTLY these fields:

        1. enhanced_query:
        - Rewrite the financial question to improve clarity and context.
            Rules:
            - If user requests a PDF (pdf, report, export, document):
                "rewrite query mentioning that give structured report. Make the report detailed and well-formatted. Add tables if useful."
            - If user requests an email (email, mail, send):
                "rewrite query mentioning that give draft an email only. Include subject + body. Do NOT add anything else."
            - If user asks for BOTH:
                "rewrite query mentioning that give answer in structured report format. Then draft a separate email summarizing the report."
            - If neither PDF nor email:
                "rewrite query mentioning that provide a clear and concise answer."
        - Add relevant financial keywords to improve search

        3. task_type:
        One of:
        - Budget Planning
        - Risk Profiling
        - Debt Management
        - Savings Optimization
        - General Q&A

        4. tools_needed:
        List of tools required
        Examples: ["send_email_sendgrid"], ["generate_pdf_anvil"], []

        5. apis_needed:
        List of APIs required BEFORE answering
        Examples: ["stock_api", "news_api", "col_api"], []

        6. actions:
        What final output user requested
        Examples: ["generate_pdf"], ["send_email"], ["generate_pdf","send_email"], []

        7. confidence:
        Integer 0–100

        Return JSON ONLY. No comments. No explanation. No text outside JSON.
        """

        return prompt


    def _generate_response(self, prompt: str) -> str:
        if self.pipe:
            out = self.pipe(prompt, max_length=512, temperature=0.2, do_sample=False, return_full_text=False)
            return out[0]["generated_text"] if out and isinstance(out, list) else ""
        elif self.model and self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512, temperature=0.2)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            return ""

    # ------------------------------
    # Helpers
    # ------------------------------
    def _try_parse_json(self, raw: str) -> Optional[Dict]:
        """Try to locate a JSON object inside model output and parse it."""
        if not raw:
            return None
        # Find first '{' and last '}' to attempt extraction
        start = raw.find('{')
        end = raw.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        json_str = raw[start:end + 1]
        try:
            parsed = json.loads(json_str)
            return parsed
        except Exception:
            # try to fix common issues (single quotes)
            try:
                fixed = json_str.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                return None

    def _map_task_name_to_type(self, task_name: Optional[str]):
        if not task_name:
            return None
        tn = task_name.lower()
        mapping = {
            'budget': TaskType.BUDGET_PLANNING,
            'budget planning': TaskType.BUDGET_PLANNING,
            'risk': TaskType.RISK_PROFILING,
            'risk profiling': TaskType.RISK_PROFILING,
            'debt': TaskType.DEBT_MANAGEMENT,
            'savings': TaskType.SAVINGS_OPTIMIZATION,
            'general': TaskType.GENERAL_QA,
            'q&a': TaskType.GENERAL_QA,
            'question': TaskType.GENERAL_QA,
        }
        for k, v in mapping.items():
            if k in tn:
                return v
        return None

    def _fallback_detection(self, query: str) -> Dict:
        """Simple keyword-based fallback returning task_type and confidence."""
        q = query.lower()
        scores = {}
        for tt, info in self.task_definitions.items():
            kw = info.get("keywords", [])
            matches = sum(1 for w in kw if w in q)
            scores[tt] = matches / (len(kw) if kw else 1)
        best = max(scores, key=scores.get)
        return {"task_type": best, "confidence": scores[best]}

# Expose class
__all__ = ["DragonLLMTaskDetector"]
