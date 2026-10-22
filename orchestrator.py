import asyncio
import torch
import logging
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
from config import (
    Config, TaskType, IssueType, AgentResponse, logger
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from specialists import (
    IssueDetector, DataSpecialist, LogicSpecialist,
    ComplianceSpecialist, CompletionSpecialist
)
from dragonllm_task_detector import DragonLLMTaskDetector
from advanced_rag_pinecone import (
    PineconeConnector, init_embedding_model, init_reranker, HybridRetriever
)
from live_data_providers import DataAggregator
from output_generators import PDFGenerator, EmailSender
from specialized_task_handlers import (
    FinancialQAHandler,
    BudgetPlannerHandler,
    RiskProfilerHandler
)
import markdown
from weasyprint import HTML

try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    from langchain.llms.base import LLM as LangChainLLMBase
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False


logger = logging.getLogger(__name__)

# Ensure logs and chat_history directories exist
LOG_DIR = "./logs"
CHAT_HISTORY_DIR = "./chat_history"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

EMAIL_BODIES_BY_TASK = {
    "budget_planning": {
        "subject": "Your Personalized Budget Plan",
        "body": (
            "Hi,\n\n"
            "Attached is your personalized budget plan tailored to your income, spending habits, and financial goals.\n\n"
            "This plan outlines:\n"
            "- Core living expenses\n"
            "- Recommended savings contributions\n"
            "- Discretionary spending limits\n\n"
            "It's designed to help you manage your money more effectively while ensuring future stability.\n"
            "Review it at your convenience and feel free to reach out if you’d like to adjust any part of the plan.\n\n"
            "Best regards,\n"
            "Your Financial Assistant\n"
            "AgentX"
        )
    },
    "risk_profiling": {
        "subject": "Your Investment Risk Profile and Allocation Strategy",
        "body": (
            "Hi,\n\n"
            "Based on your responses, we've assessed your investment risk tolerance and created a recommended asset allocation.\n\n"
            "This report covers:\n"
            "- Your risk category (e.g., Conservative, Balanced, Aggressive)\n"
            "- Suggested asset mix across equities, bonds, and cash\n"
            "- Notes on risk-return tradeoffs\n\n"
            "Use this as a reference point for portfolio planning. Reach out if you'd like to explore adjustments or deeper analysis.\n\n"
            "Regards,\n"
            "Your Financial Assistant\n"
            "AgentX"
        )
    },
    "financial_qa": {
        "subject": "Summary of Your Financial Question and Response",
        "body": (
            "Hi,\n\n"
            "Here’s the response to your recent financial question, with context and relevant data included in the attached report.\n\n"
            "If the topic involves markets, investments, or regulations, remember this is general guidance and not a guarantee.\n\n"
            "Let us know if you'd like to explore this further or get a personalized breakdown.\n\n"
            "Thanks,\n"
            "Your Financial Assistant\n"
            "AgentX"
        )
    }
}


class MultiAgentOrchestratorV4:
    def __init__(self, llm_handler):
        self.llm = llm_handler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Detector
        self.task_detector = DragonLLMTaskDetector(device=self.device, use_pipeline=True)

        # Pinecone connector
        self.pinecone_connector = PineconeConnector(api_key=Config.PINECONE_API_KEY)
        self.embed_model = init_embedding_model(device=self.device)
        self.reranker = init_reranker(device=self.device)
        if self.pinecone_connector.is_connected() and self.embed_model:
            self.hybrid_retriever = HybridRetriever(
                pinecone_index=self.pinecone_connector.index,
                embed_model=self.embed_model,
                reranker=self.reranker,
                top_k=5,
                alpha=0.6,
                device=self.device
            )
        else:
            self.hybrid_retriever = None

        # Verifier and specialists
        self.issue_detector = IssueDetector(llm_handler)
        self.specialists = {
            IssueType.FACTUAL_ERROR.value: DataSpecialist(llm_handler, self.hybrid_retriever),
            IssueType.LOGICAL_ERROR.value: LogicSpecialist(llm_handler, self.hybrid_retriever),
            IssueType.DOMAIN_ERROR.value: ComplianceSpecialist(llm_handler, self.hybrid_retriever),
            IssueType.COMPLETENESS_ERROR.value: CompletionSpecialist(llm_handler, self.hybrid_retriever),
        }

        # Live data & outputs
        self.data_aggregator = DataAggregator()
        self.pdf_generator = PDFGenerator()
        self.email_sender = EmailSender()

                # --- LangChain tools (optional) ---
        self.langchain_agent = None
        if HAS_LANGCHAIN:
            try:
                self._init_langchain_tools()
                logger.info("✓ LangChain agent ready")
            except Exception as e:
                logger.warning("LangChain tools initialization failed: %s", e)
                self.langchain_agent = None
        else:
            logger.info("LangChain not installed; skipping tool agent init")


        # Task handlers
        self.qa_handler = FinancialQAHandler(llm_handler, self.data_aggregator)
        self.budget_handler = BudgetPlannerHandler(llm_handler, self.data_aggregator)
        self.risk_handler = RiskProfilerHandler(llm_handler, self.data_aggregator)

        self.max_iterations = getattr(Config, "MAX_VERIFICATION_ITERATIONS", 3)
        self.auto_execute_tools = getattr(Config, "AUTO_EXECUTE_TOOLS", True)

        logger.info("✓ Orchestrator V4 initialized successfully")

    def _load_chat_history(self, user_id: str) -> list:
        """
        Load chat history for user and convert to LangChain message format.
        Returns list of (role, content) tuples for MessagesPlaceholder.
        """
        safe_user = user_id or "default_user"
        path = os.path.join(CHAT_HISTORY_DIR, f"{safe_user}.json")
        
        messages = []
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                
                # Convert to LangChain message format
                for entry in history[-10:]:  # Last 10 entries for context window
                    messages.append(("human", entry.get("question", "")))
                    messages.append(("assistant", entry.get("answer", "")))
        except Exception as e:
            logger.error("Failed to load chat history: %s", e)
        
        return messages

    # -------------------------
    # Logging & chat helpers
    # -------------------------
    def _write_log(self, log_obj: Dict[str, Any]) -> str:
        """Write full trace log to timestamped JSON file, return path."""
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        filename = f"log_{ts}.json"
        path = os.path.join(LOG_DIR, filename)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(log_obj, f, indent=2, default=str)
            logger.info("Wrote pipeline log to %s", path)
        except Exception as e:
            logger.error("Failed to write log: %s", e)
        return path
    
    def _init_langchain_tools(self):
        if not HAS_LANGCHAIN:
            return

        tools = []

        # Tool: get_stock_data
        def tc_get_stock(symbol: str):
            import asyncio
            return asyncio.run(self.data_aggregator.get_stock_data(symbol))
        tools.append(Tool(
            name="get_stock_data",
            func=tc_get_stock,
            description="Get current stock data for a symbol."
        ))

        # Tool: web_search
        def tc_web_search(q: str):
            import asyncio
            return asyncio.run(self.data_aggregator.web_search(q))
        tools.append(Tool(
            name="web_search",
            func=tc_web_search,
            description="Search the web or news."
        ))

        # Tool: generate_pdf
        def tc_generate_pdf(payload: dict):
            t = payload.get("type")
            if t == "budget":
                b = payload.get("budget_data", {})
                return self.pdf_generator.generate_budget_plan_pdf(
                    b.get("user_data", {}), b.get("plan", {})
                )
            elif t == "risk":
                u = payload.get("user_data", {})
                a = payload.get("assessment", {})
                return self.pdf_generator.generate_risk_profile_pdf(u, a)
            else:
                return self.pdf_generator.generate_financial_report_pdf(
                    payload.get("title", "Report"),
                    payload.get("content", "")
                )
        tools.append(Tool(
            name="generate_pdf",
            func=tc_generate_pdf,
            description="Generate a PDF document."
        ))

        def tc_generate_pdf_anvil(payload: dict):
            import requests
            md_text = payload.get("markdown", "")
            filename = payload.get("filename", "report.pdf")
            api_key = Config.ANVIL_API_KEY

            url = "https://app.useanvil.com/form/iiith/sample-workflow"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "type": "markdown",
                "title": "Financial Report",
                "markdown": md_text
            }

            try:
                r = requests.post(url, headers=headers, json=data)
                if r.status_code != 200:
                    return {"error": f"Anvil failed: {r.status_code} - {r.text}"}
                pdf_bytes = r.content
                local_path = f"./logs/{filename}"
                with open(local_path, "wb") as f:
                    f.write(pdf_bytes)
                return {"pdf_path": local_path}
            except Exception as e:
                return {"error": str(e)}


        # Tool: send_email
        def tc_send_email(payload: dict):
            import asyncio
            return asyncio.run(self.email_sender.send_email_with_attachment(
                payload.get("recipient"),
                payload.get("subject"),
                payload.get("body"),
                payload.get("attachment_bytes"),
                payload.get("filename")
            ))
        tools.append(Tool(
            name="send_email",
            func=tc_send_email,
            description="Send an email with attachment."
        ))

        def tc_send_email_sendgrid(payload: dict):
            import re, base64
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

            api_key = Config.SENDGRID_API_KEY
            from_email = Config.SENDGRID_FROM_EMAIL
            text = payload.get("text", "")
            subject = payload.get("subject", "Your Financial Report")
            body = payload.get("body", "Here is your report attached.")
            filename = payload.get("filename", "report.pdf")

            # Try to extract recipient
            recipient = payload.get("recipient")
            if not recipient:
                found = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
                if found:
                    recipient = found[0]
                else:
                    return {"error": "No recipient email found"}

            # Load PDF
            pdf_path = payload.get("pdf_path")
            try:
                with open(pdf_path, "rb") as f:
                    data = f.read()
                encoded = base64.b64encode(data).decode()
            except Exception as e:
                return {"error": f"PDF read error: {e}"}

            message = Mail(
                from_email=from_email,
                to_emails=recipient,
                subject=subject,
                html_content=body
            )
            message.attachment = Attachment(
                FileContent(encoded),
                FileName(filename),
                FileType("application/pdf"),
                Disposition("attachment")
            )

            try:
                sg = SendGridAPIClient(api_key)
                response = sg.send(message)
                return {"status": response.status_code, "sent_to": recipient}
            except Exception as e:
                return {"error": str(e)}
        tools.append(Tool(
        name="generate_pdf_anvil",
        func=tc_generate_pdf_anvil,
        description="Generate a PDF using Anvil from markdown. Input: {markdown, filename}"))

        tools.append(Tool(
            name="send_email_sendgrid",
            func=tc_send_email_sendgrid,
            description="Send an email with PDF attached via SendGrid. Input: {text, subject, body, pdf_path, filename, recipient(optional)}"
        ))

        # ---------- FIXED LLM WRAPPER ----------
        from langchain.llms.base import LLM

        if isinstance(self.llm.model, LangChainLLMBase):
            lc_llm = self.llm.model
        else:
            handler_ref = self.llm  # closure reference

            class ThinLCWrapper(LLM):
                @property
                def _llm_type(self):
                    return "thin-llmhandler"

                @property
                def _identifying_params(self):
                    return {"name": "thin-llmhandler"}

                def _call(self, prompt: str, stop=None):
                    import asyncio
                    return asyncio.run(handler_ref.generate(prompt))

            lc_llm = ThinLCWrapper()
        # ---------------------------------------

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a financial expert assistant. Use the chat history to maintain context."),
            MessagesPlaceholder("history", optional=True),
            ("human", "{input}")
        ])

        self.langchain_agent = initialize_agent(
            tools,
            lc_llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )
        self.langchain_tools_dict = {tool.name: tool.func for tool in tools}
            
    def _append_chat_history(self, user_id: str, question: str, answer: str, metadata: Dict):
        """
        Append only verified final Q/A to per-user chat history.
        """
        safe_user = user_id or "default_user"
        path = os.path.join(CHAT_HISTORY_DIR, f"{safe_user}.json")
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "metadata": metadata
        }
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    arr = json.load(f)
            else:
                arr = []
            arr.append(entry)
            # Keep last N entries to avoid unbounded growth (optimize context)
            MAX_HISTORY = getattr(Config, "MAX_CHAT_HISTORY", 50)
            arr = arr[-MAX_HISTORY:]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(arr, f, indent=2, default=str)
            logger.info("Appended accepted QA to chat history for user %s", safe_user)
        except Exception as e:
            logger.error("Failed to append chat history: %s", e)

    # -------------------------
    # Public process_query
    # -------------------------
    async def process_query(self, query: str, user_context: Dict = None) -> AgentResponse:
        """
        Full pipeline:
        1) enhanced detection
        2) call APIs/tools as indicated (optional)
        3) RAG retrieval using enhanced_query
        4) assemble prompt + call LLM
        5) verify & attempt fixes via specialists
        6) execute actions (pdf/email) only after verification
        7) log everything, store accepted Q/A in chat history
        """
        user_context = user_context or {}
        user_id = user_context.get("user_id", "default_user")
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S.%fZ")

        # Prepare top-level log object
        log_obj = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "original_query": query,
            "user_id": user_id,
            "pipeline": []
        }

        # 1) Task detection & enhancement
        detect_entry = {"step": "task_detection", "input": query}
        detection = self.task_detector.detect_and_enhance(query)
        detect_entry["output"] = {
            "task_type": detection["task_type"].value if detection["task_type"] else None,
            "enhanced_query": detection["enhanced_query"],
            "tools_needed": detection["tools_needed"],
            "apis_needed": detection["apis_needed"],
            "actions": detection["actions"],
            "metadata": detection.get("metadata", {})
        }
        log_obj["pipeline"].append(detect_entry)
        try:
            print("🔍 Task Detection:", detect_entry["output"])
        except Exception:
            pass 

        # 2) Pre-LLM API/tool calls if asked by detector
        api_results = {}
        api_step = {"step": "pre_api_calls", "requested": detection.get("apis_needed", [])}
        # If detector requested apis, call them (some common keys: stock_api -> data_aggregator.get_stock_data, news_api -> search_financial_news, col_api -> get_cost_of_living)
        for api_name in detection.get("apis_needed", []) or []:
            try:
                if api_name.lower() in ("stock_api", "finnhub", "alpha_vantage", "stock"):
                    # attempt to extract symbol from query (simple heuristic)
                    symbol = None
                    for sym in ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'INFY', 'TCS', 'RELIANCE', 'HDFC']:
                        if sym.lower() in query.lower():
                            symbol = sym
                            break
                    if symbol:
                        api_results[f"stock_{symbol}"] = await self.data_aggregator.get_stock_data(symbol)
                elif api_name.lower() in ("news_api", "news"):
                    api_results["news"] = await self.data_aggregator.search_financial_news(query)
                elif api_name.lower() in ("col_api", "numbeo", "cost_of_living"):
                    # try to parse city from user_context or query
                    loc = user_context.get("location") or ""
                    if "," in query and len(query.split(",")) <= 3:
                        loc = query.split("in")[-1].strip() if " in " in query.lower() else loc
                    if loc:
                        # crude split
                        parts = [p.strip() for p in loc.split(",")]
                        city = parts[0]
                        country = parts[1] if len(parts) > 1 else "India"
                        api_results["cost_of_living"] = await self.data_aggregator.get_cost_of_living(city, country)
                else:
                    # generic web search
                    res = await self.data_aggregator.web_search.search_general(query)
                    api_results[api_name] = res
            except Exception as e:
                api_results[api_name] = {"error": str(e)}
        api_step["output"] = api_results
        log_obj["pipeline"].append(api_step)
        try:
            print("📡 Pre-API Call Results:", api_results)
        except Exception:
            pass 

        # 3) RAG retrieval using enhanced_query (with rerank-score check)
        rag_step = {"step": "rag_retrieval", "input_query": detection.get("enhanced_query", query)}
        rag_context = ""
        retrieved_docs = []
        try:
            if self.hybrid_retriever:
                docs = self.hybrid_retriever.retrieve(detection.get("enhanced_query", query))
                RERANK_THRESHOLD = getattr(Config, "RERANK_FALLBACK_THRESHOLD", 0.1)
                filtered_docs = []
                avg_rerank = 0.0
                for d in docs:
                    score = d.metadata.get("rerank_score")
                    if score is None or score >= RERANK_THRESHOLD:
                        filtered_docs.append(d)
                    avg_rerank =+d.metadata.get("rerank_score")

                avg_rerank = avg_rerank / len(docs) if docs else None

                # Build rag_context only from filtered docs
                rag_context = "\n\n".join([d.page_content for d in filtered_docs])

                retrieved_docs = [
                    {
                        "id": d.metadata.get("id"),
                        "source": d.metadata.get("source"),
                        "snippet": d.page_content[:500],
                        "rerank_score": d.metadata.get("rerank_score")
                    }
                    for d in filtered_docs
                ]

                if avg_rerank is None or avg_rerank < RERANK_THRESHOLD:
                    logger.info("Rerank avg low (%.3f) — augmenting context with web search",
                                avg_rerank or -1)

                    web_results = await self.data_aggregator.web_search.search_general(
                        detection.get("enhanced_query", query)
                    )

                    web_snippets = "\n\n".join([
                        r.get("description") or r.get("snippet") or r.get("title")
                        for r in web_results[:3]
                    ])

                    if web_snippets:
                        rag_context += ("\n\n websearch context:" + web_snippets) if rag_context else web_snippets

                        retrieved_docs.extend([
                            {
                                "id": f"web_{i}",
                                "source": r.get("source") or r.get("url"),
                                "snippet": (
                                    r.get("description") or
                                    r.get("snippet") or
                                    r.get("title")
                                )[:500],
                                "rerank_score": None
                            }
                            for i, r in enumerate(web_results[:3])
                        ])
            else:
                rag_context = ""
        except Exception as e:
            rag_step["error"] = str(e)
        rag_step["output"] = retrieved_docs
        log_obj["pipeline"].append(rag_step)
        try:
            print("📚 RAG Retrieved Documents:", [d["source"] for d in retrieved_docs])
        except Exception:
            pass

        # 4) Route to task-specific handler (use specialized handlers as primary experts)
        task_type = detection.get("task_type")
        handler_result = None
        handler_used = None

        # Map to handler
        if task_type == TaskType.BUDGET_PLANNING:
            handler_used = "budget_handler"
            # If user provided data in user_context, prefer that
            
            handler_result = await self.budget_handler.generate_budget_plan(detection.get("enhanced_query", query), rag_context)
        elif task_type == TaskType.RISK_PROFILING:
            handler_used = "risk_handler"
            questionnaire = await self.risk_handler.collect_risk_questionnaire()
            # Merge supplied fields
            # questionnaire.update(user_context.get("risk_answers", {}))
            handler_result = await self.risk_handler.calculate_risk_profile(detection.get("enhanced_query", query), questionnaire)
        else:
            handler_used = "qa_handler"
            # For generic QA, try calling FinancialQAHandler which already gathers live data
            handler_result = await self.qa_handler.handle_query(detection.get("enhanced_query", query), rag_context)

        llm_call_entry = {
            "step": "expert_handler_call",
            "handler": handler_used,
            "handler_result_preview": str(handler_result)[:1500]
        }
        log_obj["pipeline"].append(llm_call_entry)

        # Build final LLM prompt only if handler returned a 'draft' text or if further synthesis required
        if isinstance(handler_result, dict) and handler_result.get("answer"):
            # Handler returned an answer already — use it as current_answer
            llm_output = handler_result.get("answer")
            api_results.update(handler_result.get("live_data", {}))
        else:
            # If handler returned structured content (e.g., plan), synthesize a human-readable answer
            synth_prompt = "You are a financial expert. Synthesize the following structured data into a concise user-facing answer:\n\n"
            synth_prompt += json.dumps(handler_result, default=str)
            # If detector indicated tools/actions and we have langchain agent, allow agent to be used for tool-calls in the synthesis step
            if self.langchain_agent and detection.get("tools_needed"):
                try:
                    chat_history = self._load_chat_history(user_id)
                    formatted_prompt = self.prompt_template.format_messages(
                    history=chat_history,
                    input=synth_prompt)
            
                    # Run agent with history context
                    llm_output = self.langchain_agent.run(
                        input=synth_prompt,
                        history=chat_history
                    )
                except Exception:
                    llm_output = await self.llm.generate(synth_prompt)
            else:
                 # For non-tool queries, still provide history context
                chat_history = self._load_chat_history(user_id)
                try:
                    formatted_prompt = self.prompt_template.format_messages(
                        history=chat_history,
                        input=synth_prompt
                    )
                    llm_output = await self.llm.generate(str(formatted_prompt))
                except Exception:
                    llm_output = await self.llm.generate(synth_prompt)
        try:
            print(f"🤖 Expert Handler Used: {handler_used}")
            print(f"📄 Handler Result Preview: {str(handler_result)[:300]}")
        except Exception:
            pass


        # 5) Verification & recovery loop (improved)
        verification_entry = {"step": "verification", "iterations": []}
        current_answer = llm_output
        current_context = rag_context
        iteration = 0
        specialists_used = []
        has_issues = False 

        while iteration < self.max_iterations:
            verification_result = await self.issue_detector.detect_issues(current_answer, current_context, query)
            it_log = {
                "iteration": iteration + 1,
                "has_issues": verification_result.has_issues,
                "issues": [i.reason for i in verification_result.issues] if verification_result.issues else []
            }
            verification_entry["iterations"].append(it_log)
            if not verification_result.has_issues:
                has_issues = False
                break  # accepted

            # route to specialist
            primary_issue = verification_result.primary_issue
            specialist_type = primary_issue.issue_type.value
            specialist = self.specialists.get(specialist_type)
            if not specialist:
                logger.warning("No specialist for %s", specialist_type)
                has_issues = True
                break

            try:
                # Let the specialist use targeted retrieval and tools. Provide the LangChain agent reference
                # so specialist can call web_search/get_stock_data/generate_pdf if needed.
                if hasattr(specialist, "langchain_agent") is False:
                    # inject agent (if available)
                    setattr(specialist, "langchain_agent", self.langchain_agent)

                specialist_result = await specialist.fix_issue(current_answer, query, current_context, primary_issue)
                # Expect specialist_result to include 'fixed_answer' and optional 'confidence' (0-1)
                current_answer = specialist_result.get("fixed_answer", current_answer)
                if current_answer == "":
                    logger.warning("Specialist returned empty answer; retaining previous")
                    current_answer = llm_output 
                    
                specialists_used.append(specialist_result.get("specialist", specialist_type))
                it_log["applied_fix"] = specialist_result.get("fix_description", "")
                it_log["answer_preview_after"] = current_answer[:500]
                it_log["specialist_confidence"] = specialist_result.get("confidence", None)

                # If the specialist reports a high confidence, accept early
                sp_conf = specialist_result.get("confidence")
                if sp_conf is not None and sp_conf >= 0.70:
                    has_issues = False
                    break

            except Exception as e:
                it_log["specialist_error"] = str(e)

            iteration += 1


        verification_entry["final_answer_preview"] = current_answer[:1000]
        verification_entry["specialists_used"] = specialists_used
        # escalate if still issues
        if iteration >= self.max_iterations and verification_result.has_issues:
            verification_entry["escalation"] = {
                "note": "Max verification iterations reached. Manual human review required.",
                "issues": [i.reason for i in verification_result.issues]
            }
            verification_entry["accepted"] = False
            has_issues = True
        else:
            verification_entry["accepted"] = not has_issues

        log_obj["pipeline"].append(verification_entry)

        print("🧠 Final Synthesized Answer Preview:", llm_output[:300])
        try:
            print("🛠️ Specialist Applied:", specialist_result.get("specialist"))
            print("🛠️ Fix Description:", specialist_result.get("fix_description"))
        except Exception:
            pass


        # 6) Actions execution (PDF/Email) — only if accepted
        actions_entry = {"step": "action_execution", "requested_actions": detection.get("actions", []), "executed": []}
        action_results = {}
        if verification_entry["accepted"]:
            filename = f"{user_id}_{run_id}_report.pdf"
            # prepare minimal content for PDFs/emails (either use LLM produced document_body or use full answer)
            # simple extraction: look for a 'document_body' marker else use current_answer
            document_body = None
            if "document_body" in current_answer.lower():
                # naive extraction — in practice define a strict schema
                document_body = current_answer
            else:
                document_body = current_answer

            markdown_text = document_body #self._answer_to_markdown(document_body)  # convert if needed

            # decide whether to execute automatically
            if self.auto_execute_tools:
                confirm = True
            else:
                # If not auto, prefer the user_context to contain user approval; otherwise skip and put in log
                confirm = user_context.get("auto_confirm", False)

            for act in detection.get("actions", []) or []:
                print("⚙️ Running Action:", act)
                if act == "generate_pdf":
                    if confirm:
                        try:
                            payload = {
                                "markdown": markdown_text,
                                "filename": filename}
                            # which pdf? infer from task_type
                            pdf_result = self.langchain_tools_dict["generate_pdf_anvil"](payload)
                            if isinstance(pdf_result, dict) and pdf_result.get("pdf_path"):
                                pdf_path = pdf_result["pdf_path"]
                                action_results["generate_pdf"] = {"path": pdf_path}
                                actions_entry["executed"].append("generate_pdf")
                            else:
                                action_results["generate_pdf"] = {"error": "PDF tool failed"}
                                html = markdown.markdown(markdown_text)
                                HTML(string=html).write_pdf(filename)

                        except Exception as e:
                            action_results["generate_pdf"] = {"error": str(e)}
                    else:
                        action_results["generate_pdf"] = {"skipped": "user_confirmation_required"}
                if act == "send_email":
                    
                    html = markdown.markdown(markdown_text)
                    HTML(string=html).write_pdf("/home2/sathya.marisetti/lma_major/agentx_karan/data/reports/" + filename)

                    recipient = user_context.get("email") or self._extract_email_from_text(query)
                    if not recipient:
                        action_results.setdefault("send_email", {}).update({"error": "no_recipient"})
                        print("❌ No recipient email found; skipping email sending.")
                        recipient = input("Please enter recipient email: ").strip()
                        continue
                    if confirm:
                        try:
                            payload = {
                                "recipient": recipient,
                                "text": document_body,
                                "subject": EMAIL_BODIES_BY_TASK.get(
                                    detection.get("task_type").value if detection.get("task_type") else "financial_qa",
                                    EMAIL_BODIES_BY_TASK["financial_qa"]
                                ).get("subject"),
                                "body": EMAIL_BODIES_BY_TASK.get(
                                    detection.get("task_type").value if detection.get("task_type") else "financial_qa",
                                    EMAIL_BODIES_BY_TASK["financial_qa"]
                                ).get("body"),
                                "filename": filename,
                                "pdf_path": "/home2/sathya.marisetti/lma_major/agentx_karan/data/reports/" + filename
                            }
                            email_result = self.langchain_tools_dict["send_email_sendgrid"](payload)
                            if isinstance(email_result, dict) and email_result.get("sent_to"):
                                action_results["send_email"] = {"sent": email_result["sent_to"]}
                                actions_entry["executed"].append("send_email")
                            else:
                                action_results["send_email"] = {"error": "SendGrid failed or not sent"}
                        except Exception as e:
                            action_results["send_email"] = {"error": str(e)}
                    else:
                        action_results["send_email"] = {"skipped": "user_confirmation_required"}
            try:
                print("📤 Email Result:", email_result)
                print("📄 PDF Path:", pdf_path)
            except Exception:
                pass

        else:
            actions_entry["note"] = "Answer not accepted; actions skipped"
        actions_entry["results"] = action_results
        log_obj["pipeline"].append(actions_entry)

        # 7) Finalize response object and write logs / chat history
        final_answer = current_answer
        accepted = verification_entry["accepted"]
        log_obj["final"] = {
            "answer_preview": final_answer[:2000],
            "accepted": accepted,
            "specialists_used": specialists_used,
            "action_results": action_results
        }
        log_path = self._write_log(log_obj)

        # store accepted QA in chat history (only)
        if accepted:
            qa_meta = {
                "task_type": detection["task_type"].value if detection["task_type"] else None,
                "log_path": log_path,
                "actions_executed": actions_entry.get("executed", [])
            }
            # Append to chat history (only verified final)
            self._append_chat_history(user_id, query, final_answer, qa_meta)

        # Build AgentResponse (reuse earlier AgentResponse dataclass shape)
        response = AgentResponse(
            query=query,
            answer=final_answer,
            confidence=detection.get("metadata", {}).get("confidence", 0.8),
            task_type=detection["task_type"],
            iterations=iteration + 1,
            verification_log=verification_entry,
            specialists_used=specialists_used,
            sources=[d.get("id") for d in retrieved_docs],
            reasoning_chain=[],  # could be extended
            metadata={
                'task_detection': detection,
                'api_results': api_results,
                'rag_documents': retrieved_docs,
                'log_path': log_path,
                'accepted': accepted,
                'action_results': action_results
            }
        )

        return response

    # small helper: naive email extraction
    def _extract_email_from_text(self, text: str) -> Optional[str]:
        import re
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        return emails[0] if emails else None


class FinancialAgentOrchestratorV4:
    def __init__(self, llm_handler):
        self.llm = llm_handler
        self.verifier = MultiAgentOrchestratorV4(llm_handler)
        logger.info("✓ Financial Agent Orchestrator V4 initialized")

    async def process_query(self, query: str, user_context: Dict = None) -> AgentResponse:
        return await self.verifier.process_query(query, user_context)
