# ============================================================================
# TASK DETECTION & INTENT CLASSIFICATION MODULE
# Using FinBERT for intelligent task classification instead of simple if/else
# ============================================================================

import logging
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from config import TaskType, logger

logger = logging.getLogger(__name__)


class TaskDetector:
    """
    Intelligent task detection using FinBERT
    Replaces simple if/else with ML-based classification
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize task detector with FinBERT
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Task labels for classification
        self.task_labels = {
            0: TaskType.GENERAL_QA,
            1: TaskType.BUDGET_PLANNING,
            2: TaskType.RISK_PROFILING,
            # 3: TaskType.DEBT_MANAGEMENT,
            # 4: TaskType.SAVINGS_OPTIMIZATION
        }
        
        # Keywords for each task (for similarity matching if model fails)
        self.task_keywords = {
            TaskType.BUDGET_PLANNING: [
                'budget', 'expense', 'allocation', 'spending', 'distribute',
                'monthly', 'income', 'breakdown', 'plan', 'divide', '50/30/20'
            ],
            TaskType.RISK_PROFILING: [
                'risk', 'profile', 'tolerance', 'invest', 'portfolio', 'asset',
                'allocation', 'aggressive', 'conservative', 'stock', 'bond'
            ],
            TaskType.DEBT_MANAGEMENT: [
                'debt', 'loan', 'credit', 'card', 'repay', 'interest', 'emi',
                'payoff', 'refinance', 'consolidate', 'owe'
            ],
            TaskType.SAVINGS_OPTIMIZATION: [
                'save', 'saving', 'emergency', 'fund', 'goal', 'target',
                'accumulate', 'build', 'corpus', 'amount', 'years'
            ],
            TaskType.GENERAL_QA: [
                'what', 'how', 'why', 'when', 'where', 'compound', 'interest',
                'financial', 'concept', 'explain', 'understand'
            ]
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize FinBERT model for financial text classification"""
        try:
            logger.info("Loading FinBERT for task classification...")
            
            # Use FinBERT - specifically tuned for financial text
            model_name = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3  # 5 task types
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ FinBERT loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            logger.info("Will use keyword-based fallback")
            self.model = None
            self.tokenizer = None
    
    def detect_task(self, query: str) -> Tuple[TaskType, Dict[str, any]]:
        """
        Detect task type from query using FinBERT
        
        Args:
            query: User's query string
        
        Returns:
            Tuple of (TaskType, metadata with confidence and reasoning)
        """
        
        if self.model is not None and self.tokenizer is not None:
            return self._classify_with_model(query)
        else:
            return self._classify_with_keywords(query)
    
    def _classify_with_model(self, query: str) -> Tuple[TaskType, Dict]:
        """
        Classify using FinBERT model
        
        Args:
            query: User query
        
        Returns:
            Tuple of (TaskType, metadata)
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)[0]
            probabilities = probabilities.cpu().numpy()
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1]
            
            # Primary prediction
            primary_idx = top_indices[0]
            primary_task = self.task_labels[primary_idx]
            primary_confidence = float(probabilities[primary_idx])
            
            # Secondary prediction
            secondary_idx = top_indices[1]
            secondary_task = self.task_labels[secondary_idx]
            secondary_confidence = float(probabilities[secondary_idx])
            
            # Build metadata
            metadata = {
                'model': 'FinBERT',
                'confidence': primary_confidence,
                'secondary_task': secondary_task,
                'secondary_confidence': secondary_confidence,
                'all_scores': {
                    self.task_labels[i].value: float(probabilities[i])
                    for i in range(len(self.task_labels))
                },
                'reasoning': f"FinBERT classification (confidence: {primary_confidence:.2%})"
            }
            
            logger.info(f"Task detection: {primary_task.value} (confidence: {primary_confidence:.2%})")
            
            return primary_task, metadata
            
        except Exception as e:
            logger.error(f"Model classification failed: {e}")
            logger.info("Falling back to keyword-based classification")
            return self._classify_with_keywords(query)
    
    def _classify_with_keywords(self, query: str) -> Tuple[TaskType, Dict]:
        """
        Classify using keyword matching (fallback)
        
        Args:
            query: User query
        
        Returns:
            Tuple of (TaskType, metadata)
        """
        query_lower = query.lower()
        
        # Calculate scores for each task type
        task_scores = {}
        
        for task_type, keywords in self.task_keywords.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            # Calculate score as percentage of keywords that matched
            score = matches / len(keywords) if keywords else 0
            task_scores[task_type] = score
        
        # Get task with highest score
        best_task = max(task_scores, key=task_scores.get)
        confidence = task_scores[best_task]
        
        # Get second best
        sorted_tasks = sorted(task_scores.items(), key=lambda x: x[1], reverse=True)
        secondary_task = sorted_tasks[1][0] if len(sorted_tasks) > 1 else None
        secondary_confidence = sorted_tasks[1][1] if len(sorted_tasks) > 1 else 0
        
        metadata = {
            'model': 'Keyword-Matching (Fallback)',
            'confidence': confidence,
            'secondary_task': secondary_task,
            'secondary_confidence': secondary_confidence,
            'all_scores': {task.value: score for task, score in task_scores.items()},
            'reasoning': f"Keyword-based classification (matched keywords: {confidence:.2%})"
        }
        
        logger.info(f"Task detection (keyword): {best_task.value} (confidence: {confidence:.2%})")
        
        return best_task, metadata
    
    def detect_complexity(self, query: str) -> Dict[str, any]:
        """
        Assess query complexity level
        
        Args:
            query: User query
        
        Returns:
            Dictionary with complexity assessment
        """
        
        query_length = len(query.split())
        has_numbers = any(char.isdigit() for char in query)
        has_calculations = any(word in query.lower() for word in ['calculate', 'how much', 'percentage', 'amount'])
        question_marks = query.count('?')
        
        # Determine complexity
        if query_length < 10:
            complexity = "simple"
        elif query_length < 30 and not has_calculations:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            'complexity_level': complexity,
            'query_length': query_length,
            'has_numbers': has_numbers,
            'has_calculations': has_calculations,
            'question_marks': question_marks,
            'requires_examples': complexity in ['moderate', 'complex'],
            'requires_step_by_step': complexity == 'complex'
        }
    
    def get_classifier_metadata(self) -> Dict:
        """Get information about the classifier being used"""
        if self.model is not None:
            return {
                'model_name': 'FinBERT (ProsusAI)',
                'type': 'ML-based',
                'device': self.device,
                'status': 'active'
            }
        else:
            return {
                'model_name': 'Keyword Matching',
                'type': 'Rule-based (Fallback)',
                'device': 'N/A',
                'status': 'fallback'
            }


__all__ = ['TaskDetector']
