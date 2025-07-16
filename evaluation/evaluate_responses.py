#!/usr/bin/env python3
"""
Enhanced Response Evaluation Module

This module evaluates chatbot responses using multiple metrics:
1. ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
2. Relevance score (semantic similarity)
3. Coherence score (text quality)
4. Guardrails metrics (profanity, topic relevance, politeness)
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

# Import our guardrails validator
from chatbot.guardrails_validator import GuardrailsValidator, GuardrailScores

logger = logging.getLogger(__name__)

@dataclass
class EvaluationScores:
    """Comprehensive evaluation scores for a response"""
    # Semantic similarity
    relevance_score: float = 0.0
    
    # Text quality
    coherence_score: float = 0.0
    
    # Guardrails scores
    profanity_score: float = 0.0
    topic_relevance_score: float = 0.0
    politeness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'relevance_score': self.relevance_score,
            'coherence_score': self.coherence_score,
            'profanity_score': self.profanity_score,
            'topic_relevance_score': self.topic_relevance_score,
            'politeness_score': self.politeness_score
        }
    
    def get_overall_score(self) -> float:
        """Calculate overall evaluation score (average of all metrics)"""
        scores = [
            self.relevance_score, self.coherence_score,
            self.profanity_score, self.topic_relevance_score, self.politeness_score
        ]
        return sum(scores) / len(scores) if scores else 0.0

class ResponseEvaluator:
    """
    Evaluates chatbot responses using multiple metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_models()
        self.guardrails_validator = GuardrailsValidator()
        
    def setup_models(self):
        """Initialize evaluation models"""
        try:
            # Initialize sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize text quality model
            self.text_quality_pipeline = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            self.logger.info("Evaluation models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up evaluation models: {e}")
            raise
    
    def calculate_relevance_score(self, query: str, response: str) -> float:
        """Calculate semantic similarity between query and response"""
        try:
            # Encode query and response
            query_embedding = self.sentence_model.encode([query])
            response_embedding = self.sentence_model.encode([response])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, response_embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
            )
            
            return float(similarity[0][0])
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance score: {e}")
            # Fallback to simple keyword matching
            return self._calculate_simple_relevance(query, response)
    
    def _calculate_simple_relevance(self, query: str, response: str) -> float:
        """Fallback relevance calculation using keyword matching"""
        try:
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if not query_words:
                return 0.0
            
            # Calculate word overlap
            overlap = len(query_words.intersection(response_words))
            relevance = overlap / len(query_words)
            
            return min(1.0, relevance)
        except Exception as e:
            self.logger.error(f"Error in simple relevance calculation: {e}")
            return 0.0
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate text coherence/quality score with improved logic"""
        try:
            if not text or len(text.strip()) < 10:
                return 0.0
            
            # Check for basic text quality indicators
            words = text.split()
            if len(words) < 5:
                return 0.2
            
            # Check for sentence structure
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 1:
                return 0.1
            
            # Calculate coherence based on multiple factors
            coherence_factors = []
            
            # 1. Text length factor (normalized)
            length_factor = min(1.0, len(words) / 100.0)
            coherence_factors.append(length_factor)
            
            # 2. Sentence structure factor
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            sentence_factor = min(1.0, avg_sentence_length / 20.0)  # Optimal ~20 words per sentence
            coherence_factors.append(sentence_factor)
            
            # 3. Vocabulary diversity factor
            unique_words = len(set(words))
            diversity_factor = min(1.0, unique_words / len(words)) if words else 0.0
            coherence_factors.append(diversity_factor)
            
            # 4. Technical content factor (for patent analysis)
            technical_terms = ['patent', 'invention', 'technology', 'system', 'method', 'device', 'apparatus', 'process']
            technical_count = sum(1 for word in words if word.lower() in technical_terms)
            technical_factor = min(1.0, technical_count / 5.0)  # Normalize by expected technical terms
            coherence_factors.append(technical_factor)
            
            # Calculate overall coherence as average of factors
            coherence = sum(coherence_factors) / len(coherence_factors)
            
            return min(1.0, coherence)
            
        except Exception as e:
            self.logger.error(f"Error calculating coherence score: {e}")
            return 0.0
    
    def evaluate_single_response(self, query: str, response: str, reference: str = None) -> EvaluationScores:
        """
        Evaluate a single response using all metrics
        
        Args:
            query: The user's query
            response: The chatbot's response
            reference: Optional reference response for ROUGE calculation
            
        Returns:
            EvaluationScores object with all metrics
        """
        scores = EvaluationScores()
        
        try:
            # Calculate relevance score
            scores.relevance_score = self.calculate_relevance_score(query, response)
            
            # Calculate coherence score
            scores.coherence_score = self.calculate_coherence_score(response)
            
            # Calculate guardrails scores
            try:
                validated_response, guardrail_scores = self.guardrails_validator.validate_response(response)
                scores.profanity_score = guardrail_scores.profanity_score
                scores.topic_relevance_score = guardrail_scores.topic_relevance_score
                scores.politeness_score = guardrail_scores.politeness_score
            except Exception as e:
                self.logger.error(f"Error calculating guardrails scores: {e}")
                # Set default guardrails scores
                scores.profanity_score = 0.0
                scores.topic_relevance_score = 0.5  # Neutral score
                scores.politeness_score = 0.5  # Neutral score
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            # Return default scores on error
            scores = EvaluationScores()
        
        return scores
    
    def evaluate_batch(self, queries: List[str], responses: List[str], 
                      references: List[str] = None) -> List[EvaluationScores]:
        """
        Evaluate a batch of responses
        
        Args:
            queries: List of user queries
            responses: List of chatbot responses
            references: Optional list of reference responses
            
        Returns:
            List of EvaluationScores objects
        """
        if references is None:
            references = [None] * len(responses)
        
        results = []
        for query, response, reference in zip(queries, responses, references):
            scores = self.evaluate_single_response(query, response, reference)
            results.append(scores)
        
        return results
    
    def get_evaluation_summary(self, queries: List[str], responses: List[str], 
                             references: List[str] = None) -> Dict:
        """
        Get a comprehensive evaluation summary
        
        Args:
            queries: List of user queries
            responses: List of chatbot responses
            references: Optional list of reference responses
            
        Returns:
            Dictionary with evaluation statistics
        """
        if not responses:
            return {
                "total_responses": 0,
                "average_scores": EvaluationScores().to_dict(),
                "overall_score": 0.0
            }
        
        evaluation_results = self.evaluate_batch(queries, responses, references)
        
        # Calculate average scores
        avg_scores = EvaluationScores()
        num_results = len(evaluation_results)
        
        for result in evaluation_results:
            avg_scores.relevance_score += result.relevance_score
            avg_scores.coherence_score += result.coherence_score
            avg_scores.profanity_score += result.profanity_score
            avg_scores.topic_relevance_score += result.topic_relevance_score
            avg_scores.politeness_score += result.politeness_score
        
        # Normalize by number of results
        avg_scores.relevance_score /= num_results
        avg_scores.coherence_score /= num_results
        avg_scores.profanity_score /= num_results
        avg_scores.topic_relevance_score /= num_results
        avg_scores.politeness_score /= num_results
        
        return {
            "total_responses": num_results,
            "average_scores": avg_scores.to_dict(),
            "overall_score": avg_scores.get_overall_score(),
            "individual_results": [
                {
                    "query": queries[i],
                    "response": responses[i],
                    "reference": references[i] if references else None,
                    "scores": result.to_dict(),
                    "overall_score": result.get_overall_score()
                }
                for i, result in enumerate(evaluation_results)
            ]
        }
    
    def print_evaluation_report(self, summary: Dict):
        """
        Print a formatted evaluation report
        
        Args:
            summary: Evaluation summary dictionary
        """
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"Total Responses Evaluated: {summary['total_responses']}")
        print(f"Overall Score: {summary['overall_score']:.3f}")
        
        print("\nAVERAGE SCORES:")
        print("-" * 30)
        scores = summary['average_scores']
        
        print(f"Relevance:      {scores['relevance_score']:.3f}")
        print(f"Coherence:      {scores['coherence_score']:.3f}")
        print(f"Profanity:      {scores['profanity_score']:.3f}")
        print(f"Topic Relevance: {scores['topic_relevance_score']:.3f}")
        print(f"Politeness:     {scores['politeness_score']:.3f}")
        
        print("\nINDIVIDUAL RESULTS:")
        print("-" * 30)
        for i, result in enumerate(summary['individual_results'][:5]):  # Show first 5
            print(f"\nResponse {i+1}:")
            print(f"  Query: {result['query'][:50]}...")
            print(f"  Response: {result['response'][:50]}...")
            print(f"  Overall Score: {result['overall_score']:.3f}")
            print(f"  Relevance: {result['scores']['relevance_score']:.3f}")
            print(f"  Guardrails: {result['scores']['profanity_score']:.3f}/{result['scores']['topic_relevance_score']:.3f}/{result['scores']['politeness_score']:.3f}")

# Example usage and testing
def test_evaluator():
    """Test the response evaluator with sample data"""
    evaluator = ResponseEvaluator()
    
    # Sample test data
    queries = [
        "What is the main claim of this patent?",
        "How does this invention work?",
        "What are the key features of this patent?"
    ]
    
    responses = [
        "This patent describes a novel method for data encryption using quantum computing principles.",
        "The invention works by utilizing advanced algorithms to process information securely.",
        "The key features include improved efficiency, enhanced security, and better performance."
    ]
    
    references = [
        "The patent claims a method for quantum encryption of data.",
        "The invention processes data using quantum algorithms.",
        "Key features are quantum encryption, efficiency, and security."
    ]
    
    print("Testing Response Evaluator...")
    
    # Evaluate individual response
    scores = evaluator.evaluate_single_response(queries[0], responses[0], references[0])
    print(f"\nIndividual Response Scores: {scores.to_dict()}")
    print(f"Overall Score: {scores.get_overall_score():.3f}")
    
    # Evaluate batch
    summary = evaluator.get_evaluation_summary(queries, responses, references)
    evaluator.print_evaluation_report(summary)

if __name__ == "__main__":
    test_evaluator() 