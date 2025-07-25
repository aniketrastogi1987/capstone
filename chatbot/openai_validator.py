#!/usr/bin/env python3
"""
OpenAI Validation Service for Local LLM Responses

This module provides validation capabilities for local LLM responses using OpenAI GPT-4o Mini.
It validates responses for hallucinations, factual accuracy, and completeness.
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import openai

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for OpenAI validation results"""
    is_valid: bool
    corrected_text: str
    validation_score: float  # 0.0 to 1.0
    hallucination_detected: bool
    corrections_made: List[str]
    validation_time: float
    error_message: Optional[str] = None

@dataclass
class QualityMetrics:
    """Container for quality metrics from OpenAI validation"""
    hallucination_rate: float
    factual_accuracy: float
    completeness: float
    technical_depth: float
    overall_quality: float

class OpenAIValidator:
    """
    OpenAI-based validator for local LLM responses
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI validator
        
        Args:
            api_key: OpenAI API key (will try to get from environment if not provided)
            model: OpenAI model to use for validation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Validation will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            
        # Critical terms that trigger validation regardless of length
        self.critical_terms = [
            "patent", "US", "EP", "WO", "claim", "prior art", "patentability", 
            "novelty", "inventive step", "patent number", "patent id", "patent application",
            "patent office", "patent examiner", "patent attorney", "patent agent"
        ]
        
        # Validation metrics tracking
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "hallucination_detections": 0,
            "average_validation_time": 0.0,
            "average_quality_score": 0.0
        }
    
    def should_validate_response(self, response: str) -> bool:
        """
        Determine if a response should be validated based on critical terms
        
        Args:
            response: The response text to check
            
        Returns:
            True if response should be validated, False otherwise
        """
        if not self.enabled:
            return False
            
        response_lower = response.lower()
        return any(term.lower() in response_lower for term in self.critical_terms)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def validate_response(self, original_response: str, user_query: str, rag_context: str = "") -> ValidationResult:
        """
        Validate a local LLM response using OpenAI
        
        Args:
            original_response: The original response from local LLM
            user_query: The user's original query
            rag_context: RAG context used for the response
            
        Returns:
            ValidationResult with validation details
        """
        if not self.enabled:
            return ValidationResult(
                is_valid=True,
                corrected_text=original_response,
                validation_score=1.0,
                hallucination_detected=False,
                corrections_made=[],
                validation_time=0.0
            )
        
        start_time = time.time()
        
        try:
            # Create validation prompt
            validation_prompt = self._create_validation_prompt(
                original_response, user_query, rag_context
            )
            
            # Call OpenAI API
            response = self._call_openai_api(validation_prompt)
            
            # Parse validation response
            validation_result = self._parse_validation_response(response, original_response)
            validation_result.validation_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"OpenAI validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                corrected_text=original_response,
                validation_score=0.0,
                hallucination_detected=True,
                corrections_made=[],
                validation_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _create_validation_prompt(self, original_response: str, user_query: str, rag_context: str) -> str:
        """
        Create validation prompt for OpenAI
        
        Args:
            original_response: The response to validate
            user_query: The original user query
            rag_context: The RAG context used
            
        Returns:
            Validation prompt
        """
        prompt = f"""You are a patent analysis expert and quality validator. Your task is to validate and improve a response about patent analysis.

ORIGINAL USER QUERY: {user_query}

RAG CONTEXT USED: {rag_context[:1000] if rag_context else "No RAG context provided"}

ORIGINAL RESPONSE TO VALIDATE:
{original_response}

VALIDATION TASK:
1. Check for factual accuracy - ensure all patent numbers and technical details are correct
2. Verify that patent IDs mentioned are actually present in the RAG context
3. Check for hallucinations - remove any invented patent numbers or technical details
4. Improve clarity and readability while maintaining the original structure
5. **IMPORTANT: Preserve the original formatting and structure as much as possible**
6. **DO NOT add excessive markdown formatting or change the text structure dramatically**

VALIDATION CRITERIA:
- Factual accuracy: All patent numbers and technical details must be verifiable
- Completeness: Response should address the user's question adequately
- Clarity: Technical information should be understandable
- Patent ID accuracy: Only include patent IDs that exist in the RAG context

RESPONSE FORMAT:
Return a JSON object with the following structure:
{{
    "is_valid": true/false,
    "corrected_text": "The improved response text (preserve original structure)",
    "validation_score": 0.0-1.0,
    "hallucination_detected": true/false,
    "corrections_made": ["List of specific corrections made"]
}}

**CRITICAL INSTRUCTIONS:**
- If the original response is mostly correct, make minimal changes
- Preserve the original text structure and formatting
- Only correct factual errors and patent ID inaccuracies
- Do not add excessive markdown formatting
- Keep the response natural and readable

JSON Response:"""
        
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Call OpenAI API with retry logic
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            API response as string
        """
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a patent analysis expert and quality validator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            api_response = response.choices[0].message.content.strip()
            
            # Add detailed logging for debugging
            logger.info(f"OpenAI API Response Length: {len(api_response)}")
            logger.info(f"OpenAI API Response Preview: {api_response[:200]}...")
            
            return api_response
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_validation_response(self, api_response: str, original_response: str) -> ValidationResult:
        """
        Parse OpenAI API response into ValidationResult
        
        Args:
            api_response: Raw API response from OpenAI
            original_response: Original response being validated
            
        Returns:
            Parsed ValidationResult
        """
        try:
            # Extract JSON from response
            json_start = api_response.find('{')
            json_end = api_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = api_response[json_start:json_end]
            
            # Clean up common JSON issues
            json_str = self._clean_json_string(json_str)
            
            # Try to parse the JSON
            try:
                validation_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to extract individual fields
                logger.warning(f"JSON parsing failed, attempting field extraction: {e}")
                validation_data = self._extract_fields_from_text(api_response)
            
            # Extract corrected_text and ensure it's not empty or malformed
            corrected_text = validation_data.get("corrected_text", original_response)
            
            # If corrected_text is empty or just whitespace, use original
            if not corrected_text or corrected_text.strip() == "":
                corrected_text = original_response
            
            # Clean up any trailing backslashes or malformed endings
            if corrected_text.endswith('\\'):
                corrected_text = corrected_text[:-1]
            
            # Remove any incomplete sentences at the end
            corrected_text = corrected_text.strip()
            
            return ValidationResult(
                is_valid=validation_data.get("is_valid", True),
                corrected_text=corrected_text,
                validation_score=validation_data.get("validation_score", 1.0),
                hallucination_detected=validation_data.get("hallucination_detected", False),
                corrections_made=validation_data.get("corrections_made", []),
                validation_time=0.0  # Will be set by caller
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse OpenAI validation response: {e}")
            logger.error(f"Response content: {api_response[:500]}...")  # Log first 500 chars for debugging
            
            # Return a fallback validation result that preserves the original text
            return ValidationResult(
                is_valid=False,
                corrected_text=original_response,  # Always preserve original on parse failure
                validation_score=0.0,
                hallucination_detected=True,
                corrections_made=["Failed to parse validation response"],
                validation_time=0.0,
                error_message=f"Parse error: {e}"
            )
    
    def _extract_fields_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract validation fields from text when JSON parsing fails
        
        Args:
            text: Raw text response
            
        Returns:
            Dictionary with extracted fields
        """
        import re
        
        result = {
            "is_valid": True,
            "corrected_text": "",
            "validation_score": 1.0,
            "hallucination_detected": False,
            "corrections_made": []
        }
        
        # Try to extract corrected text - look for the full field
        corrected_pattern = r'"corrected_text":\s*"([^"]*(?:\\"[^"]*)*)"'
        corrected_match = re.search(corrected_pattern, text, re.DOTALL)
        if corrected_match:
            corrected_text = corrected_match.group(1)
            # Clean up any escape sequences
            corrected_text = corrected_text.replace('\\"', '"')
            corrected_text = corrected_text.replace('\\n', '\n')
            corrected_text = corrected_text.replace('\\t', '\t')
            # Remove trailing backslashes
            if corrected_text.endswith('\\'):
                corrected_text = corrected_text[:-1]
            result["corrected_text"] = corrected_text
        
        # Try to extract validation score
        score_match = re.search(r'"validation_score":\s*([0-9.]+)', text)
        if score_match:
            result["validation_score"] = float(score_match.group(1))
        
        # Try to extract is_valid
        valid_match = re.search(r'"is_valid":\s*(true|false)', text, re.IGNORECASE)
        if valid_match:
            result["is_valid"] = valid_match.group(1).lower() == "true"
        
        # Try to extract hallucination_detected
        hallucination_match = re.search(r'"hallucination_detected":\s*(true|false)', text, re.IGNORECASE)
        if hallucination_match:
            result["hallucination_detected"] = hallucination_match.group(1).lower() == "true"
        
        return result
    
    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean up common JSON parsing issues
        
        Args:
            json_str: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        import re
        
        # Handle multiline strings with unescaped quotes
        # This is a more robust approach for handling complex strings
        lines = json_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Handle lines with string values
            if '"corrected_text"' in line:
                # Extract the value part and clean it
                value_start = line.find('"corrected_text":') + len('"corrected_text":')
                value_part = line[value_start:].strip()
                
                if value_part.startswith('"'):
                    # Find the end of the string value
                    start_quote = value_part.find('"')
                    if start_quote != -1:
                        # Look for the closing quote, handling escaped quotes
                        end_quote = -1
                        i = start_quote + 1
                        while i < len(value_part):
                            if value_part[i] == '"' and value_part[i-1] != '\\':
                                end_quote = i
                                break
                            i += 1
                        
                        if end_quote != -1:
                            # Extract and clean the string value
                            string_value = value_part[start_quote+1:end_quote]
                            # Clean the string value
                            cleaned_string = self._clean_string_value(string_value)
                            cleaned_lines.append(f'    "corrected_text": "{cleaned_string}",')
                        else:
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        # Reconstruct the JSON
        cleaned_json = '\n'.join(cleaned_lines)
        
        # Remove trailing commas
        cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
        
        return cleaned_json
    
    def _clean_string_value(self, string_value: str) -> str:
        """
        Clean a string value for JSON
        
        Args:
            string_value: Raw string value
            
        Returns:
            Cleaned string value
        """
        import re
        
        # Only escape quotes that aren't already escaped
        string_value = re.sub(r'(?<!\\)"', '\\"', string_value)
        
        # Handle newlines properly - don't double escape
        string_value = string_value.replace('\\\\n', '\\n')  # Fix double escaped newlines
        string_value = string_value.replace('\\\\t', '\\t')  # Fix double escaped tabs
        string_value = string_value.replace('\\\\r', '\\r')  # Fix double escaped carriage returns
        
        # Remove any control characters that could break JSON
        string_value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', string_value)
        
        # Ensure the string doesn't end with a backslash (which would escape the closing quote)
        if string_value.endswith('\\'):
            string_value = string_value[:-1]
        
        return string_value
    
    def _update_metrics(self, validation_result: ValidationResult):
        """
        Update validation metrics
        
        Args:
            validation_result: The validation result to track
        """
        self.validation_metrics["total_validations"] += 1
        
        if validation_result.error_message is None:
            self.validation_metrics["successful_validations"] += 1
            
        if validation_result.hallucination_detected:
            self.validation_metrics["hallucination_detections"] += 1
            
        # Update average validation time
        current_avg = self.validation_metrics["average_validation_time"]
        total_validations = self.validation_metrics["total_validations"]
        new_avg = (current_avg * (total_validations - 1) + validation_result.validation_time) / total_validations
        self.validation_metrics["average_validation_time"] = new_avg
        
        # Update average quality score
        current_avg_score = self.validation_metrics["average_quality_score"]
        new_avg_score = (current_avg_score * (total_validations - 1) + validation_result.validation_score) / total_validations
        self.validation_metrics["average_quality_score"] = new_avg_score
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current validation metrics
        
        Returns:
            Dictionary of validation metrics
        """
        metrics = self.validation_metrics.copy()
        
        # Calculate rates
        if metrics["total_validations"] > 0:
            metrics["validation_success_rate"] = metrics["successful_validations"] / metrics["total_validations"]
            metrics["hallucination_detection_rate"] = metrics["hallucination_detections"] / metrics["total_validations"]
        else:
            metrics["validation_success_rate"] = 0.0
            metrics["hallucination_detection_rate"] = 0.0
        
        return metrics
    
    def reset_metrics(self):
        """Reset validation metrics"""
        self.validation_metrics = {
            "total_validations": 0,
            "successful_validations": 0,
            "hallucination_detections": 0,
            "average_validation_time": 0.0,
            "average_quality_score": 0.0
        }
    
    def health_check(self) -> bool:
        """
        Check if OpenAI service is available
        
        Returns:
            True if service is available, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Simple test call
            test_prompt = "Test validation"
            self._call_openai_api(test_prompt)
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False 