#!/usr/bin/env python3
"""
OpenAI Configuration for Patent Chatbot Validation

This module contains configuration settings for OpenAI integration
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-4o-mini",  # Cost-effective model for validation
    "temperature": 0.1,  # Low temperature for consistent validation
    "max_tokens": 2000,
    "timeout": 30,  # seconds
    "retry_attempts": 3,
    "retry_delay": 4,  # seconds
}

# Critical terms that trigger validation regardless of response length
CRITICAL_TERMS = [
    "patent", "US", "EP", "WO", "claim", "prior art", "patentability", 
    "novelty", "inventive step", "patent number", "patent id", "patent application",
    "patent office", "patent examiner", "patent attorney", "patent agent",
    "patent family", "patent citation", "patent classification", "patent status"
]

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "min_quality_score": 0.7,  # Minimum acceptable quality score
    "hallucination_threshold": 0.3,  # Threshold for hallucination detection
    "correction_threshold": 0.5,  # Threshold for significant corrections
}

# Progress indicator settings
PROGRESS_INDICATOR = {
    "enabled": True,
    "message": "🔍 Validating response with OpenAI...",
    "timeout_message": "⏱️ Validation taking longer than expected...",
    "success_message": "✅ Validation complete",
    "error_message": "❌ Validation failed, using local response"
}

# Quality metrics configuration
QUALITY_METRICS = {
    "factual_accuracy_weight": 0.4,
    "completeness_weight": 0.3,
    "technical_depth_weight": 0.2,
    "overall_quality_weight": 0.1,
}

def get_openai_api_key() -> str:
    """Get OpenAI API key from environment"""
    return os.getenv("OPENAI_API_KEY")

def is_openai_enabled() -> bool:
    """Check if OpenAI is properly configured"""
    api_key = get_openai_api_key()
    return api_key is not None and len(api_key.strip()) > 0 and api_key != "your_openai_api_key_here"

def get_validation_config() -> Dict[str, Any]:
    """Get complete validation configuration"""
    return {
        "openai_config": OPENAI_CONFIG,
        "critical_terms": CRITICAL_TERMS,
        "validation_thresholds": VALIDATION_THRESHOLDS,
        "progress_indicator": PROGRESS_INDICATOR,
        "quality_metrics": QUALITY_METRICS,
        "enabled": is_openai_enabled()
    } 