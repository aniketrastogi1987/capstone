#!/usr/bin/env python3
"""
Test PostgreSQL monitoring functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from postgres_monitor import postgres_monitor
import time

def test_postgres_monitoring():
    """Test PostgreSQL monitoring functionality"""
    print("🧪 Testing PostgreSQL monitoring...")
    
    # Test recording chat metrics
    print("📝 Recording test chat metrics...")
    postgres_monitor.record_chat_metric(
        query_text="What is machine learning?",
        response_text="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        response_time_ms=2500,
        tokens_used=45,
        model_name="qwen2.5:14b-instruct",
        source_count=3,
        guardrail_scores={"profanity": 0.0, "topic_relevance": 0.9, "politeness": 0.8},
        evaluation_scores={"relevance": 0.85, "coherence": 0.9}
    )
    
    # Test recording system metrics
    print("📊 Recording test system metrics...")
    postgres_monitor.record_system_metric(
        metric_name="cpu_usage",
        metric_value=45.2,
        metric_unit="percent",
        additional_data={"cores": 8, "temperature": 65}
    )
    
    postgres_monitor.record_system_metric(
        metric_name="memory_usage",
        metric_value=2048,
        metric_unit="MB",
        additional_data={"total": 8192, "available": 6144}
    )
    
    # Test recording performance metrics
    print("⚡ Recording test performance metrics...")
    postgres_monitor.record_performance_metric(
        component="lightrag",
        operation="document_retrieval",
        duration_ms=1200,
        success=True,
        additional_data={"documents_found": 5, "query_type": "semantic_search"}
    )
    
    postgres_monitor.record_performance_metric(
        component="ollama",
        operation="text_generation",
        duration_ms=3500,
        success=True,
        additional_data={"tokens_generated": 150, "model": "qwen2.5:14b-instruct"}
    )
    
    # Wait a moment for data to be written
    time.sleep(1)
    
    # Test retrieving metrics
    print("📈 Retrieving metrics...")
    
    # Get chat metrics
    chat_metrics = postgres_monitor.get_chat_metrics(hours=1)
    print(f"📝 Found {len(chat_metrics)} chat metrics in the last hour")
    
    # Get system metrics
    system_metrics = postgres_monitor.get_system_metrics(hours=1)
    print(f"📊 Found {len(system_metrics)} system metrics in the last hour")
    
    # Get performance metrics
    performance_metrics = postgres_monitor.get_performance_metrics(hours=1)
    print(f"⚡ Found {len(performance_metrics)} performance metrics in the last hour")
    
    # Get summary
    summary = postgres_monitor.get_metrics_summary(hours=1)
    print("📋 Metrics Summary:")
    print(f"  - Total queries: {summary.get('chat_metrics', {}).get('total_queries', 0)}")
    print(f"  - Avg response time: {summary.get('chat_metrics', {}).get('avg_response_time_ms', 0)}ms")
    print(f"  - Performance components: {list(summary.get('performance_metrics', {}).keys())}")
    
    print("✅ PostgreSQL monitoring test completed successfully!")
    return True

if __name__ == "__main__":
    test_postgres_monitoring() 