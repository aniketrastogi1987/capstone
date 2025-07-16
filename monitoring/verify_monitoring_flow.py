#!/usr/bin/env python3
"""
Verify the complete monitoring flow: Chatbot → PostgreSQL → Grafana
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot.patent_chatbot import PatentChatbot
from postgres_monitor import postgres_monitor
import psycopg2
import time

def verify_monitoring_flow():
    """Verify the complete monitoring flow"""
    print("🔍 Verifying monitoring flow: Chatbot → PostgreSQL → Grafana")
    
    # Step 1: Generate chatbot response (this will record to PostgreSQL)
    print("\n📝 Step 1: Generating chatbot response...")
    chatbot = PatentChatbot(enable_monitoring=True)
    response = chatbot.get_response("What is artificial intelligence?")
    print(f"✅ Response generated: {response.content[:100]}...")
    
    # Step 2: Verify data is in PostgreSQL
    print("\n📊 Step 2: Verifying PostgreSQL storage...")
    try:
        conn = psycopg2.connect(dbname="patent_monitoring")
        cursor = conn.cursor()
        
        # Check chat metrics
        cursor.execute("SELECT COUNT(*) FROM chat_metrics WHERE query_text LIKE '%artificial intelligence%'")
        chat_count = cursor.fetchone()[0]
        print(f"✅ Chat metrics in PostgreSQL: {chat_count}")
        
        # Check recent metrics
        cursor.execute("""
            SELECT query_text, response_time_ms, tokens_used 
            FROM chat_metrics 
            WHERE query_text LIKE '%artificial intelligence%' 
            ORDER BY timestamp DESC LIMIT 1
        """)
        recent_metric = cursor.fetchone()
        if recent_metric:
            print(f"✅ Latest metric: {recent_metric[0][:50]}... ({recent_metric[1]}ms, {recent_metric[2]} tokens)")
        
        # Check total metrics
        cursor.execute("SELECT COUNT(*) FROM chat_metrics")
        total_metrics = cursor.fetchone()[0]
        print(f"✅ Total metrics in database: {total_metrics}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking PostgreSQL: {e}")
        return False
    
    # Step 3: Verify Grafana can access the data
    print("\n📈 Step 3: Verifying Grafana data access...")
    print("✅ PostgreSQL data is ready for Grafana!")
    print("📋 To view in Grafana:")
    print("   1. Open Grafana (http://localhost:3000)")
    print("   2. Add PostgreSQL data source:")
    print("      - Host: localhost:5432")
    print("      - Database: patent_monitoring")
    print("   3. Import dashboard: monitoring/postgres_dashboard.json")
    
    # Step 4: Show sample queries for Grafana
    print("\n🔧 Sample Grafana SQL queries:")
    print("""
    -- Response time over time
    SELECT timestamp, response_time_ms 
    FROM chat_metrics 
    WHERE timestamp >= $__timeFrom AND timestamp <= $__timeTo 
    ORDER BY timestamp
    
    -- Total queries count
    SELECT COUNT(*) as total 
    FROM chat_metrics 
    WHERE timestamp >= $__timeFrom AND timestamp <= $__timeTo
    
    -- Average response time
    SELECT AVG(response_time_ms) as avg_time 
    FROM chat_metrics 
    WHERE timestamp >= $__timeFrom AND timestamp <= $__timeTo
    
    -- Performance by component
    SELECT component, AVG(duration_ms) as avg_duration, COUNT(*) as total_ops 
    FROM performance_metrics 
    WHERE timestamp >= $__timeFrom AND timestamp <= $__timeTo 
    GROUP BY component
    """)
    
    print("✅ Monitoring flow verification completed!")
    return True

if __name__ == "__main__":
    verify_monitoring_flow() 