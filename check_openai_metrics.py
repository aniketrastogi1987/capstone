#!/usr/bin/env python3
"""
Script to check OpenAI metrics data in PostgreSQL
"""

import psycopg2
import json
from datetime import datetime, timedelta

def check_openai_metrics():
    """Check if OpenAI metrics data exists in PostgreSQL"""
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            dbname="patent_monitoring",
            host="localhost",
            port=5432,
            user=None,
            password=None
        )
        
        cursor = conn.cursor()
        
        # Check if chat_metrics table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_metrics'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        if not table_exists:
            print("❌ chat_metrics table does not exist")
            return False
        
        # Check for OpenAI metrics columns
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'chat_metrics' 
            AND column_name LIKE 'openai_%';
        """)
        
        openai_columns = [row[0] for row in cursor.fetchall()]
        print(f"✅ Found OpenAI columns: {openai_columns}")
        
        # Check for recent data
        cursor.execute("""
            SELECT COUNT(*) 
            FROM chat_metrics 
            WHERE timestamp >= NOW() - INTERVAL '24 hours';
        """)
        
        total_records = cursor.fetchone()[0]
        print(f"📊 Total records in last 24 hours: {total_records}")
        
        # Check for OpenAI validation data
        cursor.execute("""
            SELECT COUNT(*) 
            FROM chat_metrics 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            AND openai_validation_score IS NOT NULL;
        """)
        
        openai_records = cursor.fetchone()[0]
        print(f"🤖 OpenAI validation records: {openai_records}")
        
        # Get sample data
        cursor.execute("""
            SELECT 
                timestamp,
                openai_validation_score,
                openai_hallucination_detected,
                openai_validation_time,
                openai_corrections_applied,
                openai_validation_success
            FROM chat_metrics 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            AND openai_validation_score IS NOT NULL
            ORDER BY timestamp DESC 
            LIMIT 5;
        """)
        
        sample_data = cursor.fetchall()
        if sample_data:
            print("\n📋 Sample OpenAI metrics data:")
            for row in sample_data:
                timestamp, score, hallucination, time, corrections, success = row
                print(f"  {timestamp}: Score={score}, Hallucination={hallucination}, Time={time}s, Corrections={corrections}, Success={success}")
        else:
            print("\n⚠️ No OpenAI metrics data found in the last 24 hours")
            print("💡 This means either:")
            print("   - No LLM responses have been validated with OpenAI")
            print("   - The validation data is not being recorded properly")
        
        # Check for model usage
        cursor.execute("""
            SELECT model_name, COUNT(*) 
            FROM chat_metrics 
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY model_name;
        """)
        
        model_usage = cursor.fetchall()
        if model_usage:
            print("\n🤖 Model usage in last 24 hours:")
            for model, count in model_usage:
                print(f"  {model}: {count} records")
        
        cursor.close()
        conn.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Cannot connect to PostgreSQL: {e}")
        print("💡 Make sure PostgreSQL is running:")
        print("   brew services start postgresql")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🔍 OpenAI Metrics Data Checker")
    print("=" * 40)
    
    success = check_openai_metrics()
    
    if success:
        print("\n✅ Database check completed!")
        print("\n📋 If Grafana panels still show errors:")
        print("1. Make sure Grafana is running: brew services start grafana")
        print("2. Import the dashboard: python3 import_openai_dashboard.py")
        print("3. Check PostgreSQL datasource in Grafana")
        print("4. Verify the datasource UID is 'postgres'")
    else:
        print("\n❌ Database check failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Start PostgreSQL: brew services start postgresql")
        print("2. Check database connection settings")
        print("3. Verify the patent_monitoring database exists")

if __name__ == "__main__":
    main() 