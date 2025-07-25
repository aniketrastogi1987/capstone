#!/usr/bin/env python3
"""
Script to import the OpenAI metrics dashboard with stat panels
"""

import json
import os

def main():
    """Main function"""
    print("📊 OpenAI Metrics Dashboard - Stat Panels")
    print("=" * 50)
    
    # Check if the stats JSON exists
    if not os.path.exists('monitoring/grafana_openai_stats.json'):
        print("❌ Stats dashboard JSON not found!")
        return
    
    print("✅ Stats dashboard JSON found")
    print("📊 Data verification:")
    
    # Test database connection and show current stats
    import subprocess
    try:
        # Test total validations
        result = subprocess.run([
            'psql', '-d', 'patent_monitoring', '-c',
            "SELECT COUNT(*) as total FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL '24 hours' AND openai_validation_score IS NOT NULL;"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                count = lines[2].strip()
                print(f"✅ Total OpenAI validations (24h): {count}")
            else:
                print("⚠️ Could not parse record count")
        else:
            print("❌ Database connection failed")
            return
            
        # Test success rate
        result = subprocess.run([
            'psql', '-d', 'patent_monitoring', '-c',
            "SELECT (COUNT(*) FILTER (WHERE openai_validation_success = true) * 100.0 / COUNT(*)) as success_rate FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL '24 hours' AND openai_validation_success IS NOT NULL;"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                rate = lines[2].strip()
                print(f"✅ Success rate (24h): {rate}%")
            else:
                print("⚠️ Could not parse success rate")
        else:
            print("❌ Success rate query failed")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return
    
    print("\n📋 Import Instructions:")
    print("1. Open Grafana: http://localhost:3000")
    print("2. Login with your credentials")
    print("3. Go to Dashboards → Import")
    print("4. Upload: monitoring/grafana_openai_stats.json")
    print("5. Select your existing PostgreSQL datasource")
    print("6. Click 'Import'")
    
    print("\n📊 Dashboard Features:")
    print("- 6 stat panels showing current 24h metrics")
    print("- Color-coded thresholds (green/yellow/red)")
    print("- Auto-refresh every 30 seconds")
    print("- No time series complexity")
    
    print("\n📈 Expected Panels:")
    print("1. Avg OpenAI Validation Score (0-100 scale)")
    print("2. Avg OpenAI Validation Time (in milliseconds)")
    print("3. Total Hallucination Detections (count)")
    print("4. Total Corrections Applied (count)")
    print("5. OpenAI Validation Success Rate (percentage)")
    print("6. Total OpenAI Validations (count)")
    
    print("\n🎯 Why This Should Work:")
    print("- Uses stat panels instead of time series")
    print("- Simple SQL queries with single values")
    print("- Matches your working dashboard format")
    print("- No complex aggregations or time formatting")

if __name__ == "__main__":
    main() 