#!/usr/bin/env python3
"""
Script to import the fixed OpenAI metrics dashboard
"""

import json
import os

def main():
    """Main function"""
    print("🚀 OpenAI Metrics Dashboard - Fixed Version")
    print("=" * 50)
    
    # Check if the fixed JSON exists
    if not os.path.exists('monitoring/grafana_openai_fixed.json'):
        print("❌ Fixed dashboard JSON not found!")
        return
    
    print("✅ Fixed dashboard JSON found")
    print("📊 Data verification:")
    
    # Test database connection
    import subprocess
    try:
        result = subprocess.run([
            'psql', '-d', 'patent_monitoring', '-c',
            "SELECT COUNT(*) as openai_records FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL '24 hours' AND openai_validation_score IS NOT NULL;"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                count = lines[2].strip()
                print(f"✅ OpenAI records in last 24h: {count}")
            else:
                print("⚠️ Could not parse record count")
        else:
            print("❌ Database connection failed")
            return
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return
    
    print("\n📋 Import Instructions:")
    print("1. Open Grafana: http://localhost:3000")
    print("2. Login with your credentials")
    print("3. Go to Dashboards → Import")
    print("4. Upload: monitoring/grafana_openai_fixed.json")
    print("5. Select your existing PostgreSQL datasource")
    print("6. Click 'Import'")
    
    print("\n🔧 If import fails:")
    print("1. Check that your PostgreSQL datasource UID is 'postgres'")
    print("2. Verify the datasource is working with other dashboards")
    print("3. Try importing with a different dashboard name")
    
    print("\n📊 Expected Results:")
    print("- 5 panels with OpenAI metrics (removed Model Usage Distribution)")
    print("- Real-time data from the last 24 hours")
    print("- No error messages in panels")
    print("- Proper formatting and units")
    print("- Working: Validation Score, Validation Time, Hallucination Detections, Corrections Applied, Success Rate")
    
    print("\n💡 Alternative: Copy panels to existing dashboard")
    print("1. Open your working dashboard")
    print("2. Add new panels manually")
    print("3. Use the SQL queries from the JSON file")
    
    print("\n🔍 Fixed Issues:")
    print("- ✅ Proper time series format for aggregated data")
    print("- ✅ Correct column names (time, value)")
    print("- ✅ Removed Model Usage Distribution panel")
    print("- ✅ Fixed SQL queries for hourly aggregations")

if __name__ == "__main__":
    main() 