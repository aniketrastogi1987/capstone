#!/usr/bin/env python3
"""
Script to import OpenAI metrics dashboard into Grafana
"""

import requests
import json
import os

def import_openai_dashboard():
    """Import the OpenAI metrics dashboard into Grafana"""
    
    # Grafana configuration
    grafana_url = "http://localhost:3000"  # Default Grafana URL
    username = "admin"  # Default Grafana username
    password = "admin"   # Default Grafana password
    
    # Check if environment variables are set
    if os.getenv('GRAFANA_URL'):
        grafana_url = os.getenv('GRAFANA_URL')
    if os.getenv('GRAFANA_USERNAME'):
        username = os.getenv('GRAFANA_USERNAME')
    if os.getenv('GRAFANA_PASSWORD'):
        password = os.getenv('GRAFANA_PASSWORD')
    
    # Read the dashboard JSON
    try:
        with open('monitoring/grafana_openai_metrics.json', 'r') as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print("❌ Error: grafana_openai_metrics.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in dashboard file: {e}")
        return False
    
    # Prepare the import payload
    import_payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "inputs": [
            {
                "name": "DS_POSTGRES",
                "type": "datasource",
                "pluginId": "postgres",
                "value": "postgres"
            }
        ]
    }
    
    # Import the dashboard
    try:
        response = requests.post(
            f"{grafana_url}/api/dashboards/import",
            json=import_payload,
            auth=(username, password),
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ OpenAI metrics dashboard imported successfully!")
            print(f"📊 Dashboard URL: {grafana_url}{result.get('importedUrl', '')}")
            print(f"🆔 Dashboard ID: {result.get('imported', {}).get('id', 'Unknown')}")
            return True
        else:
            print(f"❌ Error importing dashboard: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to Grafana. Make sure Grafana is running on", grafana_url)
        print("💡 To start Grafana: brew services start grafana")
        return False
    except requests.exceptions.Timeout:
        print("❌ Error: Request timeout. Grafana might be slow to respond.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_grafana_status():
    """Check if Grafana is running"""
    try:
        response = requests.get("http://localhost:3000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Grafana is running")
            return True
        else:
            print("❌ Grafana is not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Grafana is not running on http://localhost:3000")
        print("💡 To start Grafana: brew services start grafana")
        return False

def main():
    """Main function"""
    print("🚀 OpenAI Metrics Dashboard Import Tool")
    print("=" * 50)
    
    # Check Grafana status
    if not check_grafana_status():
        print("\n📋 To start Grafana:")
        print("1. Install Grafana: brew install grafana")
        print("2. Start Grafana: brew services start grafana")
        print("3. Access Grafana: http://localhost:3000")
        print("4. Default credentials: admin/admin")
        return
    
    # Import the dashboard
    print("\n📥 Importing OpenAI metrics dashboard...")
    success = import_openai_dashboard()
    
    if success:
        print("\n🎉 Dashboard import completed!")
        print("\n📋 Next steps:")
        print("1. Open Grafana: http://localhost:3000")
        print("2. Navigate to the 'OpenAI Metrics Dashboard'")
        print("3. Verify the panels are showing data")
        print("4. If panels show errors, check PostgreSQL connection")
    else:
        print("\n❌ Dashboard import failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Grafana is running")
        print("2. Check Grafana credentials")
        print("3. Verify PostgreSQL datasource is configured")
        print("4. Check if PostgreSQL is running and accessible")

if __name__ == "__main__":
    main() 