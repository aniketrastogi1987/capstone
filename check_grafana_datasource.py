#!/usr/bin/env python3
"""
Script to check Grafana datasource configuration
"""

import requests
import json

def check_grafana_datasources():
    """Check available datasources in Grafana"""
    
    # Try different authentication methods
    auth_methods = [
        ("admin", "admin"),
        ("admin", "password"),
        (None, None)  # No auth
    ]
    
    for username, password in auth_methods:
        try:
            if username and password:
                auth = (username, password)
            else:
                auth = None
            
            response = requests.get(
                "http://localhost:3000/api/datasources",
                auth=auth,
                timeout=10
            )
            
            if response.status_code == 200:
                datasources = response.json()
                print(f"✅ Successfully connected to Grafana (auth: {username or 'none'})")
                print("\n📊 Available datasources:")
                
                postgres_datasources = []
                for ds in datasources:
                    print(f"  - {ds.get('name', 'Unknown')}: {ds.get('type', 'Unknown')} (UID: {ds.get('uid', 'Unknown')})")
                    if ds.get('type') == 'postgres':
                        postgres_datasources.append(ds)
                
                if postgres_datasources:
                    print(f"\n🎯 Found {len(postgres_datasources)} PostgreSQL datasource(s):")
                    for ds in postgres_datasources:
                        print(f"  - Name: {ds.get('name')}")
                        print(f"    UID: {ds.get('uid')}")
                        print(f"    URL: {ds.get('url', 'Unknown')}")
                        print(f"    Database: {ds.get('jsonData', {}).get('database', 'Unknown')}")
                else:
                    print("\n❌ No PostgreSQL datasource found!")
                    print("💡 You need to add a PostgreSQL datasource in Grafana")
                
                return postgres_datasources
            else:
                print(f"❌ Failed with auth {username or 'none'}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error with auth {username or 'none'}: {e}")
    
    print("\n❌ Could not connect to Grafana with any authentication method")
    return []

def create_simple_dashboard(uid):
    """Create a simple dashboard with the correct UID"""
    
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "OpenAI Metrics - Simple",
            "tags": ["openai"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "OpenAI Validation Score",
                    "type": "timeseries",
                    "targets": [
                        {
                            "refId": "A",
                            "rawSql": "SELECT timestamp as time, COALESCE(openai_validation_score, 0) as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL '24 hours' ORDER BY timestamp",
                            "format": "time_series"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "short"
                        }
                    }
                },
                {
                    "id": 2,
                    "title": "OpenAI Validation Time",
                    "type": "timeseries",
                    "targets": [
                        {
                            "refId": "A",
                            "rawSql": "SELECT timestamp as time, COALESCE(openai_validation_time, 0) * 1000 as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL '24 hours' ORDER BY timestamp",
                            "format": "time_series"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "ms"
                        }
                    }
                }
            ]
        },
        "inputs": [
            {
                "name": "DS_POSTGRES",
                "type": "datasource",
                "pluginId": "postgres",
                "value": uid
            }
        ],
        "overwrite": True
    }
    
    # Save the dashboard
    with open('monitoring/grafana_openai_simple.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"\n📁 Created simple dashboard with UID: {uid}")
    print("📄 File: monitoring/grafana_openai_simple.json")
    
    return dashboard

def main():
    """Main function"""
    print("🔍 Grafana Datasource Checker")
    print("=" * 40)
    
    postgres_datasources = check_grafana_datasources()
    
    if postgres_datasources:
        # Use the first PostgreSQL datasource
        uid = postgres_datasources[0]['uid']
        print(f"\n🎯 Using PostgreSQL datasource UID: {uid}")
        
        # Create a simple dashboard with the correct UID
        dashboard = create_simple_dashboard(uid)
        
        print("\n📋 Manual Import Instructions:")
        print("1. Open Grafana: http://localhost:3000")
        print("2. Go to Dashboards → Import")
        print("3. Upload: monitoring/grafana_openai_simple.json")
        print("4. Select the PostgreSQL datasource")
        print("5. Click 'Import'")
        
        print("\n🔧 Alternative: Use the API")
        print(f"curl -X POST http://localhost:3000/api/dashboards/import \\")
        print(f"  -H 'Content-Type: application/json' \\")
        print(f"  -d @monitoring/grafana_openai_simple.json")
        
    else:
        print("\n❌ No PostgreSQL datasource found!")
        print("\n🔧 To add PostgreSQL datasource:")
        print("1. Open Grafana: http://localhost:3000")
        print("2. Go to Configuration → Data Sources")
        print("3. Click 'Add data source'")
        print("4. Select 'PostgreSQL'")
        print("5. Configure:")
        print("   - Host: localhost:5432")
        print("   - Database: patent_monitoring")
        print("   - User: (your postgres user)")
        print("   - Password: (your postgres password)")
        print("6. Test connection and save")

if __name__ == "__main__":
    main() 