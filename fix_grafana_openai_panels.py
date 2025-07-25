 #!/usr/bin/env python3
"""
Script to fix OpenAI panel queries in Grafana dashboard
"""

import json
import re

def fix_openai_panels():
    """Fix OpenAI panel queries to handle NULL values properly"""
    
    # Read the current dashboard
    with open('monitoring/grafana_comprehensive_dashboard.json', 'r') as f:
        dashboard = json.load(f)
    
    # Fix queries for OpenAI panels
    for panel in dashboard['panels']:
        if 'targets' in panel and panel['targets']:
            for target in panel['targets']:
                if 'rawSql' in target:
                    sql = target['rawSql']
                    
                    # Fix OpenAI Validation Score panel
                    if 'openai_validation_score' in sql and 'openai_validation_score > 0' in sql:
                        new_sql = sql.replace(
                            'openai_validation_score as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_validation_score > 0',
                            'COALESCE(openai_validation_score, 0) as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL \'24 hours\''
                        )
                        target['rawSql'] = new_sql
                        print(f"Fixed OpenAI Validation Score query")
                    
                    # Fix OpenAI Validation Time panel
                    elif 'openai_validation_time' in sql and 'openai_validation_time > 0' in sql:
                        new_sql = sql.replace(
                            'openai_validation_time * 1000 as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_validation_time > 0',
                            'COALESCE(openai_validation_time, 0) * 1000 as value FROM chat_metrics WHERE timestamp >= NOW() - INTERVAL \'24 hours\''
                        )
                        target['rawSql'] = new_sql
                        print(f"Fixed OpenAI Validation Time query")
                    
                    # Fix OpenAI Hallucination Detections panel
                    elif 'openai_hallucination_detected = true' in sql:
                        new_sql = sql.replace(
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_hallucination_detected = true',
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_hallucination_detected IS NOT NULL'
                        )
                        target['rawSql'] = new_sql
                        print(f"Fixed OpenAI Hallucination Detections query")
                    
                    # Fix OpenAI Corrections Applied panel
                    elif 'openai_corrections_applied = true' in sql:
                        new_sql = sql.replace(
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_corrections_applied = true',
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_corrections_applied IS NOT NULL'
                        )
                        target['rawSql'] = new_sql
                        print(f"Fixed OpenAI Corrections Applied query")
                    
                    # Fix OpenAI Validation Success Rate panel
                    elif 'openai_validation_success = true' in sql:
                        new_sql = sql.replace(
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_validation_score > 0',
                            'WHERE timestamp >= NOW() - INTERVAL \'24 hours\' AND openai_validation_score IS NOT NULL'
                        )
                        target['rawSql'] = new_sql
                        print(f"Fixed OpenAI Validation Success Rate query")
    
    # Write the updated dashboard
    with open('monitoring/grafana_comprehensive_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("✅ Fixed all OpenAI panel queries in Grafana dashboard")

if __name__ == "__main__":
    fix_openai_panels()