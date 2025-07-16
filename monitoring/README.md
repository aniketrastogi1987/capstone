# 📊 Chatbot Performance Monitoring

This monitoring system tracks comprehensive performance metrics for the patent chatbot and can export data to Grafana for visualization.

## 🎯 What We Monitor

### **Response Performance**
- **Response Time**: Average, 95th percentile, max response times
- **Throughput**: Requests per minute/second
- **Success Rate**: Percentage of successful vs failed requests
- **Timeout Rate**: Percentage of requests that timeout

### **Model Performance**
- **Token Usage**: Input/output tokens per request
- **Model Latency**: Time spent in LLM inference
- **Model Accuracy**: Guardrails scores (profanity, topic, politeness)
- **Evaluation Metrics**: ROUGE, relevance, coherence scores

### **System Health**
- **LightRAG Server Status**: Health checks, uptime
- **Neo4j Database**: Connection status, query performance
- **Ollama Service**: Model availability, memory usage
- **Resource Usage**: CPU, memory, disk I/O

### **User Experience**
- **Active Sessions**: Number of concurrent users
- **Query Patterns**: Most common questions, query length
- **User Satisfaction**: Based on guardrails scores
- **Error Types**: Timeout, connection, validation errors

### **Business Metrics**
- **Patent Coverage**: Number of patents indexed
- **Query Volume**: Daily/weekly/monthly trends
- **Popular Topics**: Most queried patent categories
- **System Utilization**: Peak usage times

## 🚀 Quick Start

### 1. **Enable Monitoring in Chatbot**
```bash
# Interactive mode with monitoring
python main.py
# Select option 3 (Run interactive chatbot)
# Enable monitoring when prompted
```

### 2. **View Real-time Dashboard**
```bash
# From the main menu, select option 9
# "Show monitoring dashboard"
```

### 3. **Save Monitoring Data**
```bash
# From the main menu, select option 10
# "Save monitoring data"
```

## 📈 Grafana Integration

### **Setup Grafana**

1. **Install Grafana**:
   ```bash
   # macOS
   brew install grafana
   
   # Start Grafana
   brew services start grafana
   ```

2. **Access Grafana**:
   - Open http://localhost:3000
   - Default credentials: admin/admin
   - Change password when prompted

3. **Add Data Source**:
   - Go to Configuration → Data Sources
   - Add Prometheus (if using Prometheus)
   - Or use JSON files for direct import

### **Import Dashboard**

1. **Copy Dashboard Configuration**:
   ```bash
   cp monitoring/grafana_dashboard.json /tmp/
   ```

2. **Import in Grafana**:
   - Go to Dashboards → Import
   - Upload the JSON file
   - Configure data source
   - Save dashboard

### **Dashboard Panels**

The dashboard includes these key panels:

1. **Response Performance**: Average response time
2. **Success Rate**: Percentage of successful requests
3. **Response Time Distribution**: P95 and P50 response times
4. **System Health**: LightRAG, Neo4j, Ollama status
5. **Resource Usage**: CPU, memory, disk usage
6. **Guardrails Performance**: Profanity, topic, politeness scores
7. **Active Sessions**: Current user sessions
8. **Token Usage**: Average tokens per request
9. **Query Categories**: Distribution of query types
10. **Error Rate**: Request error frequency

## 🔧 Configuration

### **Monitoring Settings**

The monitoring system can be configured via the chatbot initialization:

```python
from chatbot.patent_chatbot import PatentChatbot

# Enable monitoring
chatbot = PatentChatbot(
    with_guardrails=True,
    enable_monitoring=True
)

# Disable monitoring
chatbot = PatentChatbot(
    with_guardrails=True,
    enable_monitoring=False
)
```

### **Data Retention**

- **Response Metrics**: Last 10,000 responses
- **System Health**: Last 1,000 health checks
- **Business Metrics**: Last 100 snapshots
- **Performance Data**: Rolling 1,000 samples

### **Health Check Frequency**

- **System Health**: Every 30 seconds
- **Business Metrics**: Every 5 minutes (10 health checks)
- **Background Monitoring**: Continuous in separate thread

## 📊 Metrics Export

### **JSON Export Format**

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "performance": {
    "period_hours": 1,
    "total_requests": 150,
    "success_rate": 0.95,
    "avg_response_time": 2.3,
    "p95_response_time": 4.1,
    "guardrail_scores": {
      "profanity_score": 0.98,
      "topic_relevance_score": 0.92,
      "politeness_score": 0.89
    }
  },
  "system_health": {
    "current_status": {
      "lightrag": true,
      "neo4j": true,
      "ollama": true,
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "active_connections": 3
    }
  },
  "real_time": {
    "active_sessions": 3,
    "total_requests": 150,
    "error_count": 7,
    "avg_response_time": 2.3
  }
}
```

### **Grafana Export**

The system provides a `export_metrics_for_grafana()` method that returns data in a format suitable for Grafana dashboards.

## 🛠️ Advanced Usage

### **Custom Metrics**

You can extend the monitoring system by adding custom metrics:

```python
from monitoring.chatbot_monitor import ChatbotMonitor

monitor = ChatbotMonitor()

# Add custom metric
monitor.record_custom_metric("custom_metric", value)
```

### **Alerting**

Set up Grafana alerts for critical metrics:

- **Response Time > 10s**: High latency alert
- **Success Rate < 90%**: Error rate alert
- **CPU Usage > 80%**: Resource alert
- **System Down**: Service availability alert

### **Data Analysis**

Use the saved JSON files for offline analysis:

```python
import json

with open('chatbot_metrics_20240115_103000.json', 'r') as f:
    data = json.load(f)
    
# Analyze performance trends
performance = data['performance_summary']
print(f"Success rate: {performance['success_rate']:.1%}")
```

## 🔍 Troubleshooting

### **Common Issues**

1. **No Monitoring Data**:
   - Ensure monitoring is enabled when starting chatbot
   - Check that the chatbot has been used recently

2. **Grafana Connection Issues**:
   - Verify Grafana is running on port 3000
   - Check data source configuration
   - Ensure JSON export format matches dashboard expectations

3. **High Memory Usage**:
   - Monitoring stores data in memory
   - Consider reducing retention limits for long-running sessions

### **Performance Impact**

- **Memory**: ~10MB for 10,000 responses
- **CPU**: Minimal impact (background thread)
- **Storage**: JSON exports only when explicitly saved

## 📈 Best Practices

1. **Enable Monitoring**: Always enable for production use
2. **Regular Exports**: Save monitoring data periodically
3. **Dashboard Review**: Check Grafana dashboard regularly
4. **Alert Setup**: Configure alerts for critical metrics
5. **Data Retention**: Clean up old monitoring files periodically

## 🔗 Integration with Existing Systems

The monitoring system can be integrated with:

- **Prometheus**: For time-series metrics
- **InfluxDB**: For high-frequency metrics
- **Elasticsearch**: For log analysis
- **Slack/Discord**: For alerting
- **Email**: For critical alerts

## 📚 API Reference

### **ChatbotMonitor Class**

```python
monitor = ChatbotMonitor(lightrag_url="http://localhost:9621")

# Record a response
monitor.record_response(metrics)

# Get performance summary
summary = monitor.get_performance_summary(hours=24)

# Export for Grafana
grafana_data = monitor.export_metrics_for_grafana()

# Save to file
filename = monitor.save_metrics_to_file()
```

### **ResponseMetrics Class**

```python
metrics = ResponseMetrics(
    timestamp=datetime.now(),
    query="What is this patent about?",
    response_time=2.5,
    success=True,
    input_tokens=10,
    output_tokens=150,
    guardrail_scores={'profanity_score': 1.0, 'topic_relevance_score': 0.9},
    source_count=3,
    response_length=500
)
``` 