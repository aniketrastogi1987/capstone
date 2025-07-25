# Grafana OpenAI Metrics Dashboard Import Guide

## Problem
The OpenAI panels in Grafana are showing errors because:
1. Authentication issues with the API import
2. Incorrect datasource UID in the dashboard JSON
3. Missing or misconfigured PostgreSQL datasource

## Solution

### Step 1: Check Grafana Status
```bash
# Check if Grafana is running
brew services list | grep grafana

# If not running, start it
brew services start grafana
```

### Step 2: Access Grafana
1. Open browser: http://localhost:3000
2. Login with default credentials: `admin` / `admin`
3. If prompted to change password, use: `admin` / `admin`

### Step 3: Configure PostgreSQL Datasource
1. Go to **Configuration** → **Data Sources**
2. Click **"Add data source"**
3. Select **"PostgreSQL"**
4. Configure:
   - **Name**: `postgres`
   - **Host**: `localhost:5432`
   - **Database**: `patent_monitoring`
   - **User**: (your postgres username, usually your system username)
   - **Password**: (your postgres password, or leave blank if none)
   - **SSL Mode**: `disable`
5. Click **"Test connection"** - should show "Data source is working"
6. Click **"Save & test"**

### Step 4: Import the Dashboard
1. Go to **Dashboards** → **Import**
2. Click **"Upload JSON file"**
3. Select: `monitoring/grafana_openai_simple.json`
4. Click **"Load"**
5. In the import screen:
   - **Name**: `OpenAI Metrics Dashboard`
   - **Folder**: (leave default)
   - **Datasource**: Select the PostgreSQL datasource you created
6. Click **"Import"**

### Step 5: Verify the Dashboard
1. The dashboard should load with 6 panels:
   - OpenAI Validation Score
   - OpenAI Validation Time
   - OpenAI Hallucination Detections
   - OpenAI Corrections Applied
   - OpenAI Validation Success Rate
   - Model Usage Distribution

### Step 6: Check Data
If panels show "No data":
1. Run the data check: `python3 check_openai_metrics.py`
2. Verify PostgreSQL is running: `brew services start postgresql`
3. Check database connection: `psql -d patent_monitoring -c "SELECT COUNT(*) FROM chat_metrics;"`

## Alternative: API Import
If you want to use the API instead:

```bash
# First, get the correct datasource UID
curl -u admin:admin http://localhost:3000/api/datasources

# Then import with the correct UID
curl -X POST http://localhost:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d @monitoring/grafana_openai_simple.json
```

## Troubleshooting

### Error: "Invalid username or password"
- Try the default credentials: `admin` / `admin`
- If changed, reset Grafana: `brew services restart grafana`

### Error: "Data source not found"
- Make sure PostgreSQL datasource is configured correctly
- Check the datasource UID matches in the JSON file

### Error: "No data"
- Verify PostgreSQL is running: `brew services start postgresql`
- Check database exists: `psql -l | grep patent_monitoring`
- Run data check: `python3 check_openai_metrics.py`

### Error: "Connection refused"
- Start PostgreSQL: `brew services start postgresql`
- Check PostgreSQL is listening: `lsof -i :5432`

## Files Created
- `monitoring/grafana_openai_simple.json` - Simple dashboard for manual import
- `check_openai_metrics.py` - Script to verify data exists
- `check_grafana_datasource.py` - Script to check Grafana configuration

## Expected Results
After successful import, you should see:
- 6 panels with OpenAI metrics
- Real-time data from the last 24 hours
- Proper formatting and units for each metric
- No error messages in the panels 