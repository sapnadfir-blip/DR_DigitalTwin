# Grafana Dashboard — DR Digital Twin

Real-time analytics dashboard for the Diabetic Retinopathy Digital Twin project.
All data comes from real OLIVES patient data — no simulations.

## Dashboard Panels

### Row 1: Key Metrics (stat panels)
- Best Model name + Macro F1 score
- Total patient sequences from OLIVES
- Significant causal edges discovered
- Progression model MAE
- Number of test patients evaluated

### Row 2: Data Distribution
- Biomarker prevalence (% of visits each biomarker is active)
- Patient severity distribution (histogram)
- Visit count distribution (how many visits per patient)

### Row 3: Causal Analysis
- Significant causal edges table (source → target, effect, p-value)
- Discovered causal chains table (multi-hop pathways, strength)

### Row 4: Treatment Analysis
- Treatment recommendation pie chart (which drug recommended most)
- Per-patient treatment effectiveness table
- Patient details table (severity, active biomarkers, visits)

### Row 5: Full Causal Matrix
- Complete 6×6 causal effect matrix with color-coded values

## Setup

### 1. Install Grafana
```bash
# Windows (winget)
winget install GrafanLabs.Grafana

# macOS
brew install grafana

# Linux
sudo apt-get install grafana
```

### 2. Install JSON API Plugin
```bash
grafana-cli plugins install marcusolsson-json-datasource
# Restart Grafana after installing
```

### 3. Run the Pipeline First
```bash
# Train models on OLIVES data (generates metrics files)
python run_pipeline.py
```

### 4. Start the Metrics Server
```bash
python grafana/metrics_server.py
# Runs on http://localhost:3001
```

### 5. Configure Grafana Datasource

**Option A: Auto-provision**
Copy `datasource.yaml` to your Grafana provisioning directory:
- Windows: `C:\Program Files\GrafanaLabs\grafana\conf\provisioning\datasources\`
- macOS: `/opt/homebrew/etc/grafana/provisioning/datasources/`
- Linux: `/etc/grafana/provisioning/datasources/`

**Option B: Manual**
1. Open Grafana (http://localhost:3000)
2. Go to Configuration → Data Sources → Add data source
3. Search "JSON API" and select it
4. Set URL: `http://localhost:3001`
5. Save & Test

### 6. Import Dashboard
1. Open Grafana → Dashboards → Import
2. Upload `dashboard.json`
3. Select the "DR Digital Twin Metrics" datasource
4. Click Import

## API Endpoints

The metrics server exposes these endpoints:

| Endpoint | Description |
|----------|-------------|
| `/overview` | Summary stats (F1, MAE, counts) |
| `/biomarker-prevalence` | Per-biomarker active percentage |
| `/patient-severity` | Per-patient severity + active BMs |
| `/causal-edges` | Significant directed causal links |
| `/causal-chains` | Multi-hop causal pathways |
| `/causal-matrix` | Full 6×6 causal effect matrix |
| `/treatment-recommendations` | Treatment recommendation counts |
| `/treatment-effectiveness` | Per-patient per-treatment severity |
| `/visit-distribution` | Visit count histogram data |
| `/pipeline-report` | Full pipeline report JSON |
| `/patient-reports` | All patient evaluation reports |
