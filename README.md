# DataSense AI – Automated Data Analysis & Business Intelligence Platform
> **Turn Raw Data Into Business Intelligence.**

DataSense AI is an enterprise-grade automated data analysis, business intelligence, and AI-assisted interpretation platform built with Python and Streamlit. Users can upload business datasets in CSV or Excel format (`.xlsx`, `.xls`), and the platform automatically performs data profiling, anomaly detection, non-destructive data cleaning, exploratory data analysis (EDA), dynamic visualization, KPI discovery, domain-aware business insight generation, natural language conversational data querying, and executive report export (PDF, Excel, CSV).

---

## 🌟 Key Capabilities & Modules

### 1. 📊 Executive Dashboard
- Unified executive overview of the active dataset.
- Top calculated business KPIs with benchmark deltas.
- **Dataset Health Score (0–100)** with transparent scoring rubric breakdown.
- Primary automated business charts and strategic highlight cards.

### 2. 📁 Dataset Upload Engine
- Drag-and-drop file uploader supporting `.csv`, `.xlsx`, and `.xls` (up to 50 MB).
- Multi-sheet Excel workbook support with worksheet selection dropdown.
- Automatic encoding detection fallback (`utf-8`, `latin-1`, `iso-8859-1`, `cp1252`) and CSV delimiter sniffing.
- Pre-loaded one-click business sample datasets (Retail Sales, HR Workforce Analytics, E-Commerce Orders).
- Session isolation: the original raw dataset is kept strictly immutable and is never overwritten.

### 3. 🔍 Automated Data Profiling
- Structural analysis: row/col count, memory footprint, duplicate records, missing cells.
- Column-level audit: inferred roles, data types, uniqueness, min/max, mean, median, constant feature alerts.
- **Transparent Dataset Health Score Formula**:
  - Cell Completeness (35 pts max)
  - Row Uniqueness (25 pts max)
  - Feature Richness / Variance (20 pts max)
  - Severe Column Missingness Penalty (20 pts max)

### 4. 🧹 Interactive & Non-Destructive Data Cleaning
- **Missing Value Treatment**: Numerical imputation (median, mean, zero, forward/back fill) and categorical imputation (mode, custom string, 'Unknown').
- **Deduplication**: Detection and removal with keep-first/last controls.
- **Data Type Normalization**: Converts formatted currency strings (`$1,200.50`), percentages (`15%`), and date strings to standardized analytical types.
- **Outlier Detection & Capping**: IQR (Interquartile Range) and parametric Z-score detection with interactive review and Winsorization (capping) or dropping.
- **Text Standardization**: Whitespace trimming and case normalization (`title`, `lower`, `upper`).
- **Audit Trail**: Real-time before-and-after metrics summary and full operation log with 1-click revert to original dataset.

### 5. 📈 Automated Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Mean, median, mode, standard deviation, variance, quartiles (Q1, Q3, IQR), 5th/95th percentiles, skewness, and kurtosis.
- **Categorical Distributions**: Frequency distributions, dominant vs least frequent classes, diversity metrics.
- **Correlation Engine**: Pearson and Spearman correlation matrices with strongest pairwise relationship rankings.
- **Feature Inspector**: Deep-dive interactive histograms with marginal box plots for any variable.

### 6. 📉 Dynamic Data Visualization Studio
- **Automated Recommendations**: Generates domain-aware visualizations based on detected column roles:
  - Date + Metric → Chronological Area/Line trend with auto-resampling.
  - Category + Metric → Viridis-graded ranking bar charts.
  - Profit vs Revenue → Scatter matrix with OLS trendline.
  - Composition → Interactive donut charts.
  - Numeric features → Multi-variable correlation heatmap.
- **Custom Chart Studio**: Build custom interactive Plotly visualizations selecting chart type (Bar, Horizontal Bar, Line, Area, Scatter, Pie/Donut, Histogram, Box Plot, Treemap), X/Y dimensions, color grouping, and aggregations (`sum`, `mean`, `median`, `count`, `min`, `max`).

### 7. 💡 Intelligent Business Insight Engine (Hybrid 3-Layer Architecture)
- **Layer 1 (Python Analytical Engine)**: Calculates exact statistical findings using Pandas, NumPy, and SciPy.
- **Layer 2 (Business Rule Engine)**: Evaluates deterministic heuristics:
  - Category dominance & Pareto 80/20 concentration risk.
  - Margin leakage (segments with high sales but negative or razor-thin profits).
  - Geographic/Regional disparities.
  - Chronological velocity (period-over-period expansion vs contraction).
  - HR workload vs attrition correlation.
- **Layer 3 (AI / Rule-Based Interpretation)**: Generates structured 5-part executive cards:
  1. *Observation (Measured Fact)*
  2. *Business Implication (Operational Impact)*
  3. *Strategic Recommendation (Managerial Action)*
  4. *Opportunity / Risk Classification*
  5. *Underlying Evidence Metrics*
- **Zero Hallucination Guarantee**: Figures are strictly computed; AI never invents numbers. Works 100% out of the box in deterministic mode without an external API key.

### 8. 💬 Chat With Your Data
- Natural language query processor for tabular data.
- **Strict Security**: Zero `eval()` or `exec()` of LLM code. Uses an allowlisted set of validated Pandas operations.
- Translates questions into structured analytical responses:
  - Plain English answers (e.g., *"Which category generated the highest revenue?"*, *"What is the average order value?"*, *"Show monthly revenue trends"*).
  - Tabular proof dataframe.
  - Dynamic interactive Plotly charts.

### 9. 📑 Export Reports
- **Executive PDF Report** (via ReportLab): Management-ready document with dataset overview, health scorecard, KPIs, strategic findings, cleaning audit log, and limitations/disclaimers.
- **Multi-Tab Excel Workbook** (via OpenPyXL): Contains `Cleaned_Data`, `Executive_KPIs`, `Column_Quality_Profile`, `Descriptive_Statistics`, and `Cleaning_Audit_Log`.
- **Cleaned CSV Export**: Machine-readable cleaned dataset ready for downstream analytics or ML models.

### 10. ⚙️ Adaptability & Settings
- **Modular AI Providers**: Switch between **Ollama (local on-device)**, **OpenAI**, **Groq**, and **Gemini**.
- **Semantic Role Mapping**: Override or fine-tune auto-detected column roles (Date, Identifier, Revenue, Cost, Profit, Quantity, Price, Discount, Category, Customer, Region, Status, etc.).
- **Session Cache Management**: One-click reset of session state.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend Framework** | Streamlit (Python) |
| **Data Processing** | Pandas, NumPy |
| **Data Visualization** | Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib |
| **Statistical Analysis** | SciPy, Statsmodels, Scikit-learn |
| **Report Generation** | ReportLab (PDF), OpenPyXL (Excel), CSV |
| **AI Integration** | Modular REST / Ollama / OpenAI / Groq / Gemini (with offline rule-based fallback) |
| **Testing** | Pytest |

---

## 📂 Project Architecture

```
DataSense-AI/
├── app.py                      # Primary Streamlit application entry point
├── backend/
│   ├── __init__.py
│   ├── ai_interpreter.py       # Modular LLM interface & deterministic fallback
│   ├── business_rules.py       # Layer 2 business heuristics & pattern engine
│   ├── chat_engine.py          # Safe natural language query engine
│   ├── cleaning_engine.py      # Non-destructive data cleaning engine & audit log
│   ├── column_detector.py      # Semantic column role detection & adaptability
│   ├── data_loader.py          # CSV/XLSX loader with sheet & encoding detection
│   ├── data_profiler.py        # Structural profiler & Dataset Health Scorecard
│   ├── eda_engine.py           # Descriptive statistics, distributions, correlations
│   ├── insight_engine.py       # 3-Layer hybrid insight orchestrator
│   ├── kpi_engine.py           # Domain-specific business KPI engine
│   ├── preprocessing.py        # Date feature extraction & type mappers
│   ├── report_generator.py     # PDF (ReportLab) & Excel (OpenPyXL) exporter
│   └── visualization_engine.py # Recommended & custom Plotly charts
├── views/
│   ├── dashboard_view.py       # 1. Dashboard
│   ├── upload_view.py          # 2. Upload Dataset & sample loaders
│   ├── profiling_view.py       # 3. Data Profiling & health score
│   ├── cleaning_view.py        # 4. Interactive Data Cleaning
│   ├── eda_view.py             # 5. Exploratory Data Analysis
│   ├── visualization_view.py   # 6. Data Visualization Studio
│   ├── insights_view.py        # 7. Business Insights
│   ├── chat_view.py            # 8. Chat With Your Data
│   ├── reports_view.py         # 9. Export Reports
│   └── settings_view.py        # 10. AI & Column Mapping Settings
├── utils/
│   ├── __init__.py
│   ├── constants.py            # Color palettes, roles, provider constants
│   ├── helpers.py              # Currency, number, and memory formatters
│   ├── ui_components.py        # Modern SaaS CSS, KPI cards, health widgets
│   └── validators.py           # File size, extension, and dataframe validation
├── sample_datasets/
│   ├── generate_samples.py     # Reproducible sample generator script
│   ├── retail_sales_sample.csv # Retail sales dataset with messy anomalies
│   ├── hr_analytics_sample.csv # HR workforce analytics dataset
│   ├── ecommerce_orders_sample.csv # E-commerce orders dataset
│   └── business_analytics_multisheet.xlsx # Multi-sheet Excel workbook
├── tests/
│   ├── test_app_flow.py        # End-to-end integration simulation test
│   ├── test_chat_engine.py     # Natural language query unit tests
│   ├── test_cleaning.py        # Cleaning transformations & audit log tests
│   ├── test_column_detector.py # Semantic role detector tests
│   ├── test_data_loader.py     # Multi-format data loader tests
│   ├── test_eda.py             # Descriptive statistics & correlation tests
│   ├── test_insights.py        # Business rules & KPI tests
│   ├── test_profiling.py       # Data profiler & health score tests
│   └── test_report_generator.py # PDF and Excel generation tests
├── .streamlit/
│   └── config.toml             # Headless server & dark theme configuration
├── .env.example                # Example environment variables
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # Comprehensive documentation
```

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone or Navigate to Directory
```bash
cd /Users/atharvavitthaljawalkar/Desktop/Data_Analyst_Project
```

### 3. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## ⚙️ Configuring an AI / LLM Provider (Optional)

DataSense AI functions completely without an external API key using its built-in Python analytical rule engine. If you want LLM-generated narrative synthesis, configure any of the following:

### Option A: Local Inference via Ollama (Free & Private)
1. Install [Ollama](https://ollama.ai) and pull a model:
   ```bash
   ollama run llama3:8b
   ```
2. In DataSense AI, navigate to **Settings → LLM Provider Configuration**:
   - Provider: `ollama`
   - Model: `llama3:8b`
   - Base URL: `http://localhost:11434`
3. Click **Test Connection** and **Save AI Settings**.

### Option B: Cloud API (OpenAI / Groq / Gemini)
1. In **Settings → LLM Provider Configuration**:
   - Select `openai`, `groq`, or `gemini`.
   - Enter your API Key.
   - Enter model (e.g., `gpt-4o-mini`, `llama-3.1-70b-versatile`, `gemini-1.5-flash`).
2. Alternatively, set them in your `.env` file:
   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-...
   LLM_MODEL=gpt-4o-mini
   ```

---

## 🧪 Running Automated Tests

DataSense AI includes a complete Pytest suite covering all modules:
```bash
source .venv/bin/activate
export PYTHONPATH=.
pytest tests/ -v
```

### Test Suite Summary:
- `test_app_flow.py`: Full end-to-end integration test across all 3 business domains.
- `test_chat_engine.py`: Safe query parser, aggregation, ranking, and trend handling.
- `test_cleaning.py`: Missing values, deduplication, formatted string parsing, outlier Winsorizing.
- `test_column_detector.py`: Role inference and manual override verification.
- `test_data_loader.py`: CSV delimiter sniffing, multi-sheet Excel loading, extension validation.
- `test_eda.py`: Skewness, kurtosis, quartiles, and correlation matrices.
- `test_insights.py`: KPI calculation, Pareto rules, margin leakage rules.
- `test_profiling.py`: Profile statistics and health score rubric calculation.
- `test_report_generator.py`: PDF binary structure (`%PDF`) and Excel binary structure (`PK`).

---

## 🔒 Security & Privacy Guarantees

1. **No Untrusted Code Execution**: Chat and query interpretation use allowlisted Pandas analytical pipelines. Never executes arbitrary `eval()` or `exec()`.
2. **Data Privacy**: Raw tabular records are never transmitted to external LLM APIs. Only aggregated summary figures (e.g. top category sum, gross margin %) are shared when LLM interpretation is active.
3. **Session State Isolation**: Datasets remain in memory for the active user session and are not persisted to shared external databases.
4. **File Safety**: Uploads are restricted by extension (`.csv`, `.xlsx`, `.xls`) and validated for size limits (up to 50 MB) before parsing.

---

## 🚢 Deployment Instructions

### Deploy to Streamlit Community Cloud
1. Push repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect repository and select `app.py` as main file path.
4. Set optional secrets in Streamlit Cloud Dashboard for `OPENAI_API_KEY` or `GROQ_API_KEY`.

### Deploy with Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t datasense-ai .
docker run -p 8501:8501 datasense-ai
```

