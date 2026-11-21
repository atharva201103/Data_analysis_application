"""
Constants and configurations for DataSense AI.
"""

APP_NAME = "DataSense AI"
APP_TAGLINE = "Turn Raw Data Into Business Intelligence"
APP_VERSION = "1.0.0"

# Color Palette (Modern SaaS)
THEME_COLORS = {
    "primary": "#4F46E5",       # Indigo 600
    "primary_light": "#818CF8", # Indigo 400
    "primary_dark": "#3730A3",  # Indigo 800
    "secondary": "#06B6D4",     # Cyan 500
    "accent": "#F59E0B",        # Amber 500
    "success": "#10B981",       # Emerald 500
    "warning": "#F59E0B",       # Amber 500
    "danger": "#EF4444",        # Red 500
    "info": "#3B82F6",          # Blue 500
    "bg_dark": "#0F172A",       # Slate 900
    "card_dark": "#1E293B",     # Slate 800
    "border_dark": "#334155",   # Slate 700
    "text_dark": "#F8FAFC",     # Slate 50
    "text_muted": "#94A3B8"     # Slate 400
}

# Plotly Color Sequences
CHART_COLOR_SEQUENCE = [
    "#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EC4899",
    "#8B5CF6", "#14B8A6", "#F97316", "#6366F1", "#84CC16"
]

# Supported File Formats
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Semantic Column Roles
ROLE_DATE = "Date"
ROLE_ID = "Identifier"
ROLE_REVENUE = "Revenue"
ROLE_COST = "Cost"
ROLE_PROFIT = "Profit"
ROLE_QUANTITY = "Quantity"
ROLE_PRICE = "Price"
ROLE_DISCOUNT = "Discount"
ROLE_CATEGORY = "Category"
ROLE_SUBCATEGORY = "Subcategory"
ROLE_CUSTOMER = "Customer"
ROLE_REGION = "Region"
ROLE_PRODUCT = "Product"
ROLE_STATUS = "Status"
ROLE_PERCENTAGE = "Percentage"
ROLE_NUMERICAL = "Numerical"
ROLE_CATEGORICAL = "Categorical"

SEMANTIC_ROLES = [
    ROLE_DATE,
    ROLE_ID,
    ROLE_REVENUE,
    ROLE_COST,
    ROLE_PROFIT,
    ROLE_QUANTITY,
    ROLE_PRICE,
    ROLE_DISCOUNT,
    ROLE_CATEGORY,
    ROLE_SUBCATEGORY,
    ROLE_CUSTOMER,
    ROLE_REGION,
    ROLE_PRODUCT,
    ROLE_STATUS,
    ROLE_PERCENTAGE,
    ROLE_NUMERICAL,
    ROLE_CATEGORICAL
]

# LLM Providers
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_GROQ = "groq"
PROVIDER_GEMINI = "gemini"
PROVIDER_ANTHROPIC = "anthropic"

LLM_PROVIDERS = [
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_GROQ,
    PROVIDER_GEMINI,
    PROVIDER_ANTHROPIC
]

