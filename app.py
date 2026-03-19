"""
ServiceNow AI Analytics Dashboard
- Upload ServiceNow data via Excel
- Natural language querying with Llama 3.3 70B (Groq)
- Agentic AI with AutoGen
- Auto chart/graph generation

Requirements:
    pip install streamlit pandas openpyxl groq pyautogen plotly matplotlib seaborn
"""

import os
import re
import json
import traceback
from typing import Optional, Dict, Any, Tuple, List
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from groq import Groq
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ServiceNow AI Analytics",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#0078D4; margin-bottom:0.2rem;}
    .sub-header  {font-size:1rem;  color:#555; margin-bottom:1.5rem;}
    .metric-card {background:#f0f4ff; border-radius:10px; padding:1rem; text-align:center;}
    .chat-user   {background:#DCF8C6; border-radius:10px; padding:0.8rem; margin:0.4rem 0;}
    .chat-ai     {background:#F1F0F0; border-radius:10px; padding:0.8rem; margin:0.4rem 0;}
    .stButton>button {background:#0078D4; color:white; border-radius:8px; border:none; padding:0.5rem 1.2rem;}
    .stButton>button:hover {background:#005a9e;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "df": None,
    "chat_history": [],
    "groq_client": None,
    "file_name": "",
    "debug_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def init_groq(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def get_df_schema(df: pd.DataFrame) -> str:
    """Get detailed schema with samples and statistics."""
    info = []
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Add more context for better AI understanding
        extra_info = f"unique={unique_count}, nulls={null_count}"
        info.append(f"  - {col} ({dtype}): {extra_info}, sample={sample}")
    return "\n".join(info)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DataFrame for better analytics."""
    df_processed = df.copy()
    
    # Auto-detect and convert date columns
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Check if column name suggests it's a date
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'opened', 'closed', 'resolved', 'modified']):
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                except:
                    pass
    
    return df_processed


def parse_date_range(date_str: str) -> Optional[Tuple[str, str]]:
    """Parse natural language date ranges."""
    from datetime import datetime, timedelta
    import calendar
    
    date_str_lower = date_str.lower()
    today = datetime.now()
    
    # Month and year patterns
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Try to parse "month of MONTH YEAR" pattern
    for month_name, month_num in months.items():
        if month_name in date_str_lower:
            # Extract year
            import re
            year_match = re.search(r'\b(20\d{2})\b', date_str)
            if year_match:
                year = int(year_match.group(1))
                last_day = calendar.monthrange(year, month_num)[1]
                start_date = f"{year}-{month_num:02d}-01"
                end_date = f"{year}-{month_num:02d}-{last_day}"
                return (start_date, end_date)
    
    # Relative dates
    if 'last month' in date_str_lower:
        first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day_last_month = today.replace(day=1) - timedelta(days=1)
        return (first_day_last_month.strftime('%Y-%m-%d'), last_day_last_month.strftime('%Y-%m-%d'))
    
    if 'this month' in date_str_lower:
        first_day = today.replace(day=1)
        return (first_day.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    
    if 'last week' in date_str_lower:
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
    
    if 'this week' in date_str_lower:
        start = today - timedelta(days=today.weekday())
        return (start.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
    
    return None


def build_system_prompt(df: pd.DataFrame) -> str:
    schema = get_df_schema(df)
    shape  = df.shape
    cols   = list(df.columns)
    
    # Identify column types for better guidance
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    date_cols = df.select_dtypes(include='datetime').columns.tolist()
    
    # Try to identify date columns from object types (common in ServiceNow exports)
    potential_date_cols = [col for col in cat_cols if any(keyword in col.lower()
                          for keyword in ['date', 'time', 'created', 'updated', 'opened', 'closed', 'resolved'])]
    
    # Identify common ServiceNow columns
    tower_cols = [col for col in cols if any(keyword in col.lower() for keyword in ['tower', 'group', 'team', 'assignment'])]
    priority_cols = [col for col in cols if 'priority' in col.lower()]
    status_cols = [col for col in cols if any(keyword in col.lower() for keyword in ['status', 'state', 'stage'])]
    
    return f"""You are an EXPERT ServiceNow data analyst and visualization specialist with deep knowledge of analytics.

DATASET INFORMATION:
- Total Records: {shape[0]:,}
- Total Columns: {shape[1]}

DETAILED SCHEMA:
{schema}

COLUMN CATEGORIES:
- Categorical Columns: {cat_cols[:15]}
- Numeric Columns: {num_cols[:15]}
- Date/Time Columns: {date_cols[:10]}
- Potential Date Columns: {potential_date_cols[:10]}
- Tower/Group Columns: {tower_cols}
- Priority Columns: {priority_cols}
- Status Columns: {status_cols}

═══════════════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

**IMPORTANT:** Determine if the user wants a visualization or just an answer:

1. **TEXT-ONLY QUERIES** (No chart needed):
   - "How many tickets were created in May 2025?"
   - "What is the total count of high priority tickets?"
   - "How many incidents are open?"
   - "What's the average resolution time?"
   
   **Response:** Provide a direct, concise answer with the number/statistic.
   Example: "There were 1,247 tickets created in May 2025."

2. **VISUALIZATION QUERIES** (Chart needed):
   - "Show me...", "Display...", "Chart of...", "Graph of..."
   - "Weekly trend", "Monthly breakdown", "Distribution of..."
   - "Compare...", "Visualize...", "Plot..."
   
   **Response:** Provide JSON chart specification (see below).

═══════════════════════════════════════════════════════════════════════════════
CHART SPECIFICATION FORMAT (Only for visualization queries)
═══════════════════════════════════════════════════════════════════════════════

When user requests visualization, respond with JSON in ```json ``` block:

{{
  "chart_type": "bar|line|pie|scatter|histogram|heatmap|box|funnel|treemap",
  "title": "Clear, descriptive title",
  "x": "column_name",
  "y": "column_name_or_null",
  "color": "column_name_or_null",
  "agg": "count|sum|mean|median|nunique",
  "filters": {{}},
  "time_grouping": "day|week|month|quarter|year|null",
  "insight": "One sentence insight about the data pattern"
}}

═══════════════════════════════════════════════════════════════════════════════
CHART TYPE SELECTION GUIDE
═══════════════════════════════════════════════════════════════════════════════

📊 BAR CHART - Categorical comparisons
   Use when: Comparing counts/values across categories
   Example: "tickets by priority", "incidents by tower"
   Setup: x=category_column, agg="count", chart_type="bar"

📈 LINE CHART - Time trends (BEST for time-based queries)
   Use when: Showing trends over time, temporal patterns
   Example: "weekly tickets", "monthly trend", "daily count"
   Setup: x=date_column, time_grouping="week/month/day", chart_type="line"

🥧 PIE CHART - Proportions/percentages
   Use when: Showing parts of a whole (max 10 categories)
   Example: "distribution of priorities", "ticket breakdown by status"
   Setup: x=category_column, agg="count", chart_type="pie"

🔵 SCATTER - Numeric relationships
   Use when: Exploring correlation between two numeric variables
   Example: "resolution time vs priority", "age vs count"
   Setup: x=numeric_col, y=numeric_col, chart_type="scatter"

📊 HISTOGRAM - Distribution of single variable
   Use when: Understanding data distribution
   Example: "distribution of resolution times"
   Setup: x=numeric_column, chart_type="histogram"

🔥 HEATMAP - 2D categorical patterns
   Use when: Showing intensity across two categorical dimensions
   Example: "tickets by day of week and hour", "priority vs status"
   Setup: x=category1, y=category2, chart_type="heatmap"

═══════════════════════════════════════════════════════════════════════════════
TIME-BASED ANALYTICS (CRITICAL FOR TEMPORAL QUERIES)
═══════════════════════════════════════════════════════════════════════════════

🕐 DAILY Analysis:
   Query: "daily tickets", "tickets per day"
   JSON: {{"chart_type": "line", "x": "{date_cols[0] if date_cols else 'created_on'}", "time_grouping": "day", "agg": "count"}}

📅 WEEKLY Analysis:
   Query: "weekly count", "tickets by week", "weekly trend"
   JSON: {{"chart_type": "line", "x": "{date_cols[0] if date_cols else 'created_on'}", "time_grouping": "week", "agg": "count"}}

📆 MONTHLY Analysis:
   Query: "monthly tickets", "trend by month"
   JSON: {{"chart_type": "line", "x": "{date_cols[0] if date_cols else 'created_on'}", "time_grouping": "month", "agg": "count"}}

📊 QUARTERLY/YEARLY:
   Use time_grouping="quarter" or "year" for longer periods

═══════════════════════════════════════════════════════════════════════════════
FILTERING & DATE RANGES (ESSENTIAL)
═══════════════════════════════════════════════════════════════════════════════

📍 CATEGORICAL FILTERS:
   "security tower" → filters={{"{tower_cols[0] if tower_cols else 'tower'}": "security"}}
   "high priority" → filters={{"{priority_cols[0] if priority_cols else 'priority'}": "high"}}
   "open status" → filters={{"{status_cols[0] if status_cols else 'state'}": "open"}}

📅 DATE RANGE FILTERS:
   "May 2025" → filters={{"{date_cols[0] if date_cols else 'created_on'}": ["2025-05-01", "2025-05-31"]}}
   "Q1 2025" → filters={{"{date_cols[0] if date_cols else 'created_on'}": ["2025-01-01", "2025-03-31"]}}
   "last month" → filters{{"{date_cols[0] if date_cols else 'created_on'}": ["2024-12-01", "2024-12-31"]}}
   "2025" → filters={{"{date_cols[0] if date_cols else 'created_on'}": ["2025-01-01", "2025-12-31"]}}

═══════════════════════════════════════════════════════════════════════════════
MULTI-DIMENSIONAL ANALYSIS (COLOR GROUPING)
═══════════════════════════════════════════════════════════════════════════════

🎨 Use COLOR parameter to show multiple categories in one chart:

"weekly tickets for ALL towers" →
{{
  "chart_type": "line",
  "x": "{date_cols[0] if date_cols else 'created_on'}",
  "color": "{tower_cols[0] if tower_cols else 'tower'}",
  "time_grouping": "week",
  "agg": "count"
}}

"monthly count by priority" →
{{
  "chart_type": "line",
  "x": "{date_cols[0] if date_cols else 'created_on'}",
  "color": "{priority_cols[0] if priority_cols else 'priority'}",
  "time_grouping": "month",
  "agg": "count"
}}

═══════════════════════════════════════════════════════════════════════════════
TEXT-ONLY QUERY EXAMPLES (No JSON, just answer)
═══════════════════════════════════════════════════════════════════════════════

❓ "How many tickets were created in May 2025?"
✅ Response: "There were 1,247 tickets created in May 2025."

❓ "What is the total count of high priority tickets?"
✅ Response: "There are 342 high priority tickets in the dataset."

❓ "How many open incidents are in security tower?"
✅ Response: "There are 89 open incidents in the security tower."

❓ "What's the average resolution time?"
✅ Response: "The average resolution time is 4.2 days."

❓ "Count of tickets by priority"
✅ Response: "High: 342, Medium: 567, Low: 891"

═══════════════════════════════════════════════════════════════════════════════
VISUALIZATION QUERY EXAMPLES (Provide JSON chart spec)
═══════════════════════════════════════════════════════════════════════════════

❓ "Show me weekly count of tickets for all towers for the month of May 2025"
✅ {{
  "chart_type": "line",
  "title": "Weekly Ticket Count by Tower - May 2025",
  "x": "{date_cols[0] if date_cols else 'created_on'}",
  "y": null,
  "color": "{tower_cols[0] if tower_cols else 'tower'}",
  "agg": "count",
  "filters": {{"{date_cols[0] if date_cols else 'created_on'}": ["2025-05-01", "2025-05-31"]}},
  "time_grouping": "week",
  "insight": "Displays weekly ticket trends across all towers during May 2025"
}}

❓ "Monthly high priority incidents in 2025"
✅ {{
  "chart_type": "line",
  "title": "Monthly High Priority Incidents - 2025",
  "x": "{date_cols[0] if date_cols else 'created_on'}",
  "color": null,
  "agg": "count",
  "filters": {{
    "{date_cols[0] if date_cols else 'created_on'}": ["2025-01-01", "2025-12-31"],
    "{priority_cols[0] if priority_cols else 'priority'}": "high"
  }},
  "time_grouping": "month"
}}

❓ "Compare ticket volume across towers last quarter"
✅ {{
  "chart_type": "bar",
  "title": "Ticket Volume by Tower - Q4 2024",
  "x": "{tower_cols[0] if tower_cols else 'tower'}",
  "agg": "count",
  "filters": {{"{date_cols[0] if date_cols else 'created_on'}": ["2024-10-01", "2024-12-31"]}}
}}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
═══════════════════════════════════════════════════════════════════════════════

✅ ALWAYS use actual column names from the schema above
✅ For time-based queries, ALWAYS use LINE chart with time_grouping
✅ For "all towers/groups", use color parameter to show each separately
✅ For date ranges, use filters with [start_date, end_date] format
✅ Match filter values to actual data (check samples in schema)
✅ Use descriptive titles that explain what the chart shows
✅ Provide actionable insights based on the query

Available columns: {cols}
"""


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from various code fence formats or raw JSON."""
    # Try multiple patterns
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```JSON\s*(.*?)\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception as e:
                st.warning(f"JSON parse error: {e}")
                continue
    
    # Try parsing raw JSON without code fences
    try:
        # Look for JSON object in text
        json_match = re.search(r'\{[^{}]*"chart_type"[^{}]*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
    
    return None


def validate_chart_spec(spec: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate chart specification against DataFrame."""
    if not spec:
        return False, "No chart specification provided"
    
    x_col = spec.get("x")
    y_col = spec.get("y")
    color = spec.get("color")
    chart_type = spec.get("chart_type", "").lower()
    
    # Validate chart type
    valid_types = ["bar", "line", "pie", "scatter", "histogram", "heatmap", "box", "funnel", "treemap"]
    if chart_type not in valid_types:
        return False, f"Invalid chart type '{chart_type}'. Valid types: {', '.join(valid_types)}"
    
    # Validate columns exist
    for col_name, col_val in [("x", x_col), ("y", y_col), ("color", color)]:
        if col_val and col_val not in df.columns:
            available = ", ".join(df.columns[:10].tolist())
            return False, f"Column '{col_val}' not found. Available columns: {available}..."
    
    # Chart-specific validations
    if chart_type in ["scatter", "box"] and y_col:
        if not pd.api.types.is_numeric_dtype(df[y_col]):
            return False, f"Chart type '{chart_type}' requires numeric y column, but '{y_col}' is {df[y_col].dtype}"
    
    if chart_type == "heatmap":
        if not x_col or not y_col:
            return False, "Heatmap requires both x and y columns"
    
    if chart_type == "pie" and not x_col:
        return False, "Pie chart requires x column"
    
    return True, "Valid"


def render_plotly_chart(spec: Dict[str, Any], df: pd.DataFrame) -> Optional[go.Figure]:
    """Turn a JSON spec into a Plotly figure with robust error handling."""
    # Validate spec first
    is_valid, msg = validate_chart_spec(spec, df)
    if not is_valid:
        st.error(f"❌ Chart Validation Error: {msg}")
        return None
    
    ct      = spec.get("chart_type", "bar").lower()
    title   = spec.get("title", "Chart")
    x_col   = spec.get("x")
    y_col   = spec.get("y")
    color   = spec.get("color")
    agg     = spec.get("agg", "count")
    filters = spec.get("filters", {})
    time_grouping = spec.get("time_grouping")  # day, week, month, quarter, year

    # Apply filters safely
    dff: pd.DataFrame = df.copy()
    
    # Handle time grouping for date columns
    if time_grouping and x_col:
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(dff[x_col]):
                dff[x_col] = pd.to_datetime(dff[x_col], errors='coerce')
            
            # Create time-grouped column
            if time_grouping == "day":
                dff['_time_group'] = dff[x_col].dt.date
            elif time_grouping == "week":
                dff['_time_group'] = dff[x_col].dt.to_period('W').dt.start_time
            elif time_grouping == "month":
                dff['_time_group'] = dff[x_col].dt.to_period('M').dt.start_time
            elif time_grouping == "quarter":
                dff['_time_group'] = dff[x_col].dt.to_period('Q').dt.start_time
            elif time_grouping == "year":
                dff['_time_group'] = dff[x_col].dt.to_period('Y').dt.start_time
            
            # Use the grouped column as x
            x_col = '_time_group'
        except Exception as e:
            st.warning(f"⚠️ Time grouping failed: {str(e)}. Using original column.")
    
    for col, val in filters.items():
        if col not in dff.columns:
            st.warning(f"⚠️ Filter column '{col}' not found, skipping filter")
            continue

        try:
            # Multi-select filter
            if isinstance(val, list):
                # Date range filter
                if (
                    pd.api.types.is_datetime64_any_dtype(dff[col])
                    and len(val) == 2
                ):
                    start = pd.to_datetime(val[0], errors="coerce")
                    end = pd.to_datetime(val[1], errors="coerce")
                    
                    if pd.isna(start) or pd.isna(end):
                        st.warning(f"⚠️ Invalid date range for '{col}', skipping filter")
                        continue
                    
                    filtered = dff[(dff[col] >= start) & (dff[col] <= end)]
                    dff = filtered if isinstance(filtered, pd.DataFrame) else dff
                else:
                    filtered = dff[dff[col].isin(val)]
                    dff = filtered if isinstance(filtered, pd.DataFrame) else dff
            # Single value filter
            else:
                filtered = dff[dff[col] == val]
                dff = filtered if isinstance(filtered, pd.DataFrame) else dff

        except Exception as e:
            st.warning(f"⚠️ Filter error for '{col}': {str(e)}")
            continue
    
    # Check if filtering resulted in empty DataFrame
    if len(dff) == 0:
        st.error("❌ No data remaining after applying filters. Try adjusting your query.")
        return None
    
    # Aggregation helper
    def agg_df(grp_cols: List[str]) -> pd.DataFrame:
        try:
            # Remove None values from group columns
            grp_cols = [c for c in grp_cols if c is not None]
            if not grp_cols:
                return dff
            
            if agg == "count":
                result = dff.groupby(grp_cols, dropna=False).size()
                return result.reset_index(name="count")  # type: ignore
            elif agg in ("sum", "mean", "median", "nunique"):
                if y_col and y_col in dff.columns:
                    return dff.groupby(grp_cols, dropna=False)[y_col].agg(agg).reset_index()
            return dff
        except Exception as e:
            st.error(f"Aggregation error: {str(e)}")
            return dff

    try:
        fig = None
        
        if ct == "bar":
            if x_col:
                grp = [x_col] + ([color] if color and color != x_col else [])
                adf  = agg_df(grp)
                if len(adf) == 0:
                    st.error("❌ No data to display after aggregation")
                    return None
                yval = "count" if agg == "count" else y_col
                fig  = px.bar(adf, x=x_col, y=yval,
                             color=color if color and color in adf.columns else None,
                             title=title, barmode="group")
                             
        elif ct == "line":
            if x_col:
                grp  = [x_col] + ([color] if color and color != x_col else [])
                adf  = agg_df(grp)
                if len(adf) == 0:
                    st.error("❌ No data to display after aggregation")
                    return None
                yval = "count" if agg == "count" else y_col
                fig  = px.line(adf, x=x_col, y=yval,
                              color=color if color and color in adf.columns else None,
                              title=title)
                              
        elif ct == "pie":
            if x_col:
                adf = dff[x_col].value_counts().reset_index()
                adf.columns = [x_col, "count"]
                if len(adf) == 0:
                    st.error("❌ No data to display")
                    return None
                fig = px.pie(adf, names=x_col, values="count", title=title)
                
        elif ct == "scatter":
            if x_col and y_col:
                fig = px.scatter(dff, x=x_col, y=y_col,
                               color=color if color and color in dff.columns else None,
                               title=title, opacity=0.7)
                               
        elif ct == "histogram":
            if x_col:
                fig = px.histogram(dff, x=x_col,
                                 color=color if color and color in dff.columns else None,
                                 title=title)
                                 
        elif ct == "box":
            if x_col and y_col:
                fig = px.box(dff, x=x_col, y=y_col,
                           color=color if color and color in dff.columns else None,
                           title=title)
                           
        elif ct == "heatmap":
            if x_col and y_col:
                pivot = pd.crosstab(dff[x_col], dff[y_col])
                if pivot.empty:
                    st.error("❌ No data for heatmap")
                    return None
                fig = px.imshow(pivot, title=title, aspect="auto",
                              color_continuous_scale="Blues")
                              
        elif ct == "funnel":
            if x_col:
                adf  = agg_df([x_col])
                if len(adf) == 0:
                    st.error("❌ No data to display")
                    return None
                yval = "count" if agg == "count" else y_col
                fig  = px.funnel(adf, x=yval, y=x_col, title=title)
                
        elif ct == "treemap":
            if x_col:
                path = [x_col] + ([color] if color and color != x_col else [])
                adf  = agg_df(path)
                if len(adf) == 0:
                    st.error("❌ No data to display")
                    return None
                yval = "count" if agg == "count" else y_col
                fig  = px.treemap(adf, path=path, values=yval, title=title)

        if fig:
            fig.update_layout(
                template="plotly_white",
                title_font_size=16,
                hovermode='closest'
            )
            return fig
        else:
            st.error(f"❌ Could not generate {ct} chart. Check if required columns are specified.")
            
    except Exception as e:
        st.error(f"❌ Chart render error: {str(e)}")
        with st.expander("🔍 Debug Info"):
            st.write(f"**Chart Type:** {ct}")
            st.write(f"**X Column:** {x_col}")
            st.write(f"**Y Column:** {y_col}")
            st.write(f"**Color Column:** {color}")
            st.write(f"**Aggregation:** {agg}")
            st.write(f"**Data Shape:** {dff.shape}")
            st.write(f"**Error:** {traceback.format_exc()}")
    
    return None


# ══════════════════════════════════════════════════════════════════════════════
# AutoGen Agentic Layer
# ══════════════════════════════════════════════════════════════════════════════

def run_autogen_analysis(query: str, df: pd.DataFrame, groq_api_key: str) -> str:
    """
    Use AutoGen multi-agent system:
      - DataAnalystAgent  → interprets the query and plans analysis
      - PythonCodeAgent   → writes Python/Pandas snippets
      - InsightAgent      → summarizes findings
    All backed by Groq Llama 3.3 70B via OpenAI-compatible endpoint.
    """
    schema   = get_df_schema(df)
    df_stats = df.describe(include="all").to_string()

    llm_config = {
        "config_list": [{
            "model": "llama-3.3-70b-versatile",
            "api_key": groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
        }],
        "temperature": 0.3,
        "max_tokens": 1024,
        "cache_seed": None,
    }

    context = f"""
ServiceNow DataFrame Schema:
{schema}

Summary Statistics:
{df_stats}

User Query: {query}
"""

    analyst = AssistantAgent(
        name="DataAnalyst",
        system_message="""You are a ServiceNow data analyst.
Given a user query and DataFrame schema, decompose the analysis into clear steps.
Be concise. Do NOT write code. Focus on WHAT to analyse.""",
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
    )

    coder = AssistantAgent(
        name="PythonCoder",
        system_message="""You are a Python/Pandas expert for ServiceNow data.
Given an analysis plan and DataFrame schema, write the pandas logic (as pseudocode or snippet)
to answer the query. Keep it short and precise.""",
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
    )

    insight_agent = AssistantAgent(
        name="InsightAgent",
        system_message="""You synthesize ServiceNow data analysis into a crisp, actionable insight.
Summarize findings in 2-3 sentences. End with one recommendation.""",
        llm_config=llm_config,
        max_consecutive_auto_reply=1,
    )

    user_proxy = UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        default_auto_reply="TERMINATE",
    )

    groupchat = GroupChat(
        agents=[user_proxy, analyst, coder, insight_agent],
        messages=[],
        max_round=6,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_proxy.initiate_chat(manager, message=context, silent=True)

    # Collect all assistant messages
    messages = groupchat.messages
    result_parts = []
    for m in messages:
        if m.get("role") == "assistant" or m.get("name") in ("DataAnalyst", "PythonCoder", "InsightAgent"):
            name    = m.get("name", "Agent")
            content = m.get("content", "")
            if content and content.strip() not in ("", "TERMINATE"):
                result_parts.append(f"**{name}:** {content}")

    return "\n\n".join(result_parts) if result_parts else "No agent response generated."


def simple_groq_query(query: str, df: pd.DataFrame, client: Groq, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Direct Groq call for chart spec + text answers with conversation history."""
    sys_prompt = build_system_prompt(df)
    sample     = df.head(50).to_csv(index=False)
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": sys_prompt}]
    
    # Add recent chat history for context (last 3 exchanges)
    if chat_history:
        recent_history = chat_history[-6:]  # Last 3 user + 3 assistant messages
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                # Only include text content, not figures
                content = msg.get("content", "")
                if content:
                    messages.append({"role": msg["role"], "content": content[:500]})  # Limit length
    
    # Add current query with data sample
    user_msg = f"Data sample (first 50 rows):\n{sample}\n\nUser question: {query}"
    messages.append({"role": "user", "content": user_msg})
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,  # type: ignore
            temperature=0.2,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content
        return content if content else ""
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "⚠️ **Rate Limit Reached**\n\nYou've hit the Groq API rate limit. Please:\n1. Wait 10-20 minutes and try again\n2. Or upgrade your Groq plan at https://console.groq.com/settings/billing\n3. Or use a different API key"
        else:
            return f"⚠️ **API Error:** {error_msg}"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/57/ServiceNow_logo.svg", width=180)
    st.markdown("### ⚙️ Configuration")

    groq_api_key = st.text_input("🔑 Groq API Key", type="password",
                                  placeholder="gsk_xxxxxxxxxxxx",
                                  help="Get your key at console.groq.com")
    if groq_api_key:
        st.session_state.groq_client = init_groq(groq_api_key)
        st.success("✅ Groq connected")

    st.divider()
    st.markdown("### 📁 Upload Data")
    uploaded = st.file_uploader("Upload ServiceNow Excel", type=["xlsx", "xls"],
                                  help="Export your ServiceNow table as Excel")
    if uploaded:
        try:
            xl   = pd.ExcelFile(uploaded)
            tabs = xl.sheet_names
            sheet = st.selectbox("Select Sheet", tabs)
            df_raw = pd.read_excel(uploaded, sheet_name=sheet)

            # Clean column names
            df_raw.columns = [str(c).strip().replace(" ", "_").lower() for c in df_raw.columns]
            
            # Preprocess data (auto-convert dates, etc.)
            df_processed = preprocess_dataframe(df_raw)
            
            st.session_state.df        = df_processed
            st.session_state.file_name = uploaded.name
            
            # Show data info
            date_cols_found = df_processed.select_dtypes(include='datetime').columns.tolist()
            st.success(f"✅ Loaded **{df_processed.shape[0]:,}** rows × **{df_processed.shape[1]}** cols")
            if date_cols_found:
                st.info(f"📅 Auto-detected date columns: {', '.join(date_cols_found[:5])}")
        except Exception as e:
            st.error(f"Upload error: {e}")

    st.divider()
    mode = st.radio("🤖 AI Mode",
                    ["Direct (Groq)", "Agentic (AutoGen + Groq)"],
                    help="Direct = fast single LLM call. Agentic = multi-agent reasoning chain.")

    st.divider()
    st.session_state.debug_mode = st.checkbox("🐛 Debug Mode",
                                               value=st.session_state.debug_mode,
                                               help="Show detailed error messages and chart specifications")
    
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Model:** Llama 3.3 70B Versatile  \n**Provider:** Groq  \n**Framework:** AutoGen")


# ══════════════════════════════════════════════════════════════════════════════
# Main Area
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🎫 ServiceNow AI Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Natural language insights & visualizations powered by Llama 3.3 70B + AutoGen</div>',
            unsafe_allow_html=True)

df = st.session_state.df

# ── Data overview ─────────────────────────────────────────────────────────────
if df is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Chat & Visualize", "🔍 Raw Data"])

    # ── Tab 1: Auto-dashboard ─────────────────────────────────────────────────
    with tab1:
        st.markdown(f"### 📂 `{st.session_state.file_name}` — Quick Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{df.shape[0]:,}")
        c2.metric("Columns",       df.shape[1])
        c3.metric("Numeric Cols",  df.select_dtypes(include="number").shape[1])
        c4.metric("Text Cols",     df.select_dtypes(include="object").shape[1])

        st.divider()
        st.markdown("#### 🔎 Auto-Generated Charts")

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()

        auto_cols = cat_cols[:4]  # show top-4 categorical breakdowns
        cols_grid  = st.columns(2)
        for i, col in enumerate(auto_cols):
            vc  = df[col].value_counts().head(10).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count",
                         title=f"Distribution of {col.replace('_',' ').title()}",
                         color="count", color_continuous_scale="Blues")
            fig.update_layout(template="plotly_white", showlegend=False)
            with cols_grid[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

        # Numeric correlation heatmap if ≥2 numeric cols
        if len(num_cols) >= 2:
            st.markdown("#### 🌡️ Numeric Correlation")
            numeric_df = df[num_cols]
            corr = numeric_df.corr()  # pyright: ignore[reportCallIssue]
            fig  = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu_r", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Chat ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 💬 Ask Anything About Your Data")
        st.caption("Examples: *Show a pie chart of incidents by priority* · *What is the average resolution time?* · *Bar chart of tickets by category last month*")

        # Display history
        for msg in st.session_state.chat_history:
            role  = msg["role"]
            color = "#DCF8C6" if role == "user" else "#F1F0F0"
            icon  = "🧑" if role == "user" else "🤖"
            st.markdown(f"""
            <div style="background:{color};border-radius:10px;padding:0.8rem;margin:0.4rem 0;">
                <b>{icon} {'You' if role=='user' else 'AI Agent'}:</b><br>{msg['content']}
            </div>""", unsafe_allow_html=True)
            if "figure" in msg:
                st.plotly_chart(msg["figure"], use_container_width=True)
            if "insight" in msg:
                st.info(f"💡 **Insight:** {msg['insight']}")
            if "debug_spec" in msg and st.session_state.debug_mode:
                with st.expander("🐛 Chart Specification"):
                    st.json(msg["debug_spec"])

        # Input
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("Your question", height=80,
                                       placeholder="e.g. Show a bar chart of incidents by state and priority")
            submitted  = st.form_submit_button("🚀 Send")

        if submitted and user_input.strip():
            if not st.session_state.groq_client:
                st.error("❌ Please enter your Groq API key in the sidebar.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                with st.spinner("🤖 Agents thinking..."):
                    try:
                        if "agentic" in mode.lower():
                            # AutoGen multi-agent call
                            agent_response = run_autogen_analysis(
                                user_input, df, groq_api_key)
                            # Also get chart spec via direct Groq call with history
                            direct_response = simple_groq_query(
                                user_input, df, st.session_state.groq_client,
                                st.session_state.chat_history)
                        else:
                            agent_response  = None
                            direct_response = simple_groq_query(
                                user_input, df, st.session_state.groq_client,
                                st.session_state.chat_history)

                        # Parse chart JSON if present
                        spec    = extract_json_block(direct_response)
                        fig_obj = None
                        insight = None

                        # Show debug info if enabled
                        if st.session_state.debug_mode:
                            with st.expander("🔍 Debug: AI Response"):
                                st.text_area("Raw AI Response", direct_response, height=200)
                                if spec:
                                    st.json(spec)
                                else:
                                    st.warning("No JSON chart spec found in response")

                        if spec:
                            fig_obj = render_plotly_chart(spec, df)
                            insight = spec.get("insight")
                        elif "chart" in user_input.lower() or "graph" in user_input.lower() or "plot" in user_input.lower():
                            st.warning("⚠️ Chart requested but no valid chart specification received from AI. Try rephrasing your question.")

                        # Clean text (remove JSON block)
                        clean_text = re.sub(r"```json.*?```", "", direct_response, flags=re.DOTALL).strip()
                        clean_text = re.sub(r"```JSON.*?```", "", clean_text, flags=re.DOTALL | re.IGNORECASE).strip()

                        # Compose AI message
                        ai_content = ""
                        if agent_response:
                            ai_content += f"**🤖 AutoGen Agent Chain:**\n\n{agent_response}\n\n---\n\n"
                        if clean_text:
                            ai_content += clean_text

                        entry: Dict[str, Any] = {"role": "assistant", "content": ai_content}
                        if fig_obj:
                            entry["figure"]  = fig_obj
                        if insight:
                            entry["insight"] = insight
                        if st.session_state.debug_mode and spec:
                            entry["debug_spec"] = spec

                        st.session_state.chat_history.append(entry)
                        st.rerun()

                    except Exception:
                        err = traceback.format_exc()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"⚠️ Error: ```\n{err}\n```"
                        })
                        st.rerun()

    # ── Tab 3: Raw data ────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔍 Raw Data Explorer")
        c1, c2 = st.columns([3, 1])
        with c1:
            search = st.text_input("🔎 Filter rows (searches all columns)", "")
        with c2:
            n_rows = st.selectbox("Rows to show", [50, 100, 500, 1000], index=0)

        display_df = df.copy()
        if search:
            mask       = display_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        st.dataframe(display_df.head(n_rows), use_container_width=True, height=500)
        st.caption(f"Showing {min(n_rows, len(display_df)):,} of {len(display_df):,} filtered rows")

        # Download
        csv = display_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered CSV", csv, "servicenow_export.csv", "text/csv")

else:
    # Landing state
    st.info("👈 Upload a ServiceNow Excel file and enter your Groq API key in the sidebar to get started.")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
**1. Upload Data**
Export any ServiceNow table (Incidents, Problems, Changes, Assets…) as Excel and upload it.

**2. Auto Dashboard**
Instant charts and a correlation heatmap are generated automatically.

**3. Chat with your Data**
Type natural language questions like:
- *"Show a pie chart of P1 vs P2 incidents"*
- *"What is the average resolution time by category?"*
- *"Bar chart of open tickets by assignee"*
- *"How many incidents were created this month?"*

**4. AI Modes**
- **Direct (Groq):** Fast single-call inference → Llama 3.3 70B on Groq
- **Agentic (AutoGen):** 3-agent chain: DataAnalyst → PythonCoder → InsightAgent, all backed by Groq

**5. Charts are auto-rendered**
The AI returns a structured JSON spec and the app renders an interactive Plotly chart.
        """)
    with st.expander("📋 Supported Chart Types"):
        types = ["bar", "line", "pie", "scatter", "histogram", "heatmap", "box", "funnel", "treemap"]
        st.write(", ".join(f"`{t}`" for t in types))
