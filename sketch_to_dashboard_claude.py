from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import random
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anthropic
import os
import base64


# ==================== PYDANTIC MODELS ====================

class ChartOptions(BaseModel):
    """Configuration options for charts"""
    show_legend: bool = True
    color_scheme: Optional[str] = None
    orientation: Optional[Literal["horizontal", "vertical"]] = "vertical"
    stacked: bool = False


class ChartDefinition(BaseModel):
    """Defines a single chart in the dashboard"""
    type: Literal["pie", "bar", "line", "metric", "timeseries"]
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right", "full-width"]
    title: str
    data_field: str
    aggregation: Optional[Literal["sum", "count", "avg", "min", "max"]] = "count"
    group_by: Optional[str] = None
    time_field: Optional[str] = "@timestamp"
    options: ChartOptions = Field(default_factory=ChartOptions)


class DashboardLayout(BaseModel):
    """Complete dashboard specification"""
    layout: Literal["1x1", "2x2", "3x3", "2x1", "1x2"] = "2x2"
    title: str = "Analytics Dashboard"
    charts: List[ChartDefinition]


# ==================== MOCK DATA GENERATOR ====================

def generate_mock_data(num_records: int = 1000) -> pd.DataFrame:
    """Generate mock data matching"""

    countries = ["US", "UK", "DE", "FR", "JP", "BR", "IN", "AU"]
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)"
    ]
    extensions = ["jpg", "css", "js", "html", "png", "gif", "zip"]
    os_types = ["windows", "linux", "osx", "ios", "android"]
    responses = ["200", "404", "500", "301", "304"]

    base_time = datetime.now() - timedelta(days=30)

    data = []
    for i in range(num_records):
        timestamp = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        country = random.choice(countries)

        record = {
            "@timestamp": timestamp,
            "agent": random.choice(agents),
            "bytes": random.randint(100, 500000),
            "bytes_counter": random.randint(1000, 1000000),
            "bytes_gauge": random.randint(50, 100),
            "clientip": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "event.dataset": "web_logs",
            "extension": random.choice(extensions),
            "geo.dest": country,
            "geo.src": random.choice(countries),
            "geo.coordinates": {
                "lat": random.uniform(-90, 90),
                "lon": random.uniform(-180, 180)
            },
            "host": f"server-{random.randint(1, 10)}.example.com",
            "index": f"logs-{timestamp.strftime('%Y.%m.%d')}",
            "ip": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "machine.os": random.choice(os_types),
            "machine.ram": random.choice([4, 8, 16, 32, 64]) * 1024,
            "memory": random.uniform(0.1, 0.9),
            "message": f"Request {i} processed",
            "phpmemory": random.randint(100000, 500000),
            "referer": f"https://example.com/page{random.randint(1, 100)}",
            "request": f"/api/v1/endpoint{random.randint(1, 20)}",
            "response": random.choice(responses),
            "tags": random.choice(["production", "staging", "development"]),
            "url": f"https://example.com/page{random.randint(1, 100)}",
            "utc_time": timestamp
        }
        data.append(record)

    return pd.DataFrame(data)


# ==================== VISION MODEL SIMULATOR ====================

def analyze_sketch_with_claude(image_path: str, data_schema: Dict) -> DashboardLayout:
    """
    Calls Claude's vision API to analyze the dashboard sketch.

    Args:
        image_path: Path to the sketch image file
        data_schema: Dictionary containing available data fields

    Returns:
        DashboardLayout: Parsed and validated dashboard specification

    Requires:
        - ANTHROPIC_API_KEY environment variable to be set
        - anthropic package: pip install anthropic
    """

    # Initialize Claude client
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # Read and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    # Determine image media type
    image_extension = image_path.lower().split('.')[-1]
    media_type_map = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    media_type = media_type_map.get(image_extension, 'image/jpeg')

    # Create detailed prompt
    prompt = f"""Analyze this hand-drawn dashboard sketch and extract the visualization specifications.

Available data fields:
{json.dumps(data_schema, indent=2)}

Identify each chart/visualization in the sketch and return ONLY a JSON object (no markdown, no extra text) with this exact structure:
{{
  "layout": "2x2",
  "title": "Dashboard Title from sketch",
  "charts": [
    {{
      "type": "pie|bar|line|metric|timeseries",
      "position": "top-left|top-right|bottom-left|bottom-right",
      "title": "Chart title from sketch",
      "data_field": "field name from schema above",
      "aggregation": "count|sum|avg|min|max",
      "group_by": "field to group by (optional)",
      "time_field": "@timestamp",
      "options": {{
        "show_legend": true,
        "orientation": "vertical"
      }}
    }}
  ]
}}

Important:
- Use exact field names from the schema
- Match chart types to what you see (pie, bar, timeseries, metric) and choose one only
- Determine position from layout (top-left, top-right, bottom-left, bottom-right)
- Only use a specific available field for the data_field and do not use any field that is not available
- data_field should not be @timestamp 
- Return ONLY valid JSON, no additional text or markdown code blocks"""

 # uncomment to see the full prompt
 #   print(prompt)
 #   Call Claude's vision API
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5 - best for vision tasks
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    # Extract response text
    response_text = message.content[0].text

# Sample response
#     response_text = """{
#   "layout": "2x2",
#   "title": "Log Analytics Dashboard",
#   "charts": [
#     {
#       "type": "pie",
#       "position": "top-left",
#       "title": "Visits by country",
#       "data_field": "geo.src",
#       "aggregation": "count",
#       "time_field": "@timestamp",
#       "options": {
#         "show_legend": true
#       }
#     },
#     {
#       "type": "timeseries",
#       "position": "top-right",
#       "title": "Number of logs over time",
#       "data_field": "bytes_counter",
#       "aggregation": "count",
#       "time_field": "@timestamp",
#       "options": {
#         "show_legend": false
#       }
#     },
#     {
#       "type": "metric",
#       "position": "bottom-left",
#       "title": "Total Logs",
#       "data_field": "bytes_counter",
#       "aggregation": "count",
#       "time_field": "@timestamp",
#       "options": {}
#     },
#     {
#       "type": "bar",
#       "position": "bottom-right",
#       "title": "Logs per hour",
#       "data_field": "bytes_counter",
#       "aggregation": "count",
#       "group_by": "hour",
#       "time_field": "@timestamp",
#       "options": {
#         "show_legend": false,
#         "orientation": "vertical"
#       }
#     }
#   ]
# }"""

    response_text = response_text.strip()

    # Parse and validate with Pydantic
    try:
        dashboard_layout = DashboardLayout.model_validate_json(response_text)
        return dashboard_layout
    except Exception as e:
        print(f"Error parsing Claude response: {e}")
        print(f"Response was: {response_text}")
        raise





# ==================== CHART RENDERERS ====================

class ChartRenderer:
    """Renders different chart types using Plotly"""

    @staticmethod
    def render_pie(df: pd.DataFrame, chart: ChartDefinition) -> go.Figure:
        """Render pie chart"""
        grouped = df[chart.data_field].value_counts()

        fig = go.Figure(data=[go.Pie(
            labels=grouped.index,
            values=grouped.values,
            hole=0.3
        )])

        fig.update_layout(
            title=chart.title,
            showlegend=chart.options.show_legend,
            height=400
        )

        return fig

    @staticmethod
    def render_bar(df: pd.DataFrame, chart: ChartDefinition) -> go.Figure:
        """Render bar chart"""
        if chart.group_by == "hour":
            df['hour'] = pd.to_datetime(df['@timestamp']).dt.hour
            grouped = df['hour'].value_counts().sort_index()
            x_data = [f"{h}:00" for h in grouped.index]
            y_data = grouped.values
        else:
            grouped = df[chart.data_field].value_counts()
            x_data = grouped.index
            y_data = grouped.values

        orientation = 'v' if chart.options.orientation == 'vertical' else 'h'

        if orientation == 'h':
            fig = go.Figure(data=[go.Bar(y=x_data, x=y_data, orientation='h')])
        else:
            fig = go.Figure(data=[go.Bar(x=x_data, y=y_data)])

        fig.update_layout(
            title=chart.title,
            xaxis_title=chart.group_by or chart.data_field,
            yaxis_title="Count",
            height=400
        )

        return fig

    @staticmethod
    def render_timeseries(df: pd.DataFrame, chart: ChartDefinition) -> go.Figure:
        """Render time series line chart"""
        df_sorted = df.copy()
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['@timestamp'])
        df_sorted = df_sorted.sort_values('timestamp')

        # Group by day
        daily_counts = df_sorted.groupby(df_sorted['timestamp'].dt.date).size()

        fig = go.Figure(data=[go.Scatter(
            x=daily_counts.index,
            y=daily_counts.values,
            mode='lines+markers',
            line=dict(color='#636EFA', width=2),
            marker=dict(size=6)
        )])

        fig.update_layout(
            title=chart.title,
            xaxis_title="Date",
            yaxis_title="Count",
            height=400
        )

        return fig

    @staticmethod
    def render_metric(df: pd.DataFrame, chart: ChartDefinition) -> go.Figure:
        """Render metric card"""
        if chart.aggregation == "count":
            value = len(df)
        elif chart.aggregation == "sum":
            value = df[chart.data_field].sum()
        elif chart.aggregation == "avg":
            value = df[chart.data_field].mean()
        elif chart.aggregation == "min":
            value = df[chart.data_field].min()
        elif chart.aggregation == "max":
            value = df[chart.data_field].max()
        else:
            value = len(df)

        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="number",
            value=value,
            title={'text': chart.title, 'font': {'size': 24}},
            number={'font': {'size': 60}},
            domain={'x': [0, 1], 'y': [0, 1]}
        ))

        fig.update_layout(height=400)

        return fig


# ==================== DASHBOARD GENERATOR ====================

class DashboardGenerator:
    """Generates complete dashboards from specifications"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.renderer = ChartRenderer()

    def generate(self, layout: DashboardLayout) -> go.Figure:
        """Generate complete dashboard"""

        # Determine subplot layout
        if layout.layout == "2x2":
            rows, cols = 2, 2
        elif layout.layout == "3x3":
            rows, cols = 3, 3
        elif layout.layout == "2x1":
            rows, cols = 2, 1
        elif layout.layout == "1x2":
            rows, cols = 1, 2
        else:
            rows, cols = 1, 1

        # Map positions to grid coordinates
        position_map = {
            "top-left": (1, 1),
            "top-right": (1, 2),
            "bottom-left": (2, 1),
            "bottom-right": (2, 2)
        }

        # Create specs based on chart types
        specs = []
        chart_index = 0
        for r in range(rows):
            row_specs = []
            for c in range(cols):
                if chart_index < len(layout.charts):
                    chart_type = layout.charts[chart_index].type
                    if chart_type == "pie":
                        row_specs.append({"type": "domain"})
                    elif chart_type == "metric":
                        row_specs.append({"type": "indicator"})
                    else:
                        row_specs.append({"type": "xy"})
                    chart_index += 1
                else:
                    row_specs.append({"type": "domain"})
            specs.append(row_specs)

        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.title for chart in layout.charts[:rows * cols]],
            specs=specs
        )
        # Render each chart
        for chart in layout.charts[:rows * cols]:
            if chart.type == "pie":
                chart_fig = self.renderer.render_pie(self.data, chart)
            elif chart.type == "bar":
                chart_fig = self.renderer.render_bar(self.data, chart)
            elif chart.type == "timeseries":
                chart_fig = self.renderer.render_timeseries(self.data, chart)
            elif chart.type == "metric":
                chart_fig = self.renderer.render_metric(self.data, chart)
            else:
                continue

            # Get grid position
            row, col = position_map.get(chart.position, (1, 1))

            # Add traces to subplot
            for trace in chart_fig.data:
                fig.add_trace(trace, row=row, col=col)

        # Update layout
        fig.update_layout(
            title_text=layout.title,
            title_font_size=24,
            showlegend=True,
            height=800,
            width=1200
        )

        return fig


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution flow"""

    print("=" * 60)
    print("SKETCH TO DASHBOARD PROTOTYPE")
    print("=" * 60)

    # Step 1: Generate mock data
    print("\n[1/4] Generating mock  data...")
    df = generate_mock_data(num_records=2000)
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Date range: {df['@timestamp'].min()} to {df['@timestamp'].max()}")
    print(f"✓ Columns: {', '.join(df.columns[:5])}... ({len(df.columns)} total)")

    # Step 2: Analyze sketch with vision model
    print("\n[2/4] Analyzing sketch with LLaVA...")
    sketch_path = "dashboard_drawing.jpeg"
    data_schema = {
        "fields": list(df.columns),
        "sample_values": {col: str(df[col].iloc[0]) for col in df.columns[:5]}
    }

    dashboard_spec = analyze_sketch_with_claude(sketch_path, data_schema)

    # Step 3: Print Pydantic model output
    print("\n[3/4] Dashboard Specification (Pydantic Model):")
    print(json.dumps(dashboard_spec.model_dump(), indent=2, default=str))

    # Step 4: Generate dashboard
    print("\n[4/4] Generating dashboard...")
    generator = DashboardGenerator(df)
    dashboard = generator.generate(dashboard_spec)

    # Save to HTML
    output_file = "dashboard.html"
    dashboard.write_html(output_file)
    print(f"✓ Dashboard saved to {output_file}")

    print("\n" + "=" * 60)
    print("DASHBOARD GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nOpen {output_file} in your browser to view the dashboard.")

    return dashboard


if __name__ == "__main__":
    dashboard = main()