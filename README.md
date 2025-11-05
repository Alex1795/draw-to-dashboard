# Draw to Dashboard  
  
An AI-powered tool that converts hand-drawn dashboard sketches into functional, interactive web dashboards. Simply draw your dashboard layout on paper, take a photo, and let Claude's vision API transform it into a live Plotly dashboard.  
  
## Features  
  
- **Sketch-to-Dashboard Conversion**: Upload a hand-drawn dashboard sketch and get a fully functional HTML dashboard  
- **AI-Powered Analysis**: Uses Claude Sonnet 4.5 vision model to interpret your sketches  
- **Multiple Chart Types**: Supports pie charts, bar charts, time series, and metric displays  
- **Flexible Layouts**: Create 1x1, 2x2, 3x3, 2x1, or 1x2 grid layouts  
- **Interactive Visualizations**: Generates Plotly charts with hover interactions and legends  
- **Self-Contained Output**: Produces a single HTML file with all data and visualizations embedded  
  
## Prerequisites  
  
- Python 3.7+  
- Anthropic API key (for Claude vision API access)  
  
## Installation  
  
1. Clone the repository:  
```bash  
git clone https://github.com/Alex1795/draw-to-dashboard.git  
cd draw-to-dashboard
```
Install dependencies:
```
pip install -r requirements.txt
```
Set up your Anthropic API key:
```
export ANTHROPIC_API_KEY='your-api-key-here'
```
## Usage
1. Draw your dashboard layout on paper, including:

- Chart titles
- Chart types (pie, bar, line, metric)
- Overall dashboard title

2. Take a photo and save it as dashboard_drawing.jpeg in the project directory

3. Run the script:
```
python sketch_to_dashboard_claude.py
```
4. Open the generated dashboard.html file in your browser

## How It Works
The system follows a four-stage pipeline:

- Data Generation: Creates mock web log data with fields like timestamps, bytes, countries, response codes, etc.
- Vision Analysis: Claude's vision API analyzes your sketch and extracts chart specifications
- Validation: Pydantic models ensure the AI-generated specification is valid
- Dashboard Generation: Plotly renders the charts into an interactive HTML dashboard sketch_to_dashboard_claude.py:108-131


### Supported Chart Types
- Pie Charts: For categorical data distribution
- Bar Charts: For comparing values across categories
- Time Series: For trends over time
- Metrics: For single KPI values (count, sum, average, etc.) sketch_to_dashboard_claude.py:24-33

### Data Schema

The system generates mock web log data.


## Architecture
The codebase consists of:

- Pydantic Models: Type-safe data validation (DashboardLayout, ChartDefinition, ChartOptions)
- ChartRenderer: Renders individual chart types using Plotly
- DashboardGenerator: Orchestrates the complete dashboard assembly
- Vision Integration: Claude API client for sketch analysis sketch_to_dashboard_claude.py:36-41

## Example Output
The system produces a self-contained HTML file (dashboard.html) with:

- Interactive Plotly visualizations
- Responsive layout
- Hover tooltips
- Legend controls
- Embedded data (no external dependencies)

## Dependencies
- anthropic: Claude API client
- pydantic: Data validation
- pandas: Data manipulation
- plotly: Interactive visualizations
