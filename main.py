import uvicorn
import os
import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from typing import List, Dict, Any, Union, Optional
from dotenv import load_dotenv
from charts_config import charts_config

# --- Load .env ---
load_dotenv()

# --- Configuration & API Client Setup ---
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
MODEL = os.getenv("OPENROUTER_MODEL")
DATALAKE_BASE_URL = os.getenv("DATALAKE_BASE_URL", "http://localhost:8080/api/v1")


if not API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in .env!")

CLIENT = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Chart Generation Assistant API",
    description="An API that suggests charts and builds queries based on user prompts and metadata.",
    version="2.0.0"
)
# --- Data-Lakehouse Integration ---
async def execute_query_on_datalake(query_json: Dict) -> Dict:
    """Send generated query to data-lakehouse for execution and wait for completion"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Submit query
            response = await client.post(
                f"{DATALAKE_BASE_URL}/query",
                json=query_json
            )
            response.raise_for_status()
            result = response.json()
            
            job_id = result.get("jobId")
            if not job_id:
                raise HTTPException(status_code=500, detail="No jobId returned from data-lakehouse")
            
            # Poll for query completion
            for attempt in range(60):
                status_response = await client.get(
                    f"{DATALAKE_BASE_URL}/query/{job_id}"
                )
                status_response.raise_for_status()
                status_data = status_response.json()
                
                if status_data.get("status") == "completed":
                    print(f"Query {job_id} completed with {status_data.get('rowCount', 0)} rows")
                    return status_data
                elif status_data.get("status") == "failed":
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Query failed: {status_data.get('message', 'Unknown error')}"
                    )
                
                await asyncio.sleep(1)
            
            raise HTTPException(status_code=500, detail="Query execution timeout")
            
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Data-lakehouse error: {str(e)}")


# --- Logic Classes (Adapted from prompt2.py) ---

class ChartSuggester:
    def __init__(self, charts_config: List[Dict], model: str = MODEL):
        self.model = model
        self.minimal_config = [
            {
                "id": chart.get("chart_id"),
                "name": chart.get("name"),
                "why": chart.get("why"),
                "use_cases": chart.get("use_cases")
            }
            for chart in charts_config
        ]
        self.system_prompt = """
        You are a data visualization assistant.
        You are given a list of chart configurations (id, name, why, use_cases).
        Your task:
        1. Read the user's request carefully.
        2. Compare it with the provided chart configurations.
        3. Choose ALL charts relevant to the user's request. Do not pick just the most obvious.
        4. If no chart is relevant, return {"chosen_charts": []}.
        5. Return ONLY JSON, in this exact format:

        {
          "chosen_charts": [
            {"id": "<chart_id>", "name": "<chart_name>"}
          ]
        }

        Do not include any extra text, explanation, or markdown.
        Do not make assumptions about the dataset yet.
        """

    async def suggest(self, user_prompts: List[str]) -> List[Dict]:
        results = []

        for prompt in user_prompts:
            try:
                response = await CLIENT.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {
                            "role": "user",
                            "content": f"User request: {prompt}\nCharts config: {json.dumps(self.minimal_config, separators=(',', ':'))}"
                        }
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content
                print(f"Raw response content: {content}")
                # Handle potential JSON parsing errors or wrapping
                try:
                    # Extract JSON from wrapper tags if present
                    if "[OUT]" in content and "[/OUT]" in content:
                        content = content.split("[OUT]")[1].split("[/OUT]")[0].strip()
                    chosen_charts = json.loads(content)["chosen_charts"]
                except (KeyError, json.JSONDecodeError) as e:
                    # Fallback or empty if parsing failed
                    chosen_charts = []
                    print(f'Failed to parse JSON: {e}')

                results.append({
                    "user_prompt": prompt,
                    "chosen_charts": chosen_charts
                })
            except Exception as e:
                # Log error and return empty for this prompt
                print(f"Error processing prompt '{prompt}': {e}")
                results.append({
                    "user_prompt": prompt,
                    "chosen_charts": []
                })

        return results


class ChartValidatorAndQueryBuilder:
    def __init__(self, charts_config: List[Dict], model: str = MODEL):
        self.model = model
        self.minimal_config = [
            {
                "id": chart.get("chart_id"),
                "name": chart.get("name"),
                "data_requirements": chart.get("data_requirements", {}),
            }
            for chart in charts_config
        ]
        self.system_prompt = """
        You are a data visualization assistant that outputs only JSON.

        Inputs you will receive:

        user_prompt: the user’s request text

        dataset_metadata: list of columns with name + data_type (+ optional description)

        recommended_charts: list of chart specs. Each spec contains:

        chart_id

        chart_type

        requirements: required roles (e.g., numeric_measure, categorical_dimension, datetime) and any constraints

        encoding_template: which encodings are expected (x, y, color)

        Your job:
        For each chart in recommended_charts:

        Check if dataset_metadata satisfies every requirement.

        If satisfied, choose exact column names from dataset_metadata for each role.

        If not satisfied, skip the chart (do not guess or invent columns).

        Output format:
        Return ONLY this JSON object (no markdown, no commentary):
        {
        "intent": "visualization",
        "charts": []
        }

        If at least one chart is applicable, each item in "charts" MUST be:
        {
        "user_prompt": "<copy user_prompt exactly>",
        "chart_id": "<chart_id>",
        "chart_type": "<chart_type>",
        "query": {
        "source": "uploaded_file",
        "select": [
        {"column": "<dataset_column>", "as": "<alias>"},
        {"column": "<dataset_column>", "aggregation": "<sum|avg|min|max|count|count_distinct>", "as": "<alias>"}
        ],
        "filters": [
        {"column": "<dataset_column>", "operator": "<=|>=|=|!=|in|between|contains>", "value": "<value_or_list>"}
        ],
        "groupBy": ["<alias_or_column>"],
        "orderBy": [
        {"column": "<alias_or_column>", "direction": "asc"}
        ],
        "limit": null
        },
        "encoding": {"x": "<alias_or_column>", "y": "<alias_or_column>", "color": "<alias_or_column_or_empty_string>"}
        }

        Hard rules:

        Output must start with { and end with }.

        Use only columns that exist in dataset_metadata.

        Always include query.select, query.filters, query.groupBy, query.orderBy, query.limit even if empty.

        select MUST be a list of objects, never strings.

        orderBy MUST be a list of objects, never a single object.

        If no charts apply, return {"intent":"visualization","charts":[]} exactly.


        
        """

    async def build_final_charts(self, dataset_metadata: Dict, recommended_charts_with_prompts: List[Dict]) -> Dict:
        try:
            response = await CLIENT.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Dataset metadata: {json.dumps(dataset_metadata)}\nRecommended charts with prompts: {json.dumps(recommended_charts_with_prompts)}\nChart configurations: {json.dumps(self.minimal_config)}"
                    }
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            print(f"Raw response content: {content}")
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            return json.loads(content)
        except json.JSONDecodeError:
            return {"intent": "visualization", "charts": []}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in Query Builder: {str(e)}")

# ...existing code...
async def fetch_table_columns(project_id: str, table_name: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch schema from data-lakehouse and return {"columns": resultData}.
    - Surfaces datalake error body for easier debugging.
    - Polls /query/{jobId} if the schema request is queued.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{DATALAKE_BASE_URL}/schema/{project_id}/{table_name}")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Network error contacting data-lakehouse: {e}")

        # try to parse JSON body even on non-2xx so we can show server message
        try:
            payload = resp.json()
        except Exception:
            payload = {"_raw_text": resp.text}

        # Surface server errors with body
        if resp.status_code >= 500:
            detail = payload.get("error") or payload.get("message") or resp.text
            raise HTTPException(status_code=502, detail=f"Data-lakehouse schema endpoint error {resp.status_code}: {detail}")
        if resp.status_code >= 400:
            detail = payload.get("error") or payload.get("message") or resp.text
            raise HTTPException(status_code=400, detail=f"Data-lakehouse schema endpoint returned {resp.status_code}: {detail}")

        # If the request returned a queued job, poll the job status
        job_id = payload.get("jobId")
        status = payload.get("status")
        if job_id and status in ("queued", "running"):
            for _ in range(timeout):
                await asyncio.sleep(1)
                try:
                    status_resp = await client.get(f"{DATALAKE_BASE_URL}/query/{job_id}")
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"Error polling schema job: {e}")

                try:
                    status_payload = status_resp.json()
                except Exception:
                    status_payload = {"_raw_text": status_resp.text}

                if status_resp.status_code >= 500:
                    raise HTTPException(status_code=502, detail=f"Schema job status endpoint error {status_resp.status_code}: {status_resp.text}")
                if status_payload.get("status") == "completed":
                    payload = status_payload
                    break
                if status_payload.get("status") == "failed":
                    msg = status_payload.get("message") or status_payload.get("error") or status_resp.text
                    raise HTTPException(status_code=400, detail=f"Schema job failed: {msg}")
            else:
                raise HTTPException(status_code=504, detail="Timed out waiting for schema job to complete")

        # Normalize and return the resultData
        result_data = payload.get("resultData") or payload.get("result_data") or []
        return {"columns": result_data}



# --- Pydantic Models ---

class SuggestChartsRequest(BaseModel):
    user_prompts: List[str]

class SuggestChartsResponse(BaseModel):
    suggestions: List[Dict[str, Any]]

class BuildQueriesRequest(BaseModel):
    dataset_metadata: Dict[str, Any]
    suggestions: List[Dict[str, Any]]

class BuildQueriesResponse(BaseModel):
    intent: str
    charts: List[Dict[str, Any]]
class ExecutePromptRequest(BaseModel):
    user_prompts: List[str]
    project_id: str
    table_name: str
class ExecutePromptResponse(BaseModel):
    intent: str
    charts: List[Dict[str, Any]]
# --- API Endpoints ---

@app.get("/charts-config", summary="Get Full Chart Configuration")
async def get_charts_config():
    """Returns the complete charts_config JSON object."""
    return charts_config

@app.post("/suggest-charts", response_model=SuggestChartsResponse, summary="Suggest Charts from Prompts")
async def api_suggest_charts(request: SuggestChartsRequest):
    """
    Takes a list of natural language prompts and returns suggested chart types 
    relevant to each request using 'Model 1' logic.
    """
    suggester = ChartSuggester(charts_config)
    results = await suggester.suggest(request.user_prompts)
    return {"suggestions": results}

# @app.post("/build-queries", response_model=BuildQueriesResponse, summary="Validate & Build Chart Queries")
# async def api_build_queries(request: BuildQueriesRequest):
#     """
#     Takes dataset metadata and suggested charts, validates them against requirements,
#     and builds the final query/encoding JSON using 'Model 2' logic.
#     """
#     builder = ChartValidatorAndQueryBuilder(charts_config)
#     final_result = await builder.build_final_charts(request.dataset_metadata, request.suggestions)
#     return final_result
@app.post("/build-queries", response_model=BuildQueriesResponse, summary="Build & Execute Chart Queries")
async def api_build_queries(request: BuildQueriesRequest):
    """Build queries and execute on data-lakehouse"""
    validator = ChartValidatorAndQueryBuilder(charts_config, MODEL)
    
    # Build queries from suggestions
    # charts_with_prompts = [
    #     {"chart_id": s.chart_id, "chart_name": s.chart_name, "user_prompt": f"Visualize using {s.chart_name}"}
    #     for s in request.suggestions
    # ]
    
    result = await validator.build_final_charts(request.dataset_metadata, request.suggestions)
    
    # Execute each query on data-lakehouse
    # final_charts = []
    # Get projectId and tableName from dataset_metadata
    # project_id = request.dataset_metadata.get("projectId")
    # table_name = request.dataset_metadata.get("tableName")
    
    # if not project_id or not table_name:
    #     raise HTTPException(
    #         status_code=400, 
    #         detail="dataset_metadata must include 'projectId' and 'tableName'"
    #     )
    
    # # Replace source placeholder with actual source
    # source_name = f"{project_id}.{table_name}"
    
    for chart in result.get("charts", []):
        try:
            # Convert to QuerySpec format
            query_spec = chart["query"]
            query_spec["source"] = "elm4r7a.sales"
            print(f"Executing query for chart {chart['chart_id']}: {query_spec}")
            execution_result = await execute_query_on_datalake(query_spec)
            print(f"Execution result for chart {chart['chart_id']}: {execution_result}")
            chart["data"] = execution_result
            chart["error"] = None
        except HTTPException as e:
            print(f"HTTPException during execution for chart {chart['chart_id']}: {e.detail}")
            chart["error"] = str(e.detail)
        except Exception as e:
            print(f"General exception during execution for chart {chart['chart_id']}: {str(e)}")
            chart["error"] = f"Execution error: {str(e)}"
        
        # final_charts.append(ChartWithQuery(**chart))
    
    # return BuildQueriesResponse(intent="visualization", charts=final_charts)
    return result

@app.post("/execute-prompt", response_model=ExecutePromptResponse, summary=" Execute Chart of Prompt")
async def api_build_queries(request: ExecutePromptRequest):
    """Suggest charts from prompts"""
    suggester = ChartSuggester(charts_config)
    suggestions = await suggester.suggest(request.user_prompts)
    dataset_metadata=await fetch_table_columns(request.project_id, request.table_name)
    """Build queries and execute on data-lakehouse"""
    validator = ChartValidatorAndQueryBuilder(charts_config, MODEL)
    
    # Build queries from suggestions
    # charts_with_prompts = [
    #     {"chart_id": s.chart_id, "chart_name": s.chart_name, "user_prompt": f"Visualize using {s.chart_name}"}
    #     for s in request.suggestions
    # ]
    
    result = await validator.build_final_charts(dataset_metadata, suggestions)
    
    # Execute each query on data-lakehouse
    # final_charts = []
    # Get projectId and tableName from dataset_metadata
    # project_id = request.dataset_metadata.get("projectId")
    # table_name = request.dataset_metadata.get("tableName")
    
    # if not project_id or not table_name:
    #     raise HTTPException(
    #         status_code=400, 
    #         detail="dataset_metadata must include 'projectId' and 'tableName'"
    #     )
    
    # # Replace source placeholder with actual source
    # source_name = f"{project_id}.{table_name}"
    
    for chart in result.get("charts", []):
        try:
            # Convert to QuerySpec format
            query_spec = chart["query"]
            query_spec["source"] = f"{request.project_id}.{request.table_name}"
            print(f"Executing query for chart {chart['chart_id']}: {query_spec}")
            execution_result = await execute_query_on_datalake(query_spec)
            print(f"Execution result for chart {chart['chart_id']}: {execution_result}")
            chart["data"] = execution_result
            chart["error"] = None
        except HTTPException as e:
            print(f"HTTPException during execution for chart {chart['chart_id']}: {e.detail}")
            chart["error"] = str(e.detail)
        except Exception as e:
            print(f"General exception during execution for chart {chart['chart_id']}: {str(e)}")
            chart["error"] = f"Execution error: {str(e)}"
        
        # final_charts.append(ChartWithQuery(**chart))
    
    # return BuildQueriesResponse(intent="visualization", charts=final_charts)
    return result

@app.get("/schema/{project_id}/{table_name}/columns", summary="Get table columns as {'columns': resultData}")
async def api_get_table_columns(project_id: str, table_name: str):
    """
    Return {"columns": resultData} where resultData comes from data-lakehouse /query/{jobId}.
    """
    try:
        return await fetch_table_columns(project_id, table_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# --- Run the Application ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("API documentation available at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)