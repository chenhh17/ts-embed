#!/usr/bin/env python3
"""
Simple FastAPI Hello World application with Jinja2 frontend.
Displays a stylish "Hello World" message.
"""

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

# Create FastAPI app
app = FastAPI(
    title="Hello World App",
    description="A simple FastAPI app with stylish Hello World",
    version="1.0.0"
)

# Setup templates directory (create if not exists)
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")

os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def hello_world(request: Request):
    """Display the stylish Hello World page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": "Hello World!",
        "subtitle": "Welcome to FastAPI with Jinja2"
    })

@app.get("/api/hello")
async def api_hello():
    """Simple API endpoint"""
    return {
        "message": "Hello World!",
        "status": "success",
        "timestamp": "2024"
    }

if __name__ == "__main__":

    uvicorn.run(
        "hello_world_app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )
