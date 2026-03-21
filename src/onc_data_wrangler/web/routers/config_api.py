"""Config CRUD API router."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


class SaveConfigRequest(BaseModel):
    path: str
    config: dict[str, Any]


class ValidateConfigRequest(BaseModel):
    config: dict[str, Any]


@router.get("/load")
async def load_config_endpoint(path: str) -> JSONResponse:
    """Load a YAML config file and return as JSON."""
    config_path = Path(path)
    if not config_path.exists():
        raise HTTPException(404, f"Config file not found: {path}")
    try:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return JSONResponse(data)
    except Exception as exc:
        raise HTTPException(400, f"Failed to load config: {exc}")


@router.put("/save")
async def save_config_endpoint(req: SaveConfigRequest) -> JSONResponse:
    """Save a config dict as YAML."""
    try:
        import yaml

        config_path = Path(req.path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(req.config, f, default_flow_style=False, sort_keys=False)
        logger.info("Config saved to %s", config_path)
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        raise HTTPException(400, f"Failed to save config: {exc}")


@router.post("/validate")
async def validate_config_endpoint(req: ValidateConfigRequest) -> JSONResponse:
    """Validate a config dict."""
    errors: list[str] = []
    project = req.config.get("project", {})
    if not project.get("name"):
        errors.append("project.name is required")
    if not project.get("output_dir"):
        errors.append("project.output_dir is required")
    return JSONResponse({"valid": len(errors) == 0, "errors": errors})


@router.get("/ontologies")
async def list_ontologies() -> JSONResponse:
    """List available ontologies from the registry."""
    try:
        from ...ontologies.registry import OntologyRegistry

        result = []
        for ont in OntologyRegistry.get_all():
            result.append(
                {
                    "id": ont.ontology_id,
                    "display_name": ont.display_name,
                    "description": getattr(ont, "description", ""),
                    "version": getattr(ont, "version", ""),
                }
            )
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Failed to list ontologies: %s", exc, exc_info=True)
        return JSONResponse([])


@router.get("/ontology/{ontology_id}/fields")
async def get_ontology_fields(ontology_id: str) -> JSONResponse:
    """Get field definitions for a specific ontology."""
    try:
        from ...ontologies.registry import OntologyRegistry

        ont = OntologyRegistry.get(ontology_id)
        if ont is None:
            raise HTTPException(404, f"Ontology not found: {ontology_id}")

        categories = []
        base_items = ont.get_base_items()
        for cat in base_items:
            items = []
            for item in cat.items:
                items.append(
                    {
                        "id": item.id,
                        "name": item.name,
                        "data_type": item.data_type,
                        "description": getattr(item, "description", ""),
                    }
                )
            categories.append(
                {"id": cat.id, "name": cat.name, "items": items}
            )
        return JSONResponse({"categories": categories})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to get ontology fields: {exc}")
