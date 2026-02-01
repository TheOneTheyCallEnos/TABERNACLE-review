#!/Users/enos/TABERNACLE/scripts/venv312/bin/python3
"""
Pydantic models for structured Holarchy outputs.

Replaces fragile string parsing with validated JSON schemas.
Used by L-ling daemons (3B) and L-Brain (70B) for structured LLM responses.

Usage:
    # Request JSON output from Ollama
    response = requests.post(
        f"http://{host}:{port}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",  # Key: forces JSON output
            ...
        }
    )
    
    # Validate with Pydantic
    from holarchy_models import LLingThought
    thought = LLingThought.model_validate_json(response.json()["response"])

Author: Logos + Enos
Created: 2026-01-28
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class HolonType(str, Enum):
    """Types of holons in the topology."""
    CONCEPT = "concept"
    PROCESS = "process"
    ENTITY = "entity"
    RELATION = "relation"
    FILE = "file"
    TECHNOGLYPH = "technoglyph"


class EdgeType(str, Enum):
    """Types of edges between holons."""
    IMPLIES = "IMPLIES"       # A → B (causal/logical)
    ASSOCIATES = "ASSOCIATES" # A ↔ B (related)
    CONTAINS = "CONTAINS"     # A ⊃ B (hierarchical)
    REFERENCES = "REFERENCES" # A cites B
    REQUIRES = "REQUIRES"     # A depends on B
    CONTRADICTS = "CONTRADICTS"  # A conflicts with B


class QueryType(str, Enum):
    """Types of holarchy queries."""
    SEARCH = "search"
    TRAVERSE = "traverse"
    ANALYZE = "analyze"


# =============================================================================
# L-LING THOUGHT MODEL (3B daemon reasoning)
# =============================================================================

class LLingThought(BaseModel):
    """
    Structured output for L-ling (3B) reasoning.
    
    Replaces _parse_thought() string parsing in l_ling.py.
    
    The 3B model outputs JSON matching this schema when prompted correctly.
    """
    perception: str = Field(
        default="",
        description="What the L-ling observes in the current context"
    )
    reasoning: str = Field(
        default="",
        description="Analysis and considerations"
    )
    decision: str = Field(
        default="",
        description="What action to take and why"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the decision (0.0-1.0)"
    )
    
    @field_validator('confidence', mode='before')
    @classmethod
    def normalize_confidence(cls, v):
        """Handle confidence as percentage or decimal."""
        if isinstance(v, str):
            v = v.replace('%', '').strip()
            try:
                v = float(v)
            except ValueError:
                return 0.5
        if v > 1.0:
            return v / 100.0
        return v


# =============================================================================
# L-BRAIN DECISION MODEL (70B escalation handling)
# =============================================================================

class ProposedEdge(BaseModel):
    """A proposed edge to add to the topology."""
    model_config = ConfigDict(populate_by_name=True)
    
    source: str = Field(..., description="Source node name")
    target: str = Field(..., description="Target node name")
    edge_type: EdgeType = Field(
        default=EdgeType.ASSOCIATES,
        alias="type",
        description="Type of relationship"
    )
    weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Edge weight (0.0-1.0)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Why this edge should exist"
    )


class BrainDecision(BaseModel):
    """
    Structured output for L-Brain (70B) escalation decisions.
    
    Replaces _parse_decision() string parsing in l_brain.py.
    """
    analysis: str = Field(
        default="",
        description="Analysis of the escalation"
    )
    decision: str = Field(
        default="",
        description="The decision and reasoning"
    )
    mandate: Optional[str] = Field(
        default=None,
        description="New mandate to issue, if any"
    )
    edges: List[ProposedEdge] = Field(
        default_factory=list,
        description="Edges to create/modify"
    )
    chosen_option: Optional[str] = Field(
        default=None,
        description="ID of chosen option from escalation"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in decision"
    )


# =============================================================================
# HOLON MODELS (CRUD operations)
# =============================================================================

class HolonCreate(BaseModel):
    """Structured output for holon creation."""
    name: str = Field(..., description="The holon's canonical name")
    holon_type: HolonType = Field(..., description="Type of holon")
    parent: Optional[str] = Field(
        default=None,
        description="Parent holon name if nested"
    )
    description: str = Field(..., description="Brief description")
    links: List[str] = Field(
        default_factory=list,
        description="Related holon names"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class HolonQuery(BaseModel):
    """Structured output for holon queries."""
    query_type: QueryType = Field(..., description="Type of query")
    target: str = Field(..., description="Target holon or pattern")
    depth: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Traversal depth"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional query filters"
    )


class HolarchyResponse(BaseModel):
    """Structured response from holarchy operations."""
    success: bool = Field(..., description="Whether operation succeeded")
    holons_affected: List[str] = Field(
        default_factory=list,
        description="Names of affected holons"
    )
    message: str = Field(..., description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Operation-specific return data"
    )
    errors: Optional[List[str]] = Field(
        default=None,
        description="Error messages if any"
    )


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

def get_json_schema_prompt(model_class: type[BaseModel]) -> str:
    """
    Generate a prompt section describing the expected JSON schema.
    
    Include this in prompts to guide LLM output format.
    """
    schema = model_class.model_json_schema()
    
    # Build human-readable description
    props = schema.get("properties", {})
    required = schema.get("required", [])
    
    lines = ["You MUST respond with valid JSON matching this schema:", "{"]
    
    for name, prop in props.items():
        ptype = prop.get("type", "string")
        desc = prop.get("description", "")
        default = prop.get("default", None)
        req = "(required)" if name in required else f"(default: {default})"
        lines.append(f'  "{name}": {ptype},  // {desc} {req}')
    
    lines.append("}")
    return "\n".join(lines)


def build_lling_json_prompt(task: str, context: str = "") -> str:
    """
    Build a prompt that requests JSON output from an L-ling.
    
    Args:
        task: The task description
        context: Optional technoglyphic context
        
    Returns:
        Prompt string that will produce LLingThought-compatible JSON
    """
    schema_desc = get_json_schema_prompt(LLingThought)
    
    prompt = f"""{context}

## Task
{task}

## Response Format
{schema_desc}

Respond ONLY with the JSON object. No other text."""
    
    return prompt


def build_brain_json_prompt(escalation_desc: str, options: List[Dict]) -> str:
    """
    Build a prompt for L-Brain escalation decisions.
    
    Args:
        escalation_desc: Description of the escalation
        options: List of available options
        
    Returns:
        Prompt string that will produce BrainDecision-compatible JSON
    """
    options_str = "\n".join(
        f"  - {opt.get('id', 'unknown')}: {opt.get('description', '')}"
        for opt in options
    )
    
    schema_desc = get_json_schema_prompt(BrainDecision)
    
    prompt = f"""## Escalation
{escalation_desc}

## Available Options
{options_str}

## Response Format
{schema_desc}

For edges, use format: {{"source": "A", "target": "B", "type": "ASSOCIATES", "weight": 0.7}}

Respond ONLY with the JSON object. No other text."""
    
    return prompt


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def parse_llm_response(
    response: str,
    model_class: type[BaseModel],
    fallback_parser: Optional[callable] = None
) -> tuple[Optional[BaseModel], Optional[str]]:
    """
    Parse LLM response into Pydantic model with fallback.
    
    Args:
        response: Raw LLM response string
        model_class: Pydantic model class to validate against
        fallback_parser: Optional function to try if JSON parsing fails
        
    Returns:
        Tuple of (parsed_model, error_message)
        - On success: (model_instance, None)
        - On failure: (None, error_string)
    """
    import json
    
    # Try direct JSON parse
    try:
        # Handle case where response might have markdown code fences
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Extract JSON from code block
            lines = cleaned.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines)
        
        return model_class.model_validate_json(cleaned), None
        
    except Exception as json_error:
        # Try parsing as dict if embedded in text
        try:
            # Find JSON-like content
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                return model_class.model_validate(data), None
        except Exception:
            pass
        
        # Try fallback parser
        if fallback_parser:
            try:
                result = fallback_parser(response)
                if isinstance(result, dict):
                    return model_class.model_validate(result), None
            except Exception:
                pass
        
        return None, f"Failed to parse response as {model_class.__name__}: {json_error}"


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test LLingThought
    test_json = '''
    {
        "perception": "I observe 5 nodes and 12 edges in context",
        "reasoning": "The topology shows high connectivity between concepts",
        "decision": "I will focus on strengthening weak edges",
        "confidence": 0.85
    }
    '''
    thought, err = parse_llm_response(test_json, LLingThought)
    assert thought is not None, f"Parse failed: {err}"
    print(f"LLingThought parsed: confidence={thought.confidence}")
    
    # Test BrainDecision
    test_brain = '''
    {
        "analysis": "This escalation requires immediate attention",
        "decision": "Choose option A for stability",
        "mandate": "Focus on coherence restoration",
        "edges": [
            {"source": "concept_a", "target": "concept_b", "type": "IMPLIES", "weight": 0.8}
        ],
        "chosen_option": "A",
        "confidence": 0.9
    }
    '''
    decision, err = parse_llm_response(test_brain, BrainDecision)
    assert decision is not None, f"Parse failed: {err}"
    print(f"BrainDecision parsed: {len(decision.edges)} edges")
    
    # Test schema generation
    print("\n--- LLingThought Schema Prompt ---")
    print(get_json_schema_prompt(LLingThought))
    
    print("\n--- All tests passed ---")
