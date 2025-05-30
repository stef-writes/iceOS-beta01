import json
from app.models.node_models import NodeConfig
from app.models.config import LLMConfig
from app.utils.token_counter import TokenCounter
from app.nodes.constants import TOOL_INSTRUCTION

# Standalone function for building the tool preamble

def build_tool_preamble(tools: list) -> str:
    """Generate a human-readable preamble describing available tools, their parameters, and usage examples."""
    if not tools:
        return ""
    lines = [TOOL_INSTRUCTION, "\nYou have access to the following tools:"]
    for tool in tools:
        params = tool.get('parameters_schema', {}).get('properties', {})
        required = set(tool.get('parameters_schema', {}).get('required', []))
        param_strs = []
        for k, v in params.items():
            typ = v.get('type', 'unknown')
            desc = v.get('description', '')
            req = 'required' if k in required else 'optional'
            param_strs.append(f"{k} ({typ}, {req}){': ' + desc if desc else ''}")
        param_str = ", ".join(param_strs) if param_strs else "No parameters."
        lines.append(f"- {tool['name']}: {tool['description']} Parameters: {param_str}")
        # Add usage example if present
        usage_example = tool.get('usage_example')
        if usage_example:
            lines.append("  Example:")
            lines.append("  " + usage_example.strip().replace("\n", "\n  "))
    return "\n".join(lines) + "\n\n"

# Standalone function for preparing the prompt

def prepare_prompt(config: NodeConfig, context_manager, llm_config: LLMConfig, tool_service, inputs: dict) -> str:
    template = config.prompt
    selected_contexts = {}
    # Filter inputs based on selection
    if config.input_selection:
        selected_contexts = {
            k: v for k, v in inputs.items() 
            if k in config.input_selection
        }
    else:
        selected_contexts = inputs
    # Apply context rules and format inputs
    formatted_inputs = {}
    for input_id, context in selected_contexts.items():
        rule = config.context_rules.get(input_id)
        if rule and rule.include:
            formatted_inputs[input_id] = context_manager.format_context(
                context,
                rule,
                config.format_specifications.get(input_id)
            )
        elif not rule or rule.include:
            formatted_inputs[input_id] = context
    # Replace placeholders in template
    for input_id, content in formatted_inputs.items():
        placeholder = f"{{{input_id}}}"
        if placeholder in template:
            template = template.replace(placeholder, str(content))
    # Add tool preamble if tools are available
    tools = None
    if config.tools:
        tools = [tool.model_dump() for tool in config.tools]
    else:
        tools = tool_service.list_tools_with_schemas() if tool_service else []
    preamble = build_tool_preamble(tools) if tools else ""
    prompt_with_preamble = preamble + template
    # Validate total token count
    try:
        total_tokens = TokenCounter.count_tokens(
            prompt_with_preamble,
            llm_config.model,
            llm_config.provider
        )
        if total_tokens > llm_config.max_context_tokens:
            # Truncate prompt if needed
            prompt_with_preamble = prompt_with_preamble[:llm_config.max_context_tokens * 4]  # rough estimate
    except ValueError:
        pass
    return prompt_with_preamble 