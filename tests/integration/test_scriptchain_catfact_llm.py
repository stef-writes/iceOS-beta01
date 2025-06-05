import os
import asyncio
from datetime import datetime
from app.services.tool_service import ToolService
from app.models.node_models import NodeConfig, NodeMetadata, NodeExecutionResult
from app.chains.script_chain import ScriptChain
from app.nodes.factory import node_factory
from app.tools.base import BaseTool
from pydantic import BaseModel
from app.models.config import LLMConfig
from app.utils.context import GraphContextManager
from app.utils.callbacks import ScriptChainCallback

# --- Tool Definition ---
class WordCounterParams(BaseModel):
    fact: str

class WordCounterTool(BaseTool):
    name = "WordCounter"
    description = "Counts the number of words in a cat fact."
    parameters_schema = WordCounterParams

    def run(self, fact):
        text = str(fact)
        return {"word_count": len(text.split()), "text": text}

def build_chain():
    tool_service = ToolService()
    tool_service.register_tool(WordCounterTool())
    now = datetime.utcnow()
    api_node_config = NodeConfig(
        id='get_cat_fact',
        type='api',
        name='GetCatFact',
        model='api',
        prompt='Fetch a random cat fact from the Cat Facts API.',
        templates={
            'url': 'https://catfact.ninja/fact',
            'method': 'GET',
            'headers': {'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        },
        dependencies=[],
        input_schema={},
        output_schema={'fact': 'str', 'length': 'int'},
        metadata=NodeMetadata(
            node_id='get_cat_fact',
            node_type='api',
            name='GetCatFact',
            version='1.0.0',
            owner='test_user',
            created_at=now,
            modified_at=now,
            description='Fetches a random cat fact from the Cat Facts API.',
            error_type=None,
            timestamp=now,
            start_time=now,
            end_time=None,
            duration=None,
            provider=None
        )
    )
    tool_node_config = NodeConfig(
        id='count_words',
        type='tool',
        name='WordCounter',
        model='tool',
        prompt='Count the number of words in the cat fact.',
        dependencies=['get_cat_fact'],
        input_schema={'fact': 'str'},
        output_schema={'word_count': 'int', 'text': 'str'},
        metadata=NodeMetadata(
            node_id='count_words',
            node_type='tool',
            name='WordCounter',
            version='1.0.0',
            owner='test_user',
            created_at=now,
            modified_at=now,
            description='Counts the number of words in the provided cat fact.',
            error_type=None,
            timestamp=now,
            start_time=now,
            end_time=None,
            duration=None,
            provider=None
        )
    )
    llm_config = LLMConfig(
        provider='openai',
        model='gpt-3.5-turbo',
        api_key=os.getenv('OPENAI_API_KEY'),
        max_context_tokens=2048
    )
    ai_node_config = NodeConfig(
        id='generate_tweet',
        type='ai',
        name='TweetGenerator',
        model='gpt-3.5-turbo',
        prompt='Write a witty tweet about this cat fact: {fact} (It has {word_count} words!)',
        dependencies=['get_cat_fact', 'count_words'],
        input_schema={'fact': 'str', 'word_count': 'int'},
        output_schema={'tweet': 'str'},
        output_format='json',
        metadata=NodeMetadata(
            node_id='generate_tweet',
            node_type='ai',
            name='TweetGenerator',
            version='1.0.0',
            owner='test_user',
            created_at=now,
            modified_at=now,
            description='Generates a witty tweet using the cat fact and word count.',
            error_type=None,
            timestamp=now,
            start_time=now,
            end_time=None,
            duration=None,
            provider='openai'
        ),
        llm_config=llm_config
    )
    nodes = [api_node_config, tool_node_config, ai_node_config]
    return nodes, tool_service

class RealisticScriptChain(ScriptChain):
    def __init__(self, nodes, context_manager=None, callbacks=None, max_parallel=5, persist_intermediate_outputs=True, tool_service=None, initial_context=None):
        self.node_configs = {node.id: node for node in nodes}
        self.nodes = {}
        self.global_context_manager = context_manager or GraphContextManager()
        self.callbacks = callbacks or []
        self.max_parallel = max_parallel
        self.chain_id = os.urandom(8).hex()
        self.name = f"chain-{self.chain_id[:8]}"
        self.persist_intermediate_outputs = persist_intermediate_outputs
        self.metrics = {
            'total_tokens': 0,
            'node_execution_times': {},
            'provider_usage': {},
            'token_usage': {},
            'chain_name': self.name
        }
        self.tool_service = tool_service
        self.initial_context = initial_context or {}
        import networkx as nx
        self.graph = nx.DiGraph()
        for node_id, node_config in self.node_configs.items():
            self.graph.add_node(node_id, level=0)
            for dep in node_config.dependencies:
                if dep not in self.node_configs:
                    raise ValueError(f"Dependency {dep} not found for node {node_id}")
                self.graph.add_edge(dep, node_id)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                cycle_str = " -> ".join(cycles[0])
                raise Exception(f"Circular dependency detected: {cycle_str}")
        except nx.NetworkXNoCycle:
            pass
        for node_id in nx.topological_sort(self.graph):
            node_config = self.node_configs[node_id]
            node_config.level = max(
                (self.node_configs[dep].level for dep in node_config.dependencies),
                default=-1
            ) + 1
        self.levels = {}
        for node_id, node_config in self.node_configs.items():
            level = node_config.level
            if level not in self.levels:
                self.levels[level] = []
            self.levels[level].append(node_id)
            self.nodes[node_id] = node_factory(
                node_config,
                self.global_context_manager,
                getattr(node_config, 'llm_config', None),
                self.callbacks,
                self.tool_service
            )
    async def _execute_level(self, level, accumulated_results):
        level_start_time = datetime.utcnow()
        semaphore = asyncio.Semaphore(self.max_parallel)
        async def process_node(node_id):
            async with semaphore:
                start_time = datetime.utcnow()
                node = self.nodes[node_id]
                context = {}
                missing_dependencies = []
                validation_errors = []
                for dep_id in node.config.dependencies:
                    dep_result = accumulated_results.get(dep_id)
                    if not dep_result or not dep_result.success:
                        missing_dependencies.append(dep_id)
                        continue
                    if not dep_result.output:
                        validation_errors.append(f"Dependency '{dep_id}' produced no output")
                        continue
                    if isinstance(dep_result.output, dict):
                        if dep_id == 'get_cat_fact' and 'body' in dep_result.output and 'fact' in dep_result.output['body']:
                            context['fact'] = dep_result.output['body']['fact']
                        if dep_id == 'count_words' and 'word_count' in dep_result.output:
                            context['word_count'] = dep_result.output['word_count']
                        context.update(dep_result.output)
                    else:
                        context[dep_id] = dep_result.output
                if not node.config.dependencies:
                    context = {**self.initial_context, **context}
                if missing_dependencies:
                    error_msg = f"Node '{node.id}' skipped: Required dependencies failed or missing: {missing_dependencies}"
                    print(error_msg)
                    error_result = NodeExecutionResult(
                        success=False,
                        error=error_msg,
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.config.type,
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type="MissingDependencyError",
                            provider=node.config.provider
                        )
                    )
                    return node_id, error_result
                if validation_errors:
                    error_msg = f"Node '{node.id}' validation failed:\n" + "\n".join(validation_errors)
                    print(error_msg)
                    error_result = NodeExecutionResult(
                        success=False,
                        error=error_msg,
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.config.type,
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type="ValidationError",
                            provider=node.config.provider
                        )
                    )
                    return node_id, error_result
                result = await node.execute(context)
                return node_id, result
        tasks = [process_node(node_id) for node_id in self.levels[level]]
        current_level_node_outputs = dict(await asyncio.gather(*tasks))
        return current_level_node_outputs

async def test_scriptchain_catfact_llm():
    nodes, tool_service = build_chain()
    chain = RealisticScriptChain(nodes=nodes, tool_service=tool_service)
    result = await chain.execute()
    assert result.success, f"ScriptChain failed: {result.error}"
    assert 'get_cat_fact' in result.output
    assert 'count_words' in result.output
    assert 'generate_tweet' in result.output
    tweet = result.output['generate_tweet'].output.get('tweet')
    print("\n=== ScriptChain Cat Fact LLM Test Result ===")
    print(f"Tweet: {tweet}")
    print(result)
    print("===========================================\n")

if __name__ == "__main__":
    asyncio.run(test_scriptchain_catfact_llm()) 