import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef

logger = logging.getLogger(__name__)


class SimpleLlmCallFunctionConfig(FunctionBaseConfig, name="simple_llm_call"):
    llm_name: LLMRef

@register_function(config_type=SimpleLlmCallFunctionConfig)
async def simple_llm_call_function(
    config: SimpleLlmCallFunctionConfig, builder: Builder
):
    # Implement your function logic here
    async def _response_fn(input_message: str) -> str:
        # Process the input_message and generate output
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        response = await llm.ainvoke(f"You are a helpful assistant. Respond to the following input: {input_message}")
    
        return response.content

    try:
        yield FunctionInfo.from_fn(_response_fn, description="Simple LLM Call Workflow")
    except GeneratorExit:
        print("Workflow exited early!")
    finally:
        print("Cleaning up simple_llm_call workflow.")