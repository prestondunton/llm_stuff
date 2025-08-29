import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM 

MODEL = "llama3.1:8b"
# MODEL = "gpt-oss:20b"
MODEL_FILE_DIR = os.path.join(os.getcwd(), 'model_file_sandbox')

app = MCPApp(name="jd_matcher")


async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context
        logger.info("Current config:", data=context.config.model_dump())

        if not os.path.exists(MODEL_FILE_DIR):
            logger.info(f'Model file dir `{MODEL_FILE_DIR}` does not exist.  Creating...')
            os.mkdir(MODEL_FILE_DIR)

        context.config.mcp.servers["filesystem"].args.extend([MODEL_FILE_DIR])




        curator_agent = Agent(
            name="finder",
            instruction="""
            You are an agent with access to the filesystem and
            the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.

            IMPORTANT: When asked to fetch or list content from a URL, you MUST:
            1. Use the fetch tool to retrieve the webpage content
            2. Display the actual text/content you retrieve.  Do NOT truncate content.
            3. Be thorough and show what you found
            4. Do NOT write code to accomplish the task.  Instead use the tools available to you, the agent, through MCP.
            5. Do NOT provide a command or arguments that the user can run to accomplish the task.  It is YOUR job to retrieve the content.
            6. Do NOT provide commentary on whether or not you accomplished the task.  Provide any outputs accomplished, even if partial.
            7. It is always possible to translate text/html content to markdown.


            FORBIDDEN:
            1. Never summarize content without explicitly being asked to.
            CRITICAL: When using the fetch tool, pass boolean parameters as actual booleans (true/false), not as strings ("true"/"false").
            Always use the appropriate tools when asked to access web content.
            """,
            server_names=["fetch", "filesystem"],
        )

        async with curator_agent:
            # logger.info("finder: Connected to server, calling list_tools...")
            # result = await curator_agent.list_tools()
            # tools_overview = result.model_dump()['tools']
            # tools_overview = {tool['name']: tool['description'] for tool in result.model_dump()['tools']}
            # logger.info("Tools available:", data=tools_overview)

            llm = await curator_agent.attach_llm(OllamaAugmentedLLM)


            for step in [
                'Fetch all the content https://www.sandboxaq.com/job-openings using the fetch tool.  Use start_index parameters if necessary and concatenate the result.',
                'Create a list of all the links on the page in a markdown bulleted list of links.',
            ]:
                logger.info(f'Running prompt {step}')
                result = await llm.generate_str(
                    message=step,
                    request_params=RequestParams(model=MODEL, max_iterations=10),
                )
                logger.info(f"Result: {result}")


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
