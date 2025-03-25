# ========= Copyright 2023-2024 @ TommeyChang. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ TommeyChang. All Rights Reserved. =========
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI

TOOL_SYS_PROMPT = """
You are a helpful assistant, and you prefer to response to the user with 
the tools providing by himself.
If the user provide the result, you should answer the user directly.
"""

TOOL_PROMPT = """
## Available Tools:

{tools}
## Task:

{query}

## Output Format:
If you decide to answer the user with a tool, you should return the 
`tool_name` and the `tool_args` in JSON format as following:
```json
{
    "tool_name": "tool_name",
    "tool_args": {
        "arg1": value1,
        "arg2": value2,
        ...
    }
}
```
"""

RESULT_PROMPT = """
The result `{results}` is obtain by tools you proposed to solve the {query}.
Please check the result and answer me according to the result directly without using tools.
"""


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        self.tools_prompt = TOOL_PROMPT

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # add the tools to the prompt
        tools_list = ""
        for idx, tool in enumerate(response.tools):
            tools_list += f"tool_name: {tool.name}\ndescription: {tool.description}\ninput_schema: {tool.inputSchema}\n\n"
        self.tools_prompt = TOOL_PROMPT.replace("{tools}", tools_list)

    async def process_query(self, query: str) -> str:
        # init the messages
        messages = [
            {
                "rolse": "system",
                "content": TOOL_SYS_PROMPT,
            }
        ]

        # append the query to the prompt with tools
        query_with_tools = self.tools_prompt.replace("{query}", query)
        messages.append(
            {
                "role": "user",
                "content": query_with_tools,
            }
        )

        # Process a query using available tools
        response = self.client.chat.completions.create(
            messages=messages,
            model="qwq",
            max_tokens=64000,
        )

        # Process response and handle tool calls
        content = response.choices[0].message.content.lower()
        messages.append({"role": "assistant", "content": content})

        # extract tool calls from the content with JSON format
        tool_calls = []
        while "```json" in content:
            json_start = re.search(r"```json", content).span()[1]
            json_end = re.search(r"```", content[json_start:]).span()[0]
            json_end = json_end + json_start
            tool_josn = content[json_start:json_end].strip("\n")
            tool_calls.append(json.loads(tool_josn))
            content = content[json_end:]

        if len(tool_calls) == 0:
            return content
        else:
            # call the tool
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["tool_name"]
                tool_args = tool_call["tool_args"]
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({tool_name: result.content[0].text})
            tool_results = json.dumps(tool_results)

            # feed the llm with the results to obtain the final answer
            result_prompt = RESULT_PROMPT.replace("{query}", query)
            result_prompt = result_prompt.replace("{results}", tool_results)

            messages.append({"role": "user", "content": result_prompt})

            response = self.client.chat.completions.create(
                messages=messages,
                model="qwq",
                max_tokens=64000,
            )

            return response.choices[0].message.content

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server("calculator_server.py")
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
