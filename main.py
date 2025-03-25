import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from pydantic import BaseModel

TOOLS_SYS_PROMPT ="""
You are a helpful assistant, and you are equipped with a set of tools to help the user.
You always try to answer the user with a tool as much as possible.
If you decide to answer the user with a tool, you should return the `tool_name` and the `tool_args` in JSON format.

## Available Tools:

{tools}
## Output Format:

```JSON
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

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama"
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:

        # list all tools
        response = await self.session.list_tools()
        tools_list = ""
        for idx, tool in enumerate(response.tools):
            tools_list += f"tool_name: {tool.name}\ndescription: {tool.description}\ninput_schema: {tool.inputSchema}\n\n"
        
        sys_prompt = TOOLS_SYS_PROMPT.replace("{tools}", tools_list) 
        
        # Process a query using available tools
        response = self.client.chat.completions.create(
            messages=[
                {
                    'role': 'system',
                    'content': sys_prompt,
                },
                {
                    'role': 'user',
                    'content': query,
                    },
            ],
            model='qwq',
            max_tokens=64000,
            )

        # Process response and handle tool calls
        final_text = []
        content = response.choices[0].message.content
        tool_calls = []

        # extract tool calls from the content with JSON format
        while "```JSON" in content:
            josn_start = re.search(r'```JSON', content).span()[1]
            josn_end = re.search(r'```', content[josn_start:]).span()[0] + josn_start
            tool_josn = content[josn_start:josn_end].strip('\n')
            tool_calls.append(json.loads(tool_josn))
            content = content[josn_end:]

        if len(tool_calls) == 0:
            final_text.append(content)
        else:
            for tool_call in tool_calls:
                tool_name = tool_call['tool_name']
                tool_args = tool_call['tool_args']
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"Calling tool {tool_name} with args {tool_args} get the result: {result.content[0].text}.")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
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
    import sys
    asyncio.run(main())