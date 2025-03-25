# MCP-toolkit

In this repository, I implement the MCP calling for models without function calling ability.

The core idea is using LLM to identify the `tool_name` and `tool_args` and encapsulate them into a JSON object.

Then we extract the JSON object and call the tool.

I have proposed a PR that implemented this function into [Camel](https://github.com/camel-ai/camel).
