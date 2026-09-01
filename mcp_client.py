from __future__ import annotations

import json
from typing import Dict, List

import deps


class MCPClient:
    def __init__(self):
        self.connected = False
        self.command = ""
        self.args: List[str] = []
        self.env: Dict[str, str] = {}

    def connect(self, command: str, args: str, env_json: str) -> str:
        self.command = command.strip()
        self.args = [a for a in args.strip().split() if a]
        try:
            self.env = json.loads(env_json) if env_json.strip() else {}
            if not isinstance(self.env, dict):
                return "❌ env must be a JSON object (key/value)."
        except Exception as e:
            return f"❌ Invalid env JSON: {e}"
        self.connected = True
        if deps._MCP_AVAILABLE:
            return "✅ MCP is installed. Connected settings saved (this UI does not auto-spawn servers)."
        return "ℹ️ MCP package not installed. Settings saved; install `pip install mcp` to enable real connections."

    def list_tools(self) -> str:
        if not self.connected:
            return "Not connected. Provide command/args/env and click Connect."
        if deps._MCP_AVAILABLE:
            return "MCP installed: This demo UI does not enumerate tools automatically. Integrate your client here."
        return "MCP not available: cannot list tools."

    def run_tool(self, name: str, args_json: str) -> str:
        if not self.connected:
            return "Not connected. Provide command/args/env and click Connect."
        try:
            parsed = json.loads(args_json.strip() or "{}")
        except Exception as e:
            return f"❌ Invalid JSON args: {e}"
        return f"🛠️ (Demo) Would run MCP tool `{name}` with args:\n```json\n{json.dumps(parsed, indent=2)}\n```"


MCP = MCPClient()
