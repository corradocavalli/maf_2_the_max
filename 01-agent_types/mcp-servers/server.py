from dataclasses import dataclass
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(name="Payment Server 01")

balance: float = 100.0


@mcp.tool()
def get_balance() -> str:
    """Get the balance for a given user."""
    return f"The balance for is ${balance}."


@mcp.tool()
async def make_payment(
    amount: Annotated[float, Field(description="The amount to pay.")],
    reason: Annotated[str, Field(description="The reason for the payment.")],
) -> str:
    global balance
    balance -= amount
    return f"Processing payment of ${amount} for {reason}."


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8000)
