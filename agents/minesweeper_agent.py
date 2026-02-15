def build_prompt(self, game_state: Dict[str, Any]) -> tuple[str, str]:
    """
    Generate prompt for LLM from game state.

    Args:
        game_state: Dictionary containing board, rows, cols, mines, etc.

    Returns:
        (prompt, system_prompt)
    """
    # ✅ CHANGE: Match training system prompt EXACTLY
    sys_prompt = "You are a Minesweeper AI. Output ONLY valid JSON. No explanations, no reasoning, just {\"type\":\"reveal\",\"row\":N,\"col\":N}."

    # ✅ CHANGE: Ultra-minimal prompt (matches training)
    board = game_state.get("board", [])
    rows = game_state.get("rows", len(board))
    cols = game_state.get("cols", len(board[0]) if board else 0)
    mines = game_state.get("mines", 0)
    
    # Format board
    board_str = "\n".join(" ".join(row) for row in board)
    
    # Ultra-minimal prompt matching training format
    prompt = f"""{rows}x{cols} {mines}mines
{board_str}
Reply ONLY {{"type":"reveal","row":N,"col":N}} or {{"type":"flag","row":N,"col":N}}
Example: {{"type":"reveal","row":2,"col":3}}
DO NOT explain. Just the JSON."""

    return prompt, sys_prompt
