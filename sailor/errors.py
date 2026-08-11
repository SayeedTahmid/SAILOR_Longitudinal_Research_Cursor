"""Integrity failures and the binding STOP protocol."""

from __future__ import annotations


class StopProtocolError(RuntimeError):
    """A scientific-integrity failure that must stop pipeline execution."""

    def __init__(self, problem: str, impact: str, recommended_fix: str) -> None:
        self.problem = problem
        self.impact = impact
        self.recommended_fix = recommended_fix
        super().__init__(self.render())

    def render(self) -> str:
        return (
            f"PROBLEM:         {self.problem}\n"
            f"IMPACT:          {self.impact}\n"
            f"RECOMMENDED FIX: {self.recommended_fix}"
        )
