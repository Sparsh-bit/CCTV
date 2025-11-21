"""Local stub for torch_geometric to satisfy static analysis when PyG is not installed.
This is a lightweight shim and should NOT be used for production — install `torch-geometric` for full functionality.
"""
__all__ = ["nn", "data"]
