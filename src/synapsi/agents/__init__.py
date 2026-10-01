"""Agents and built-in roles."""

from synapsi.agents.base import Agent
from synapsi.agents.roles import RoleSpec, get_role, list_roles, register_role

__all__ = ["Agent", "RoleSpec", "get_role", "list_roles", "register_role"]
