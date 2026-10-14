"""PROCESS_ARGS is the single list of `process` args: every config field, API argument and launcher field must match it."""

import dataclasses
import inspect
import re
from pathlib import Path

import pytest

import pixel_patrol_base
from pixel_patrol_base import api
from pixel_patrol_base.core.processing_config import USER_MINIMUMS, ProcessingConfig
from pixel_patrol_base.launch_server import _get_process_args
from pixel_patrol_base.process_args import PROCESS_ARGS

ARG_NAMES = [a.name for a in PROCESS_ARGS]

# Where a ProcessingConfig / API name differs from the arg name.
CONFIG_TO_ARG = {
    "processors_included": "processors_include",
    "processors_excluded": "processors_exclude",
    "selected_file_extensions": "file_extensions",
}
ARG_TO_API = {arg: name for name, arg in CONFIG_TO_ARG.items()}

# Args handled outside api.process_files (project creation, paths, Dask client, viewer).
NOT_PROCESS_FILES_ARGS = {"base_directory", "loader", "output", "paths", "name", "scheduler", "open_viewer"}
# Launcher form field names that differ from the arg name.
ARG_TO_FORM = {"name": "project_name", "output": "output_path"}
NOT_IN_LAUNCHER = {"open_viewer"}

LAUNCHER_HTML = (Path(pixel_patrol_base.__file__).parent / "launch_assets" / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(ProcessingConfig) if f.name != "metadata"])
def test_config_field_is_a_process_arg(field):
    assert CONFIG_TO_ARG.get(field, field) in ARG_NAMES, f"ProcessingConfig.{field} has no entry in PROCESS_ARGS"


@pytest.mark.parametrize("arg", [n for n in ARG_NAMES if n not in NOT_PROCESS_FILES_ARGS])
def test_process_arg_is_documented_api_argument(arg):
    name = ARG_TO_API.get(arg, arg)
    assert name in inspect.signature(api.process_files).parameters, f"api.process_files has no argument '{name}'"
    assert re.search(rf"^\s*{name}:", api.process_files.__doc__, re.MULTILINE), (
        f"api.process_files docstring does not document '{name}'"
    )


@pytest.mark.parametrize("arg", [n for n in ARG_NAMES if n not in NOT_IN_LAUNCHER])
def test_process_arg_is_a_launcher_field(arg):
    form_name = ARG_TO_FORM.get(arg, arg)
    assert f'name="{form_name}"' in LAUNCHER_HTML, f"launcher form has no field '{form_name}'"


@pytest.mark.parametrize("arg, minimum", USER_MINIMUMS.items())
def test_launcher_gets_the_enforced_minimum(arg, minimum):
    assert _get_process_args()[arg]["minimum"] == minimum


def test_launcher_gets_no_minimum_for_unlimited_args():
    assert _get_process_args()["description"]["minimum"] is None


def test_launcher_help_refers_to_real_args():
    unknown = set(re.findall(r'data-arg="([^"]+)"', LAUNCHER_HTML)) - set(ARG_NAMES)
    assert not unknown, f"launcher data-arg refers to unknown args: {sorted(unknown)}"
