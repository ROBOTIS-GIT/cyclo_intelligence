"""Thin adapter shared by the offline worker and collection API."""
import json

SCHEMA = 'qwen_mission_annotation_v3'
if __package__:
    from .mission_decisions import SYSTEM, build_input, parse_decision, system_for, sample_for as build_sample
else:  # Offline worker mounts this module directly.
    from mission_decisions import SYSTEM, build_input, parse_decision, system_for, sample_for as build_sample


def prompt_for(frame, catalog):
    return json.dumps(build_input(frame.get('context') or {}, catalog), ensure_ascii=False)


def sample_for(frame, catalog, target, image):
    return build_sample(frame.get('context') or {}, catalog, target, image)


def parse_target(text, context=None):
    return parse_decision(text, context)
