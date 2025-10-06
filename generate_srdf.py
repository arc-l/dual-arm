#!/usr/bin/env python3
"""Generate a SRDF file from a URDF description.

The script inspects the kinematic graph of the provided URDF, heuristically
selects reasonable manipulator chains, and emits a Semantic Robot Description
Format (SRDF) file that is compatible with MoveIt-style tooling. The defaults
can be overridden via command-line arguments to tailor the output to a specific
robot configuration.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set
import sys
import xml.etree.ElementTree as ET


@dataclass
class JointInfo:
    """Lightweight container for joint connectivity."""

    name: str
    type: str
    parent: str
    child: str


@dataclass
class ChainInfo:
    """Represents a kinematic chain from the chosen root to a tip link."""

    tip_link: str
    joints: List[JointInfo]
    active_joints: List[JointInfo]
    base_link: str

    @property
    def has_actuation(self) -> bool:
        return any(j.type != "fixed" for j in self.joints)


@dataclass
class RobotModel:
    name: str
    links: Set[str]
    joints: List[JointInfo]
    child_to_joint: Dict[str, JointInfo]
    parent_children: Dict[str, List[str]]


DEFAULT_TIP_KEYWORDS: Sequence[str] = (
    "tool",
    "tcp",
    "tip",
    "gripper",
    "ee",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="Path to the URDF file to parse.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination path for the generated SRDF. Defaults to <urdf_basename>.srdf in the same directory.",
    )
    parser.add_argument(
        "--root-link",
        dest="root_link",
        help="Name of the root link to use when building chains. Required if multiple roots exist in the URDF.",
    )
    parser.add_argument(
        "--base-link",
        dest="forced_base_link",
        help="Optional override for the base link used in every chain (skips automatic inference).",
    )
    parser.add_argument(
        "--tips",
        nargs="*",
        help="Explicit list of tip links to include. If omitted, the script will heuristically pick reasonable tips.",
    )
    parser.add_argument(
        "--exclude-tips",
        nargs="*",
        default=[],
        help="List of tip links to drop from consideration.",
    )
    parser.add_argument(
        "--min-active-joints",
        type=int,
        default=1,
        help="Discard chains with fewer actuated joints than this threshold (after removing fixed joints).",
    )
    parser.add_argument(
        "--tip-keywords",
        nargs="*",
        default=list(DEFAULT_TIP_KEYWORDS),
        help="Keywords that mark preferred tip links when choosing heuristically.",
    )
    parser.add_argument(
        "--group-prefix",
        default="group_",
        help="Prefix applied to the generated group names.",
    )
    parser.add_argument(
        "--world-frame",
        default="world",
        help="Frame name used as the parent in generated virtual joints.",
    )
    parser.add_argument(
        "--virtual-joint-type",
        default="fixed",
        choices=["fixed", "floating", "planar"],
        help="Type of virtual joint that anchors each base link to the world frame.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print a summary; do not write the SRDF file.",
    )
    return parser.parse_args()


def parse_urdf(urdf_path: Path) -> RobotModel:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected <robot> as the URDF root element, got <{root.tag}> instead.")

    links: Set[str] = set()
    for link_elem in root.findall("link"):
        name = link_elem.attrib.get("name")
        if not name:
            raise ValueError("Encountered a <link> element without a name attribute.")
        links.add(name)

    joints: List[JointInfo] = []
    child_to_joint: Dict[str, JointInfo] = {}
    parent_children: Dict[str, List[str]] = {}

    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib.get("name")
        if not name:
            raise ValueError("Encountered a <joint> element without a name attribute.")
        joint_type = joint_elem.attrib.get("type", "fixed")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint '{name}' is missing a parent or child element.")

        parent_link = parent_elem.attrib.get("link")
        child_link = child_elem.attrib.get("link")
        if not parent_link or not child_link:
            raise ValueError(f"Joint '{name}' has an unnamed parent or child link.")

        joint_info = JointInfo(name=name, type=joint_type, parent=parent_link, child=child_link)
        joints.append(joint_info)
        child_to_joint[child_link] = joint_info
        parent_children.setdefault(parent_link, []).append(child_link)

    return RobotModel(
        name=root.attrib.get("name", "robot"),
        links=links,
        joints=joints,
        child_to_joint=child_to_joint,
        parent_children=parent_children,
    )


def infer_root_link(model: RobotModel, user_root: Optional[str]) -> str:
    if user_root:
        if user_root not in model.links:
            raise ValueError(f"Requested root link '{user_root}' does not exist in the URDF.")
        return user_root

    child_links = set(model.child_to_joint.keys())
    candidate_roots = model.links - child_links
    if not candidate_roots:
        raise ValueError("Unable to locate a root link in the URDF.")
    if len(candidate_roots) > 1:
        raise ValueError(
            "Multiple root links detected ("
            + ", ".join(sorted(candidate_roots))
            + ") – please specify one via --root-link."
        )
    return next(iter(candidate_roots))


def build_chains(model: RobotModel, root_link: str) -> List[ChainInfo]:
    chains: List[ChainInfo] = []
    leaves = [link for link in model.links if link not in model.parent_children]

    for tip in leaves:
        joints: List[JointInfo] = []
        active: List[JointInfo] = []
        current = tip

        while True:
            joint = model.child_to_joint.get(current)
            if joint is None:
                break
            joints.append(joint)
            if joint.type != "fixed":
                active.append(joint)
            current = joint.parent

        if not joints:
            continue
        joints.reverse()
        active.reverse()

        if joints[0].parent != root_link:
            # Chain does not originate at the chosen root; skip it.
            continue

        base_link = infer_chain_base(root_link, joints)
        chains.append(ChainInfo(tip_link=tip, joints=joints, active_joints=active, base_link=base_link))

    return chains


def infer_chain_base(root_link: str, joints: Sequence[JointInfo]) -> str:
    last_fixed_parent = root_link
    last_fixed_child = root_link
    for joint in joints:
        if joint.type == "fixed":
            last_fixed_parent = joint.parent
            last_fixed_child = joint.child
            continue
        # First actuated joint encountered.
        if last_fixed_child != root_link:
            return last_fixed_parent if last_fixed_parent != root_link else last_fixed_child
        return joint.parent
    # All joints were fixed; fall back to the deepest link reached.
    return last_fixed_child if last_fixed_child != root_link else last_fixed_parent


def select_tip_chains(
    chains: Sequence[ChainInfo],
    tips: Optional[Sequence[str]],
    excluded: Sequence[str],
    min_active: int,
    tip_keywords: Sequence[str],
) -> List[ChainInfo]:
    if not chains:
        raise ValueError("No kinematic chains could be constructed from the URDF.")

    chain_by_tip = {chain.tip_link: chain for chain in chains}
    missing_excludes = [tip for tip in excluded if tip not in chain_by_tip]
    if missing_excludes:
        raise ValueError(
            "Cannot exclude unknown tip links: " + ", ".join(sorted(missing_excludes))
        )

    filtered_chains = [chain for chain in chains if chain.tip_link not in excluded]
    if not filtered_chains:
        raise ValueError("All candidate chains were filtered out; relax the exclusion settings.")

    if tips:
        missing = [tip for tip in tips if tip not in chain_by_tip]
        if missing:
            raise ValueError("The following requested tips do not exist: " + ", ".join(sorted(missing)))
        return [chain_by_tip[tip] for tip in tips]

    preferred_keywords = [kw.lower() for kw in tip_keywords]

    def is_keyword_match(chain: ChainInfo) -> bool:
        tip_lower = chain.tip_link.lower()
        return any(keyword in tip_lower for keyword in preferred_keywords)

    keyword_matches = [chain for chain in filtered_chains if is_keyword_match(chain) and len(chain.active_joints) >= min_active]
    if keyword_matches:
        return keyword_matches

    min_active_matches = [chain for chain in filtered_chains if len(chain.active_joints) >= min_active]
    if min_active_matches:
        return min_active_matches

    return filtered_chains


def build_srdf_tree(
    robot_name: str,
    chains: Sequence[ChainInfo],
    group_prefix: str,
    world_frame: str,
    virtual_joint_type: str,
    forced_base_link: Optional[str],
) -> ET.Element:
    srdf_root = ET.Element("robot", attrib={"name": robot_name})

    base_links: Set[str] = set()

    for chain in chains:
        base_link = forced_base_link or chain.base_link
        base_links.add(base_link)
        group_name = f"{group_prefix}{chain.tip_link}"
        group_elem = ET.SubElement(srdf_root, "group", attrib={"name": group_name})
        ET.SubElement(
            group_elem,
            "chain",
            attrib={
                "base_link": base_link,
                "tip_link": chain.tip_link,
            },
        )

    for base_link in sorted(base_links):
        ET.SubElement(
            srdf_root,
            "virtual_joint",
            attrib={
                "name": f"world_to_{base_link}",
                "type": virtual_joint_type,
                "parent_frame": world_frame,
                "child_link": base_link,
            },
        )

    indent_xml(srdf_root)
    return srdf_root


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    line_break = "\n"
    indent = "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = f"{line_break}{'  ' * (level + 1)}"
        for child in elem:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = f"{line_break}{'  ' * (level + 1)}"
        last_child = elem[-1]
        if not last_child.tail or not last_child.tail.strip():
            last_child.tail = f"{line_break}{indent}"
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = f"{line_break}{indent}"


def write_srdf(output_path: Path, srdf_root: ET.Element) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(srdf_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def describe_chains(chains: Iterable[ChainInfo]) -> str:
    lines = []
    for chain in chains:
        active_count = len(chain.active_joints)
        joint_names = ", ".join(j.name for j in chain.active_joints) or "<none>"
        lines.append(
            f"- tip '{chain.tip_link}' with base '{chain.base_link}' and {active_count} active joints: {joint_names}"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_arguments()
    model = parse_urdf(args.urdf)
    root_link = infer_root_link(model, args.root_link)
    chains = build_chains(model, root_link)
    selected_chains = select_tip_chains(
        chains,
        tips=args.tips,
        excluded=args.exclude_tips,
        min_active=args.min_active_joints,
        tip_keywords=args.tip_keywords,
    )

    srdf_root = build_srdf_tree(
        robot_name=model.name,
        chains=selected_chains,
        group_prefix=args.group_prefix,
        world_frame=args.world_frame,
        virtual_joint_type=args.virtual_joint_type,
        forced_base_link=args.forced_base_link,
    )

    if args.dry_run:
        print("Dry run requested; SRDF content will not be written.")
        print(describe_chains(selected_chains))
        return

    output_path = args.output or args.urdf.with_suffix(".srdf")
    write_srdf(output_path, srdf_root)
    print(f"SRDF written to {output_path}")
    print(describe_chains(selected_chains))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI helper
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
