"""Voice changer endpoint setup tool (module F): enumerate/enable VB-CABLE
endpoints (Disabled ones included) and point the default recording device
(all three roles) at the CABLE capture endpoint.

Examples: --list, --enable, --set-default-mic (combinable).
Names and the dependency dir are configurable (CLI wins, env second):
  --driver-name/BSP_CABLE_DRIVER, --capture-name/BSP_CABLE_CAPTURE,
  --deps-dir/BSP_DEPS_DIR
Enabling writes HKLM\\...\\MMDevices DeviceState=1 (admin required);
AudioEndpointBuilder re-enumerates within ~0.5 s, no service restart needed.
DEVICE_STATE_ACTIVE_ALL (0xF) is what also lists Disabled endpoints.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DEPS_DIRS = (
    PROJECT_ROOT / "downloads" / "repair_deps",
)

DEFAULT_DRIVER_NAME = "VB-Audio Virtual Cable"
DEFAULT_CAPTURE_CANDIDATES = ("变声麦克风", "变声器麦克风", "CABLE Output")
DEFAULT_RENDER_CANDIDATES = ("变声器输出", "CABLE Input")

FLOW_RENDER = 0
FLOW_CAPTURE = 1
FLOW_NAMES = {FLOW_RENDER: "render", FLOW_CAPTURE: "capture"}

DEVICE_STATE_ACTIVE = 0x1
DEVICE_STATE_DISABLED = 0x2
DEVICE_STATE_NOTPRESENT = 0x4
DEVICE_STATE_UNPLUGGED = 0x8
DEVICE_STATE_ACTIVE_ALL = 0xF  # spec's DEVICE_STATE_ACTIVE_ALL = DEVICE_STATEMASK_ALL
STATE_NAMES = {
    DEVICE_STATE_ACTIVE: "Active",
    DEVICE_STATE_DISABLED: "Disabled",
    DEVICE_STATE_NOTPRESENT: "NotPresent",
    DEVICE_STATE_UNPLUGGED: "Unplugged",
}

ROLES = ((0, "Console"), (1, "Multimedia"), (2, "Communications"))

_ENDPOINT_GUID_RE = re.compile(
    r"\{[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}"
)

try:
    import winreg  # type: ignore
except ImportError:  # pragma: no cover - non-Windows only
    winreg = None  # type: ignore


# --------------------------------------------------------------------------- #
# Dependencies and basic helpers
# --------------------------------------------------------------------------- #
def ensure_core_audio_deps(deps_dir: Optional[str] = None) -> None:
    """Make pycaw (Windows Core Audio) importable, adding the dependency dir to sys.path when needed."""
    try:
        import pycaw  # noqa: F401

        return
    except ImportError:
        pass

    candidates: List[Path] = []
    if deps_dir:
        candidates.append(Path(deps_dir))
    for env_name in ("BSP_DEPS_DIR", "PYCAW_DEPS_DIR"):
        value = os.environ.get(env_name, "").strip()
        if value:
            candidates.append(Path(value))
    candidates.extend(DEFAULT_DEPS_DIRS)

    for path in candidates:
        if not (path / "pycaw").is_dir():
            continue
        sys.path.insert(0, str(path))
        try:
            import pycaw  # noqa: F401,F811

            return
        except ImportError:
            sys.path.remove(str(path))

    raise RuntimeError(
        "找不到 pycaw 依赖：请用 --deps-dir 或环境变量 BSP_DEPS_DIR 指定含 pycaw/ 的目录"
    )


def _log_to(log: Optional[Callable[[str], None]], message: str) -> None:
    if log is not None:
        log(message)


def is_admin() -> bool:
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # pragma: no cover - non-Windows only
        return False


def state_name(state: int) -> str:
    return STATE_NAMES.get(state & 0xF, "State0x%x" % state)


# --------------------------------------------------------------------------- #
# Endpoint enumeration
# --------------------------------------------------------------------------- #
def _friendly_name(device) -> str:
    from comtypes import GUID
    from pycaw.api.mmdeviceapi import PROPERTYKEY

    key = PROPERTYKEY()
    key.fmtid = GUID("{a45c254e-df1c-4efd-8020-67d146a850e0}")  # PKEY_Device_FriendlyName
    key.pid = 14
    try:
        value = device.OpenPropertyStore(0).GetValue(key).GetValue()
    except Exception:
        return ""
    return str(value).strip()


def enumerate_endpoints(
    driver_name: Optional[str] = None,
    state_mask: int = DEVICE_STATE_ACTIVE_ALL,
) -> List[Dict]:
    """Enumerate render + capture endpoints; when driver_name is non-empty, keep only endpoints whose friendly name contains it."""
    ensure_core_audio_deps()
    from comtypes import CoCreateInstance
    from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
    from pycaw.constants import CLSID_MMDeviceEnumerator

    enumerator = CoCreateInstance(CLSID_MMDeviceEnumerator, interface=IMMDeviceEnumerator)
    endpoints: List[Dict] = []
    needle = (driver_name or "").lower()
    for flow in (FLOW_RENDER, FLOW_CAPTURE):
        collection = enumerator.EnumAudioEndpoints(flow, state_mask)
        for index in range(collection.GetCount()):
            device = collection.Item(index)
            name = _friendly_name(device)
            if needle and needle not in name.lower():
                continue
            try:
                state = int(device.GetState())
            except Exception:
                state = 0
            endpoints.append(
                {
                    "flow": flow,
                    "flow_label": FLOW_NAMES[flow],
                    "state": state,
                    "state_label": state_name(state),
                    "name": name or "(无名称)",
                    "id": device.GetId(),
                }
            )
    return endpoints


def default_endpoint_id(flow: int, role: int = 0) -> Optional[str]:
    """Read the default endpoint id; returns None on failure. role: 0=Console, 1=Multimedia, 2=Communications."""
    ensure_core_audio_deps()
    from comtypes import CoCreateInstance
    from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
    from pycaw.constants import CLSID_MMDeviceEnumerator

    enumerator = CoCreateInstance(CLSID_MMDeviceEnumerator, interface=IMMDeviceEnumerator)
    try:
        return enumerator.GetDefaultAudioEndpoint(flow, role).GetId()
    except Exception:
        return None


def pick_endpoint(
    endpoints: Sequence[Dict],
    flow: int,
    candidates: Optional[Sequence[str]] = None,
) -> Optional[Dict]:
    """Pick an endpoint in the given direction by candidate name substring: prefer Active, then candidate order."""
    pool = [ep for ep in endpoints if ep["flow"] == flow]
    if not pool:
        return None
    for candidate in candidates or ():
        hits = [ep for ep in pool if candidate.lower() in ep["name"].lower()]
        if hits:
            active = [ep for ep in hits if ep["state"] == DEVICE_STATE_ACTIVE]
            return (active or hits)[0]
    if len(pool) == 1:
        return pool[0]
    active = [ep for ep in pool if ep["state"] == DEVICE_STATE_ACTIVE]
    return active[0] if len(active) == 1 else None


def describe_endpoint(endpoint: Dict) -> str:
    return "%s [%s, %s] %s" % (
        endpoint["name"],
        endpoint["flow_label"],
        endpoint["state_label"],
        endpoint["id"],
    )


# --------------------------------------------------------------------------- #
# Enabling (registry DeviceState)
# --------------------------------------------------------------------------- #
def endpoint_registry_path(flow: int, device_id: str) -> str:
    matches = _ENDPOINT_GUID_RE.findall(device_id)
    if not matches:
        raise ValueError("无法从端点 ID 解析 GUID：%s" % device_id)
    sub_key = "Render" if flow == FLOW_RENDER else "Capture"
    return "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\MMDevices\\Audio\\%s\\%s" % (
        sub_key,
        matches[-1],
    )


def read_device_state(registry_path: str) -> Optional[int]:
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
            value, _ = winreg.QueryValueEx(key, "DeviceState")
        return int(value)
    except OSError:
        return None


def write_device_state(registry_path: str, value: int = DEVICE_STATE_ACTIVE) -> None:
    if winreg is None:
        raise RuntimeError("winreg 不可用（当前不是 Windows 环境）")
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, registry_path, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, "DeviceState", 0, winreg.REG_DWORD, int(value))
    except PermissionError as exc:
        raise RuntimeError("写注册表被拒绝，需要管理员权限：%s" % registry_path) from exc


def _wait_state(device_id: str, want_state: int, timeout: float = 3.0) -> Optional[int]:
    """Poll the endpoint state until it equals want_state or times out; returns the last state read."""
    deadline = time.monotonic() + timeout
    state: Optional[int] = None
    while True:
        for endpoint in enumerate_endpoints(driver_name=None):
            if endpoint["id"] == device_id:
                state = endpoint["state"]
                break
        if state == want_state or time.monotonic() >= deadline:
            return state
        time.sleep(0.25)


def _restart_audio_services(log: Optional[Callable[[str], None]] = None) -> bool:
    """Fallback: restart the audio service to force re-enumeration when the state still hasn't changed after the registry write."""
    commands = (
        ("AudioEndpointBuilder", ["net", "stop", "AudioEndpointBuilder", "/y"]),
        ("AudioEndpointBuilder", ["net", "start", "AudioEndpointBuilder"]),
        ("Audiosrv", ["net", "start", "Audiosrv"]),
    )
    ok = True
    for service, command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=60, shell=False
            )
        except Exception as exc:  # noqa: BLE001 - fallback path only
            _log_to(log, "  重启音频服务 %s 失败：%s" % (service, exc))
            ok = False
            continue
        if completed.returncode != 0:
            _log_to(
                log,
                "  重启音频服务 %s 返回 %d：%s"
                % (service, completed.returncode, (completed.stdout or "").strip()[:200]),
            )
            ok = False
    return ok


def enable_cable_endpoints(
    driver_name: str = DEFAULT_DRIVER_NAME,
    refresh: str = "auto",
    log: Optional[Callable[[str], None]] = print,
) -> Dict:
    """Enable Disabled CABLE render / capture endpoints.

    Returns {"ok": bool, "enabled": [...], "already_active": [...], "failed": [...]};
    each entry is {name, id, flow_label, state_before, state_after, registry, note}.
    """
    endpoints = enumerate_endpoints(driver_name=driver_name)
    result: Dict = {
        "ok": False,
        "driver": driver_name,
        "enabled": [],
        "already_active": [],
        "failed": [],
    }
    if not endpoints:
        _log_to(
            log,
            "未枚举到匹配 %r 的音频端点：驱动可能未安装（见 downloads/VBCABLE_Driver_Pack.zip）"
            % driver_name,
        )
        return result

    admin = is_admin()
    for endpoint in endpoints:
        record = {
            "name": endpoint["name"],
            "id": endpoint["id"],
            "flow_label": endpoint["flow_label"],
            "state_before": endpoint["state_label"],
            "state_after": endpoint["state_label"],
            "registry": "",
            "note": "",
        }
        if endpoint["state"] == DEVICE_STATE_ACTIVE:
            result["already_active"].append(record)
            continue
        if endpoint["state"] != DEVICE_STATE_DISABLED:
            record["note"] = "状态 %s 无法通过注册表 DeviceState 启用（设备不存在/未插入）" % (
                endpoint["state_label"],
            )
            result["failed"].append(record)
            continue
        if not admin:
            record["note"] = "需要管理员权限（写 HKLM\\...\\MMDevices 注册表）"
            result["failed"].append(record)
            continue

        try:
            registry_path = endpoint_registry_path(endpoint["flow"], endpoint["id"])
            record["registry"] = registry_path
            record["registry_state_before"] = read_device_state(registry_path)
            write_device_state(registry_path, DEVICE_STATE_ACTIVE)
        except Exception as exc:  # noqa: BLE001 - aggregated for the caller
            record["note"] = str(exc)
            result["failed"].append(record)
            continue

        state_after = _wait_state(endpoint["id"], DEVICE_STATE_ACTIVE, timeout=3.0)
        if state_after != DEVICE_STATE_ACTIVE and refresh != "none":
            _log_to(log, "  %s 状态未变化，兜底重启音频服务" % endpoint["name"])
            if _restart_audio_services(log):
                state_after = _wait_state(endpoint["id"], DEVICE_STATE_ACTIVE, timeout=10.0)
        record["state_after"] = state_name(state_after or 0)
        if state_after == DEVICE_STATE_ACTIVE:
            result["enabled"].append(record)
        else:
            record["note"] = (record["note"] + "；" if record["note"] else "") + (
                "写入注册表后状态仍为 %s" % record["state_after"]
            )
            result["failed"].append(record)

    result["ok"] = not result["failed"]
    return result


# --------------------------------------------------------------------------- #
# Setting the default device
# --------------------------------------------------------------------------- #
def set_default_endpoint(endpoint_id: str, roles: Iterable[Tuple[int, str]] = ROLES) -> List[Dict]:
    """Set an endpoint as the default device via IPolicyConfig.SetDefaultEndpoint (all three roles)."""
    ensure_core_audio_deps()
    from comtypes import CoCreateInstance
    from pycaw.api.policyconfig import IPolicyConfig
    from pycaw.constants import CLSID_CPolicyConfigClient

    policy = CoCreateInstance(CLSID_CPolicyConfigClient, interface=IPolicyConfig)
    results = []
    for role, label in roles:
        record = {"role": label, "ok": False, "note": ""}
        try:
            return_code = policy.SetDefaultEndpoint(endpoint_id, role)
            record["rc"] = return_code
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001 - aggregated for the caller
            record["note"] = str(exc)
        results.append(record)
    return results


def set_default_capture_endpoint(
    driver_name: str = DEFAULT_DRIVER_NAME,
    capture_candidates: Optional[Sequence[str]] = None,
    log: Optional[Callable[[str], None]] = print,
) -> Dict:
    """Point the default recording device (Console/Multimedia/Communications roles) at the CABLE capture endpoint."""
    endpoints = enumerate_endpoints(driver_name=driver_name)
    target = pick_endpoint(
        endpoints, FLOW_CAPTURE, capture_candidates or DEFAULT_CAPTURE_CANDIDATES
    )
    result: Dict = {"ok": False, "target": None, "previous": {}, "roles": [], "note": ""}
    if target is None:
        result["note"] = "未找到匹配 %r 的 CABLE 捕获端点（候选名：%s）" % (
            driver_name,
            "、".join(capture_candidates or DEFAULT_CAPTURE_CANDIDATES),
        )
        _log_to(log, result["note"])
        return result
    result["target"] = target
    if target["state"] != DEVICE_STATE_ACTIVE:
        result["note"] = "CABLE 捕获端点处于 %s，请先运行 --enable" % target["state_label"]
        _log_to(log, result["note"])
        return result

    all_endpoints = enumerate_endpoints(driver_name=None)
    for role, label in ROLES:
        current_id = default_endpoint_id(FLOW_CAPTURE, role)
        result["previous"][label] = next(
            (ep["name"] for ep in all_endpoints if ep["id"] == current_id), current_id
        )

    result["roles"] = set_default_endpoint(target["id"])
    # Read-back verification (trust only what the enumerator reports)
    verified = {}
    for role, label in ROLES:
        verified[label] = default_endpoint_id(FLOW_CAPTURE, role) == target["id"]
    result["verified"] = verified
    result["ok"] = all(verified.values())
    if not result["ok"]:
        result["note"] = "部分角色未生效：%s" % (
            "、".join(label for label, ok in verified.items() if not ok),
        )
    return result


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - display_width(text))


def cmd_list(args: argparse.Namespace) -> int:
    endpoints = enumerate_endpoints(driver_name=None if args.all else args.driver_name)
    print("=== 端点列表（%s）" % ("全部驱动" if args.all else "驱动名含 %r" % args.driver_name))
    if not endpoints:
        print("  （没有匹配的端点）")
    else:
        default_ids = {flow: default_endpoint_id(flow) for flow in (FLOW_RENDER, FLOW_CAPTURE)}
        header = ("方向", "状态", "名称", "端点ID", "默认")
        rows = []
        for endpoint in sorted(endpoints, key=lambda ep: (ep["flow"], ep["name"])):
            rows.append(
                (
                    endpoint["flow_label"],
                    endpoint["state_label"],
                    endpoint["name"],
                    endpoint["id"],
                    "是" if default_ids.get(endpoint["flow"]) == endpoint["id"] else "",
                )
            )
        widths = [
            max(display_width(header[i]), max(display_width(row[i]) for row in rows))
            for i in range(len(header))
        ]
        print("  " + "  ".join(pad(header[i], widths[i]) for i in range(len(header))))
        for row in rows:
            print("  " + "  ".join(pad(row[i], widths[i]) for i in range(len(row))))

    print("=== 当前默认设备（角色 Console）")
    all_endpoints = enumerate_endpoints(driver_name=None)
    for flow in (FLOW_RENDER, FLOW_CAPTURE):
        device_id = default_endpoint_id(flow, 0)
        name = next(
            (ep["name"] for ep in all_endpoints if ep["id"] == device_id), "(未匹配)"
        )
        print("  %-8s = %s  %s" % (FLOW_NAMES[flow], name, device_id))
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    print("=== 启用 CABLE 端点（驱动名含 %r）" % args.driver_name)
    result = enable_cable_endpoints(
        driver_name=args.driver_name, refresh=args.refresh, log=None
    )
    for record in result["enabled"]:
        print(
            "  [已启用] %s  (%s)  %s -> %s"
            % (record["name"], record["flow_label"], record["state_before"], record["state_after"])
        )
        print("           启用后名称：%s" % record["name"])
        print("           注册表：HKLM\\%s  DeviceState=1" % record["registry"])
    for record in result["already_active"]:
        print("  [已在用] %s  (%s)" % (record["name"], record["flow_label"]))
    for record in result["failed"]:
        print(
            "  [失败]   %s  (%s)  %s：%s"
            % (record["name"], record["flow_label"], record["state_before"], record["note"])
        )
    if not result["enabled"] and not result["failed"]:
        print("  无需启用：CABLE 端点已经全部处于 Active")
    print(
        "小结：已启用 %d，已在用 %d，失败 %d"
        % (len(result["enabled"]), len(result["already_active"]), len(result["failed"]))
    )
    return 0 if result["ok"] else 1


def split_candidates(text: Optional[str]) -> Optional[List[str]]:
    """Split a comma-separated candidate-name list; empty string returns None (use the built-in default candidates)."""
    if not text:
        return None
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items or None


def cmd_set_default_mic(args: argparse.Namespace) -> int:
    candidates = split_candidates(args.capture_name)
    print("=== 设置默认录音设备 -> CABLE 捕获端点（驱动名含 %r）" % args.driver_name)
    result = set_default_capture_endpoint(
        driver_name=args.driver_name, capture_candidates=candidates, log=None
    )
    if not result["target"]:
        print("  [失败] %s" % result["note"])
        return 1
    target = result["target"]
    print("  目标端点：%s  %s" % (target["name"], target["id"]))
    if result["previous"]:
        for role, label in ROLES:
            print("  %-14s 原默认 = %s" % (label, result["previous"].get(label)))
    for record in result["roles"]:
        print("  SetDefaultEndpoint(%-14s) rc=%s %s" % (record["role"], record.get("rc"), record["note"]))
    verified = result.get("verified") or {}
    if verified:
        print(
            "  生效校验：%s"
            % "、".join("%s=%s" % (label, "是" if ok else "否") for label, ok in verified.items())
        )
    if result["note"]:
        print("  [失败] %s" % result["note"])
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devsetup",
        description="变声器 · VB-CABLE 端点准备（枚举 / 启用 / 设为默认录音设备）",
        epilog="示例：python tools/devsetup.py --list；--enable --set-default-mic 可同时执行。",
    )
    parser.add_argument("--list", action="store_true", help="列出 CABLE 端点（含 Disabled）")
    parser.add_argument("--enable", action="store_true", help="启用处于 Disabled 的 CABLE 端点（需管理员）")
    parser.add_argument(
        "--set-default-mic",
        dest="set_default_mic",
        action="store_true",
        help="把默认录音设备设为 CABLE 捕获端点（需管理员）",
    )
    parser.add_argument(
        "--driver-name",
        default=os.environ.get("BSP_CABLE_DRIVER", "").strip() or DEFAULT_DRIVER_NAME,
        help="CABLE 驱动名子串（默认 %(default)s，可用环境变量 BSP_CABLE_DRIVER 覆盖）",
    )
    parser.add_argument(
        "--capture-name",
        default=os.environ.get("BSP_CABLE_CAPTURE", "").strip(),
        help="CABLE 捕获端点名子串，逗号分隔按序匹配；留空则用默认候选 %s"
        % "、".join(DEFAULT_CAPTURE_CANDIDATES),
    )
    parser.add_argument("--all", action="store_true", help="--list 时列出全部端点，不按驱动名过滤")
    parser.add_argument(
        "--refresh",
        choices=("auto", "none", "service"),
        default="auto",
        help="注册表写入后状态未变化时的兜底动作（service=重启音频服务）",
    )
    parser.add_argument("--deps-dir", default=None, help="pycaw 所在目录（默认按 BSP_DEPS_DIR / 内置候选）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        ensure_core_audio_deps(args.deps_dir)
    except RuntimeError as exc:
        print("[错误] %s" % exc)
        return 2

    if not (args.list or args.enable or args.set_default_mic):
        parser.print_help()
        print("\n[提示] 至少要指定 --list / --enable / --set-default-mic 之一")
        return 2

    exit_code = 0
    if args.list:
        exit_code |= cmd_list(args)
    if args.enable:
        exit_code |= cmd_enable(args)
    if args.set_default_mic:
        exit_code |= cmd_set_default_mic(args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
