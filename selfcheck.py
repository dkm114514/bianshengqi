"""Voice changer silent-link self-check (module F): input mic, CABLE render
endpoint, render-to-capture loopback tone, default recording device. No UI;
no system-state changes by default (--fix enables CABLE endpoints and sets
the default mic first, then checks). Loopback results go to
out_dev/cable-loopback.json.

Usage: python selfcheck.py [--fix] [--mic NAME] [--hostapi "Windows WASAPI"]
Device names are configurable (CLI args win, env vars second):
  --mic/BSP_MIC, --cable-render/BSP_CABLE_RENDER,
  --cable-capture/BSP_CABLE_CAPTURE, --driver-name/BSP_CABLE_DRIVER,
  --deps-dir/BSP_DEPS_DIR
Exit codes: 0 = all pass, 1 = some failed, 2 = environment/dependency problem.
Bluetooth headsets are excluded from loopback; loopback always uses VB-CABLE.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import devsetup  # noqa: E402  (needs the sys.path tweak above)

try:
    import numpy as np
    import sounddevice as sd
except ImportError:  # main() reports a readable error below
    np = None  # type: ignore
    sd = None  # type: ignore

DEFAULT_MIC = "麦克风 (USBAudio1.0)"
DEFAULT_CABLE_RENDER = ",".join(devsetup.DEFAULT_RENDER_CANDIDATES)
DEFAULT_CABLE_CAPTURE = ",".join(devsetup.DEFAULT_CAPTURE_CANDIDATES)
DEFAULT_HOSTAPI = "Windows WASAPI"
DEFAULT_OUT_JSON = PROJECT_ROOT / "out_dev" / "cable-loopback.json"

PASS, FAIL, SKIP = "通过", "失败", "跳过"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def split_candidates(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-12))


def endpoint_volume_hint(endpoint_id: str) -> str:
    """Best-effort read of the endpoint's software mute/volume as a diagnostic hint on check-1 failure; returns "" when unreadable."""
    try:
        devsetup.ensure_core_audio_deps()
        from pycaw.pycaw import AudioUtilities

        with warnings.catch_warnings():  # pycaw warns when reading ghost-endpoint properties; silence it here
            warnings.simplefilter("ignore")
            devices = list(AudioUtilities.GetAllDevices())
        for device in devices:
            if device.id != endpoint_id:
                continue
            volume = device.EndpointVolume
            return "端点软件状态：静音=%s 音量=%.0f%%（%.1f dB）" % (
                "是" if volume.GetMute() else "否",
                volume.GetMasterVolumeLevelScalar() * 100.0,
                volume.GetMasterVolumeLevel(),
            )
    except Exception:  # noqa: BLE001 - diagnostic hint only, skip when unreadable
        pass
    return ""


# --------------------------------------------------------------------------- #
# sounddevice device lookup and measurement
# --------------------------------------------------------------------------- #
def find_sd_device(
    candidates: Sequence[str],
    hostapi_substr: str,
    want_input: bool = False,
    want_output: bool = False,
) -> Tuple[Optional[int], Optional[dict], str]:
    """Find a device in the given hostapi by candidate name substring; returns (index, device_info, api_name)."""
    hostapis = sd.query_hostapis()
    for candidate in candidates:
        for index, device in enumerate(sd.query_devices()):
            api_name = hostapis[device["hostapi"]]["name"]
            if hostapi_substr.lower() not in api_name.lower():
                continue
            if candidate.lower() not in device["name"].lower():
                continue
            if want_input and device["max_input_channels"] < 1:
                continue
            if want_output and device["max_output_channels"] < 1:
                continue
            return index, device, api_name
    return None, None, ""


def capture_input(
    index: int, samplerate: int, seconds: float, channels: int = 1
) -> Tuple["np.ndarray", List[str]]:
    blocks: List["np.ndarray"] = []
    statuses: List[str] = []

    def callback(indata, frames, time_info, status):
        blocks.append(indata.copy())
        if status:
            statuses.append(str(status))

    with sd.InputStream(
        device=index, channels=channels, samplerate=samplerate, dtype="float32", callback=callback
    ):
        sd.sleep(int(round(seconds * 1000)))
    data = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.zeros((0, channels), dtype="float32")
    )
    return data, statuses


def block_rms_db(mono: "np.ndarray", samplerate: int, block_seconds: float = 0.1) -> List[float]:
    block = max(1, int(samplerate * block_seconds))
    values = []
    for start in range(0, len(mono), block):
        chunk = mono[start : start + block]
        if len(chunk) == 0:
            continue
        values.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))
    return values


def measure_loopback(
    render_index: int,
    capture_index: int,
    samplerate: int,
    render_channels: int,
    capture_channels: int,
    freq: float,
    level: float,
    preroll: float,
    seconds: float,
) -> Dict:
    """Feed the test tone into CABLE render while grabbing the CABLE capture endpoint; returns loopback level metrics."""
    start_sample = int(round(preroll * samplerate))
    floor_rms: List[float] = []
    tone_rms: List[float] = []
    statuses: List[str] = []
    phase_index = 0
    started = time.monotonic()
    total = preroll + seconds

    def callback(indata, outdata, frames, time_info, status):
        nonlocal phase_index
        first, last = phase_index, phase_index + frames
        positions = np.arange(first, last, dtype=np.float64)
        wave = (level * np.sin(2.0 * np.pi * freq * positions / samplerate)).astype(np.float32)
        wave[positions < start_sample] = 0.0
        outdata[:] = wave[:, None]
        phase_index = last
        if status:
            statuses.append(str(status))
        value = float(np.sqrt(np.mean(np.square(indata.astype(np.float64)))))
        (floor_rms if last <= start_sample else tone_rms).append(value)
        if time.monotonic() - started >= total:
            raise sd.CallbackStop

    stream = sd.Stream(
        device=(capture_index, render_index),
        samplerate=samplerate,
        channels=(capture_channels, render_channels),
        dtype="float32",
        callback=callback,
    )
    with stream:
        sd.sleep(int(round((total + 0.7) * 1000)))

    tone_mean = float(np.mean(tone_rms)) if tone_rms else 0.0
    tone_max = float(np.max(tone_rms)) if tone_rms else 0.0
    floor_mean = float(np.mean(floor_rms)) if floor_rms else 0.0
    # Digital silence (all zeros) computes to -240 dBFS; clamp the display to -120 so the margin stays readable
    floor_for_margin = max(dbfs(floor_mean), -120.0)
    return {
        "expected_tone_dbfs": dbfs(level / math.sqrt(2.0)),
        "loopback_rms_dbfs": dbfs(tone_mean),
        "loopback_block_peak_dbfs": dbfs(tone_max),
        "noise_floor_dbfs": floor_for_margin,
        "noise_floor_is_digital_silence": floor_mean == 0.0,
        "margin_db": dbfs(tone_mean) - floor_for_margin,
        "tone_blocks": len(tone_rms),
        "floor_blocks": len(floor_rms),
        "callback_status": sorted(set(statuses)),
    }


# --------------------------------------------------------------------------- #
# The four checks
# --------------------------------------------------------------------------- #
def step1_input_mic(args: argparse.Namespace) -> Dict:
    result = {"key": "①", "title": "输入麦克风", "status": SKIP, "lines": [], "data": {}}
    candidates = split_candidates(args.mic)
    endpoints = devsetup.enumerate_endpoints(driver_name=None)
    endpoint = devsetup.pick_endpoint(endpoints, devsetup.FLOW_CAPTURE, candidates)
    if endpoint is None:
        result["status"] = FAIL
        result["lines"].append("未枚举到名称含 %s 的录音端点（用 --mic 指定）" % "、".join(candidates))
        return result
    result["data"]["endpoint"] = endpoint
    result["lines"].append("端点：%s" % devsetup.describe_endpoint(endpoint))
    if endpoint["state"] != devsetup.DEVICE_STATE_ACTIVE:
        result["status"] = FAIL
        result["lines"].append("端点状态是 %s，不是 Active" % endpoint["state_label"])
        return result

    index, device, api_name = find_sd_device(candidates, args.hostapi, want_input=True)
    if index is None:
        result["status"] = FAIL
        result["lines"].append("sounddevice 在 %s 下找不到该设备" % args.hostapi)
        return result
    channels = max(1, min(2, int(device["max_input_channels"])))
    result["lines"].append(
        "抓取：%s index=%d  %s  x%dch @%d Hz"
        % (api_name, index, device["name"], channels, args.samplerate)
    )
    try:
        data, statuses = capture_input(index, args.samplerate, args.mic_seconds, channels)
    except Exception as exc:  # noqa: BLE001 - the self-check must surface the error
        result["status"] = FAIL
        result["lines"].append("打开输入流失败：%s" % exc)
        return result

    mono = data.mean(axis=1) if data.ndim == 2 and data.shape[1] > 1 else data.reshape(-1)
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2))) if mono.size else 0.0
    blocks = block_rms_db(mono, args.samplerate)
    block_max = max(blocks) if blocks else 0.0
    loud_blocks = sum(1 for value in blocks if value > 0 and dbfs(value) > args.silence_db)
    result["data"].update(
        {
            "device_index": index,
            "device_name": device["name"],
            "hostapi": api_name,
            "channels": channels,
            "seconds": args.mic_seconds,
            "rms_dbfs": dbfs(rms),
            "block_max_dbfs": dbfs(block_max),
            "loud_blocks": loud_blocks,
            "loud_blocks_required": 3,
            "threshold_dbfs": args.silence_db,
            "callback_status": sorted(set(statuses)),
        }
    )
    result["lines"].append(
        "实测：%.1fs  RMS %.1f dBFS，100ms 块最大 %.1f dBFS（阈值 %.1f dBFS），"
        "%d 块中有 %d 块高于阈值"
        % (args.mic_seconds, dbfs(rms), dbfs(block_max), args.silence_db,
           len(blocks), loud_blocks)
    )
    if loud_blocks >= 3:
        result["status"] = PASS
        result["lines"].append("结论：麦收到有效电平")
    else:
        result["status"] = FAIL
        result["lines"].append("结论：3 秒内有效电平不足 3 个块 —— 麦没收音（检查 USB 麦/系统录音权限/是否被独占；房间安静时对麦说话再测）")
        hint = endpoint_volume_hint(endpoint["id"])
        if hint:
            result["lines"].append(hint)
    return result


def step2_cable_render(args: argparse.Namespace) -> Dict:
    result = {"key": "②", "title": "CABLE render 端点", "status": SKIP, "lines": [], "data": {}}
    candidates = split_candidates(args.cable_render)
    endpoints = devsetup.enumerate_endpoints(driver_name=args.driver_name)
    endpoint = devsetup.pick_endpoint(endpoints, devsetup.FLOW_RENDER, candidates)
    if endpoint is None:
        result["status"] = FAIL
        result["lines"].append(
            "未找到匹配 %r 且名称含 %s 的渲染端点" % (args.driver_name, "、".join(candidates))
        )
        return result
    result["data"]["endpoint"] = endpoint
    result["lines"].append("端点：%s" % devsetup.describe_endpoint(endpoint))
    if endpoint["state"] != devsetup.DEVICE_STATE_ACTIVE:
        result["status"] = FAIL
        result["lines"].append(
            "端点处于 %s：请运行 tools/devsetup.py --enable（或 selfcheck.py --fix）"
            % endpoint["state_label"]
        )
        return result

    index, device, api_name = find_sd_device(candidates, args.hostapi, want_output=True)
    if index is None:
        result["status"] = FAIL
        result["lines"].append("sounddevice 在 %s 下找不到该渲染端点" % args.hostapi)
        return result
    channels = max(1, min(2, int(device["max_output_channels"])))
    try:
        with sd.OutputStream(
            device=index, channels=channels, samplerate=args.samplerate, dtype="float32"
        ):
            sd.sleep(300)
    except Exception as exc:  # noqa: BLE001
        result["status"] = FAIL
        result["lines"].append("%d Hz 输出流打开失败：%s" % (args.samplerate, exc))
        return result

    latency_ms = float(device.get("default_low_output_latency", 0.0) or 0.0) * 1000.0
    result["data"].update(
        {
            "device_index": index,
            "device_name": device["name"],
            "hostapi": api_name,
            "channels": channels,
            "samplerate": args.samplerate,
            "low_latency_ms": latency_ms,
        }
    )
    result["status"] = PASS
    result["lines"].append(
        "实测：%s index=%d，%d Hz x%dch 输出流打开成功（默认低延迟 %.1f ms）"
        % (api_name, index, args.samplerate, channels, latency_ms)
    )
    return result


def step3_loopback(args: argparse.Namespace, step2: Dict) -> Dict:
    result = {
        "key": "③",
        "title": "CABLE 环回（render -> 捕获端点）",
        "status": SKIP,
        "lines": [],
        "data": {},
    }
    payload: Dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "check": "cable-loopback",
        "hostapi": args.hostapi,
        "samplerate": args.samplerate,
        "tone": {
            "freq_hz": args.tone_freq,
            "amplitude": args.tone_level,
            "seconds": args.tone_seconds,
            "preroll_seconds": args.tone_preroll,
        },
        "passed": False,
        "reason": "",
    }
    out_path = Path(args.out_json)

    render_index, render_device, render_api = find_sd_device(
        split_candidates(args.cable_render), args.hostapi, want_output=True
    )
    capture_index, capture_device, capture_api = find_sd_device(
        split_candidates(args.cable_capture), args.hostapi, want_input=True
    )
    if render_index is None or capture_index is None:
        result["status"] = FAIL
        result["lines"].append(
            "环回设备缺失：render=%s capture=%s（检查 CABLE 端点是否启用）"
            % (render_device and render_device["name"], capture_device and capture_device["name"])
        )
        payload["reason"] = "cable endpoints not found in %s" % args.hostapi
        payload["render_device"] = render_device and render_device["name"]
        payload["capture_device"] = capture_device and capture_device["name"]
        payload["passed"] = False
        payload["written_from"] = "selfcheck.py"
        _write_json(out_path, payload)
        result["data"]["json"] = str(out_path)
        result["lines"].append("结果写入：%s" % out_path)
        return result
    if step2.get("status") != PASS:
        result["lines"].append("提示：② 未通过，环回结果仅供参考")

    render_channels = max(1, min(2, int(render_device["max_output_channels"])))
    capture_channels = max(1, min(2, int(capture_device["max_input_channels"])))
    result["lines"].append(
        "渲染：%s index=%d（%s）" % (render_device["name"], render_index, render_api)
    )
    result["lines"].append(
        "捕获：%s index=%d（%s）" % (capture_device["name"], capture_index, capture_api)
    )
    result["lines"].append(
        "注入：%.0f Hz 幅度 %.3f（%.1f dBFS）%.1fs，前置静音 %.1fs"
        % (
            args.tone_freq,
            args.tone_level,
            dbfs(args.tone_level / math.sqrt(2.0)),
            args.tone_seconds,
            args.tone_preroll,
        )
    )
    try:
        metrics = measure_loopback(
            render_index=render_index,
            capture_index=capture_index,
            samplerate=args.samplerate,
            render_channels=render_channels,
            capture_channels=capture_channels,
            freq=args.tone_freq,
            level=args.tone_level,
            preroll=args.tone_preroll,
            seconds=args.tone_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        result["status"] = FAIL
        result["lines"].append("环回测量失败：%s" % exc)
        payload.update(
            {
                "render_device": {"index": render_index, "name": render_device["name"], "api": render_api},
                "capture_device": {
                    "index": capture_index,
                    "name": capture_device["name"],
                    "api": capture_api,
                },
                "error": str(exc),
                "written_from": "selfcheck.py",
            }
        )
        _write_json(out_path, payload)
        result["data"]["json"] = str(out_path)
        result["lines"].append("结果写入：%s" % out_path)
        return result

    passed = (
        metrics["loopback_rms_dbfs"] >= args.loopback_db
        and metrics["margin_db"] >= 6.0
        and metrics["tone_blocks"] > 0
    )
    result["data"].update(metrics)
    result["data"]["json"] = str(out_path)
    result["lines"].append(
        "实测：环回 RMS %.1f dBFS，块峰 %.1f dBFS；本底 %.1f dBFS；余量 %.1f dB（阈值 %.1f dBFS）"
        % (
            metrics["loopback_rms_dbfs"],
            metrics["loopback_block_peak_dbfs"],
            metrics["noise_floor_dbfs"],
            metrics["margin_db"],
            args.loopback_db,
        )
    )
    result["status"] = PASS if passed else FAIL
    if not passed:
        result["lines"].append(
            "结论：CABLE render -> CABLE Output 环回不通（声音没到捕获端点，或电平过低）"
        )
    payload.update(
        {
            "render_device": {
                "index": render_index,
                "name": render_device["name"],
                "api": render_api,
                "channels": render_channels,
            },
            "capture_device": {
                "index": capture_index,
                "name": capture_device["name"],
                "api": capture_api,
                "channels": capture_channels,
            },
            "metrics": metrics,
            "threshold_dbfs": args.loopback_db,
            "passed": bool(passed),
            "reason": "" if passed else "loopback level below threshold",
            "written_from": "selfcheck.py",
        }
    )
    _write_json(out_path, payload)
    result["lines"].append("结果写入：%s" % out_path)
    return result


def step4_default_mic(args: argparse.Namespace) -> Dict:
    result = {"key": "④", "title": "默认录音设备", "status": SKIP, "lines": [], "data": {}}
    endpoints = devsetup.enumerate_endpoints(driver_name=args.driver_name)
    target = devsetup.pick_endpoint(
        endpoints, devsetup.FLOW_CAPTURE, split_candidates(args.cable_capture)
    )
    if target is None:
        result["status"] = FAIL
        result["lines"].append("找不到 CABLE 捕获端点，无法比较默认录音设备")
        return result
    checks = {}
    names = {}
    for role, label in devsetup.ROLES:
        current_id = devsetup.default_endpoint_id(devsetup.FLOW_CAPTURE, role)
        checks[label] = current_id == target["id"]
        names[label] = next(
            (ep["name"] for ep in devsetup.enumerate_endpoints(driver_name=None) if ep["id"] == current_id),
            current_id,
        )
    result["data"].update({"target": target, "verified": checks, "current": names})
    result["lines"].append("当前(Console)：%s" % names.get("Console"))
    result["lines"].append("期望：CABLE 捕获端点 %s" % target["name"])
    result["lines"].append(
        "三角色：%s"
        % "  ".join("%s=%s" % (label, "是" if ok else "否") for label, ok in checks.items())
    )
    if all(checks.values()):
        result["status"] = PASS
    else:
        result["status"] = FAIL
        result["lines"].append(
            "结论：默认录音设备不是 CABLE 捕获端点（未生效角色：%s）—— "
            "运行 tools/devsetup.py --set-default-mic（或 --fix）"
            % "、".join(label for label, ok in checks.items() if not ok)
        )
    return result


# --------------------------------------------------------------------------- #
# Reporting and fix
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fix(args: argparse.Namespace) -> None:
    print("=== --fix：先跑 devsetup（enable + set-default-mic）")
    enable = devsetup.enable_cable_endpoints(driver_name=args.driver_name, log=None)
    print(
        "  enable：已启用 %d / 已在用 %d / 失败 %d"
        % (len(enable["enabled"]), len(enable["already_active"]), len(enable["failed"]))
    )
    for record in enable["enabled"]:
        print("    [已启用] %s (%s) %s -> %s" % (record["name"], record["flow_label"], record["state_before"], record["state_after"]))
    for record in enable["failed"]:
        print("    [失败] %s (%s)：%s" % (record["name"], record["flow_label"], record["note"]))
    default = devsetup.set_default_capture_endpoint(driver_name=args.driver_name, log=None)
    if default["target"]:
        print("  set-default-mic：目标 = %s" % default["target"]["name"])
    verified = default.get("verified") or {}
    if verified:
        print(
            "    生效校验：%s"
            % "  ".join(
                "%s=%s" % (label, "是" if ok else "否") for label, ok in verified.items()
            )
        )
    if default.get("note"):
        print("  set-default-mic 失败注记：%s" % default["note"])


def print_report(results: Sequence[Dict]) -> int:
    print("")
    print("=== 变声器自检结论 ===")
    for result in results:
        print("%s %s：%s" % (result["key"], result["title"], result["status"]))
        for line in result["lines"]:
            print("     - %s" % line)
    passed = sum(1 for r in results if r["status"] == PASS)
    failed = [r for r in results if r["status"] == FAIL]
    print("--- 合计 %d/%d 通过" % (passed, len(results)))
    if failed:
        print("--- 失败项：%s" % "、".join("%s %s" % (r["key"], r["title"]) for r in failed))
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="selfcheck",
        description="变声器 · 静默链路自检（麦克风 / CABLE 端点 / 环回 / 默认录音设备）",
        epilog="示例：python selfcheck.py；python selfcheck.py --fix。蓝牙耳机不参与环回。",
    )
    parser.add_argument("--fix", action="store_true", help="先自动启用 CABLE 端点并设默认录音设备，再自检")
    parser.add_argument("--mic", default=_env("BSP_MIC", DEFAULT_MIC), help="输入麦克风名子串，逗号分隔候选（默认 %(default)s）")
    parser.add_argument(
        "--cable-render",
        default=_env("BSP_CABLE_RENDER", DEFAULT_CABLE_RENDER),
        help="CABLE 渲染端点名子串（默认 %(default)s）",
    )
    parser.add_argument(
        "--cable-capture",
        default=_env("BSP_CABLE_CAPTURE", DEFAULT_CABLE_CAPTURE),
        help="CABLE 捕获端点名子串（默认 %(default)s）",
    )
    parser.add_argument(
        "--driver-name",
        default=_env("BSP_CABLE_DRIVER", devsetup.DEFAULT_DRIVER_NAME),
        help="CABLE 驱动名子串（默认 %(default)s）",
    )
    parser.add_argument("--hostapi", default=DEFAULT_HOSTAPI, help="使用的 host API（默认 %(default)s）")
    parser.add_argument("--samplerate", type=int, default=48000, help="采样率（默认 %(default)s）")
    parser.add_argument("--mic-seconds", type=float, default=3.0, help="① 抓麦时长秒（默认 %(default)s）")
    parser.add_argument("--silence-db", type=float, default=-60.0, help="① 判“麦没收音”的块最大 RMS 阈值 dBFS")
    parser.add_argument("--tone-seconds", type=float, default=5.0, help="③ 注入测试音时长秒（默认 %(default)s）")
    parser.add_argument("--tone-freq", type=float, default=440.0, help="③ 测试音频率 Hz（默认 %(default)s）")
    parser.add_argument("--tone-level", type=float, default=0.05, help="③ 测试音幅度 0~1（默认 %(default)s，低电平免打扰）")
    parser.add_argument("--tone-preroll", type=float, default=0.5, help="③ 测试音前置静音秒，用于测本底（默认 %(default)s）")
    parser.add_argument("--loopback-db", type=float, default=-50.0, help="③ 环回 RMS 通过阈值 dBFS（默认 %(default)s）")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON), help="③ 结果 JSON 路径（默认 %(default)s）")
    parser.add_argument("--deps-dir", default=None, help="pycaw 所在目录（默认按 BSP_DEPS_DIR / 内置候选）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if np is None or sd is None:
        print("[错误] 缺少 sounddevice / numpy：请用 RVC 自带 runtime 运行本脚本")
        return 2
    try:
        devsetup.ensure_core_audio_deps(args.deps_dir)
    except RuntimeError as exc:
        print("[错误] %s" % exc)
        return 2

    print("=== 变声器静默链路自检（host API = %s，%d Hz）" % (args.hostapi, args.samplerate))
    if args.fix:
        run_fix(args)

    results = [
        step1_input_mic(args),
        step2_cable_render(args),
    ]
    results.append(step3_loopback(args, results[1]))
    results.append(step4_default_mic(args))
    exit_code = print_report(results)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
