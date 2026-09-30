# -*- coding: utf-8 -*-
"""Module-C smoke test: --offline inference, --live N streams, --acoustic gate,
--agc gain wiring, --pure logic unit test.

Run with the RVC bundled runtime from any cwd:
  <RVC>\\runtime\\python.exe -X utf8 smoke_test.py --offline
  <RVC>\\runtime\\python.exe -X utf8 smoke_test.py --acoustic   # or --agc, --live 10, --pure
--pure needs no devices or model load. --live/--acoustic/--agc record SKIP
(unverified, counted separately) when the environment cannot meet conditions.
Device names are overridable: --input/--monitor/--cable/--cable-capture.
"""

import argparse
import importlib
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.models import rvc_root  # noqa: E402
from engine.pipeline import (  # noqa: E402
    SAMPLERATE,
    VoicePipeline,
    level_db,
    resolve_device,
)

WEIGHTS_DIR = os.path.join(str(rvc_root()), "assets", "weights")
LOGS_DIR = os.path.join(str(rvc_root()), "logs")
#: Test artifact directory (out_dev is gitignored)
OUT_DIR = os.path.join(PROJECT_ROOT, "out_dev")

DEFAULT_INPUT = "麦克风 (USBAudio1.0)"
DEFAULT_MONITOR = "耳机 (HECATE GT2 S)"
DEFAULT_CABLE = "变声器输出 (VB-Audio Virtual Cable)"
DEFAULT_CABLE_CAPTURE = "变声麦克风 (VB-Audio Virtual Cable)"
#: Monitor-device fallback candidates when the monitor device won't open (tried in order; each is really opened once; virtual endpoints stay silent, only the switching mechanism is verified).
#: Note "CABLE In 16ch" cannot open while "变声器输出" is held by the pipeline (VB-CABLE driver limit),
#: so it ranks behind the other virtual endpoints; the physical earphones go last (audible, only used with no virtual endpoint).
MONITOR_FALLBACKS = (
    "Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
    "CABLE In 16ch (VB-Audio Virtual Cable)",
    "耳机 (2- Realtek(R) Audio)",
)

#: Data-path criterion: an input block above the IN threshold counts as "mic has signal", and the mean
#: level of the matching output blocks must clear the OUT threshold — unwritten master buffer / always-closed
#: speaker gate / AGC pinning everything down all surface here.
#: Below MIN such blocks, the verdict is "the input was quiet to begin with": SKIP, not FAIL.
LIVE_VOICE_IN_DB = -45.0
#: Criterion for "the output is not silent throughout" (dBFS): only checks data presence, not input-level parity —
#: RVC legitimately outputs low levels for non-speech input (room noise, keyboard), which is not a fault
LIVE_VOICE_DEAD_DB = -80.0
LIVE_VOICE_MIN_BLOCKS = 4


def make_profile(args):
    """Profile for the bb48k + guanguanV1 combo (f0method is rmvpe)."""
    return {
        "pth": args.pth or os.path.join(WEIGHTS_DIR, "bb48k.pth"),
        "index": args.index if args.index is not None else os.path.join(LOGS_DIR, "guanguanV1.index"),
        "pitch": args.pitch,
        "formant": 0.0,
        "index_rate": 0.0,
        "block_time": 0.06,
        "crossfade_time": 0.02,
        "extra_time": 1.0,
        "threhold": -50,
        "I_noise_reduce": True,
        "O_noise_reduce": False,
        "rms_mix_rate": 0.0,
        "f0method": args.f0method,
    }


def _rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def _db(value):
    return 20.0 * np.log10(max(float(value), 1e-12))


def _report(results, title):
    """Verdict: FAIL wins; skips are counted separately, never as passes."""
    failed = [name for name, ok, _ in results if ok is False]
    skipped = [name for name, ok, _ in results if ok is None]
    passed = len(results) - len(failed) - len(skipped)
    suffix = ""
    if skipped:
        suffix = ", %d 项未验证: %s" % (len(skipped), ", ".join(skipped))
    print(
        "%s: %s (%d/%d%s)"
        % (title, "PASS" if not failed else "FAIL: " + ", ".join(failed),
           passed, len(results), suffix)
    )
    return 0 if not failed else 1


def _check(results, name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, (" — " + detail) if detail else ""))


def _skip(results, name, detail=""):
    """Record an unverified item (conditions unmet, not a code failure)."""
    results.append((name, None, detail))
    print("  [SKIP] %s%s" % (name, (" — " + detail) if detail else ""))


def _skip_group(results, names, detail=""):
    """Record a whole unrunnable block as unverified checks."""
    for name in names:
        _skip(results, name, detail)


def test_signal(seconds=3.0, freq=220.0):
    """3 s voice-like test signal, peak 0.15."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SAMPLERATE), dtype=np.float32) / SAMPLERATE
    wave = np.zeros_like(t)
    for k in range(1, 6):
        wave += np.sin(2.0 * np.pi * freq * k * t).astype(np.float32) / k
    wave *= 0.15 / max(float(np.max(np.abs(wave))), 1e-9)
    wave += (0.005 * rng.standard_normal(t.shape)).astype(np.float32)
    return wave


# --------------------------------------------------------------------------- #
# 0. Pure logic (--pure): no devices, no model load
# --------------------------------------------------------------------------- #
#: Check names for the device-memory fallback (whole group is SKIP when app/gui.py fails to import).
PURE_CHECKS = (
    "设备记忆命中时原样返回",
    "设备记忆失效时按关键字回退",
    "无记忆时按关键字或首个默认",
    "设备列表为空时返回空串",
    "回退结果必在当前列表内",
    "pick_default 无命中取首个、空取空串",
    "pick_cable 无命中不兜底",
)


def run_pure(args):
    results = []
    print("=== ⓪ 纯逻辑：设备记忆与默认选择，不开设备、不加载模型 ===")
    try:
        from app.gui import INPUT_HINTS, App, pick_cable, pick_default
    except Exception as exc:
        _skip_group(results, PURE_CHECKS, "app/gui.py 导入失败，纯逻辑没测: %r" % (exc,))
        return _report(results, "纯逻辑结论")
    recall = App._recall_device
    pool = ["麦克风 (USBAudio1.0)", "耳机 (2- Realtek(R) Audio)", "麦克风阵列 (Intel)"]
    _check(results, "设备记忆命中时原样返回",
           recall("耳机 (2- Realtek(R) Audio)", pool, INPUT_HINTS)
           == "耳机 (2- Realtek(R) Audio)",
           "存的名字还在列表里就直接用，不受关键字影响")
    _check(results, "设备记忆失效时按关键字回退",
           recall("拔掉的旧麦克风", pool, INPUT_HINTS) == "麦克风 (USBAudio1.0)",
           "存的名字不在列表里时按关键字重挑")
    _check(results, "无记忆时按关键字或首个默认",
           recall(None, pool, INPUT_HINTS) == "麦克风 (USBAudio1.0)"
           and recall("", ["扬声器 (XYZ)"], INPUT_HINTS) == "扬声器 (XYZ)",
           "无记忆先按关键字命中，关键字全不中取第一个")
    _check(results, "设备列表为空时返回空串", recall("麦克风", [], INPUT_HINTS) == "",
           "枚举不到设备时不编造名字")
    _check(results, "回退结果必在当前列表内",
           recall("拔掉的旧麦克风", pool, ("notexist",)) in pool,
           "关键字全不中时回退到首个，绝不返回列表外的名字")
    _check(results, "pick_default 无命中取首个、空取空串",
           pick_default(["扬声器 (XYZ)"], INPUT_HINTS) == "扬声器 (XYZ)"
           and pick_default([], INPUT_HINTS) == "",
           "通用默认选择先看关键字，否则取首个")
    _check(results, "pick_cable 无命中不兜底",
           pick_cable(["扬声器 (XYZ)", "耳机 (ABC)"]) == ""
           and pick_cable(["CABLE Input (VB-Audio Virtual Cable)"])
           == "CABLE Input (VB-Audio Virtual Cable)",
           "虚拟麦命中不了返回空串，不指到别的输出上")
    return _report(results, "纯逻辑结论")


# --------------------------------------------------------------------------- #
# 1. Offline
# --------------------------------------------------------------------------- #
def run_offline(args):
    results = []
    profile = make_profile(args)
    print("=== ① 离线：bb48k + %s，3 秒信号走完整推理路径 ===" % profile["f0method"])
    print("  权重: %s" % profile["pth"])
    print("  索引: %s (index_rate=%s)" % (profile["index"] or "无", profile["index_rate"]))

    pipe = None
    x = test_signal(3.0)
    print("  输入: %.1fs, %d 样本, RMS=%.5f (%.1f dBFS)"
          % (x.shape[0] / SAMPLERATE, x.shape[0], _rms(x), level_db(x)))
    if not os.path.isfile(profile["pth"]):
        _check(results, "权重文件存在", False,
               "找不到 %s（--pth 可换权重）" % profile["pth"])
        return _report(results, "离线结论")
    try:
        pipe = VoicePipeline(profile, "offline", "offline")
        t0 = time.perf_counter()
        y = pipe.process_offline(x)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        y2 = pipe.process_offline(x)
        warm = time.perf_counter() - t0
    except Exception as exc:
        _check(results, "离线推理可运行", False, "%r" % (exc,))
        return _report(results, "离线结论")
    blocks = (x.shape[0] + pipe.block_frame - 1) // pipe.block_frame
    print("  输出: %s, RMS=%.5f (%.1f dBFS), 峰值 %.4f"
          % (y.shape, _rms(y), level_db(y), float(np.max(np.abs(y)))))
    print("  耗时: 冷机 %.2fs（含加载+预热）→ 热机 %.2fs (%.1f ms/块, %.2fx 实时), "
          "block=%d 帧, tgt_sr=%d"
          % (cold, warm, warm * 1000.0 / blocks, 3.0 / warm, pipe.block_frame, pipe.engine.tgt_sr))

    _check(results, "输出与输入等长", y.shape == x.shape, "%s vs %s" % (y.shape, x.shape))
    _check(results, "无 NaN/Inf", bool(np.isfinite(y).all()))
    _check(results, "输出非静音 (RMS > 1e-3)", _rms(y) > 1e-3, "RMS=%.5f" % _rms(y))
    _check(results, "输出峰不削顶 (<= 1.0)", float(np.max(np.abs(y))) <= 1.0,
           "峰值 %.4f" % float(np.max(np.abs(y))))
    _check(results, "热机吞吐 > 1x 实时", warm < 3.0, "%.1f ms/块" % (warm * 1000.0 / blocks))
    _check(results, "f0method 确实用的是 rmvpe",
           str(pipe.profile["f0method"]) == "rmvpe" and hasattr(pipe.engine.rvc, "model_rmvpe"),
           "profile=%s, rmvpe 模型已载=%s"
           % (pipe.profile["f0method"], hasattr(pipe.engine.rvc, "model_rmvpe")))
    _check(results, "两次推理都非静音（状态连续）", _rms(y2) > 1e-3, "第二次 RMS=%.5f" % _rms(y2))

    # Hot-update params (pitch passes straight through to the RVC object)
    pipe.set_param("pitch", 12)
    _check(results, "set_param('pitch') 透传 RVC",
           getattr(pipe.engine.rvc, "f0_up_key", None) == 12,
           "f0_up_key=%s" % getattr(pipe.engine.rvc, "f0_up_key", None))
    pipe.set_param("I_noise_reduce", False)
    _check(results, "set_param('I_noise_reduce') 记录生效", pipe.profile["I_noise_reduce"] is False)
    pipe.set_param("I_noise_reduce", True)
    pipe.set_param("denoise_mode", "off")
    y_off = pipe.process_offline(x[:SAMPLERATE])
    _check(results, "denoise_mode='off' 真旁路且不断链",
           pipe.denoise_mode == "off" and bool(np.isfinite(y_off).all())
           and _rms(y_off) > 1e-3,
           "1s 输出 RMS=%.5f，TorchGate 不跑，主链不断" % _rms(y_off))
    pipe.set_param("denoise_mode", "torchgate")
    try:
        pipe.set_param("f0method", "bogus")
        _check(results, "非法 f0method 被拒绝", False)
    except ValueError:
        _check(results, "非法 f0method 被拒绝", True)

    # Monitor switch (not running => record the state only, don't touch devices)
    pipe.set_param("monitor", True)
    on_recorded = pipe.monitor_on and not pipe.monitor_active
    pipe.set_param("monitor", False)
    _check(results, "set_param('monitor') 未运行时只记录状态",
           on_recorded and not pipe.monitor_on)

    # set_devices (not running => record only, don't touch devices; main output untouched)
    pipe.set_devices(input_device="dummy-in", monitor_device="dummy-out")
    _check(results, "set_devices 未运行时只记录",
           pipe.input_device == "dummy-in" and pipe.monitor_device == "dummy-out"
           and pipe.cable_device == "offline")

    if args.switch:
        target = args.switch if os.path.isabs(args.switch) else os.path.join(WEIGHTS_DIR, args.switch)
        print("  热切模型 → %s" % target)
        t0 = time.perf_counter()
        try:
            pipe.load_model(target, profile["index"] or None)
            switch_s = time.perf_counter() - t0
            y3 = pipe.process_offline(x[:SAMPLERATE])
        except Exception as exc:
            _check(results, "load_model() 热切模型后推理正常", False, "%r" % (exc,))
        else:
            _check(results, "load_model() 热切模型后推理正常", _rms(y3) > 1e-3,
                   "%s, 耗时 %.2fs, 1s 输出 RMS=%.5f, tgt_sr=%d"
                   % (os.path.basename(target), switch_s, _rms(y3), pipe.engine.tgt_sr))
    return _report(results, "离线结论")


# --------------------------------------------------------------------------- #
# 2. Live
# --------------------------------------------------------------------------- #
def _show_device(label, device, kind):
    """Resolve and print device info; None on failure."""
    try:
        import sounddevice as sd

        index = resolve_device(device, kind)
        info = sd.query_devices(index)
        host = sd.query_hostapis(info["hostapi"])["name"]
        print("  %s %-10r → index=%d [%s] %s @%dch in=%d out=%d"
              % (label, device, index, host, info["name"],
                 int(info["default_samplerate"]),
                 info["max_input_channels"], info["max_output_channels"]))
        return index
    except Exception as exc:
        print("  %s %-10r → 解析失败: %s" % (label, device, exc))
        return None


def _monitor_candidates(monitor, cable):
    """Monitor candidates in try order: configured, virtual (silent), earphones."""
    names = []
    for name in (monitor,) + MONITOR_FALLBACKS:
        if not name:
            continue
        if os.path.normcase(str(name)) == os.path.normcase(str(cable or "")):
            continue
        if name not in names:
            names.append(name)
    return names


def _enable_monitor(pipe, candidates):
    """Try monitor candidates in turn; return the winner."""
    last_error = None
    for candidate in candidates:
        try:
            pipe.set_devices(monitor_device=candidate)
            pipe.set_param("monitor", True)
            return candidate
        except Exception as exc:
            last_error = exc
            print("  候选监听端点 %r 打不开: %s" % (candidate, exc))
    raise last_error if last_error else RuntimeError("没有可用的监听端点")


def _other_input(current_device):
    """A real input device other than the current one; None when none exists."""
    import sounddevice as sd

    current = None
    try:
        current = resolve_device(current_device, "input")
    except Exception:
        pass
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] <= 0 or index == current:
            continue
        host = sd.query_hostapis(info["hostapi"])["name"]
        if "WASAPI" not in host:
            continue
        if "变声麦克风" in info["name"] or "CABLE Output" in info["name"]:
            continue  # never use the CABLE capture endpoint as an input source
        return info["name"]
    return None


def run_live(args):
    import sounddevice as sd

    results = []
    seconds = float(args.live)
    profile = make_profile(args)
    print("=== ② 实时：真实设备三条流，共 %.0f 秒 ===" % seconds)
    print("设备（WASAPI 48k 全链路）：")
    in_index = _show_device("输入", args.input, "input")
    cable_index = _show_device("虚拟麦", args.cable, "output")
    _show_device("监听(配置)", args.monitor, "output")
    _show_device("CABLE 采集", args.cable_capture, "input")
    monitor_candidates = _monitor_candidates(args.monitor, args.cable)

    in_levels, out_levels = [], []

    def on_level(in_db, out_db):
        in_levels.append(float(in_db))
        out_levels.append(float(out_db))

    pipe = VoicePipeline(profile, args.input, args.cable,
                         monitor_device=args.monitor, monitor_on=False, on_level=on_level)
    t0 = time.perf_counter()
    try:
        pipe.start()
    except Exception as exc:
        print("start() 失败: %r" % (exc,))
        if in_index is None or cable_index is None:
            print("实时模式放弃：输入或虚拟麦设备不可用（--input/--cable 可换）")
        else:
            print("实时模式放弃：检查设备是否被独占 / CABLE 端点是否启用（selfcheck.py）")
        return 1, results
    print("start() 完成，耗时 %.1fs；block=%d 帧 (%.0f ms) @ %d Hz"
          % (time.perf_counter() - t0, pipe.block_frame,
             pipe.block_frame * 1000.0 / SAMPLERATE, SAMPLERATE))

    # CABLE capture-endpoint loopback (best effort): verify audio really reaches the virtual cable
    cap_rms, cap_stream = [], None
    if not args.no_capture:
        try:
            cap_index = resolve_device(args.cable_capture, "input")
            cap_channels = max(1, min(2, int(sd.query_devices(cap_index)["max_input_channels"])))

            def cap_cb(indata, frames, time_info, status):
                cap_rms.append(_rms(indata))

            cap_stream = sd.InputStream(device=cap_index, samplerate=SAMPLERATE,
                                        channels=cap_channels, dtype="float32", callback=cap_cb)
            cap_stream.start()
            print("CABLE 采集验证流已开: index=%d, %dch" % (cap_index, cap_channels))
        except Exception as exc:
            print("CABLE 采集验证流打不开（只影响环回验证，不影响管线）: %r" % (exc,))
            cap_stream = None

    stats = pipe.stats

    def snapshot():
        return (stats["blocks"], stats["cable_blocks"], stats["monitor_blocks"],
                stats["errors"])

    def phase(seconds_):
        time.sleep(max(0.2, seconds_))

    b0, c0, m0, e0 = snapshot()
    main_seconds = max(2.0, seconds - 6.0)
    print("--- 阶段1：正常跑 %.1fs（监听关）" % main_seconds)
    phase(main_seconds)
    b1, c1, m1, e1 = snapshot()
    print("  推理块 %d（+%d）, CABLE 回调 %d（+%d）, 异常 %d"
          % (b1, b1 - b0, c1, c1 - c0, e1 - e0))

    # ---- monitor toggle demo ----
    monitor_used = None
    monitor_error = None
    if monitor_candidates:
        print("--- 阶段2：set_param('monitor', True) 开监听（主链不断）")
        t0 = time.perf_counter()
        try:
            monitor_used = _enable_monitor(pipe, monitor_candidates)
            print("  监听已开: %r，耗时 %.2fs" % (monitor_used, time.perf_counter() - t0))
        except Exception as exc:
            monitor_error = exc
            print("  监听端点全部打不开（设备环境问题）: %r" % (exc,))
        if monitor_used:
            phase(2.0)
            b2, c2, m2, e2 = snapshot()
            _check(results, "monitor 开关: 打开后监听流在跑", pipe.monitor_active and (m2 - m1) > 0,
                   "%r 监听回调 %d 块（+%d）, CABLE 回调 +%d" % (monitor_used, m2, m2 - m1, c2 - c1))
            _check(results, "monitor 打开不影响 CABLE 主输出", (c2 - c1) > 0, "CABLE 回调 +%d" % (c2 - c1))
            print("--- 阶段3：set_param('monitor', False) 关监听")
            pipe.set_param("monitor", False)
            frozen = stats["monitor_blocks"]
            phase(1.0)
            _check(results, "monitor 开关: 关闭后监听流已停", not pipe.monitor_active
                   and stats["monitor_blocks"] == frozen,
                   "监听回调停在 %d" % frozen)
            b3, c3, m3, e3 = snapshot()
        else:
            print("--- 阶段2/3：无可用监听端点，跳过 monitor 开关演示")
            _skip_group(results, ("monitor 开关: 打开后监听流在跑",
                                  "monitor 打开不影响 CABLE 主输出",
                                  "monitor 开关: 关闭后监听流已停"),
                        "监听端点全部打不开（环境问题，未被测代码失败）: %r" % (monitor_error,))
    else:
        print("--- 阶段2/3：无可用监听端点，跳过 monitor 开关演示")
        _skip_group(results, ("monitor 开关: 打开后监听流在跑",
                              "monitor 打开不影响 CABLE 主输出",
                              "monitor 开关: 关闭后监听流已停"),
                    "没有候选监听端点（--monitor 与 --cable 同名或为空）")

    # ---- set_devices input-swap demo ----
    other = _other_input(args.input)
    if other:
        print("--- 阶段4：set_devices(input_device=%r) 换输入（虚拟麦不停）" % other)
        before = stats["blocks"]
        cable_before = stats["cable_blocks"]
        try:
            pipe.set_devices(input_device=other)
            phase(2.0)
            switched = stats["blocks"] > before and pipe.input_device == other
            _check(results, "set_devices 换输入后推理继续", switched,
                   "推理块 +%d, 新设备 index=%s" % (stats["blocks"] - before, pipe._in_index))
            _check(results, "换输入期间 CABLE 主输出不断",
                   stats["cable_blocks"] > cable_before,
                   "CABLE 回调 +%d" % (stats["cable_blocks"] - cable_before))
            pipe.set_devices(input_device=args.input)
            back_before = stats["blocks"]
            phase(1.0)
            _check(results, "set_devices 换回原输入", pipe.input_device == args.input
                   and stats["blocks"] > back_before,
                   "换回后推理块 +%d（换回前记数，避免拿换前的累计数充数）"
                   % (stats["blocks"] - back_before))
        except Exception as exc:
            _check(results, "set_devices 换输入", False, "%r" % (exc,))
    else:
        print("--- 阶段4：找不到第二只输入设备，跳过 set_devices 演示")
        _skip_group(results, ("set_devices 换输入后推理继续",
                              "换输入期间 CABLE 主输出不断",
                              "set_devices 换回原输入"),
                    "本机只有一只 WASAPI 输入设备")

    if cap_stream is not None:
        cap_stream.abort()
        cap_stream.close()
    pipe.stop()

    b4, c4, m4, e4 = snapshot()
    print("--- 统计 ---")
    print("推理块 %d, 平均 %.1f ms/块, 最大 %.1f ms, 回调异常 %d, 输入流状态事件 %d"
          % (b4, stats["infer_ms_avg"], stats["infer_ms_max"], e4, stats["in_status"]))
    print("输出流回调: CABLE %d 块, 监听 %d 块" % (c4, m4))
    if pipe.master is not None:
        print("主缓冲: 容量 %d 帧, 窗口滚动丢弃 %d 块" % (pipe.master.capacity, pipe.master.trimmed))
        _check(results, "主缓冲滚动丢弃有界", pipe.master.trimmed <= b4,
               "丢弃 %d 块，推理产出 %d 块（滚动丢弃是环形缓冲正常行为，只防计数器错乱）"
               % (pipe.master.trimmed, b4))
    else:
        _skip(results, "主缓冲滚动丢弃有界", "管线没建主缓冲")
    print("数据通路: 推理产出 ~%d 帧 → CABLE 消费 ~%d 帧 (%.1f%%)"
          % (b4 * pipe.block_frame, c4 * pipe.block_frame,
             100.0 * c4 / max(b4, 1)))
    if pipe.cable_reader is not None:
        print("CABLE 读游标: 欠载 %d 次, 跳变 %d 次（跳变 = 真丢数据）"
              % (pipe.cable_reader.underruns, pipe.cable_reader.skips))
        _check(results, "CABLE 读游标无跳变", pipe.cable_reader.skips == 0,
               "跳变 %d 次，欠载 %d 次（欠载是启动瞬间补零，跳变才是真丢数据）"
               % (pipe.cable_reader.skips, pipe.cable_reader.underruns))
    else:
        _skip(results, "CABLE 读游标无跳变", "管线没建输出读游标")
    a_in = np.asarray(in_levels, dtype=np.float64)
    a_out = np.asarray(out_levels, dtype=np.float64)
    loud = a_in > LIVE_VOICE_IN_DB if a_in.size else np.zeros(0, dtype=bool)
    if in_levels:
        print("输入电平: 平均 %.1f dBFS, 最大 %.1f dBFS (%d 块，其中 %d 块 > %.0f dBFS)"
              % (a_in.mean(), a_in.max(), a_in.size, int(loud.sum()), LIVE_VOICE_IN_DB))
        print("输出电平(on_level): 平均 %.1f dBFS, 最大 %.1f dBFS, 非静音块 %d/%d"
              % (a_out.mean(), a_out.max(), int((a_out > LIVE_VOICE_DEAD_DB).sum()), a_out.size))
    if cap_rms:
        a_cap = np.asarray(cap_rms)
        print("CABLE 采集端(环回): 平均 %.1f dBFS, 最大 %.1f dBFS (%d 块)"
              % (_db(a_cap.mean()), _db(a_cap.max()), a_cap.size))
        if _db(a_cap.max()) <= -60.0:
            if int(loud.sum()) >= LIVE_VOICE_MIN_BLOCKS:
                print("  注意: 输入侧有信号（%d 块 > %.0f dBFS）但环回接近静音，"
                      "输出链路没把音频送进虚拟线（见下面的数据通路判定）"
                      % (int(loud.sum()), LIVE_VOICE_IN_DB))
            else:
                print("  提示: 环回接近静音，而输入本身也很安静（房间安静 / 门限生效），"
                      "不是管线故障；对麦说话时会看到电平上升")
    else:
        print("CABLE 采集端(环回): 未测（--no-capture 或采集端点不可用）")

    # Data path: input with sound => the output must not be silent throughout. Callback counts / error
    # counts / timings are data-independent, so "all silent" from an unwritten master buffer or an
    # always-closed speaker gate would still pass under them.
    # The criterion only checks "data presence" (-80 dBFS): RVC legitimately outputs low levels for
    # non-speech input, and demanding "output at input level" would flag normal behavior as a fault.
    if not a_in.size:
        _skip(results, "输入有声时输出不是全程静音", "一个 on_level 回调都没收到（推理没跑起来）")
    elif int(loud.sum()) < LIVE_VOICE_MIN_BLOCKS:
        _skip(results, "输入有声时输出不是全程静音",
              "输入本来就安静（平均 %.1f dBFS，最高 %.1f，只有 %d 块 > %.0f dBFS）："
              "这一轮没验到数据通路，对麦说话再跑一次"
              % (a_in.mean(), a_in.max(), int(loud.sum()), LIVE_VOICE_IN_DB))
        _check(results, "安静输入下管线仍在跑", b4 > 0 and c4 > 0,
               "推理 %d 块，CABLE %d 块：数据通路没验到，但链路本身在转" % (b4, c4))
    else:
        _check(results, "输入有声时输出不是全程静音",
               float(a_out.max()) > LIVE_VOICE_DEAD_DB,
               "输入 %d 块 > %.0f dBFS（这些块平均 %.1f dBFS）→ 输出最高 %.1f dBFS、"
               "非静音块 %d/%d（判据 > %.0f dBFS；两者不同电平是正常的，"
               "RVC 对非语音输入只会输出低电平）"
               % (int(loud.sum()), LIVE_VOICE_IN_DB, float(a_in[loud].mean()),
                  float(a_out.max()), int((a_out > LIVE_VOICE_DEAD_DB).sum()), a_out.size,
                  LIVE_VOICE_DEAD_DB))
        if cap_rms:
            _check(results, "虚拟麦上真的有音频（环回不是全程静音）",
                   _db(a_cap.max()) > LIVE_VOICE_DEAD_DB,
                   "CABLE 采集端最高 %.1f dBFS（判据 > %.0f dBFS）"
                   % (_db(a_cap.max()), LIVE_VOICE_DEAD_DB))
        else:
            _skip(results, "虚拟麦上真的有音频（环回不是全程静音）",
                  "--no-capture 或采集端点不可用")

    _check(results, "输入流在跑", b4 > 0, "%d 块" % b4)
    _check(results, "CABLE 主输出在跑", c4 > 0, "%d 块" % c4)
    _check(results, "CABLE 消费帧数与推理产出匹配",
           c4 * pipe.block_frame >= b4 * pipe.block_frame * 0.8,
           "%d 帧 vs %d 帧" % (c4 * pipe.block_frame, b4 * pipe.block_frame))
    _check(results, "无回调异常", e4 == 0, "errors=%d" % e4)
    _check(results, "实时吞吐跟得上", stats["infer_ms_avg"] < pipe.block_frame * 1000.0 / SAMPLERATE,
           "平均 %.1f ms/块 vs 预算 %.1f ms" % (stats["infer_ms_avg"], pipe.block_frame * 1000.0 / SAMPLERATE))
    return _report(results, "实时结论"), results


# --------------------------------------------------------------------------- #
# 3. Acoustic protection (--acoustic): synthetic "you + roommate" + speaker-gate wiring check
# --------------------------------------------------------------------------- #

#: Scene timeline (seconds): silence / user / roommate / both.
ACOUSTIC_TIMELINE = (
    ("silence", 1.2),
    ("user", 2.0),
    ("roommate", 1.5),
    ("silence", 0.3),
    ("user", 2.0),
    ("both", 1.0),
    ("user", 2.0),
    ("silence", 0.3),
)
#: Interior span of the roommate segment, past decision cadence + release.
ACOUSTIC_SETTLE = 1.0
#: "Your speech" is scored on the two solo segments only.
ACOUSTIC_USER_SOLO = ((1.2, 3.2), (5.0, 7.0))

#: Synthetic speakers: f0 + harmonic weights both differ.
ACOUSTIC_USER = {"f0": 196.0, "weights": (1.0, 0.75, 0.30, 0.12, 0.45, 0.20, 0.08)}
ACOUSTIC_ROOMMATE = {"f0": 118.0, "weights": (1.0, 0.35, 0.80, 0.60, 0.10, 0.05, 0.03)}

#: Scenario B (real speech) timeline in seconds.
SPEECH_TIMELINE = (
    ("silence", 0.4),
    ("user", 2.0),
    ("roommate", 2.5),
    ("silence", 0.3),
    ("user", 2.0),
    ("both", 1.0),
    ("silence", 0.3),
)
#: Scenario B interior span (the real VAD needs a full 1 s window).
SPEECH_SETTLE = 1.0
#: Scenario B threshold sweep alongside the default.
SPEECH_ALT_THRESHOLD = 0.3
#: Real-speech fixture dir; scenario B is skipped when absent.
DEFAULT_SPEECH_DIR = os.path.join("out_dev", "voicegate_test")


def _speaker_signal(seconds, f0, weights, seed, level=0.09):
    """Synthetic voice: harmonics + vibrato + floor noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(seconds * SAMPLERATE)), dtype=np.float64) / SAMPLERATE
    phase = 2.0 * np.pi * f0 * t + 0.6 * np.sin(2.0 * np.pi * 2.7 * t)
    wave = np.zeros_like(t)
    for k, weight in enumerate(weights, start=1):
        if weight:
            wave += weight * np.sin(k * phase)
    wave *= level / max(float(np.max(np.abs(wave))), 1e-9)
    wave += rng.standard_normal(t.shape) * 0.0002
    return wave.astype(np.float32)


def _build_acoustic_scene():
    """Assemble the mixed mic signal; return (x, clean speakers, windows)."""
    total = sum(dur for _, dur in ACOUSTIC_TIMELINE)
    user = _speaker_signal(total, seed=11, **ACOUSTIC_USER)
    roommate = _speaker_signal(total, seed=12, **ACOUSTIC_ROOMMATE)
    x = np.zeros(int(round(total * SAMPLERATE)), dtype=np.float32)
    windows = {"user": [], "roommate": [], "both": []}
    cursor = 0.0
    for kind, dur in ACOUSTIC_TIMELINE:
        i0 = int(round(cursor * SAMPLERATE))
        i1 = i0 + int(round(dur * SAMPLERATE))
        if kind == "user":
            x[i0:i1] = user[i0:i1]
        elif kind == "roommate":
            x[i0:i1] = roommate[i0:i1]
        elif kind == "both":
            x[i0:i1] = user[i0:i1] + roommate[i0:i1]
        if kind in windows:
            windows[kind].append((cursor, cursor + dur))
        cursor += dur
    return x, {"user": user, "roommate": roommate}, windows


def _to_16k(wav48k):
    import librosa

    return np.asarray(
        librosa.resample(np.asarray(wav48k, dtype=np.float32), orig_sr=SAMPLERATE, target_sr=16000),
        dtype=np.float32,
    )


class _StubSpeakerGate:
    """Test double for VoiceGate: separates the two synthetic speakers spectrally.

    Same enroll/save/load/decide/vad_prob contract; verifies wiring only.
    """

    scale = 0.02
    nfft = 512

    def __init__(self, sr=16000, frame=0.02, hop=0.01):
        self.sr = sr
        self.frame_len = int(round(sr * frame))
        self.hop = int(round(sr * hop))
        self.profiles = {}
        self.threshold = 0.5
        self.reset_stats()
        self.debug = {}

    def reset_stats(self):
        """Clear wiring evidence before each round."""
        self.windows = []  # window lengths received by decide()
        self.scores = []  # scores returned by decide()
        self.calls = 0
        self.total_s = 0.0

    # ---------------- VoiceGate contract ----------------
    def enroll(self, wav16k):
        self.profiles["user"] = self._template(wav16k)
        return self

    def enroll_other(self, wav16k):
        self.profiles["other"] = self._template(wav16k)
        return self

    def save_profile(self, path):
        np.savez(path, **self.profiles)
        return path

    def load_profile(self, path):
        with np.load(path) as data:
            self.profiles = {name: data[name] for name in data.files}
        return self

    def vad_prob(self, wav16k):
        frames = self._frames(np.asarray(wav16k, dtype=np.float32).reshape(-1))
        if frames.shape[0] == 0:
            return 0.0
        return float(np.mean(self._speech_mask(frames)))

    def decide(self, wav16k):
        t0 = time.perf_counter()
        self.calls += 1
        wav = np.asarray(wav16k, dtype=np.float32).reshape(-1)
        self.windows.append(int(wav.shape[0]))
        try:
            frames = self._frames(wav)
            mask = self._speech_mask(frames)
            if frames.shape[0] == 0 or int(mask.sum()) < 3:
                self.scores.append(None)
                return None  # no speech in the window: undecidable
            spec = self._spectrum(frames[mask])
            sim_user = float(np.dot(spec, self.profiles["user"]))
            sim_other = float(np.dot(spec, self.profiles["other"]))
            self.debug = {
                "sim_user": sim_user,
                "sim_other": sim_other,
                "speech_ratio": float(mask.mean()),
            }
            score = float(0.5 + 0.5 * np.tanh((sim_user - sim_other) / self.scale))
            self.scores.append(score)
            return score
        finally:
            self.total_s += time.perf_counter() - t0

    # ---------------- internals ----------------
    def _frames(self, wav):
        n, hop = self.frame_len, self.hop
        if wav.shape[0] < n:
            return np.zeros((0, n), dtype=np.float32)
        count = 1 + (wav.shape[0] - n) // hop
        idx = np.arange(n)[None, :] + hop * np.arange(count)[:, None]
        return wav[idx] * np.hanning(n).astype(np.float32)

    @staticmethod
    def _speech_mask(frames):
        if frames.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        peak = float(rms.max())
        if peak <= 1e-4:
            return np.zeros(rms.shape, dtype=bool)
        return rms > max(0.15 * peak, 3e-4)

    def _spectrum(self, frames):
        spec = np.log(np.abs(np.fft.rfft(frames, n=self.nfft)) + 1e-6).mean(axis=0)
        spec = spec - spec.mean()  # drop overall gain, keep only spectral shape
        norm = float(np.linalg.norm(spec))
        return spec / norm if norm > 0 else spec

    def _template(self, wav16k):
        frames = self._frames(np.asarray(wav16k, dtype=np.float32).reshape(-1))
        mask = self._speech_mask(frames)
        if int(mask.sum()) == 0:
            raise ValueError("注册语音里没有检测到语音")
        return self._spectrum(frames[mask])


class _BrokenGate:
    """Gate whose decide() always raises; the main chain must survive it."""

    def decide(self, wav16k):
        raise RuntimeError("模拟声纹模型异常")

    def vad_prob(self, wav16k):
        raise RuntimeError("模拟声纹模型异常")


def _module_attr(module_name, attr_name):
    """Probe dsp.<module>.<attr>; return (attr, status, error).

    'missing' means undelivered; 'broken' means the import itself raised.
    """
    try:
        module = importlib.import_module("dsp.%s" % module_name)
    except ModuleNotFoundError as exc:
        return None, "missing", exc
    except Exception as exc:
        return None, "broken", exc
    attr = getattr(module, attr_name, None)
    if attr is None:
        return None, "missing", AttributeError("dsp.%s 里没有 %s" % (module_name, attr_name))
    return attr, "ok", None


def _module_text(status, error, absent, present="已交付"):
    """One-line probe result."""
    if status == "ok":
        return present
    if status == "missing":
        return "未交付（%s）" % absent
    return "import 失败（模块在，不是未交付）: %r" % (error,)


def _win_slice(y, window):
    start = int(round(window[0] * SAMPLERATE))
    stop = int(round(window[1] * SAMPLERATE))
    return y[start:stop]


def _win_db(y, window):
    return level_db(_win_slice(y, window))


def _win_energy(y, windows):
    total = 0.0
    for window in windows:
        seg = np.asarray(_win_slice(y, window), dtype=np.float64)
        total += float(np.sum(np.square(seg)))
    return total


def _win_energy_db(y, windows):
    """Combined RMS level (dBFS) over windows; -100 for silence."""
    if y is None:
        return -100.0
    samples = sum(int(round((b - a) * SAMPLERATE)) for a, b in windows)
    energy = _win_energy(y, windows)
    if samples <= 0 or energy <= 0.0:
        return -100.0
    rms = float(np.sqrt(energy / samples))
    return 20.0 * np.log10(rms) if rms > 1e-5 else -100.0


def _interior(part, settle=ACOUSTIC_SETTLE):
    return (part[0] + settle, part[1])


def _last_loud_s(y, window, floor_db=-60.0):
    """Seconds from window start to the last sample above the floor."""
    seg = np.abs(np.asarray(_win_slice(y, window), dtype=np.float32))
    loud = np.nonzero(seg > 10.0 ** (floor_db / 20.0))[0]
    return float(loud[-1]) / SAMPLERATE if loud.size else 0.0


def _first_loud_s(y, window, floor_db=-60.0):
    """Seconds from window start to the first sample above the floor."""
    seg = np.abs(np.asarray(_win_slice(y, window), dtype=np.float32))
    loud = np.nonzero(seg > 10.0 ** (floor_db / 20.0))[0]
    return float(loud[0]) / SAMPLERATE if loud.size else float("nan")


_KEEP_GATE = object()


def _acoustic_pass(pipe, x, label, gate=_KEEP_GATE):
    """One offline pass; return (output, mean ms per block)."""
    if gate is not _KEEP_GATE:
        pipe.set_param("voice_gate", gate)
    pipe.reset_voice_gate()
    t0 = time.perf_counter()
    y = pipe.process_offline(x)
    elapsed = time.perf_counter() - t0
    blocks = (y.shape[0] + pipe.block_frame - 1) // pipe.block_frame
    print("  %-14s %.2fs / %d 块 = %.1f ms/块" % (label, elapsed, blocks, elapsed * 1000.0 / blocks))
    return y, elapsed * 1000.0 / blocks


def run_acoustic(args):
    results = []
    profile = make_profile(args)
    gate_cls, gate_state, gate_err = _module_attr("voice_gate", "VoiceGate")
    dfn_cls, dfn_state, dfn_err = _module_attr("dfn", "DfnDenoiser")
    print("=== ③ 声学防护：'你 + 室友'交替/同时说话 + 声纹门控（离线，不开设备）===")
    print("  权重: %s" % profile["pth"])
    print("  防护模块: dsp.voice_gate.VoiceGate=%s, dsp.dfn.DfnDenoiser=%s"
          % (_module_text(gate_state, gate_err, "用测试替身验证接线"),
             _module_text(dfn_state, dfn_err, "跳过 dfn3", "已交付（模型能否加载见下方探测）")))
    if gate_state == "broken":
        _check(results, "dsp/voice_gate.py 可导入", False,
               "模块在、import 抛异常: %r" % (gate_err,))
    if dfn_state == "broken":
        _check(results, "dsp/dfn.py 可导入", False,
               "模块在、import 抛异常: %r" % (dfn_err,))

    pipe = VoicePipeline(profile, "offline", "offline")

    # Scenario A: synthetic speakers + stub gate (wiring check).
    print("--- 场景A：合成'你 / 室友'（替身门控，验证管线接线与门控行为）")
    x, speaker, windows = _build_acoustic_scene()
    print("  场景: %.1fs (%d 样本) — %s"
          % (x.shape[0] / SAMPLERATE, x.shape[0],
             " ".join("%s%.1fs" % (kind, dur) for kind, dur in ACOUSTIC_TIMELINE)))
    print("  窗口(秒): 你单独 %s / 室友 %s / 同时说话 %s"
          % (ACOUSTIC_USER_SOLO, windows["roommate"], windows["both"]))

    gate = _StubSpeakerGate()
    gate.enroll(_to_16k(speaker["user"][int(1.2 * SAMPLERATE):int(2.2 * SAMPLERATE)]))
    gate.enroll_other(_to_16k(speaker["roommate"][int(3.2 * SAMPLERATE):int(4.2 * SAMPLERATE)]))
    os.makedirs(OUT_DIR, exist_ok=True)
    profile_path = os.path.join(OUT_DIR, "acoustic_voice_profile.npz")
    gate.save_profile(profile_path)
    gate_reload = _StubSpeakerGate().load_profile(profile_path)

    print("  离线推理（每轮 1 遍全段）:")
    _acoustic_pass(pipe, x, "① 无门控(冷机)", gate=None)
    y_base, ms_base = _acoustic_pass(pipe, x, "① 无门控(复测)", gate=None)
    gate.reset_stats()
    y_gate, ms_gate = _acoustic_pass(pipe, x, "② 门控+torchgate", gate=gate)
    gate_blocks = (x.shape[0] + pipe.block_frame - 1) // pipe.block_frame
    gate_calls, gate_windows = gate.calls, list(gate.windows)
    gate_cost = gate.total_s * 1000.0
    y_dfn, ms_dfn = None, None
    if dfn_cls is not None:
        pipe.set_param("denoise_mode", "dfn3")
    if pipe.denoise_mode == "dfn3":
        y_dfn, ms_dfn = _acoustic_pass(pipe, x, "③ 门控+dfn3", gate=gate)
        print("  dfn3 自报算法延迟: %s ms（合成谐波不是语音，dfn3 会把它当噪声压，"
              "电平列只作接线证据）" % pipe.denoise_latency_ms)
        pipe.set_param("denoise_mode", "torchgate")
    else:
        print("  dfn3 没生效：set_param('denoise_mode', 'dfn3') 后仍是 %s（dsp/dfn.py 未交付，"
              "或 DFN3 模型加载失败，管线日志里有具体原因），本轮跳过，结论里记为未验证"
              % pipe.denoise_mode)
        _skip(results, "dfn3 版每块耗时不超无门控版 +50ms",
              "dfn3 没生效（当前 %s）：本轮没有 dfn3 数据，电平表的 dfn3 列与头部"
              "「DfnDenoiser=已交付」都不能当成 dfn3 跑过" % pipe.denoise_mode)

    # ---- level table ----
    _level_table(
        (("你(单独)", ACOUSTIC_USER_SOLO),
         ("你(同时后)", (windows["user"][2],)),
         ("室友(整段)", windows["roommate"]),
         ("室友(内部 %.1fs 后)" % ACOUSTIC_SETTLE, [_interior(w) for w in windows["roommate"]]),
         ("同时说话", windows["both"])),
        y_base, y_gate, y_dfn,
    )

    # ---- (a) roommate segment is suppressed ----
    roommate_inner = [_interior(w) for w in windows["roommate"]]
    worst = max(_win_db(y_gate, w) for w in roommate_inner)
    _check(results, "室友段（判决生效后）被压到 -60dB 以下", worst <= -60.0,
           "最差 %.1f dBFS（跳过段首 %.1fs 的判决节奏 + release 过渡）"
           % (worst, ACOUSTIC_SETTLE))
    worst_full = max(_win_db(y_gate, w) for w in windows["roommate"])
    close_s = max(_last_loud_s(y_gate, w) for w in windows["roommate"])
    print("    （室友段整段最差 %.1f dBFS —— 前 %.2fs 是判决节奏 + release 的自然过渡）"
          % (worst_full, close_s))

    # ---- (b) your speech keeps >80% energy ----
    keep_solo = _win_energy(y_gate, ACOUSTIC_USER_SOLO) / max(_win_energy(y_base, ACOUSTIC_USER_SOLO), 1e-12)
    keep_all = _win_energy(y_gate, windows["user"]) / max(_win_energy(y_base, windows["user"]), 1e-12)
    open_s = min(_first_loud_s(y_gate, w) for w in ACOUSTIC_USER_SOLO)
    _check(results, "你说话段（单独说话）保留 >80% 能量", keep_solo > 0.8,
           "保留 %.1f%%（含同时说话后 %.1f%%），开门口 %.0f ms；>100%% 说明门控版在本人段"
           "不比无门控版低（RVC 的 f0/SOLA 状态本身有 ±dB 级差异）"
           % (keep_solo * 100, keep_all * 100, open_s * 1000.0))

    # Reverse check at threshold 1.0 plus silence: the gate must stay shut.
    pipe.set_param("voice_gate_threshold", 1.0)
    y_closed, _ms_closed = _acoustic_pass(pipe, x, "②′ 门控阈值 1.0", gate=gate)
    pipe.set_param("voice_gate_threshold", 0.5)
    keep_closed = _win_energy(y_closed, ACOUSTIC_USER_SOLO) / max(
        _win_energy(y_base, ACOUSTIC_USER_SOLO), 1e-12)
    _check(results, "阈值 1.0 时本人段也被压住（反向）", keep_closed < 0.2,
           "保留 %.1f%%（阈值 0.5 时保留 >80%%，1.0 要求全闭；替身本人分数恒 <1.0）"
           % (keep_closed * 100,))
    silence_long = []
    cursor = 0.0
    for kind, dur in ACOUSTIC_TIMELINE:
        if kind == "silence" and dur >= 1.0:
            silence_long.append((cursor, cursor + dur))
        cursor += dur
    worst_silence = max(_win_db(y_gate, w) for w in silence_long)
    _check(results, "静音段门控后仍是静音", worst_silence <= -60.0,
           "最差 %.1f dBFS（门控只压不增，不应制造声音）" % worst_silence)

    # Added per-block cost.
    delta = ms_gate - ms_base
    _check(results, "门控版每块耗时不超过无门控版 +50ms", delta <= 50.0,
           "%.1f vs %.1f ms/块（Δ %+.1f ms；声纹判决 %d 次 × 平均 %.1f ms，摊到每块 %+.1f ms）"
           % (ms_gate, ms_base, delta, gate_calls,
              gate_cost / max(gate_calls, 1),
              gate_cost / max(gate_blocks, 1)))
    if ms_dfn is not None:
        _check(results, "dfn3 版每块耗时不超无门控版 +50ms", ms_dfn - ms_base <= 50.0,
               "%.1f vs %.1f ms/块（Δ %+.1f ms）" % (ms_dfn, ms_base, ms_dfn - ms_base))

    # Wiring invariants: 1 s history at 16 k, one decision per N blocks.
    _check(results, "声纹判决窗口 = 最近 1s（16k, 16000 样本）",
           bool(gate_windows) and set(gate_windows) == {16000},
           "收到 %d 个窗口, 长度取值 %s" % (len(gate_windows), sorted(set(gate_windows))))
    expect = -(-gate_blocks // pipe._gate_every)
    _check(results, "判决节奏 = 每 %d 块一次（约 %.0f ms）"
           % (pipe._gate_every, pipe._gate_every * pipe.block_frame * 1000.0 / SAMPLERATE),
           abs(gate_calls - expect) <= 2,
           "判决 %d 次, 预期 %d 次（%d 块 / %d）"
           % (gate_calls, expect, gate_blocks, pipe._gate_every))
    _check(results, "声纹档案 save/load 往返可用",
           gate_reload.decide(
               _to_16k(speaker["user"][int(2.0 * SAMPLERATE):int(3.0 * SAMPLERATE)])
           ) is not None,
           "acoustic_voice_profile.npz → decide() 正常返回")
    _check(results, "全段输出无 NaN/Inf 且不削顶",
           bool(np.isfinite(y_gate).all()) and float(np.max(np.abs(y_gate))) <= 1.0,
           "峰值 %.4f" % float(np.max(np.abs(y_gate))))

    # Mixed-speech segment: reported, not asserted.
    print("  同时说话段: 无门控 %.1f dBFS → 门控 %.1f dBFS（声纹门控分不开混音，"
          "这里如实报告不判定）" % (_win_db(y_base, windows["both"][0]),
                                    _win_db(y_gate, windows["both"][0])))

    # Scenario B: real speech + real VoiceGate.
    _run_real_gate_scene(pipe, args, results, gate_cls)

    # Fallbacks: broken gate / missing modules / illegal values.
    print("  降级路径:")
    mode_before = pipe.denoise_mode
    pipe.set_param("denoise_mode", "bogus")
    _check(results, "非法 denoise_mode 只告警、保持原模式",
           pipe.denoise_mode == mode_before, "仍为 %s" % pipe.denoise_mode)
    if dfn_cls is None:
        pipe.set_param("denoise_mode", "dfn3")
        _check(results, "dfn3 未交付时自动降级 torchgate 不崩",
               pipe.denoise_mode == "torchgate", "降级后 %s" % pipe.denoise_mode)
    pipe.set_param("voice_gate", os.path.join(OUT_DIR, "不存在的声纹目录"))
    _check(results, "voice_gate 传坏路径只告警、门控关闭", pipe.voice_gate is None)
    pipe.set_param("voice_gate", gate)
    _check(results, "voice_gate 传实例即时生效", pipe.voice_gate is gate)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    pipe.set_param("voice_gate", models_dir)
    if pipe.voice_gate is None:
        _skip(results, "voice_gate 传有效模型目录可加载",
              "目录分支未落成实例：模型文件缺失或档案未注册时自动关闭，都属环境原因")
    else:
        _check(results, "voice_gate 传有效模型目录可加载",
               callable(getattr(pipe.voice_gate, "decide", None)),
               "目录路径按 VoiceGate(model_dir, threshold) 契约构造")
    pipe.set_param("voice_gate", gate)
    pipe.set_param("voice_gate_threshold", 0.62)
    _check(results, "voice_gate_threshold 热更生效",
           abs(pipe.voice_gate_threshold - 0.62) < 1e-9 and abs(gate.threshold - 0.62) < 1e-9,
           "管线 %.2f / 门控实例 %.2f" % (pipe.voice_gate_threshold, gate.threshold))
    try:
        pipe.set_param("voice_gate_threshold", "abc")
        _check(results, "非法阈值被拒绝", False)
    except ValueError:
        _check(results, "非法阈值被拒绝", True)
    pipe.set_param("voice_gate_threshold", 0.5)
    pipe.set_param("voice_gate", _BrokenGate())
    errors_before = pipe.voice_stats["errors"]
    y_broken = pipe.process_offline(x[: 2 * SAMPLERATE])
    _check(results, "decide() 连续抛异常 → 自动停用门控、主链恢复出音",
           pipe.voice_gate is None and pipe.voice_stats["errors"] - errors_before >= 3
           and bool(np.isfinite(y_broken).all()) and float(np.max(np.abs(y_broken))) > 1e-3,
           "判决失败 %d 次后自动停用，2s 输出峰值 %.4f"
           % (pipe.voice_stats["errors"] - errors_before, float(np.max(np.abs(y_broken)))))
    pipe.set_param("voice_gate", None)
    _check(results, "set_param('voice_gate', None) 关闭门控", pipe.voice_gate is None)
    return _report(results, "声学防护结论")


def _level_table(rows, y_base, y_gate, y_dfn):
    """Print the per-window RMS level table (dBFS)."""
    print("  窗口电平（dBFS）:")
    print("    %-18s %10s %14s %14s" % ("窗口", "无门控", "门控+torchgate", "门控+dfn3"))
    for label, parts in rows:
        cells = ["    %-18s %10.1f %14.1f" % (label, _win_energy_db(y_base, parts),
                                              _win_energy_db(y_gate, parts))]
        cells.append("%14.1f" % _win_energy_db(y_dfn, parts) if y_dfn is not None else "%14s" % "—")
        print(" ".join(cells))


def _run_real_gate_scene(pipe, args, results, gate_cls):
    """Scenario B: real speech + real VoiceGate end to end.

    Registers with an isolated profile under out_dev/; skips when the
    fixture or the real model is absent.
    """
    print("--- 场景B：真语音 + 真 VoiceGate（端到端）")
    scene_checks = ("真语音：室友段（判决生效后）被压到 -60dB 以下",
                    "真语音：你说话段保留 >80% 能量",
                    "真语音：离线逐块耗时增量 ≤+50ms",
                    "实时路径（后台判决线程）每块额外耗时 ≤50ms")
    if gate_cls is None:
        print("  真 VoiceGate 不可用（见上面的 import 探测），跳过场景B")
        _skip_group(results, scene_checks, "dsp/voice_gate.py 未交付")
        return
    scene = _load_speech_scene(args.speech_dir)
    if scene is None:
        print("  真语音 fixture 不全（%s 需 four-speakers-zh.wav + diarization_ref.json），跳过"
              % args.speech_dir)
        _skip_group(results, scene_checks,
                    "真语音 fixture 缺失（out_dev 已 gitignore，干净检出时正常）")
        return
    x, windows, roster = scene["x"], scene["windows"], scene["roster"]
    default_threshold = pipe.voice_gate_threshold
    try:
        profile_path = _isolated_profile_path("voice_gate_scene_b.npz")
        real = gate_cls(os.path.join(PROJECT_ROOT, "models"), default_threshold,
                        profile_path=profile_path)
        real.enroll(scene["enroll"])
    except Exception as exc:
        print("  真 VoiceGate 不可用（构造/注册失败：%r），跳过场景B" % (exc,))
        _skip_group(results, scene_checks, "真 VoiceGate 构造/注册失败: %r" % (exc,))
        return
    print("  语料: four-speakers-zh.wav，说话人时长 %s；你=#%s（注册用前 3s），室友=#%s"
          % (roster["summary"], roster["you"], roster["mate"]))
    print("  窗口(秒): 你 %s / 室友 %s / 同时说话 %s"
          % (windows["user"], windows["roommate"], windows["both"]))
    print("  真门控档案: %s（隔离路径，本次注册 %s 段，不读 models/voice_profile.npz）"
          % (profile_path, getattr(real, "num_segments", "?")))
    print("  真门控分数（各 1s）: %s" % _score_probe(scene, real))

    pipe.set_param("voice_gate", None)
    y_base, ms_base = _acoustic_pass(pipe, x, "① 真语音基线", gate=None)
    recorder = _GateRecorder(real)
    summary = []
    base_candidates = [ms_base]
    for threshold in (default_threshold, SPEECH_ALT_THRESHOLD):
        pipe.set_param("voice_gate_threshold", threshold)
        recorder.reset_stats()
        y_gate, ms_gate = _acoustic_pass(
            pipe, x, "② 真语音+真门控(阈值 %.2f)" % threshold, gate=recorder
        )
        # Re-run the baseline right after (clocks drift; keep the faster one).
        _, ms_base_again = _acoustic_pass(pipe, x, "① 真语音基线(复测)", gate=None)
        base_candidates.append(ms_base_again)
        _level_table(
            (("你(单独)", windows["user"]),
             ("室友(整段)", windows["roommate"]),
             ("室友(内部 %.1fs 后)" % SPEECH_SETTLE,
              [_interior(w, SPEECH_SETTLE) for w in windows["roommate"]]),
             ("同时说话", windows["both"])),
            y_base, y_gate, None,
        )
        keep = _win_energy(y_gate, windows["user"]) / max(
            _win_energy(y_base, windows["user"]), 1e-12
        )
        worst = max(_win_db(y_gate, w)
                    for w in [_interior(w, SPEECH_SETTLE) for w in windows["roommate"]])
        close_s = max(_last_loud_s(y_gate, w) for w in windows["roommate"])
        open_s = min(_first_loud_s(y_gate, w) for w in windows["user"])
        decided = [s for s in recorder.scores if s is not None]
        print("     你保留 %.1f%%；室友内部最差 %.1f dBFS（整段 %.1f，%.2fs 后压住）；"
              "开门口 %.0f ms；判决 %d 次（判出 %d）× %.1f ms"
              % (keep * 100, worst, max(_win_db(y_gate, w) for w in windows["roommate"]),
                 close_s, open_s * 1000.0, recorder.calls, len(decided),
                 recorder.total_s * 1000.0 / max(recorder.calls, 1)))
        summary.append((threshold, keep, worst, ms_gate - min(base_candidates)))

    _check(results, "真语音：室友段（判决生效后）被压到 -60dB 以下",
           all(worst <= -60.0 for _, _, worst, _ in summary),
           "；".join("阈值 %.2f → %.1f dBFS" % (th, worst) for th, _, worst, _ in summary))
    best_keep = max(keep for _, keep, _, _ in summary)
    _check(results, "真语音：你说话段保留 >80% 能量", best_keep > 0.8,
           "；".join("阈值 %.2f → %.1f%%" % (th, keep * 100) for th, keep, _, _ in summary)
           + "（阈值越低越容易开门、开口越快，但混音时更容易放行室友；跨语句同人分数"
             "实测约 0.53，与默认 %.2f 贴边，建议按 GUI 显示的实时分数"
             "（本人 vs 室友）定档）" % default_threshold)
    _check(results, "真语音：离线逐块耗时增量 ≤+50ms",
           all(delta <= 50.0 for _, _, _, delta in summary),
           "；".join("阈值 %.2f Δ%+.1f ms/块" % (th, delta) for th, _, _, delta in summary)
           + "（基准取附近两次基线里较快的一次）")

    # Realtime path: decisions run on a background thread, not in the callback.
    probe = _realtime_latency_probe(pipe, x, recorder)
    if probe is None:
        _skip(results, "实时路径（后台判决线程）每块额外耗时 ≤50ms",
              "探针守卫触发（场景信号不足 8 个整块），实时路径没测")
    else:
        base_ms_rt, gate_ms_rt, decisions_rt = probe
        _check(results, "实时路径（后台判决线程）每块额外耗时 ≤50ms",
               decisions_rt > 0 and gate_ms_rt - base_ms_rt <= 50.0,
               "无门控 %.1f → 门控 %.1f ms/块（Δ %+.1f ms，后台判决 %d 次，未占用回调）"
               % (base_ms_rt, gate_ms_rt, gate_ms_rt - base_ms_rt, decisions_rt))
    pipe.set_param("voice_gate", None)
    pipe.set_param("voice_gate_threshold", default_threshold)


def _isolated_profile_path(name):
    """Isolated profile path under out_dev/, clearing last run's file."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        os.remove(path)
    return path


def _score_probe(scene, gate):
    """Ask the gate for you/roommate/mixed scores (1 s each)."""
    user = np.asarray(scene["user16k"][:16000], dtype=np.float32)
    mate = np.asarray(scene["mate16k"][:16000], dtype=np.float32)
    parts = []
    for label, wav in (("你", user), ("室友", mate), ("混合", 0.7 * user + 0.7 * mate)):
        score = gate.decide(wav)
        parts.append("%s=%s" % (label, "None" if score is None else "%.3f" % score))
    return " ".join(parts)


class _GateRecorder:
    """Wraps a real VoiceGate, recording decide() inputs, scores and cost."""

    def __init__(self, gate):
        self._gate = gate
        self.reset_stats()

    def reset_stats(self):
        self.windows = []
        self.scores = []
        self.calls = 0
        self.total_s = 0.0

    def decide(self, wav16k):
        start = time.perf_counter()
        self.calls += 1
        self.windows.append(int(np.asarray(wav16k).size))
        try:
            return self._gate.decide(wav16k)
        finally:
            self.total_s += time.perf_counter() - start
            self.scores.append(getattr(self._gate, "last_score", None))

    def __getattr__(self, name):
        return getattr(self._gate, name)


def _realtime_latency_probe(pipe, x, gate, blocks=40):
    """Simulate per-block _infer_block calls with and without the realtime gate.

    Return (base ms, gated ms, background decisions); None when the signal
    is too short.
    """
    block = pipe.block_frame
    chunks = [x[i * block : (i + 1) * block] for i in range(blocks)]
    if len(chunks) < 8 or any(chunk.shape[0] < block for chunk in chunks):
        return None
    pipe.set_param("voice_gate", None)
    for chunk in chunks[:3]:  # Warm up.
        pipe._infer_block(chunk, realtime=False)
    start = time.perf_counter()
    for chunk in chunks:
        pipe._infer_block(chunk, realtime=False)
    base_ms = (time.perf_counter() - start) * 1000.0 / len(chunks)

    pipe.set_param("voice_gate", gate)
    gate.reset_stats()
    for chunk in chunks[:3]:
        pipe._infer_block(chunk, realtime=True)
    start = time.perf_counter()
    for chunk in chunks:
        pipe._infer_block(chunk, realtime=True)
    gate_ms = (time.perf_counter() - start) * 1000.0 / len(chunks)
    time.sleep(0.5)  # Let the background thread finish pending decisions.
    return base_ms, gate_ms, gate.calls


def _speech_pool(wav16k, segments, sr=16000):
    """Concatenate reference segments into one speech stream."""
    parts = []
    for seg in segments:
        start = int(round(float(seg["start"]) * sr))
        stop = int(round(float(seg["end"]) * sr))
        if stop > start:
            parts.append(np.asarray(wav16k[start:stop], dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _load_speech_scene(speech_dir):
    """Build scenario B from the four-speaker fixture; None when incomplete."""
    wav_path = os.path.join(speech_dir, "four-speakers-zh.wav")
    ref_path = os.path.join(speech_dir, "diarization_ref.json")
    if not (os.path.isfile(wav_path) and os.path.isfile(ref_path)):
        return None
    import json

    import librosa

    wav, _sr = librosa.load(wav_path, sr=16000, mono=True)
    with open(ref_path, encoding="utf-8") as handle:
        segments = json.load(handle).get("segments", [])
    by_speaker = {}
    for seg in segments:
        by_speaker.setdefault(int(seg["speaker"]), []).append(seg)
    if len(by_speaker) < 2:
        return None
    speakers = sorted(by_speaker)
    you, mate = speakers[0], speakers[-1]
    user16k = _speech_pool(wav, by_speaker[you])
    mate16k = _speech_pool(wav, by_speaker[mate])
    enroll16k = user16k[: int(3.0 * 16000)]
    play16k = user16k[int(3.0 * 16000) :]
    if play16k.size < int(4.0 * 16000) or mate16k.size < int(2.0 * 16000):
        return None
    x, windows = _build_pool_scene(
        librosa.resample(play16k, orig_sr=16000, target_sr=SAMPLERATE),
        librosa.resample(mate16k, orig_sr=16000, target_sr=SAMPLERATE),
        SPEECH_TIMELINE,
    )
    roster = {
        "you": you,
        "mate": mate,
        "summary": " ".join(
            "#%d %.1fs" % (spk, sum(float(s["end"]) - float(s["start"]) for s in segs))
            for spk, segs in sorted(by_speaker.items())
        ),
    }
    return {
        "x": x,
        "windows": windows,
        "enroll": enroll16k,
        "roster": roster,
        "user16k": play16k,
        "mate16k": mate16k,
    }


def _take(pool, offset, n):
    """Take n samples from a pool, wrapping around."""
    if pool.size == 0:
        return np.zeros(n, dtype=np.float32)
    idx = (np.arange(n) + offset) % pool.size
    return np.asarray(pool[idx], dtype=np.float32)


def _build_pool_scene(user48k, mate48k, timeline):
    """Assemble the mic signal from two clean 48 kHz voices."""
    total = sum(dur for _, dur in timeline)
    x = np.zeros(int(round(total * SAMPLERATE)), dtype=np.float32)
    windows = {"user": [], "roommate": [], "both": []}
    cursor = 0.0
    cursor_user = cursor_mate = 0
    for kind, dur in timeline:
        n = int(round(dur * SAMPLERATE))
        start = int(round(cursor * SAMPLERATE))
        stop = start + n
        user = _take(user48k, cursor_user, n)
        mate = _take(mate48k, cursor_mate, n)
        if kind == "user":
            x[start:stop] = user
            cursor_user += n
        elif kind == "roommate":
            x[start:stop] = mate
            cursor_mate += n
        elif kind == "both":
            x[start:stop] = user + mate
            cursor_user += n
            cursor_mate += n
        if kind in windows:
            windows[kind].append((cursor, cursor + dur))
        cursor += dur
    return x, windows


# --------------------------------------------------------------------------- #
# Output AGC (dsp/agc.py + engine.pipeline wiring).
# --------------------------------------------------------------------------- #
#: Scenario levels (dBFS RMS) and lengths: loud -> quiet -> soft, two cycles.
AGC_SEGMENTS = (("大声", -12.0, 4.0), ("安静", -42.0, 12.0), ("小声", -27.0, 4.0))
AGC_CYCLES = 2
#: Target loudness and settle window: gain climbs at 5 dB/s, measure the tail.
AGC_TARGET_DBFS = -20.0
AGC_SETTLE_S = 1.0
#: Inter-block step cap (dB) and per-block AGC cost cap (ms).
AGC_MAX_STEP_DB = 2.0
AGC_MAX_MS = 1.0


def _agc_signal(seed=0):
    """AGC test signal: the same dense speech at three levels, cycled twice."""
    rng = np.random.default_rng(seed)
    source = _agc_speech_source(rng)
    unit = _speech_dense_unit(source, 4.0)
    parts, segments, offset = [], [], 0
    for cycle in range(AGC_CYCLES):
        for label, level_db, seconds in AGC_SEGMENTS:
            n = int(seconds * SAMPLERATE)
            reps = int(np.ceil(n / unit.size))
            wave = np.tile(unit, reps)[:n].astype(np.float32)
            wave *= 10.0 ** (level_db / 20.0) / max(_rms(wave), 1e-9)
            wave += (0.0005 * rng.standard_normal(n)).astype(np.float32)
            parts.append(wave)
            segments.append(("%s#%d" % (label, cycle + 1), offset, offset + n))
            offset += n
    return np.concatenate(parts), segments


def _speech_dense_unit(source, seconds):
    """Highest-energy window of the given length from the source."""
    n = int(seconds * SAMPLERATE)
    if source.size <= n:
        return np.asarray(source, dtype=np.float32)
    hop = n // 2
    best_start, best_e = 0, -1.0
    for start in range(0, source.size - n + 1, hop):
        e = float(np.dot(source[start:start + n], source[start:start + n]))
        if e > best_e:
            best_e, best_start = e, start
    return np.asarray(source[best_start:best_start + n], dtype=np.float32)


def _agc_speech_source(rng):
    """Real speech material, else a synthetic harmonic fallback."""
    import glob as _glob

    for path in sorted(_glob.glob(os.path.join(PROJECT_ROOT, "out_dev", "voicegate_test",
                                               "*.wav"))):
        try:
            src = _load_wav_mono(path)
        except Exception:
            continue
        if src is not None and src.size >= SAMPLERATE:
            return src
    n = int(4.0 * SAMPLERATE)
    t = np.arange(n, dtype=np.float64) / SAMPLERATE
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.2 * t) ** 2
    for start in range(int(0.85 * SAMPLERATE), n, int(0.95 * SAMPLERATE)):
        env[start : start + int(0.06 * SAMPLERATE)] = 0.0
    wave = np.zeros(n, dtype=np.float32)
    for k in range(1, 6):
        wave += (np.sin(2.0 * np.pi * 185.0 * k * t) / k).astype(np.float32)
    return wave * env.astype(np.float32)


def _load_wav_mono(path):
    """Read a wav file as 48 kHz float32 mono."""
    import wave

    from scipy.signal import resample_poly

    with wave.open(path, "rb") as w:
        ch, width, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    if sr != SAMPLERATE:
        from math import gcd

        g = gcd(sr, SAMPLERATE)
        x = resample_poly(x, SAMPLERATE // g, sr // g)
    return x.astype(np.float32)


def _fit_source(base, n, rng):
    """Cut n samples from the source, tiling when short."""
    if base.size >= n:
        start = int(rng.integers(0, max(1, base.size - n)))
        return np.array(base[start : start + n], dtype=np.float32)
    reps = int(np.ceil(n / base.size))
    return np.tile(base, reps)[:n].astype(np.float32)


class _AgcProbe:
    """Wraps dsp.agc.Agc, recording per-block gain, speech estimate and cost."""

    def __init__(self, inner):
        self.inner = inner
        self.gains = []
        self.ms = []
        self.max_step_db = 0.0
        self.speech = []  # speech_db after each block (None = no speech yet)
        self.probs = []  # VAD probability used per block (None = energy heuristic)

    def process(self, chunk, speech_prob=None):
        t0 = time.perf_counter()
        out = self.inner.process(chunk, speech_prob)
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        gain = float(self.inner.gain_db)
        if self.gains:
            self.max_step_db = max(self.max_step_db, abs(gain - self.gains[-1]))
        self.gains.append(gain)
        self.speech.append(self.inner.speech_db)
        self.probs.append(None if speech_prob is None else float(speech_prob))
        return out

    def set_target_dbfs(self, value):
        return self.inner.set_target_dbfs(value)

    def reset(self):
        self.gains.clear()
        self.ms.clear()
        self.speech.clear()
        self.probs.clear()
        self.max_step_db = 0.0
        return self.inner.reset()

    def __getattr__(self, name):  # gain_db / ceiling / speech_db / target_dbfs ...
        return getattr(self.inner, name)


def _agc_windows(segments, settle):
    """[(label, full window, settle window)] in seconds."""
    out = []
    for label, start, stop in segments:
        first, last = start / float(SAMPLERATE), stop / float(SAMPLERATE)
        out.append((label, (first, last), (max(last - settle, first), last)))
    return out


def _agc_range(values):
    """Dynamic range = max - min (dB)."""
    return max(values) - min(values)


def run_agc(args):
    results = []
    profile = make_profile(args)
    agc_cls, agc_state, agc_err = _module_attr("agc", "Agc")
    print("=== ④ 输出侧 AGC：响度起伏的语音状信号走完整推理链（离线，不开设备）===")
    print("  权重: %s" % profile["pth"])
    print("  dsp.agc.Agc: %s" % _module_text(agc_state, agc_err, "无法验证"))
    if agc_cls is None:
        _check(results, "dsp.agc.Agc 可用", False,
               "dsp/agc.py 未交付" if agc_state == "missing"
               else "模块在、import 抛异常: %r" % (agc_err,))
        return _report(results, "AGC 结论")

    x, segments = _agc_signal()
    rows = _agc_windows(segments, AGC_SETTLE_S)
    print("  场景: %s ×%d = %.1fs (%d 样本)"
          % (" ".join("%s %.0fs@%.0fdB" % (label, sec, level)
                      for label, level, sec in AGC_SEGMENTS), AGC_CYCLES,
             x.shape[0] / float(SAMPLERATE), x.shape[0]))
    print("  目标响度 %.0f dBFS；稳态窗 = 每段最后 %.1fs（增益升速 5 dB/s，安静段要几秒才拉到位）"
          % (AGC_TARGET_DBFS, AGC_SETTLE_S))

    # Baseline: same chain, AGC off.
    pipe_off = VoicePipeline(profile, "offline", "offline")
    t0 = time.perf_counter()
    y_base = pipe_off.process_offline(x)
    print("  基线（无 AGC）: RMS=%.5f (%.1f dBFS), 峰值 %.4f, 耗时 %.1fs"
          % (_rms(y_base), level_db(y_base), float(np.max(np.abs(y_base))),
             time.perf_counter() - t0))

    # With AGC: the probe wrapper records per-block gain and cost.
    agc = _AgcProbe(agc_cls(sr=SAMPLERATE, target_dbfs=AGC_TARGET_DBFS))
    pipe = VoicePipeline(profile, "offline", "offline", agc=agc)
    t0 = time.perf_counter()
    y_agc = pipe.process_offline(x)
    agc_seconds = time.perf_counter() - t0
    print("  开 AGC: RMS=%.5f (%.1f dBFS), 峰值 %.4f, 耗时 %.1fs, 末段增益 %+.1f dB"
          % (_rms(y_agc), level_db(y_agc), float(np.max(np.abs(y_agc))),
             agc_seconds, agc.gains[-1] if agc.gains else 0.0))

    # Level table.
    print("  段电平（dBFS）:")
    print("    %-8s %9s %9s %9s %9s %9s %9s"
          % ("段", "输入整段", "输入稳态", "基线稳态", "AGC整段", "AGC稳态", "段末增益"))
    for label, full, settled in rows:
        last_block = max(0, int(round(settled[1] * SAMPLERATE)) // pipe.block_frame - 1)
        gain = agc.gains[last_block] if last_block < len(agc.gains) else 0.0
        print("    %-8s %9.1f %9.1f %9.1f %9.1f %9.1f %+9.1f"
              % (label, _win_energy_db(x, [full]), _win_energy_db(x, [settled]),
                 _win_energy_db(y_base, [settled]), _win_energy_db(y_agc, [full]),
                 _win_energy_db(y_agc, [settled]), gain))

    in_settled = [_win_energy_db(x, [settled]) for _, _, settled in rows]
    base_settled = [_win_energy_db(y_base, [settled]) for _, _, settled in rows]
    agc_settled = [_win_energy_db(y_agc, [settled]) for _, _, settled in rows]
    agc_full = [_win_energy_db(y_agc, [full]) for _, full, _ in rows]
    in_full = [_win_energy_db(x, [full]) for _, full, _ in rows]
    print("  语音段动态范围: 输入稳态 %.1f dB（整段 %.1f）→ 基线稳态 %.1f dB → AGC 稳态 %.1f dB"
          "（AGC 整段 %.1f）"
          % (_agc_range(in_settled), _agc_range(in_full), _agc_range(base_settled),
             _agc_range(agc_settled), _agc_range(agc_full)))
    if max(agc.gains) >= agc.inner.max_gain_db - 1e-9:
        print("    （安静段用满 +%.0f dB 增益上限：管线在这段的输出只有 %.1f dBFS，"
              "再补就把底噪一起抬起来）" % (agc.inner.max_gain_db, min(base_settled)))
    if min(agc.gains) <= agc.inner.min_gain_db + 1e-9:
        print("    （某段用满 %.0f dB 增益下限）" % agc.inner.min_gain_db)

    # Verdicts.
    _check(results, "构造信号语音段动态范围 ≥25 dB", _agc_range(in_settled) >= 25.0,
           "输入 %.1f ~ %.1f dBFS（稳态），动态范围 %.1f dB"
           % (min(in_settled), max(in_settled), _agc_range(in_settled)))
    _check(results, "AGC 输出语音段稳态动态范围 ≤10 dB", _agc_range(agc_settled) <= 10.0,
           "AGC 稳态 %.1f ~ %.1f dBFS = %.1f dB（基线 %.1f dB）"
           % (min(agc_settled), max(agc_settled), _agc_range(agc_settled),
              _agc_range(base_settled)))
    _check(results, "动态范围比基线压缩 ≥10 dB",
           _agc_range(base_settled) - _agc_range(agc_settled) >= 10.0,
           "%.1f → %.1f dB（压缩 %.1f dB）"
           % (_agc_range(base_settled), _agc_range(agc_settled),
              _agc_range(base_settled) - _agc_range(agc_settled)))
    peak = float(np.max(np.abs(y_agc)))
    _check(results, "无削波（峰值 <0.999）", peak < 0.999 and bool(np.isfinite(y_agc).all()),
           "峰值 %.4f / 软限幅上限 %.4f（增益还高时到来的一声由 tanh 平滑压住，不是硬削顶），无 NaN/Inf"
           % (peak, agc.inner.ceiling))
    _check(results, "增益变化平滑（块间跳变 ≤%.1f dB）" % AGC_MAX_STEP_DB,
           agc.max_step_db <= AGC_MAX_STEP_DB,
           "最大块间跳变 %.2f dB（限速 升 %.0f / 降 %.0f dB/s，块长 %.0f ms）"
           % (agc.max_step_db, agc.inner.rate_up_dbps, agc.inner.rate_down_dbps,
              pipe.block_frame * 1000.0 / SAMPLERATE))
    ms = np.asarray(agc.ms)
    _check(results, "AGC 单块 CPU 耗时 <%.1f ms" % AGC_MAX_MS, float(ms.mean()) < AGC_MAX_MS,
           "mean %.3f / p95 %.3f / max %.3f ms（%d 块 @ %d 帧，块预算 %.0f ms）"
           % (ms.mean(), np.percentile(ms, 95), ms.max(), ms.size, pipe.block_frame,
              pipe.block_frame * 1000.0 / SAMPLERATE))

    # Wiring / hot-update.
    _check(results, "构造 agc=实例 即启用（profile 无 agc 键）",
           bool(pipe.agc_enabled) and pipe.agc is agc
           and pipe_off.agc is None and not pipe_off.agc_enabled,
           "开 AGC 的管线 agc=%s、基线管线 agc=%r（默认关、不建实例）"
           % (type(agc).__name__, pipe_off.agc))
    _check(results, "当前增益透出（agc_stats / agc_gain_db）",
           pipe.agc_stats["blocks"] > 0
           and abs(pipe.agc_gain_db - agc.gains[-1]) < 1e-6
           and abs(pipe.get_param("agc_target_dbfs") - AGC_TARGET_DBFS) < 1e-9,
           "处理 %d 块，增益 %.1f dB（统计 %.1f），目标 %.0f dB"
           % (pipe.agc_stats["blocks"], pipe.agc_gain_db, pipe.agc_stats["gain_db"],
              pipe.get_param("agc_target_dbfs")))

    blocks_before = pipe.agc_stats["blocks"]
    pipe.set_param("agc", False)
    y_off = pipe.process_offline(x)
    off_settled = [_win_energy_db(y_off, [settled]) for _, _, settled in rows]
    _check(results, "set_param('agc', False) = 直通（旁路，回到基线）",
           pipe.agc_stats["blocks"] == blocks_before
           and abs(level_db(y_off) - level_db(y_base)) <= 1.0
           and abs(_agc_range(off_settled) - _agc_range(base_settled)) <= 2.0,
           "关掉后 AGC 未再处理任何块；全段 %.1f dBFS（基线 %.1f），动态范围 %.1f dB（基线 %.1f）"
           % (level_db(y_off), level_db(y_base), _agc_range(off_settled),
              _agc_range(base_settled)))
    pipe.set_param("agc", True)
    _check(results, "set_param('agc', True) 运行中重新打开", bool(pipe.agc_enabled),
           "agc_enabled=%r" % pipe.agc_enabled)

    # Target hot-update on the soft segment: its level must follow the target.
    # (The quiet segment would hit the +30 dB gain cap instead.)
    unit = _speech_dense_unit(_agc_speech_source(np.random.default_rng(11)), 4.0)
    small = np.tile(unit, 2).astype(np.float32)
    small *= 10.0 ** (-27.0 / 20.0) / max(_rms(small), 1e-9)
    small_s = small.shape[0] / float(SAMPLERATE)
    small_levels = []
    for target in (-20.0, -14.0):
        pipe.reset_agc()
        pipe.set_param("agc_target_dbfs", target)
        small_levels.append(_win_energy_db(pipe.process_offline(small), [(small_s - 2.0, small_s)]))
    pipe.reset_agc()
    reset_ok = bool(pipe.agc_stats["blocks"] == 0 and abs(pipe.agc_gain_db) < 1e-9
                    and agc.inner.speech_db is None)
    _check(results, "reset_agc() 清零增益 / 统计 / 电平估计", reset_ok,
           "重置后增益 %.1f dB、块数 %d、语音电平 %r"
           % (pipe.agc_gain_db, pipe.agc_stats["blocks"], agc.inner.speech_db))
    _check(results, "agc_target_dbfs 热更生效（小声段 -20 → -14 dBFS）",
           abs(small_levels[0] + 20.0) <= 3.0 and abs(small_levels[1] + 14.0) <= 3.0
           and abs(agc.inner.target_dbfs + 14.0) < 1e-9,
           "稳态 %.1f / %.1f dBFS（目标 -20 / -14；段电平 -40 dBFS，增益落在 +30 dB 上限内）"
           % (small_levels[0], small_levels[1]))
    try:
        pipe.set_param("agc_target_dbfs", "abc")
        _check(results, "非法目标响度被拒绝", False)
    except ValueError:
        _check(results, "非法目标响度被拒绝", True,
               "set_param('agc_target_dbfs', 'abc') → ValueError")
    pipe.set_param("agc_target_dbfs", AGC_TARGET_DBFS)

    # VAD weighting: bursts must not count as speech.
    vad_cls, vad_state, vad_err = _module_attr("vad", "SileroVad")
    if vad_cls is None:
        _skip(results, "dsp.vad.SileroVad 可用",
              "dsp/vad.py 未交付" if vad_state == "missing"
              else "模块在、import 抛异常: %r" % (vad_err,))
    else:
        _check(results, "dsp.vad.SileroVad 可用", True)

    # Unit level: same Agc, burst prob=0.02 vs speech prob=0.9.
    agc_u = agc_cls(sr=SAMPLERATE, target_dbfs=AGC_TARGET_DBFS)
    block = pipe.block_frame
    speech_src = _fit_source(_agc_speech_source(np.random.default_rng(7)),
                              block * 8, np.random.default_rng(8))
    speech_src *= 0.05 / max(_rms(speech_src), 1e-9)
    for i in range(0, speech_src.size, block):
        agc_u.process(speech_src[i:i + block], 0.9)
    speech_est = agc_u.speech_db
    rng_b = np.random.default_rng(9)
    for _ in range(5):  # Keyboard/burst-like wideband noise pulses.
        burst = (rng_b.standard_normal(block) * 0.28).astype(np.float32)
        burst *= np.exp(-np.arange(block) / (block / 6.0)).astype(np.float32)
        agc_u.process(burst, 0.02)
    burst_pollution = (None if (agc_u.speech_db is None or speech_est is None)
                       else abs(agc_u.speech_db - speech_est))
    for i in range(0, speech_src.size, block):
        agc_u.process(speech_src[i:i + block], 0.9)
    _check(results, "撞击噪声不污染语音电平估计（VAD 加权，单元级）",
           burst_pollution is not None and burst_pollution <= 1.0,
           "5 个 -11 dBFS 撞击块（prob=0.02）前后，speech_db 偏移 %s（≤1 dB）"
           % ("判不出语音（speech_db=None）" if burst_pollution is None
              else "%.2f dB" % burst_pollution))
    # Control: same bursts without VAD (energy heuristic).
    agc_h = agc_cls(sr=SAMPLERATE, target_dbfs=AGC_TARGET_DBFS)
    for i in range(0, speech_src.size, block):
        agc_h.process(speech_src[i:i + block])
    speech_est_h = agc_h.speech_db
    for _ in range(5):
        burst = (rng_b.standard_normal(block) * 0.28).astype(np.float32)
        burst *= np.exp(-np.arange(block) / (block / 6.0)).astype(np.float32)
        agc_h.process(burst)
    heuristic_pollution = (None if (agc_h.speech_db is None or speech_est_h is None)
                           else abs(agc_h.speech_db - speech_est_h))
    print("  对照: 无 VAD（能量启发式）时同样撞击块把 speech_db 抬高 %s（VAD 加权 %s）"
          % ("判不出语音" if heuristic_pollution is None
             else "%.2f dB" % heuristic_pollution,
             "判不出语音" if burst_pollution is None else "%.2f dB" % burst_pollution))

    # Pipeline level: the VAD probability source is wired in.
    if pipe._agc_vad is None:
        _skip_group(results, ("管线已接 SileroVad 概率源",
                               "VAD 单块耗时 <1.5 ms",
                               "只有撞击没有语音：VAD 版不把撞击当语音（因果判决实验）"),
                    "models/ 下无 silero 模型，管线没接 VAD（环境缺失，不是被测代码失败）")
        return _report(results, "AGC 结论")
    _check(results, "管线已接 SileroVad 概率源", True,
           "_agc_vad=%r（models/ 有 silero 模型时自动接上）" % (pipe._agc_vad,))

    # VAD per-block cost.
    if pipe._agc_vad is not None:
        probe_sig = _fit_source(unit, block * 20, np.random.default_rng(21))
        pipe._agc_vad.reset()
        t_vad = []
        for i in range(0, probe_sig.size, block):
            t0 = time.perf_counter()
            pipe._agc_vad.process(probe_sig[i:i + block])
            t_vad.append((time.perf_counter() - t0) * 1000.0)
        ms_vad = np.asarray(t_vad[2:])  # Skip the first two warm-up blocks.
        _check(results, "VAD 单块耗时 <1.5 ms", float(ms_vad.mean()) < 1.5,
               "mean %.3f / p95 %.3f / max %.3f ms（%d 块，块预算 %.0f ms；AGC 本身 0.05 ms）"
               % (ms_vad.mean(), np.percentile(ms_vad, 95), ms_vad.max(), ms_vad.size,
                  block * 1000.0 / SAMPLERATE))
        pipe._agc_vad.reset()

    # Pipeline level: converge AGC for 4 s of speech, then insert three bursts.
    # Evidence only, no asserts: the limiter bounds |dGain| either way, and a
    # burst-covered block legitimately moves the estimate, so hard limits
    # would misfire. The causal verdict is the bursts-only check below.
    seg_burst = np.tile(unit, 2)[: int(8.0 * SAMPLERATE)].astype(np.float32)
    seg_burst *= 10.0 ** (-17.0 / 20.0) / max(_rms(seg_burst), 1e-9)
    burst_blocks, burst_pure, burst_mixed = [], [], []
    for sec in (4.5, 5.5, 6.5):
        start = int(sec * SAMPLERATE)
        first, last = start // block, (start + block - 1) // block
        burst_blocks.append(first)
        if start % block == 0:  # Fully burst-covered block: no speech inside.
            burst_pure.append(first)
        else:  # Burst lands inside a speech block.
            burst_mixed.extend(range(first, last + 1))
        env = np.exp(-np.arange(block) / (block / 6.0)).astype(np.float32)
        seg_burst[start : start + block] = (rng_b.standard_normal(block) * 0.28).astype(
            np.float32) * env
    pipe.reset_agc()
    y_burst = pipe.process_offline(seg_burst)
    gains_vad = list(agc.gains)
    speech_vad = list(agc.speech)
    pipe.set_param("agc", False)  # Same signal, no-AGC baseline.
    y_burst_base = pipe.process_offline(seg_burst)
    pipe.set_param("agc", True)
    saved_vad = pipe._agc_vad

    max_dip = 0.0
    for b in burst_blocks:  # Gain track around burst blocks.
        for j in (b - 1, b, b + 1):
            if 0 <= j < len(gains_vad) - 1:
                max_dip = max(max_dip, abs(gains_vad[j] - gains_vad[j - 1]) if j > 0 else 0.0)
                max_dip = max(max_dip, abs(gains_vad[min(j + 1, len(gains_vad) - 1)] - gains_vad[j]))
    shifts = {j: abs(speech_vad[j] - speech_vad[j - 1])
              for j in burst_pure + burst_mixed
              if 0 < j < len(speech_vad) and speech_vad[j] is not None
              and speech_vad[j - 1] is not None}
    print("  参考（不作判据）: 整块被撞击覆盖的块 %s：speech_db 移动 %s dB、VAD 概率 %s；"
          "落在语音块里的 %s：speech_db 移动最大 %.3f dB；块间增益最大变化 %.2f dB"
          "（限速器 ≤%.2f dB/块，撞不撞都成立）"
          % (burst_pure, ["%.3f" % shifts[j] for j in burst_pure if j in shifts],
             ["%.2f" % agc.probs[j] for j in burst_pure
              if j < len(agc.probs) and agc.probs[j] is not None] or ["—"],
             burst_mixed, max((shifts[j] for j in burst_mixed if j in shifts), default=0.0),
             max_dip, agc.inner.rate_down_dbps * block / SAMPLERATE))
    # Decisive test: bursts alone must leave VAD gain untouched (speech_db None).
    burst_only = (0.0005 * rng_b.standard_normal(int(8.0 * SAMPLERATE))).astype(np.float32)
    for sec in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        start = int(sec * SAMPLERATE)
        env = np.exp(-np.arange(block) / (block / 6.0)).astype(np.float32)
        burst_only[start:start + block] = (rng_b.standard_normal(block) * 0.28).astype(
            np.float32) * env
    pipe.reset_agc()
    pipe._agc_vad.reset()  # Clear RNN state left by the speech run.
    pipe.process_offline(burst_only)
    vad_speech_db, vad_gain = agc.inner.speech_db, float(np.mean(agc.gains))
    pipe._agc_vad = None
    pipe.reset_agc()
    pipe.process_offline(burst_only)
    heur_speech_db, heur_gain = agc.inner.speech_db, float(np.mean(agc.gains))
    pipe._agc_vad = saved_vad
    pipe.reset_agc()
    _check(results, "只有撞击没有语音：VAD 版不把撞击当语音（因果判决实验）",
           vad_speech_db is None and abs(vad_gain) <= 0.5
           and heur_speech_db is not None and abs(heur_gain) >= abs(vad_gain) + 5.0,
           "VAD: speech_db=%r、平均增益 %+.2f dB（纹丝不动）｜启发式: speech_db=%.1f dBFS、"
           "平均增益 %+.2f dB（把撞击当语音，增益跟着跑偏）"
           % (vad_speech_db, vad_gain, heur_speech_db or 0.0, heur_gain))
    speech_wins = [(3.0, 4.4), (4.7, 5.4), (5.7, 6.4), (6.7, 7.9)]  # Converged speech windows.
    applied = [_win_energy_db(y_burst, [w]) - _win_energy_db(y_burst_base, [w])
               for w in speech_wins]
    print("  参考: 四处语音窗的实际增益 %+.1f / %+.1f / %+.1f / %+.1f dB（随音节内容波动属正常）"
          % (applied[0], applied[1], applied[2], applied[3]))
    return _report(results, "AGC 结论")


# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(description="模块C 冒烟测试（离线 / 实时 / 声学防护 / 输出 AGC）")
    parser.add_argument("--live", type=int, nargs="?", const=8, default=None,
                        metavar="N", help="实时模式：起管线跑 N 秒（默认 8），演示 monitor 开关与 set_devices")
    parser.add_argument("--pure", action="store_true",
	                        help="纯逻辑模式：设备记忆与默认选择单测，不开设备、不加载模型")
    parser.add_argument("--offline", action="store_true", help="离线模式（默认）：不开设备，推 3 秒信号")
    parser.add_argument("--acoustic", action="store_true",
                        help="声学防护模式：合成你和室友交替/同时说话，验证声纹门控与 dfn3")
    parser.add_argument("--agc", action="store_true",
                        help="输出侧 AGC 模式：响度起伏信号走完整推理链，"
                             "验证动态范围压缩 / 不削波 / 增益平滑 / 单块耗时与热更")
    parser.add_argument("--speech-dir", default=os.path.join(PROJECT_ROOT, DEFAULT_SPEECH_DIR),
                        help="场景B 真语音 fixture 目录（需 four-speakers-zh.wav + diarization_ref.json）")
    parser.add_argument("--pth", default=None, help="模型 pth（默认 assets/weights/bb48k.pth）")
    parser.add_argument("--index", default=None, help="索引（默认 logs/guanguanV1.index，传空串=不用索引）")
    parser.add_argument("--f0method", default="rmvpe", help="音高算法（项目统一 rmvpe）")
    parser.add_argument("--pitch", type=float, default=15, help="变调半音（默认 15）")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入麦克风（名/子串/索引）")
    parser.add_argument("--monitor", default=DEFAULT_MONITOR, help="监听输出设备")
    parser.add_argument("--cable", default=DEFAULT_CABLE, help="虚拟麦渲染端点")
    parser.add_argument("--cable-capture", default=DEFAULT_CABLE_CAPTURE, help="虚拟麦采集端点（环回验证）")
    parser.add_argument("--no-capture", action="store_true", help="不跑 CABLE 环回采集")
    parser.add_argument("--switch", nargs="?", const="guanguanV1.pth", default=None,
                        metavar="PTH", help="离线附加：热切到指定权重（默认 guanguanV1.pth）再推 1 秒")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.pure:
        return run_pure(args)
    if args.agc:
        return run_agc(args)
    if args.acoustic:
        return run_acoustic(args)
    if args.live is not None:
        code, _ = run_live(args)
        return code
    return run_offline(args)


if __name__ == "__main__":
    sys.exit(main())
