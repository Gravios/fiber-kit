#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════
#  neuro_extract.py — pull channels (and time ranges) out of an interleaved
#  int16 recording: .dat, .fil, .lfp, .eeg.
#
#  neurosuite-3 already ships process_extractchannels, and its channel-spec
#  grammar is reproduced here rather than replaced -- `5`, `5*1.5` for a gain,
#  `5-2` to reference channel 5 against channel 2.  A second tool that took the
#  same arguments and meant something different by them is exactly the kind of
#  divergence this codebase keeps paying for.  What this adds is the two things
#  the C++ tool has no notion of, both needed to get an LFP subset out of a
#  session without moving a hundred gigabytes:
#
#    * a TIME RANGE, so a short segment of many channels is as cheap as a long
#      segment of few;
#    * the session yaml, so nChannels comes from the recording rather than from
#      a number retyped on the command line.  Getting nChannels wrong does not
#      fail -- it silently shears the interleave and produces a file that looks
#      like data.
#
#  Everything is chunked.  A 96-channel wideband recording of this session is
#  ~132 GB, so nothing here may assume the input fits in memory, and the output
#  is written incrementally.
# ════════════════════════════════════════════════════════════════════════════
import argparse
import os
import re
import sys

import numpy as np

DTYPE = np.int16
# The gain charset must NOT contain a bare '-', or "5*0.5-7" parses its gain as
# "0.5-7" and the reference is silently lost.  A leading minus is allowed only
# immediately after the '*', and an exponent's sign only after e/E.
_SPEC = re.compile(
    r"^\s*(\d+)\s*"
    r"(?:\*\s*(-?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)\s*)?"
    r"(?:-\s*(\d+)\s*)?$")


class ChannelSpec:
    """One entry of the channel list: index, optional gain, optional reference."""

    __slots__ = ("channel", "gain", "reference", "text")

    def __init__(self, channel, gain=1.0, reference=None, text=""):
        self.channel = int(channel); self.gain = float(gain)
        self.reference = None if reference is None else int(reference)
        self.text = text or str(channel)

    def __repr__(self):
        return f"<{self.text}: ch{self.channel}" + \
            (f" -ch{self.reference}" if self.reference is not None else "") + \
            (f" x{self.gain:g}" if self.gain != 1.0 else "") + ">"


def parse_spec(token):
    """'5' | '5*1.5' | '5-2' | '5*1.5-2'  ->  ChannelSpec.

    Refuses anything it does not fully understand instead of taking the leading
    integer and dropping the rest: a typo in a gain or a reference would
    otherwise extract the right channel with the wrong arithmetic, which is not
    detectable downstream.
    """
    m = _SPEC.match(str(token))
    if not m:
        raise ValueError(f"bad channel spec {token!r}; want N, N*gain, N-ref or N*gain-ref")
    ch, g, ref = m.group(1), m.group(2), m.group(3)
    return ChannelSpec(int(ch), float(g) if g else 1.0,
                       int(ref) if ref is not None else None, str(token))


def session_nchannels(yaml_path):
    """nChannels and sampling rates from the session yaml."""
    import yaml
    d = yaml.safe_load(open(yaml_path))
    acq = d.get("acquisitionSystem", {})
    fp = d.get("fieldPotentials", {})
    return dict(n_channels=int(acq["nChannels"]),
                sampling_rate=float(acq["samplingRate"]),
                lfp_rate=float(fp.get("lfpSamplingRate", 0)) or None,
                n_bits=int(acq.get("nBits", 16)))


def probe_geometry(probe_path, channels=None, base=0):
    """Site xy for a probe file, optionally for a channel subset.

    `base` is the global channel id of site 0 -- probe 1 of a two-probe session
    starts at 64, and the probe file itself does not know that.
    """
    import yaml
    g = np.asarray(yaml.safe_load(open(probe_path))["probeFile"]["sites"]["geometry"], float)
    if channels is None:
        return g
    idx = np.asarray(channels, int) - int(base)
    if idx.min() < 0 or idx.max() >= len(g):
        raise ValueError(f"channels {channels} map outside the probe's {len(g)} sites "
                         f"with base={base}")
    return g[idx]


def n_samples(path, n_channels):
    size = os.path.getsize(path)
    frame = n_channels * np.dtype(DTYPE).itemsize
    if size % frame:
        raise ValueError(f"{path}: {size} bytes is not a whole number of "
                         f"{n_channels}-channel frames — nChannels is probably wrong")
    return size // frame


def extract(inp, out, n_channels, specs, start=0, stop=None, reverse=False,
            chunk_frames=1 << 20, progress=None):
    """Write the selected channels of `inp` to `out`, frames [start, stop).

    Reads and writes in chunks; peak memory is chunk_frames * n_channels * 2
    bytes regardless of file size.  Returns the number of frames written.
    """
    total = n_samples(inp, n_channels)
    stop = total if stop is None else min(int(stop), total)
    start = max(int(start), 0)
    if stop <= start:
        raise ValueError(f"empty range [{start}, {stop}) of {total} frames")
    for s in specs:
        for c in (s.channel, s.reference):
            if c is not None and not (0 <= c < n_channels):
                raise ValueError(f"channel {c} outside 0..{n_channels-1}")
    sign = -1.0 if reverse else 1.0
    src = np.memmap(inp, dtype=DTYPE, mode="r", shape=(total, n_channels))
    written = 0
    with open(out, "wb") as fh:
        for a in range(start, stop, chunk_frames):
            b = min(a + chunk_frames, stop)
            blk = np.asarray(src[a:b], np.float32)
            cols = []
            for s in specs:
                v = blk[:, s.channel]
                if s.reference is not None:
                    v = v - blk[:, s.reference]
                cols.append(v * (s.gain * sign))
            o = np.stack(cols, 1)
            np.clip(o, np.iinfo(DTYPE).min, np.iinfo(DTYPE).max, out=o)
            fh.write(np.rint(o).astype(DTYPE).tobytes())
            written += b - a
            if progress:
                progress(written, stop - start)
    del src
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="neuro-extract",
        description="Extract channels and a time range from an interleaved int16 "
                    "recording (.dat/.fil/.lfp/.eeg).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("channels", nargs="+",
                    help="channel specs: N, N*gain, N-ref, N*gain-ref (0-based)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--yaml", help="session yaml; nChannels is read from it")
    src.add_argument("--nchannels", type=int, help="channel count of the INPUT file")
    ap.add_argument("--rate", type=float, default=None,
                    help="sampling rate of the input, for --start/--stop in seconds; "
                         "with --yaml, defaults to lfpSamplingRate for .lfp/.eeg "
                         "inputs and samplingRate otherwise")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--stop", type=float, default=None, help="seconds")
    ap.add_argument("--reverse", action="store_true",
                    help="negate signals (some acquisition systems)")
    ap.add_argument("--chunk", type=int, default=1 << 20, help="frames per chunk")
    ap.add_argument("--probe", default=None,
                    help="probe file; report each extracted channel's site position")
    ap.add_argument("--probe-base", type=int, default=0, dest="probe_base",
                    help="global channel id of the probe's site 0 (e.g. 64 for probe 1)")
    a = ap.parse_args(argv)

    specs = [parse_spec(t) for t in a.channels]
    if a.yaml:
        info = session_nchannels(a.yaml)
        nch = info["n_channels"]
        rate = a.rate
        if rate is None:
            is_lfp = os.path.splitext(a.input)[1].lower() in (".lfp", ".eeg")
            rate = (info["lfp_rate"] or info["sampling_rate"]) if is_lfp \
                else info["sampling_rate"]
    else:
        nch, rate = a.nchannels, a.rate
    if rate is None and (a.start or a.stop is not None):
        ap.error("--start/--stop need a rate: pass --rate or --yaml")

    total = n_samples(a.input, nch)
    s0 = int(round(a.start * rate)) if rate else 0
    s1 = int(round(a.stop * rate)) if (a.stop is not None and rate) else None

    print(f"input      {a.input}")
    print(f"  {nch} channels x {total} frames"
          + (f" = {total/rate:.0f} s at {rate:g} Hz" if rate else ""))
    print(f"extracting {len(specs)} channels: " + ", ".join(s.text for s in specs))
    if a.probe:
        try:
            xy = probe_geometry(a.probe, [s.channel for s in specs], a.probe_base)
            for s, p in zip(specs, xy):
                print(f"    ch{s.channel:<4d} site x={p[0]:8.1f}  y={p[1]:8.1f} um")
        except Exception as e:
            print(f"    (probe lookup failed: {e})", file=sys.stderr)

    def prog(done, tot):
        pct = 100.0 * done / max(tot, 1)
        print(f"\r  {pct:5.1f}%  {done}/{tot} frames", end="", flush=True)

    n = extract(a.input, a.output, nch, specs, start=s0, stop=s1,
                reverse=a.reverse, chunk_frames=a.chunk, progress=prog)
    print()
    out_size = os.path.getsize(a.output)
    print(f"output     {a.output}")
    print(f"  {len(specs)} channels x {n} frames = {out_size/1e6:.1f} MB"
          + (f" = {n/rate:.0f} s" if rate else ""))
    print(f"\nRead it back as: np.memmap(path, dtype=np.int16).reshape(-1, {len(specs)})")
    print("Column order matches the channel list above, NOT the original channel ids.")
    if rate:
        print(f"Sample index from a .res timestamp:  res * {rate:g} / <acq rate>"
              + (f" - {s0}" if s0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
