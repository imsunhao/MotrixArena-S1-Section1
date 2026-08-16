#!/usr/bin/env python3
"""Record an X11 display for a fixed wall-clock duration and finalize with EOS."""

import argparse
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", default=":98")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()

    Gst.init(None)
    pipeline = Gst.parse_launch(
        "ximagesrc name=source use-damage=false show-pointer=false do-timestamp=true "
        "! identity sync=true ! video/x-raw,framerate=30/1 ! videoconvert "
        "! vp8enc deadline=1 target-bitrate=4500000 ! webmmux "
        f"! filesink location={args.output}"
    )
    pipeline.get_by_name("source").set_property("display-name", args.display)
    source = pipeline.get_by_name("source")
    source.set_property("num-buffers", round(args.seconds * 30))
    frame_index = 0

    def rewrite_timestamp(_pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        nonlocal frame_index
        buffer = info.get_buffer()
        if buffer is not None:
            timestamp = frame_index * Gst.SECOND // 30
            buffer.pts = timestamp
            buffer.dts = timestamp
            buffer.duration = Gst.SECOND // 30
            frame_index += 1
        return Gst.PadProbeReturn.OK

    source.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, rewrite_timestamp)
    loop = GLib.MainLoop()

    def on_message(_bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"GStreamer error: {error}; {debug}")
        if message.type == Gst.MessageType.EOS:
            loop.quit()

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
