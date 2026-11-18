"""Export a recorded physical-grain demonstration without reconstructing physics.

Images come from next_obs and measurements from the same row of physics, so
every video frame is post-action aligned. Input HDF5 files are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("top1", "left_wrist", "right_wrist")
CAMERA_LABELS = ("Top camera", "Left wrist", "Right wrist")
FIELDS = ("elapsed_s", "balance_g", "mass_g", "raw_net_mass_g", "spilled_g")
WINDOW = 32
CACHE_BYTES = 16 * 1024 * 1024
PAD, GAP, HEADER, LABEL, FOOTER = 12, 8, 136, 24, 32


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_attribute(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def panel_layout(height: int, width: int):
    canvas_width = max(720, 2 * PAD + 3 * width + 2 * GAP)
    canvas_width += canvas_width % 2
    canvas_height = 2 * PAD + HEADER + LABEL + height + FOOTER
    canvas_height += canvas_height % 2
    slot = (canvas_width - 2 * PAD - 2 * GAP) // 3
    top = PAD + HEADER + LABEL
    boxes = [(PAD + i * (slot + GAP) + (slot - width) // 2, top, width, height)
             for i in range(3)]
    return (canvas_width, canvas_height), boxes


def compose_frame(images, values, *, episode, target_g, speed, fps, index, count, fonts):
    """Copy RGB panels unchanged; draw explanatory text outside those panels."""
    size, boxes = panel_layout(*images[0].shape[:2])
    canvas = Image.new("RGB", size, (18, 23, 29))
    draw = ImageDraw.Draw(canvas)
    white, muted, amber = (240, 243, 247), (185, 197, 210), (246, 199, 118)
    draw.text((PAD, PAD), f"Weighing demonstration | {episode}", font=fonts[20], fill=white)
    draw.text((PAD, PAD + 29),
              f"Simulation: {values['elapsed_s']:.2f} s   |   Playback: {speed:g}x ({fps:g} fps)",
              font=fonts[16], fill=muted)
    draw.text((PAD, PAD + 52),
              f"Measured balance: {values['balance_g']:+.4f} g   |   Target: {target_g:.4f} g",
              font=fonts[17], fill=white)
    draw.text((PAD, PAD + 77),
              f"Contained inventory (diagnostic): {values['mass_g']:.4f} g",
              font=fonts[16], fill=amber)
    draw.text((PAD, PAD + 101),
              f"Tared force / gravity: {values['raw_net_mass_g']:+.4f} g   |   Spill (diagnostic): {values['spilled_g']:.4f} g",
              font=fonts[14], fill=muted)
    for label, rgb, (left, top, width, height) in zip(CAMERA_LABELS, images, boxes):
        draw.text((left, PAD + HEADER), label, font=fonts[15], fill=white)
        canvas.paste(Image.fromarray(rgb), (left, top))
    footer_y = boxes[0][1] + boxes[0][3] + 6
    draw.text((PAD, footer_y),
              f"Frame {index + 1}/{count} | Post-action images and measurements from recorded row {index}",
              font=fonts[13], fill=muted)
    return np.asarray(canvas)


def export_demo(input_path: Path, output_path: Path, *, episode="demo_0", speed=1., overwrite=False):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == input_path or metadata_path == input_path:
        raise ValueError("The input file cannot be an output destination")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Video output must have an .mp4 extension")
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be finite and positive")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError("Output video or metadata already exists; use --overwrite explicitly")
    group_path = episode if episode.startswith("data/") else "data/" + episode
    started = time.perf_counter()
    original_stat = input_path.stat()
    unique = uuid.uuid4().hex
    temporary_video = output_path.with_name(f".{output_path.stem}.{unique}.tmp.mp4")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{unique}.tmp")
    try:
        with h5py.File(input_path, "r", rdcc_nbytes=CACHE_BYTES, rdcc_nslots=1009) as handle:
            group = handle[group_path]
            if text_attribute(handle.attrs.get("observation_timing", "")) != "pre_action":
                raise ValueError("Expected pre-action schema with explicit next_obs post-action images")
            dynamics = group.attrs.get("powder_dynamics", handle.attrs.get("powder_dynamics", ""))
            if (text_attribute(dynamics) != "physical_grains_v1"
                    or int(group.attrs.get("success", 0)) != 1
                    or int(group.attrs.get("physical_valid", 0)) != 1):
                raise ValueError("Expected a declared successful, physically valid granular demonstration")
            hz = float(group.attrs.get("control_hz", handle.attrs.get("control_hz", 10.)))
            if not math.isclose(hz, 10., rel_tol=0., abs_tol=1e-9):
                raise ValueError("This exporter expects the recorded 10 Hz physical-demo control rate")
            target_g = float(group.attrs["target_g"])
            if not math.isfinite(target_g) or target_g <= 0:
                raise ValueError("Episode target must be finite and positive")
            # Keep these Dataset objects alive for the entire export. Reopening
            # them per frame loses the HDF5 temporal-chunk cache.
            cameras = [group[f"next_obs/{name}"] for name in CAMERAS]
            fields = {name: group[f"physics/{name}"] for name in FIELDS}
            shape = cameras[0].shape
            if (len(shape) != 4 or shape[-1] != 3 or shape[0] < 1
                    or shape[1] < 1 or shape[2] < 1
                    or any(dataset.shape != shape or dataset.dtype != np.dtype("uint8") for dataset in cameras)):
                raise ValueError("All three cameras must have matching nonempty uint8 N x H x W x 3 arrays")
            count, height, width, _ = shape
            if any(dataset.shape != (count,) for dataset in fields.values()):
                raise ValueError("Every physics measurement must have exactly one scalar per image row")
            declared_count = int(group.attrs.get("num_samples", count))
            if declared_count != count:
                raise ValueError("Declared sample count differs from camera/measurement rows")
            source_sha = file_sha256(input_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fonts = {size: ImageFont.load_default(size=size) for size in (13, 14, 15, 16, 17, 20)}
            fps = hz * speed
            first_values = last_values = None
            previous_time = None
            written = batches = 0
            with imageio.get_writer(str(temporary_video), format="FFMPEG", fps=fps,
                                    codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=1, output_params=["-movflags", "+faststart"]) as writer:
                for start in range(0, count, WINDOW):
                    stop = min(start + WINDOW, count)
                    image_blocks = [dataset[start:stop] for dataset in cameras]
                    sensor_blocks = {name: dataset[start:stop] for name, dataset in fields.items()}
                    if any(not np.isfinite(block).all() for block in sensor_blocks.values()):
                        raise ValueError(f"Nonfinite measurement in rows {start}:{stop}")
                    times = np.asarray(sensor_blocks["elapsed_s"], dtype=float)
                    intervals = np.diff(np.r_[previous_time, times]) if previous_time is not None else np.diff(times)
                    if times[0] < 0 or not np.allclose(intervals, 1. / hz, rtol=1e-5, atol=1e-6):
                        raise ValueError(f"Recorded time intervals differ from 10 Hz near row {start}")
                    previous_time = float(times[-1])
                    for offset in range(stop - start):
                        values = {name: float(block[offset]) for name, block in sensor_blocks.items()}
                        frame = compose_frame([block[offset] for block in image_blocks], values,
                                              episode=group_path.rsplit("/", 1)[-1], target_g=target_g,
                                              speed=speed, fps=fps, index=start+offset, count=count, fonts=fonts)
                        writer.append_data(frame)
                        if first_values is None:
                            first_values = values.copy()
                        last_values = values.copy()
                        written += 1
                    batches += 1
                    if batches % 10 == 0 or stop == count:
                        print(json.dumps({"event": "video_export_progress", "frames": written,
                                          "total_frames": count}), flush=True)
            final_stat = input_path.stat()
            if (final_stat.st_size, final_stat.st_mtime_ns) != (original_stat.st_size, original_stat.st_mtime_ns):
                raise RuntimeError("Input HDF5 changed during export; refusing to publish the video")
            if written != count or not temporary_video.is_file() or temporary_video.stat().st_size == 0:
                raise RuntimeError("Video export did not produce all recorded frames")
            size, boxes = panel_layout(height, width)
            metadata = dict(
                input_path=str(input_path), input_sha256=source_sha, input_bytes=original_stat.st_size,
                input_open_mode="read_only", input_size_and_mtime_unchanged=True,
                episode_path=group_path, input_declared_success=True, input_declared_physical_valid=True,
                exporter_revalidates_physics=False, powder_dynamics=text_attribute(dynamics),
                source_version=text_attribute(handle.attrs.get("source_version", "")),
                output_path=str(output_path), output_sha256=file_sha256(temporary_video),
                frame_count=written, source_control_hz=hz, output_fps=fps, playback_speed=speed,
                nominal_video_duration_s=written/fps, recorded_frame_period_s=1./hz,
                first_simulation_time_s=first_values["elapsed_s"], last_simulation_time_s=last_values["elapsed_s"],
                alignment="next_obs camera row i and physics measurement row i, both after action i",
                camera_datasets=[f"{group_path}/next_obs/{name}" for name in CAMERAS],
                measurement_datasets={name:f"{group_path}/physics/{name}" for name in FIELDS},
                measured_balance_field="balance_g", target_mass_g=target_g,
                diagnostic_overlay_fields=["mass_g", "spilled_g"],
                raw_sensor_mass_field="raw_net_mass_g", target_source="episode attribute target_g",
                first_frame_measurements=first_values, last_frame_measurements=last_values,
                source_image_shape=[height,width,3], output_frame_size=list(size),
                camera_panel_boxes_xywh={name:list(box) for name,box in zip(CAMERAS,boxes)},
                source_panels_resized=False, text_drawn_outside_camera_panels=True,
                video_codec="libx264", pixel_format="yuv420p", video_encoding="lossy",
                hdf_read_window_frames=WINDOW, hdf_read_batches=batches,
                persistent_dataset_handles=True, hdf_dataset_cache_bytes=CACHE_BYTES,
                overwrite_requested=bool(overwrite), wall_seconds=time.perf_counter()-started)
        temporary_metadata.write_text(json.dumps(metadata, indent=2)+"\n")
        temporary_video.replace(output_path)
        temporary_metadata.replace(metadata_path)
        return metadata
    finally:
        temporary_video.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", default="demo_0")
    parser.add_argument("--speed", type=float, default=1., help="1 = real time (10 fps); 4 = four times speed (40 fps)")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing output video and metadata")
    args = parser.parse_args()
    metadata = export_demo(args.input, args.output, episode=args.episode,
                           speed=args.speed, overwrite=args.overwrite)
    print(json.dumps({key:metadata[key] for key in ("output_path","frame_count","output_fps","playback_speed","wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
