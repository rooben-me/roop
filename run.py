#!/usr/bin/env python3
import sys
import time
import shutil
import core.globals
import gradio as gr

if not shutil.which('ffmpeg'):
    print('ffmpeg is not installed. Read the docs: https://github.com/s0md3v/roop#installation.\n' * 10)
    quit()
if '--gpu' not in sys.argv:
    core.globals.providers = ['CPUExecutionProvider']
elif 'ROCMExecutionProvider' not in core.globals.providers:
    import torch
    if not torch.cuda.is_available():
        quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")

import glob
import argparse
import multiprocessing as mp
import os
from pathlib import Path
from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames
from core.config import get_face
import webbrowser
import psutil
import cv2
import threading
from PIL import Image

pool = None
args = {}

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='source_img')
parser.add_argument('-t', '--target', help='replace this face', dest='target_path')
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--keep-fps', help='maintain original fps', dest='keep_fps', action='store_true', default=False)
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--keep-frames', help='keep frames directory', dest='keep_frames', action='store_true', default=False)
parser.add_argument('--cores', help='number of cores to use', dest='cores_count', type=int)

for name, value in vars(parser.parse_args()).items():
    args[name] = value

if not args['cores_count']:
    args['cores_count'] = psutil.cpu_count()-1

sep = "/"
if os.name == "nt":
    sep = "\\"


def start_processing():
    start_time = time.time()
    if args['gpu']:
        process_video(args['source_img'], args["frame_paths"])
        end_time = time.time()
        print(flush=True)
        print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)
        return
    frame_paths = args["frame_paths"]
    n = len(frame_paths)//(args['cores_count'])
    processes = []
    for i in range(0, len(frame_paths), n):
        p = pool.apply_async(process_video, args=(args['source_img'], frame_paths[i:i+n],))
        processes.append(p)
    for p in processes:
        p.get()
    pool.close()
    pool.join()
    end_time = time.time()
    print(flush=True)
    print(f"Processing time: {end_time - start_time:.2f} seconds", flush=True)

def start(source_img, target_path, limit_fps=True, keep_frames=False):
    args['source_img'] = source_img.name
    args['target_path'] = target_path.name
    args['keep_fps'] = not limit_fps
    args['keep_frames'] = keep_frames

    print("DON'T WORRY. IT'S NOT STUCK/CRASHED.\n" * 5)
    if not args['source_img'] or not os.path.isfile(args['source_img']):
        return "Please select an image containing a face."
    elif not args['target_path'] or not os.path.isfile(args['target_path']):
        return "Please select a video/image to swap face in."
    if not args['output_file']:
        args['output_file'] = rreplace(args['target_path'], "/", "/swapped-", 1) if "/" in target_path else "swapped-"+target_path
    global pool
    pool = mp.Pool(args['cores_count'])
    target_path = args['target_path']
    test_face = get_face(cv2.imread(args['source_img']))
    if not test_face:
        return "No face detected in source image. Please try with another one."
    if is_img(target_path):
        process_img(args['source_img'], target_path, args['output_file'])
        return "Swap successful!"
    video_name = os.path.basename(target_path)
    video_name = os.path.splitext(video_name)[0]
    output_dir = os.path.join(os.path.dirname(target_path),video_name)
    Path(output_dir).mkdir(exist_ok=True)
    fps = detect_fps(target_path)
    if not args['keep_fps'] and fps > 30:
        this_path = output_dir + "/" + video_name + ".mp4"
        set_fps(target_path, this_path, 30)
        target_path, fps = this_path, 30
    else:
        shutil.copy(target_path, output_dir)
    extract_frames(target_path, output_dir)
    args['frame_paths'] = tuple(sorted(
        glob.glob(output_dir + f"/*.png"),
        key=lambda x: int(x.split(sep)[-1].replace(".png", ""))
    ))
    start_processing()
    create_video(video_name, fps, output_dir)
    add_audio(output_dir, target_path, args['keep_frames'], args['output_file'])
    save_path = args['output_file'] if args['output_file'] else output_dir + "/" + video_name + ".mp4"
    print("\n\nVideo saved as:", save_path, "\n\n")
    return "Swap successful!"

iface = gr.Interface(
    fn=start,
    inputs=[
        gr.inputs.Image(label="Select a face"),
        gr.inputs.Image(label="Select a target"),
        gr.inputs.Checkbox(label="Limit FPS to 30", default=True),
        gr.inputs.Checkbox(label="Keep frames dir", default=False),
    ],
    outputs="text",
    title="roop",
    description="A face swapping tool",
    allow_flagging=False,
)

if __name__ == "__main__":
    if args['source_img']:
        args['cli_mode'] = True
        start()
        quit()
    else:
        iface.launch(share=True)
