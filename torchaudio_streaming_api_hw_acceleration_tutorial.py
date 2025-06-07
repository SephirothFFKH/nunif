import os
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

from torchaudio.io import StreamReader, StreamWriter

from torchaudio._extension.utils import _init_dll_path
from torchaudio.utils import ffmpeg_utils
for path in os.environ.get("PATH", "").split(";"):
    if os.path.exists(path):
        try:
            os.add_dll_directory(path)
        except Exception:
            pass
os.add_dll_directory(r"C:\\ffmpeg\\bin\\")
_init_dll_path()
print("Library versions:")
print(ffmpeg_utils.get_versions())
print("\nBuild config:")
print(ffmpeg_utils.get_build_config())
print("\nDecoders:")
print([k for k in ffmpeg_utils.get_video_decoders().keys() if "cuvid" in k])
print("\nEncoders:")
print([k for k in ffmpeg_utils.get_video_encoders().keys() if "nvenc" in k])

"""## Benchmarking GPU Encoding and Decoding

Now that FFmpeg and the resulting libraries are ready to use, we test NVDEC/NVENC with TorchAudio. For the basics of TorchAudio's streaming APIs, please refer to [Media I/O tutorials](https://pytorch.org/audio/main/tutorials.io.html).
"""

import time
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

"""## Benchmark NVDEC with `StreamReader`

First we test hardware decoding, and we fetch video from multiple locations (local file, network file, AWS S3) and use NVDEC to decod them.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

print(boto3.__version__)

"""First, we define the functions we'll use for testing.

Funcion `test_decode` decodes the given source from start to end, and it reports the elapsed time, and returns one image frmae as a sample.
"""

result = torch.zeros((4, 2))
samples = [[None, None] for _ in range(4)]


def test_decode(src, config, i_sample):
  print("=" * 40)
  print("* Configuration:", config)
  print("* Source:", src)
  print("=" * 40)

  t0 = time.monotonic()
  s = StreamReader(src)
  s.add_video_stream(5, **config)

  num_frames = 0
  for i, (chunk, ) in enumerate(s.stream()):
    if i == 0:
      print(' - Chunk:', chunk.shape, chunk.device, chunk.dtype)
    if i == i_sample:
      sample = chunk[0]
    num_frames += chunk.shape[0]
  elapsed = time.monotonic() - t0

  print()
  print(f" - Processed {num_frames} frames.")
  print(f" - Elapsed: {elapsed} seconds.")
  print()

  return elapsed, sample

"""### Decode MP4 from local file

For the first test, we compare the time it takes for CPU and NVDEC to decode 250MB of MP4 video.
"""

local_src = "input.mp4"

i_sample = 520

"""#### CPU"""

cpu_conf = {
    "decoder": "h264",  # CPU decoding
}

elapsed, sample = test_decode(local_src, cpu_conf, i_sample)

result[0, 0] = elapsed
samples[0][0] = sample

"""#### CUDA"""

cuda_conf = {
    "decoder": "h264_cuvid",  # Use CUDA HW decoder
    "hw_accel": "cuda:0",  # Then keep the memory on CUDA:0
}

elapsed, sample = test_decode(local_src, cuda_conf, i_sample)

result[0, 1] = elapsed
samples[0][1] = sample

"""### Decode MP4 from network

Let's run the same test on the source retrieved via network on-the-fly.
"""

network_src = "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
i_sample = 750

"""#### CPU"""

elapsed, sample = test_decode(network_src, cpu_conf, i_sample)

result[1, 0] = elapsed
samples[1][0] = sample

"""#### CUDA"""

elapsed, sample = test_decode(network_src, cuda_conf, i_sample)

result[1, 1] = elapsed
samples[1][1] = sample

"""### Decode MP4 directly from S3

Using file-like object input, we can fetch a video stored on AWS S3 and decode it without saving it on local file system.
"""

bucket = "pytorch"
key = "torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"

s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
i_sample = 115

"""#### Defining Helper class

StreamReader supports file-like objects with `read` method. In addition to this,
if the file-like object has `seek` method, StreamReader attempts to use it for more reliable detection of media formats.

However, the seek method of `boto3`'s S3 client response object only raises errors to let users know that seek operation is not supported. Therefore we wrap it with a class that does not have `seek` method. This way, StreamReader won't try to use the `seek` method.

**Note**

Due to the nature of streaming, when using file-like object without seek method, some formats are not supported. For example, MP4 formats contain metadata at the beginning of file or at the end. If metadata is located at the end, without `seek` method, StreamReader cannot decode streams.
"""

# Wrapper to hide the native `seek` method of boto3, which
# only raises an error.
class UnseekableWrapper:
  def __init__(self, obj):
    self.obj = obj

  def read(self, n):
    return self.obj.read(n)

  def __str__(self):
    return str(self.obj)

"""#### CPU"""

response = s3_client.get_object(Bucket=bucket, Key=key)
src = UnseekableWrapper(response["Body"])
elapsed, sample = test_decode(src, cpu_conf, i_sample)

result[2, 0] = elapsed
samples[2][0] = sample

"""#### CUDA"""

response = s3_client.get_object(Bucket=bucket, Key=key)
src = UnseekableWrapper(response["Body"])
elapsed, sample = test_decode(src, cuda_conf, i_sample)

result[2, 1] = elapsed
samples[2][1] = sample

"""### Decoding and resizing

In the next test, we add preprocessing. NVDEC supports several preprocessing schemes, which are also performed on the chosen hardware. For CPU, we apply the same kind of software preprocessing through FFmpeg's filter graph.
"""

i_sample = 1085

"""#### CPU"""

cpu_conf = {
    "decoder": "h264",  # CPU decoding
    "filter_desc": "scale=360:240",  # Software filter
}

elapsed, sample = test_decode(local_src, cpu_conf, i_sample)

result[3, 0] = elapsed
samples[3][0] = sample

"""#### CUDA"""

cuda_conf = {
    "decoder": "h264_cuvid",  # Use CUDA HW decoder
    "decoder_option": {
        "resize": "360x240",  # Then apply HW preprocessing (resize)
    },
    "hw_accel": "cuda:0",  # Then keep the memory on CUDA:0
}

elapsed, sample = test_decode(local_src, cuda_conf, i_sample)

result[3, 1] = elapsed
samples[3][1] = sample

"""### Results

The following table summarizes the time it took to decode the same media with CPU and NVDEC.
We see significant speedup with NVDEC.
"""

res = pd.DataFrame(
    result.numpy(),
    index=["Decoding (local file)", "Decoding (network file)", "Decoding (file-like object, S3)", "Decoding + Resize"],
    columns=["CPU", "NVDEC"],
)
print(res)

"""The following code shows some frames generated by CPU decoding and NVDEC. They produce seemingly identical results."""

def yuv_to_rgb(img):
  img = img.cpu().to(torch.float)
  y = img[..., 0, :, :]
  u = img[..., 1, :, :]
  v = img[..., 2, :, :]

  y /= 255
  u = u / 255 - 0.5
  v = v / 255 - 0.5

  r = y + 1.14 * v
  g = y + -0.396 * u - 0.581 * v
  b = y + 2.029 * u

  rgb = torch.stack([r, g, b], -1)
  rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
  return rgb.numpy()

f, axs = plt.subplots(4, 2, figsize=[12.8, 19.2])
for i in range(4):
  for j in range(2):
    axs[i][j].imshow(yuv_to_rgb(samples[i][j]))
    axs[i][j].set_title(
        f"{'CPU' if j == 0 else 'NVDEC'}{' with resize' if i == 3 else ''}")
plt.plot(block=False)

"""## Benchmark NVENC with `StreamWriter`

Next, we benchmark encoding speed with StreamWriter and NVENC.


"""

def test_encode(data, dst, **config):
  print("=" * 40)
  print("* Configuration:", config)
  print("* Destination:", dst)
  print("=" * 40)

  t0 = time.monotonic()
  s = StreamWriter(dst)
  s.add_video_stream(**config)
  with s.open():
    s.write_video_chunk(0, data)
  elapsed = time.monotonic() - t0

  print()
  print(f" - Processed {len(data)} frames.")
  print(f" - Elapsed: {elapsed} seconds.")
  print()
  return elapsed

result = torch.zeros((3, 3))

"""We use ``StreamReader`` to generate test data."""

def get_data(frame_rate, height, width, format, duration=15):
  src = f"testsrc2=rate={frame_rate}:size={width}x{height}:duration={duration}"
  s = StreamReader(src=src, format="lavfi")
  s.add_basic_video_stream(-1, format=format)
  s.process_all_packets()
  video, = s.pop_chunks()
  return video

"""### Encode MP4 - 360P

For the first test, we compare the time it takes for CPU and NVENC to encode 15 seconds of video with small resolution.
"""

pict_config = {
    "height": 360,
    "width": 640,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

video = get_data(**pict_config)

"""#### CPU"""

encode_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
}

result[0, 0] = test_encode(video, "360p_cpu.mp4", **pict_config, **encode_config)

"""#### CUDA (from CPU Tensor)

Now we test NVENC. This time, the data is sent from CPU memory to GPU memory as part of encoding.
"""

encode_config = {
    "encoder": "h264_nvenc",  # Use NVENC
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},  # Run encoding on the cuda:0 device
}

result[1, 0] = test_encode(video, "360p_cuda.mp4", **pict_config, **encode_config)

"""#### CUDA (from CUDA Tensor)

If the data is already present on CUDA, then we can pass it to GPU encoder directly.
"""

device = "cuda:0"

encode_config = {
    "encoder": "h264_nvenc",  # GPU Encoder
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},  # Run encoding on the cuda:0 device
    "hw_accel": device,  # Data comes from cuda:0 device
}

result[2, 0] = test_encode(video.to(torch.device(device)), "360p_cuda_hw.mp4", **pict_config, **encode_config)

"""### Encode MP4 - 720P

Let's run the same tests on video with larger resolution.
"""

pict_config = {
    "height": 720,
    "width": 1280,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

video = get_data(**pict_config)

"""#### CPU"""

encode_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
}

result[0, 1] = test_encode(video, "720p_cpu.mp4", **pict_config, **encode_config)

"""#### CUDA (from CPU Tensor)"""

encode_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
}

result[1, 1] = test_encode(video, "720p_cuda.mp4", **pict_config, **encode_config)

"""#### CUDA (from CUDA Tensor)"""

device = "cuda:0"

encode_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},
    "hw_accel": device,
}

result[2, 1] = test_encode(video.to(torch.device(device)), "720p_cuda_hw.mp4", **pict_config, **encode_config)

"""### Encode MP4 - 1080P

We make the video with even larger.
"""

pict_config = {
    "height": 1080,
    "width": 1920,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

video = get_data(**pict_config)

"""#### CPU"""

encode_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
}

result[0, 2] = test_encode(video, "1080p_cpu.mp4", **pict_config, **encode_config)

"""#### CUDA (from CPU Tensor)"""

encode_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
}

result[1, 2] = test_encode(video, "1080p_cuda.mp4", **pict_config, **encode_config)

"""#### CUDA (from CUDA Tensor)"""

device = "cuda:0"

encode_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},
    "hw_accel": device,
}

result[2, 2] = test_encode(video.to(torch.device(device)), "1080p_cuda_hw.mp4", **pict_config, **encode_config)

"""### Result

Here is the result.
"""

labels = ["CPU", "CUDA (from CPU Tensor)", "CUDA (from CUDA Tensor)"]
columns = ["360P", "720P", "1080P"]
res = pd.DataFrame(
    result.numpy(),
    index=labels,
    columns=columns,
)
print(res)

plt.plot(result.T)
plt.legend(labels)
plt.xticks([i for i in range(3)], columns)
plt.grid(visible=True, axis='y')

"""The resulting videos look like the following."""

from IPython.display import HTML

HTML('''
<div>
  <video width=360 controls autoplay>
    <source src="https://download.pytorch.org/torchaudio/tutorial-assets/streamwriter_360p_cpu.mp4" type="video/mp4">
  </video>
  <video width=360 controls autoplay>
    <source src="https://download.pytorch.org/torchaudio/tutorial-assets/streamwriter_360p_cuda.mp4" type="video/mp4">
  </video>
  <video width=360 controls autoplay>
    <source src="https://download.pytorch.org/torchaudio/tutorial-assets/streamwriter_360p_cuda_hw.mp4" type="video/mp4">
  </video>
</div>
''')

"""## Conclusion

We looked at how to build FFmpeg libraries with NVDEC/NVENC support and use them from TorchAudio. NVDEC/NVENC provide significant speed up when saving/loading a video.
"""