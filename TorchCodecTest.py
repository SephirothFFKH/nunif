import torch

print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")
print(f"{torch.cuda.get_device_properties(0)=}")

import urllib.request

video_file = "video.mp4"
urllib.request.urlretrieve(
    "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
    video_file,
)

from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder(video_file, device="cuda")
frame = decoder[0]

print(frame.shape, frame.dtype)

print(frame.data.device)

timestamps = [12, 19, 45, 131, 180]
cpu_decoder = VideoDecoder(video_file, device="cpu")
cuda_decoder = VideoDecoder(video_file, device="cuda")
cpu_frames = cpu_decoder.get_frames_played_at(timestamps).data
cuda_frames = cuda_decoder.get_frames_played_at(timestamps).data


def plot_cpu_and_cuda_frames(cpu_frames: torch.Tensor, cuda_frames: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2.functional import to_pil_image
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(to_pil_image(cpu_frames[i].to("cpu")))
        axes[i][1].imshow(to_pil_image(cuda_frames[i].to("cpu")))

    axes[0][0].set_title("CPU decoder", fontsize=24)
    axes[0][1].set_title("CUDA decoder", fontsize=24)
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()


plot_cpu_and_cuda_frames(cpu_frames, cuda_frames)

frames_equal = torch.equal(cpu_frames.to("cuda"), cuda_frames)
mean_abs_diff = torch.mean(
    torch.abs(cpu_frames.float().to("cuda") - cuda_frames.float())
)
max_abs_diff = torch.max(torch.abs(cpu_frames.to("cuda").float() - cuda_frames.float()))
print(f"{frames_equal=}")
print(f"{mean_abs_diff=}")
print(f"{max_abs_diff=}")