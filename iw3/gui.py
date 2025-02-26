import nunif.pythonw_fix  # noqa
import locale
import sys
import os
from os import path
import traceback
import functools
from time import time
import threading
import wx
from wx.lib.delayedresult import startWorker
import wx.lib.agw.persist as persist
import wx.lib.stattext as stattext
from .utils import (
    create_parser, set_state_args, iw3_main,
    is_text, is_video, is_output_dir, is_yaml, make_output_filename,
    has_rembg_model)
from nunif.initializer import gc_collect
from nunif.device import mps_is_available, xpu_is_available
from nunif.utils.image_loader import IMG_EXTENSIONS as LOADER_SUPPORTED_EXTENSIONS
from nunif.utils.video import VIDEO_EXTENSIONS as KNOWN_VIDEO_EXTENSIONS, has_nvenc
from nunif.utils.filename import sanitize_filename
from nunif.utils.git import get_current_branch
from nunif.gui import (
    TQDMGUI, FileDropCallback, EVT_TQDM, TimeCtrl,
    EditableComboBox, EditableComboBoxPersistentHandler,
    persistent_manager_register_all, persistent_manager_unregister_all,
    persistent_manager_restore_all, persistent_manager_register,
    extension_list_to_wildcard, validate_number,
    set_icon_ex,
    VideoEncodingBox, IOPathPanel
)
from .locales import LOCALES
from . import models # noqa
from .depth_anything_model import DepthAnythingModel
from .depth_pro_model import DepthProModel
from .depth_pro_model import MODEL_FILES as DEPTH_PRO_MODELS
from . import export_config
import torch


IMAGE_EXTENSIONS = extension_list_to_wildcard(LOADER_SUPPORTED_EXTENSIONS)
VIDEO_EXTENSIONS = extension_list_to_wildcard(KNOWN_VIDEO_EXTENSIONS)
YAML_EXTENSIONS = extension_list_to_wildcard((".yml", ".yaml"))
CONFIG_DIR = path.join(path.dirname(__file__), "..", "tmp")
CONFIG_PATH = path.join(CONFIG_DIR, "iw3-gui.cfg")
PRESET_DIR = path.join(CONFIG_DIR, "presets")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)


LAYOUT_DEBUG = False


class IW3App(wx.App):
    def OnInit(self):
        main_frame = MainFrame()
        self.instance = wx.SingleInstanceChecker(main_frame.GetTitle())
        if self.instance.IsAnotherRunning():
            with wx.MessageDialog(None,
                                  message=(T("Another instance is running") + "\n" +
                                           T("Are you sure you want to do this?")),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                if dlg.ShowModal() == wx.ID_NO:
                    return False
        set_icon_ex(main_frame, path.join(path.dirname(__file__), "icon.ico"), main_frame.GetTitle())
        self.SetAppName(main_frame.GetTitle())
        main_frame.Show()
        self.SetTopWindow(main_frame)
        return True


class MainFrame(wx.Frame):
    def __init__(self):
        branch_name = get_current_branch()
        if branch_name is None or branch_name in {"master", "main"}:
            branch_tag = ""
        else:
            branch_tag = f" ({branch_name})"

        super(MainFrame, self).__init__(
            None,
            name="iw3-gui",
            title=T("iw3-gui") + branch_tag,
            size=(1100, 720),
            style=(wx.DEFAULT_FRAME_STYLE & ~wx.MAXIMIZE_BOX)
        )
        self.processing = False
        self.start_time = 0
        self.input_type = None
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self.suspend_pos = 0
        self.suspend_event.set()
        self.depth_model = None
        self.depth_model_type = None
        self.depth_model_device_id = None
        self.depth_model_height = None
        self.initialize_component()

    def initialize_component(self):
        NORMAL_FONT = wx.Font(10, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_FONT = wx.Font(8, family=wx.FONTFAMILY_MODERN, style=wx.FONTSTYLE_NORMAL, weight=wx.FONTWEIGHT_NORMAL)
        WARNING_COLOR = (0xcc, 0x33, 0x33)

        self.SetFont(NORMAL_FONT)
        self.CreateStatusBar()

        # input output panel
        input_wildcard = (f"Image and Video and YAML files|{IMAGE_EXTENSIONS};{VIDEO_EXTENSIONS};{YAML_EXTENSIONS}"
                          f"|Video files|{VIDEO_EXTENSIONS}"
                          f"|Image files|{IMAGE_EXTENSIONS}"
                          f"|YAML files|{YAML_EXTENSIONS}"
                          "|All Files|*.*")
        self.pnl_file = IOPathPanel(
            self,
            input_wildcard=input_wildcard,
            default_output_dir_name="iw3",
            resolve_output_path=self.resolve_output_path,
            translate_function=T,
        )

        self.pnl_file_option = wx.Panel(self)
        self.chk_resume = wx.CheckBox(self.pnl_file_option, label=T("Resume"), name="chk_resume")
        self.chk_resume.SetToolTip(T("Skip processing when the output file already exists"))
        self.chk_resume.SetValue(True)

        self.chk_recursive = wx.CheckBox(self.pnl_file_option, label=T("Process all subfolders"),
                                         name="chk_recursive")
        self.chk_recursive.SetValue(False)

        self.chk_exif_transpose = wx.CheckBox(self.pnl_file_option, label=T("EXIF Transpose"),
                                              name="chk_exif_transpose")
        self.chk_exif_transpose.SetValue(True)
        self.chk_exif_transpose.SetToolTip(T("Transpose images according to EXIF Orientaion Tag"))

        self.chk_metadata = wx.CheckBox(self.pnl_file_option, label=T("Add metadata to filename"),
                                        name="chk_metadata")
        self.chk_metadata.SetValue(False)

        self.sep_image_format = wx.StaticLine(self.pnl_file_option, size=(2, 16), style=wx.LI_VERTICAL)
        self.lbl_image_format = wx.StaticText(self.pnl_file_option, label=" " + T("Image Format"))
        self.cbo_image_format = wx.ComboBox(self.pnl_file_option, choices=["png", "jpeg", "webp"],
                                            style=wx.CB_READONLY, name="cbo_image_format")
        self.cbo_image_format.SetSelection(0)
        self.cbo_image_format.SetToolTip(T("Output Image Format"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.AddSpacer(4)
        layout.Add(self.chk_resume, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_recursive, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_exif_transpose, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_metadata, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sep_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_image_format, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL)
        self.pnl_file_option.SetSizer(layout)

        # options panel

        self.pnl_options = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_options.SetBackgroundColour("#cfc")

        # stereo generation settings
        # divergence, convergence, method, depth_model, mapper

        self.grp_stereo = wx.StaticBox(self.pnl_options, label=T("Stereo Generation"))

        self.lbl_divergence = wx.StaticText(self.grp_stereo, label=T("3D Strength"))
        self.cbo_divergence = EditableComboBox(self.grp_stereo, choices=["5.0", "4.0", "3.0", "2.5", "2.0", "1.0"],
                                               name="cbo_divergence")
        self.lbl_divergence_warning = stattext.GenStaticText(self.grp_stereo, label="")
        self.lbl_divergence_warning.SetFont(WARNING_FONT)
        self.lbl_divergence_warning.SetForegroundColour(WARNING_COLOR)
        self.lbl_divergence_warning.Hide()

        self.cbo_divergence.SetToolTip("Divergence")
        self.cbo_divergence.SetSelection(4)

        self.lbl_convergence = wx.StaticText(self.grp_stereo, label=T("Convergence Plane"))
        self.cbo_convergence = EditableComboBox(self.grp_stereo, choices=["0.0", "0.5", "1.0"],
                                                name="cbo_convergence")
        self.cbo_convergence.SetSelection(1)
        self.cbo_convergence.SetToolTip("Convergence")

        self.lbl_ipd_offset = wx.StaticText(self.grp_stereo, label=T("Your Own Size"))
        # SpinCtrlDouble is better, but cannot save with PersistenceManager
        self.sld_ipd_offset = wx.SpinCtrl(self.grp_stereo, value="0", min=-10, max=20, name="sld_ipd_offset")
        self.sld_ipd_offset.SetToolTip("IPD Offset")

        self.lbl_synthetic_view = wx.StaticText(self.grp_stereo, label=T("Synthetic View"))
        self.cbo_synthetic_view = wx.ComboBox(self.grp_stereo,
                                              choices=["both", "right", "left"],
                                              style=wx.CB_READONLY, name="cbo_synthetic_view")
        self.cbo_synthetic_view.SetSelection(0)

        self.lbl_method = wx.StaticText(self.grp_stereo, label=T("Method"))
        self.cbo_method = wx.ComboBox(self.grp_stereo,
                                      choices=["row_flow_v3", "row_flow_v3_sym", "row_flow_v2", "forward_fill"],
                                      style=wx.CB_READONLY, name="cbo_method")
        self.cbo_method.SetSelection(0)

        self.lbl_stereo_width = wx.StaticText(self.grp_stereo, label=T("Stereo Procesing Width"))
        self.cbo_stereo_width = EditableComboBox(self.grp_stereo,
                                                 choices=["Default", "1920", "1280", "640"],
                                                 name="cbo_stereo_width")
        self.cbo_stereo_width.SetSelection(0)
        self.cbo_stereo_width.SetToolTip(T("Only used for row_flow_v3 and row_flow_v2"))

        self.lbl_depth_model = wx.StaticText(self.grp_stereo, label=T("Depth Model"))
        depth_models = [
            "ZoeD_N", "ZoeD_K", "ZoeD_NK",
            "ZoeD_Any_N", "ZoeD_Any_K",
            "DepthPro", "DepthPro_S",
            "Any_S", "Any_B", "Any_L",
            "Any_V2_S",
        ]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_B"):
            depth_models.append("Any_V2_B")
        if DepthAnythingModel.has_checkpoint_file("Any_V2_L"):
            depth_models.append("Any_V2_L")

        depth_models += ["Any_V2_N_S", "Any_V2_N_B"]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_N_L"):
            depth_models.append("Any_V2_N_L")
        depth_models += ["Any_V2_K_S", "Any_V2_K_B"]
        if DepthAnythingModel.has_checkpoint_file("Any_V2_K_L"):
            depth_models.append("Any_V2_K_L")

        self.cbo_depth_model = wx.ComboBox(self.grp_stereo,
                                           choices=depth_models,
                                           style=wx.CB_READONLY, name="cbo_depth_model")
        self.cbo_depth_model.SetSelection(3)

        self.lbl_resolution = wx.StaticText(self.grp_stereo, label=T("Depth") + " " + T("Resolution"))
        self.cbo_resolution = EditableComboBox(self.grp_stereo,
                                               choices=["Default", "512"],
                                               name="cbo_zoed_resolution")
        self.cbo_resolution.SetSelection(0)

        self.lbl_foreground_scale = wx.StaticText(self.grp_stereo, label=T("Foreground Scale"))
        self.cbo_foreground_scale = EditableComboBox(self.grp_stereo,
                                                     choices=["-3", "-2", "-1", "0", "1", "2", "3"],
                                                     name="cbo_foreground_scale")
        self.cbo_foreground_scale.SetSelection(3)

        self.chk_edge_dilation = wx.CheckBox(self.grp_stereo, label=T("Edge Fix"), name="chk_edge_dilation")
        self.cbo_edge_dilation = EditableComboBox(self.grp_stereo,
                                                  choices=["0", "1", "2", "3", "4"],
                                                  name="cbo_edge_dilation")
        self.chk_edge_dilation.SetValue(False)

        self.cbo_edge_dilation.SetSelection(2)
        self.cbo_edge_dilation.SetToolTip(T("Reduce distortion of foreground and background edges"))

        self.lbl_stereo_format = wx.StaticText(self.grp_stereo, label=T("Stereo Format"))
        self.cbo_stereo_format = wx.ComboBox(
            self.grp_stereo,
            choices=["Full SBS", "Half SBS",
                     "Full TB", "Half TB",
                     "VR90",
                     "Cross Eyed",
                     "Anaglyph",
                     "Export", "Export disparity",
                     "Debug Depth",
                     ],
            style=wx.CB_READONLY, name="cbo_stereo_format")
        self.cbo_stereo_format.SetSelection(0)

        self.lbl_anaglyph_method = wx.StaticText(self.grp_stereo, label=T("Anaglyph Method"))
        self.cbo_anaglyph_method = wx.ComboBox(
            self.grp_stereo,
            choices=["dubois", "dubois2",
                     "color", "gray",
                     "half-color",
                     "wimmer", "wimmer2"],
            style=wx.CB_READONLY, name="cbo_anaglyph_method")
        self.cbo_anaglyph_method.SetSelection(0)
        self.lbl_anaglyph_method.Hide()
        self.cbo_anaglyph_method.Hide()

        self.chk_ema_normalize = wx.CheckBox(self.grp_stereo,
                                             label=T("Flicker Reduction"),
                                             name="chk_ema_normalize")
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))
        self.cbo_ema_decay = EditableComboBox(self.grp_stereo, choices=["0.99", "0.9", "0.75", "0.5"],
                                              name="cbo_ema_decay")
        self.cbo_ema_decay.SetSelection(2)
        self.chk_ema_normalize.SetToolTip(T("Video Only") + " " + T("(experimental)"))

        self.chk_export_depth_only = wx.CheckBox(self.grp_stereo, label=T("Depth Only"), name="chk_export_depth_only")
        self.chk_export_depth_only.SetValue(False)
        self.chk_export_depth_only.SetToolTip(T("Exporting depth images only.\n"
                                                "Note that exported data with this option cannot be imported."))
        self.chk_export_depth_only.Hide()

        self.chk_export_depth_fit = wx.CheckBox(self.grp_stereo, label=T("Resize to fit"), name="chk_export_depth_fit")
        self.chk_export_depth_fit.SetValue(False)
        self.chk_export_depth_fit.SetToolTip(T("Resize depth images to the same size as rgb images."))
        self.chk_export_depth_fit.Hide()

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.SetEmptyCellSize((0, 0))

        i = 0
        layout.Add(self.lbl_divergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_divergence, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_divergence_warning, pos=(i := i + 1, 0), span=(0, 2), flag=wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_convergence, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_convergence, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_ipd_offset, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.sld_ipd_offset, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_synthetic_view, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_synthetic_view, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_method, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_stereo_width, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_width, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_depth_model, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_depth_model, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_resolution, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_resolution, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_foreground_scale, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_foreground_scale, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_edge_dilation, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_edge_dilation, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_ema_normalize, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_ema_decay, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_stereo_format, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_stereo_format, (i, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_anaglyph_method, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_anaglyph_method, (i, 1), flag=wx.EXPAND)
        layout.Add(self.chk_export_depth_only, (i := i + 1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.chk_export_depth_fit, (i, 1), flag=wx.ALIGN_CENTER_VERTICAL)

        sizer_stereo = wx.StaticBoxSizer(self.grp_stereo, wx.VERTICAL)
        sizer_stereo.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # video encoding
        # sbs/vr180, padding
        # max-fps, crf, preset, tune
        self.grp_video = VideoEncodingBox(self.pnl_options, translate_function=T, has_nvenc=has_nvenc())

        # background removal
        self.grp_rembg = wx.StaticBox(self.pnl_options, label=T("Background Removal"))
        self.chk_rembg = wx.CheckBox(self.grp_rembg, label=T("Enable"), name="chk_rembg")
        self.lbl_bg_model = wx.StaticText(self.grp_rembg, label=T("Seg Model"))
        self.cbo_bg_model = wx.ComboBox(self.grp_rembg,
                                        choices=["u2net", "u2net_human_seg",
                                                 "isnet-general-use", "isnet-anime"],
                                        style=wx.CB_READONLY, name="cbo_bg_model")
        self.cbo_bg_model.SetSelection(1)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_rembg, (0, 0), (0, 2), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.lbl_bg_model, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_bg_model, (1, 1), flag=wx.EXPAND)
        sizer_rembg = wx.StaticBoxSizer(self.grp_rembg, wx.VERTICAL)
        sizer_rembg.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # input video filter
        # deinterlace, rotate, vf
        self.grp_video_filter = wx.StaticBox(self.pnl_options, label=T("Video Filter"))
        self.chk_start_time = wx.CheckBox(self.grp_video_filter, label=T("Start Time"),
                                          name="chk_start_time")
        self.txt_start_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                       name="txt_start_time")
        self.chk_end_time = wx.CheckBox(self.grp_video_filter, label=T("End Time"), name="chk_end_time")
        self.txt_end_time = TimeCtrl(self.grp_video_filter, value="00:00:00", fmt24hr=True,
                                     name="txt_end_time")

        self.lbl_deinterlace = wx.StaticText(self.grp_video_filter, label=T("Deinterlace"))
        self.cbo_deinterlace = wx.ComboBox(self.grp_video_filter, choices=["", "yadif"],
                                           style=wx.CB_READONLY, name="cbo_deinterlace")
        self.cbo_deinterlace.SetSelection(0)

        self.lbl_vf = wx.StaticText(self.grp_video_filter, label=T("-vf (src)"))
        self.txt_vf = wx.TextCtrl(self.grp_video_filter, name="txt_vf")

        self.lbl_rotate = wx.StaticText(self.grp_video_filter, label=T("Rotate"))
        self.cbo_rotate = wx.ComboBox(self.grp_video_filter, size=(200, -1),
                                      style=wx.CB_READONLY, name="cbo_rotate")
        self.cbo_rotate.Append("", "")
        self.cbo_rotate.Append(T("Left 90 (counterclockwise)"), "left")
        self.cbo_rotate.Append(T("Right 90 (clockwise)"), "right")
        self.cbo_rotate.SetSelection(0)

        self.lbl_pad = wx.StaticText(self.grp_video_filter, label=T("Padding"))
        self.cbo_pad = wx.ComboBox(self.grp_video_filter, choices=["", "1", "2"],
                                   style=wx.CB_DROPDOWN, name="cbo_pad")
        self.cbo_pad.SetSelection(0)

        self.lbl_max_output_size = wx.StaticText(self.grp_video_filter, label=T("Output Size Limit"))
        self.cbo_max_output_size = wx.ComboBox(self.grp_video_filter,
                                               choices=["",
                                                        "1920x1080", "1280x720", "640x360",
                                                        "1080x1920", "720x1280", "360x640"],
                                               style=wx.CB_READONLY, name="cbo_max_output_size")
        self.cbo_max_output_size.SetSelection(0)

        self.chk_keep_aspect_ratio = wx.CheckBox(self.grp_video_filter, label=T("Keep Aspect Ratio"),
                                                 name="chk_keep_aspect_ratio")
        self.chk_keep_aspect_ratio.SetValue(False)

        layout = wx.GridBagSizer(vgap=4, hgap=4)
        layout.Add(self.chk_start_time, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_start_time, (0, 1), flag=wx.EXPAND)
        layout.Add(self.chk_end_time, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_end_time, (1, 1), flag=wx.EXPAND)

        layout.Add(self.lbl_deinterlace, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_deinterlace, (2, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_vf, (3, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.txt_vf, (3, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_rotate, (4, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_rotate, (4, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_pad, (5, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_pad, (5, 1), flag=wx.EXPAND)
        layout.Add(self.lbl_max_output_size, (6, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_output_size, (6, 1), flag=wx.EXPAND)
        layout.Add(self.chk_keep_aspect_ratio, (7, 1), flag=wx.EXPAND)

        sizer_video_filter = wx.StaticBoxSizer(self.grp_video_filter, wx.VERTICAL)
        sizer_video_filter.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        # processor settings
        # device, batch-size, TTA, Low VRAM, fp16
        self.grp_processor = wx.StaticBox(self.pnl_options, label=T("Processor"))
        self.lbl_device = wx.StaticText(self.grp_processor, label=T("Device"))
        self.cbo_device = wx.ComboBox(self.grp_processor, size=(200, -1), style=wx.CB_READONLY,
                                      name="cbo_device")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_properties(i).name
                self.cbo_device.Append(device_name, i)
            if torch.cuda.device_count() > 0:
                self.cbo_device.Append(T("All CUDA Device"), -2)
        elif mps_is_available():
            self.cbo_device.Append("MPS", 0)
        elif xpu_is_available():
            for i in range(torch.xpu.device_count()):
                device_name = torch.xpu.get_device_name(i)
                self.cbo_device.Append(device_name, i)

        self.cbo_device.Append("CPU", -1)
        self.cbo_device.SetSelection(0)

        self.lbl_batch_size = wx.StaticText(self.grp_processor, label=T("Depth") + " " + T("Batch Size"))
        self.cbo_batch_size = wx.ComboBox(self.grp_processor,
                                          choices=[str(n) for n in (64, 32, 16, 8, 4, 2, 1)],
                                          style=wx.CB_READONLY, name="cbo_zoed_batch_size")
        self.cbo_batch_size.SetToolTip(T("Video Only"))
        self.cbo_batch_size.SetSelection(5)

        self.lbl_max_workers = wx.StaticText(self.grp_processor, label=T("Worker Threads"))
        self.cbo_max_workers = wx.ComboBox(self.grp_processor,
                                           choices=[str(n) for n in (16, 8, 4, 3, 2, 0)],
                                           style=wx.CB_READONLY, name="cbo_max_workers")
        self.cbo_max_workers.SetToolTip(T("Video Only"))
        self.cbo_max_workers.SetSelection(5)

        self.chk_low_vram = wx.CheckBox(self.grp_processor, label=T("Low VRAM"), name="chk_low_vram")
        self.chk_tta = wx.CheckBox(self.grp_processor, label=T("TTA"), name="chk_tta")
        self.chk_tta.SetToolTip(T("Use flip augmentation to improve depth quality (slow)"))
        self.chk_fp16 = wx.CheckBox(self.grp_processor, label=T("FP16"), name="chk_fp16")
        self.chk_fp16.SetToolTip(T("Use FP16 (fast)"))
        self.chk_fp16.SetValue(True)
        self.chk_cuda_stream = wx.CheckBox(self.grp_processor, label=T("Stream"), name="chk_cuda_stream")
        self.chk_cuda_stream.SetToolTip(T("Use per-thread CUDA Stream (experimental: fast or slow or crash)"))
        self.chk_cuda_stream.SetValue(False)

        layout = wx.GridBagSizer(vgap=5, hgap=4)
        layout.Add(self.lbl_device, (0, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_device, (0, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_batch_size, (1, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_batch_size, (1, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.lbl_max_workers, (2, 0), flag=wx.ALIGN_CENTER_VERTICAL)
        layout.Add(self.cbo_max_workers, (2, 1), (0, 3), flag=wx.EXPAND)
        layout.Add(self.chk_low_vram, (3, 0), flag=wx.EXPAND)
        layout.Add(self.chk_tta, (3, 1), flag=wx.EXPAND)
        layout.Add(self.chk_fp16, (3, 2), flag=wx.EXPAND)
        layout.Add(self.chk_cuda_stream, (3, 3), flag=wx.EXPAND)

        sizer_processor = wx.StaticBoxSizer(self.grp_processor, wx.VERTICAL)
        sizer_processor.Add(layout, 1, wx.ALL | wx.EXPAND, 4)

        layout = wx.GridBagSizer(wx.HORIZONTAL)
        layout.Add(sizer_stereo, (0, 0), (2, 0), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(self.grp_video.sizer, (0, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_rembg, (1, 1), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_video_filter, (0, 2), flag=wx.ALL | wx.EXPAND, border=4)
        layout.Add(sizer_processor, (1, 2), flag=wx.ALL | wx.EXPAND, border=4)
        self.pnl_options.SetSizer(layout)

        # preset panel
        self.pnl_preset = wx.Panel(self)
        self.lbl_preset = wx.StaticText(self.pnl_preset, label=" " + T("Preset"))
        self.cbo_app_preset = EditableComboBox(self.pnl_preset, choices=self.list_preset(),
                                               size=(200, -1),
                                               name="cbo_app_preset")
        self.cbo_app_preset.SetSelection(0)
        self.btn_load_preset = wx.Button(self.pnl_preset, label=T("Load"))
        self.btn_save_preset = wx.Button(self.pnl_preset, label=T("Save"))
        self.btn_delete_preset = wx.Button(self.pnl_preset, label=T("Delete"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.lbl_preset, 0, wx.ALIGN_CENTER_VERTICAL, 2)
        layout.Add(self.cbo_app_preset, 0, wx.ALL, 2)
        layout.Add(self.btn_load_preset, 0, wx.ALL, 2)
        layout.Add(self.btn_save_preset, 0, wx.ALL, 2)
        layout.Add(self.btn_delete_preset, 0, wx.ALL, 2)
        self.pnl_preset.SetSizer(layout)

        # processing panel
        self.pnl_process = wx.Panel(self)
        if LAYOUT_DEBUG:
            self.pnl_process.SetBackgroundColour("#fcc")
        self.prg_tqdm = wx.Gauge(self.pnl_process, style=wx.GA_HORIZONTAL)
        self.btn_start = wx.Button(self.pnl_process, label=T("Start"))
        self.btn_suspend = wx.Button(self.pnl_process, label=T("Suspend"))
        self.btn_cancel = wx.Button(self.pnl_process, label=T("Cancel"))

        layout = wx.BoxSizer(wx.HORIZONTAL)
        layout.Add(self.prg_tqdm, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 4)
        layout.Add(self.btn_start, 0, wx.ALL, 4)
        layout.Add(self.btn_suspend, 0, wx.ALL, 4)
        layout.Add(self.btn_cancel, 0, wx.ALL, 4)
        self.pnl_process.SetSizer(layout)

        # main layout

        layout = wx.BoxSizer(wx.VERTICAL)
        layout.AddSpacer(8)
        layout.Add(self.pnl_preset, 0, wx.ALL | wx.EXPAND, 2)
        layout.Add(self.pnl_file.panel, 0, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_file_option, 0, wx.ALL | wx.EXPAND, 4)
        layout.Add(self.pnl_options, 1, wx.ALL | wx.EXPAND, 8)
        layout.Add(self.pnl_process, 0, wx.ALL | wx.EXPAND, 8)
        self.SetSizer(layout)

        # bind
        self.pnl_file.bind_input_path_changed(self.on_text_changed_txt_input)
        self.pnl_file.bind_output_path_changed(self.on_text_changed_txt_output)

        self.cbo_divergence.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_synthetic_view.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.cbo_method.Bind(wx.EVT_TEXT, self.update_divergence_warning)
        self.lbl_divergence_warning.Bind(wx.EVT_LEFT_DOWN, self.on_click_divergence_warning)

        self.cbo_depth_model.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_depth_model)
        self.chk_edge_dilation.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_edge_dilation)
        self.chk_ema_normalize.Bind(wx.EVT_CHECKBOX, self.on_changed_chk_ema_normalize)

        self.cbo_stereo_format.Bind(wx.EVT_TEXT, self.on_selected_index_changed_cbo_stereo_format)

        self.btn_load_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_load_preset)
        self.btn_save_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_save_preset)
        self.btn_delete_preset.Bind(wx.EVT_BUTTON, self.on_click_btn_delete_preset)

        self.btn_start.Bind(wx.EVT_BUTTON, self.on_click_btn_start)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_click_btn_cancel)
        self.btn_suspend.Bind(wx.EVT_BUTTON, self.on_click_btn_suspend)

        self.Bind(EVT_TQDM, self.on_tqdm)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        editable_comboboxes = self.get_editable_comboboxes()

        self.SetDropTarget(FileDropCallback(self.on_drop_files))
        # Disable default drop target
        for control in (self.pnl_file.input_path_widget, self.pnl_file.output_path_widget, self.txt_vf,
                        self.cbo_pad, self.txt_start_time, self.txt_end_time, *editable_comboboxes):
            control.SetDropTarget(FileDropCallback(self.on_drop_files))

        # Fix Frame and Panel background colors are different in windows
        self.SetBackgroundColour(self.pnl_file_option.GetBackgroundColour())

        # state
        self.btn_cancel.Disable()
        self.btn_suspend.Disable()

        self.load_preset()

        self.update_start_button_state()
        self.update_rembg_state()
        self.update_input_option_state()
        self.update_anaglyph_state()
        self.update_export_option_state()

        if not self.chk_edge_dilation.IsChecked():
            self.update_model_selection()
        self.update_edge_dilation()
        self.update_ema_normalize()
        self.grp_video.update_controls()

        self.update_divergence_warning()

    def get_editable_comboboxes(self):
        editable_comboboxes = [
            self.cbo_divergence,
            self.cbo_convergence,
            self.cbo_resolution,
            self.cbo_stereo_width,
            self.cbo_edge_dilation,
            self.cbo_ema_decay,
            *self.grp_video.get_editable_comboboxes(),
            self.cbo_foreground_scale,
            self.cbo_app_preset,
        ]
        return editable_comboboxes

    def get_anaglyph_method(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            anaglyph = self.cbo_anaglyph_method.GetValue()
        else:
            anaglyph = None
        return anaglyph

    def on_close(self, event):
        self.save_preset()
        event.Skip()

    def on_drop_files(self, x, y, filenames):
        if filenames:
            self.pnl_file.set_input_path(filenames[0])
        return True

    def update_start_button_state(self):
        if not self.processing:
            if self.pnl_file.input_path and self.pnl_file.output_path:
                self.btn_start.Enable()
            else:
                self.btn_start.Disable()

    def update_rembg_state(self):
        if is_video(self.pnl_file.input_path):
            self.chk_rembg.SetValue(False)
            self.chk_rembg.Disable()
            self.cbo_bg_model.Disable()
        else:
            self.chk_rembg.Enable()
            self.cbo_bg_model.Enable()

    def update_input_option_state(self):
        input_path = self.pnl_file.input_path
        if is_yaml(input_path):
            try:
                config = export_config.ExportConfig.load(input_path)
                if config.type == export_config.IMAGE_TYPE:
                    self.chk_resume.Enable()
                    self.chk_recursive.Disable()
                else:
                    self.chk_resume.Disable()
                    self.chk_recursive.Disable()
            except:  # noqa
                self.chk_resume.Disable()
                self.chk_recursive.Disable()
        elif path.isdir(input_path) or is_text(input_path):
            self.chk_resume.Enable()
            self.chk_recursive.Enable()
        else:
            self.chk_resume.Disable()
            self.chk_recursive.Disable()
        self.chk_recursive.SetValue(False)

    def reset_time_range(self):
        self.chk_start_time.SetValue(False)
        self.chk_end_time.SetValue(False)
        self.txt_start_time.SetValue("00:00:00")
        self.txt_end_time.SetValue("00:00:00")

    def resolve_output_path(self, input_path, output_path):
        args = self.parse_args()
        video = is_video(input_path)
        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename)
        elif is_output_dir(output_path):
            output_path = path.join(
                output_path,
                make_output_filename(input_path, args, video=video))

        return output_path

    def on_text_changed_txt_input(self, event):
        self.update_start_button_state()
        self.update_rembg_state()
        self.update_input_option_state()
        self.reset_time_range()

    def on_text_changed_txt_output(self, event):
        self.update_start_button_state()

    def update_model_selection(self):
        name = self.cbo_depth_model.GetValue()
        if (DepthAnythingModel.supported(name) or DepthProModel.supported(name) or name.startswith("ZoeD_Any_")):
            self.chk_edge_dilation.SetValue(True)
            self.cbo_edge_dilation.Enable()
        else:
            self.chk_edge_dilation.SetValue(False)
            self.cbo_edge_dilation.Disable()
        if name in DEPTH_PRO_MODELS:
            self.cbo_resolution.Disable()
            self.chk_fp16.Disable()
        else:
            self.cbo_resolution.Enable()
            self.chk_fp16.Enable()

    def update_anaglyph_state(self):
        if self.cbo_stereo_format.GetValue() == "Anaglyph":
            self.lbl_anaglyph_method.Show()
            self.cbo_anaglyph_method.Show()
        else:
            self.lbl_anaglyph_method.Hide()
            self.cbo_anaglyph_method.Hide()
        self.GetSizer().Layout()

    def update_export_option_state(self):
        if self.cbo_stereo_format.GetValue() in {"Export", "Export disparity"}:
            self.chk_export_depth_only.Show()
            self.chk_export_depth_fit.Show()
        else:
            self.chk_export_depth_only.Hide()
            self.chk_export_depth_fit.Hide()
        self.GetSizer().Layout()

    def on_selected_index_changed_cbo_depth_model(self, event):
        self.update_model_selection()

    def on_selected_index_changed_cbo_stereo_format(self, event):
        self.update_input_option_state()
        self.update_anaglyph_state()
        self.update_export_option_state()

    def on_selected_index_changed_cbo_video_format(self, event):
        self.update_video_format()

    def on_selected_index_changed_cbo_video_codec(self, event):
        self.update_video_codec()

    def update_edge_dilation(self):
        if self.chk_edge_dilation.IsChecked():
            self.cbo_edge_dilation.Enable()
        else:
            self.cbo_edge_dilation.Disable()

    def on_changed_chk_edge_dilation(self, event):
        self.update_edge_dilation()

    def update_ema_normalize(self):
        if self.chk_ema_normalize.IsChecked():
            self.cbo_ema_decay.Enable()
        else:
            self.cbo_ema_decay.Disable()

    def on_changed_chk_ema_normalize(self, event):
        self.update_ema_normalize()

    def confirm_overwrite(self, args):
        input_path = args.input
        output_path = args.output
        video = is_video(input_path)
        resume = args.resume
        if args.export:
            if is_video(input_path):
                basename = (path.splitext(path.basename(input_path))[0]).strip()
                output_path = path.join(output_path, basename, export_config.FILENAME)
            else:
                output_path = path.join(output_path, export_config.FILENAME)
        else:
            if is_output_dir(output_path):
                output_path = path.join(
                    output_path,
                    make_output_filename(input_path, args, video=video))
            else:
                output_path = output_path
                resume = False

        if path.exists(output_path) and not resume:
            with wx.MessageDialog(None,
                                  message=output_path + "\n" + T("already exists. Overwrite?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                return dlg.ShowModal() == wx.ID_YES
        else:
            return True

    def show_validation_error_message(self, name, min_value, max_value):
        with wx.MessageDialog(
                None,
                message=T("`{}` must be a number {} - {}").format(name, min_value, max_value),
                caption=T("Error"),
                style=wx.OK) as dlg:
            dlg.ShowModal()

    def parse_args(self):
        if not validate_number(self.cbo_divergence.GetValue(), 0.0, 100.0):
            self.show_validation_error_message(T("3D Strength"), 0.0, 100.0)
            return None
        if not validate_number(self.cbo_convergence.GetValue(), -100.0, 100.0):
            self.show_validation_error_message(T("Convergence Plane"), -100.0, 100.0)
            return None
        if not validate_number(self.cbo_pad.GetValue(), 0.0, 10.0, allow_empty=True):
            self.show_validation_error_message(T("Padding"), 0.0, 10.0)
            return None
        if not validate_number(self.cbo_edge_dilation.GetValue(), 0, 20, is_int=True, allow_empty=False):
            self.show_validation_error_message(T("Edge Fix"), 0, 20)
            return None
        if not validate_number(self.grp_video.max_fps, 0.25, 1000.0, allow_empty=False):
            self.show_validation_error_message(T("Max FPS"), 0.25, 1000.0)
            return None
        if not validate_number(self.grp_video.crf, 0, 51, is_int=True):
            self.show_validation_error_message(T("CRF"), 0, 51)
            return None
        if not validate_number(self.cbo_ema_decay.GetValue(), 0.1, 0.999):
            self.show_validation_error_message(T("Flicker Reduction"), 0.1, 0.999)
            return None
        if not validate_number(self.cbo_foreground_scale.GetValue(), -3.0, 3.0, allow_empty=False):
            self.show_validation_error_message(T("Foreground Scale"), -3, 3)
            return None

        resolution = self.cbo_resolution.GetValue()
        if resolution == "Default" or resolution == "":
            resolution = None
        else:
            if not validate_number(resolution, 384, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Depth") + " " + T("Resolution"), 384, 8190)
                return
            resolution = int(resolution)

        stereo_width = self.cbo_stereo_width.GetValue()
        if stereo_width == "Default" or stereo_width == "":
            stereo_width = None
        else:
            if not validate_number(stereo_width, 320, 8190, is_int=True, allow_empty=False):
                self.show_validation_error_message(T("Stereo processing Width"), 320, 8190)
                return
            stereo_width = int(stereo_width)

        parser = create_parser(required_true=False)

        vr180 = self.cbo_stereo_format.GetValue() == "VR90"
        half_sbs = self.cbo_stereo_format.GetValue() == "Half SBS"
        tb = self.cbo_stereo_format.GetValue() == "Full TB"
        half_tb = self.cbo_stereo_format.GetValue() == "Half TB"
        cross_eyed = self.cbo_stereo_format.GetValue() == "Cross Eyed"
        anaglyph = self.get_anaglyph_method()
        export = self.cbo_stereo_format.GetValue() == "Export"
        export_disparity = self.cbo_stereo_format.GetValue() == "Export disparity"
        if export or export_disparity:
            export_depth_only = self.chk_export_depth_only.IsChecked()
            export_depth_fit = self.chk_export_depth_fit.IsChecked()
        else:
            export_depth_only = None
            export_depth_fit = None

        debug_depth = self.cbo_stereo_format.GetValue() == "Debug Depth"

        if self.cbo_pad.GetValue():
            pad = float(self.cbo_pad.GetValue())
        else:
            pad = None
        rot = self.cbo_rotate.GetClientData(self.cbo_rotate.GetSelection())
        rotate_left = rotate_right = None
        if rot == "left":
            rotate_left = True
        elif rot == "right":
            rotate_right = True

        vf = []
        if self.cbo_deinterlace.GetValue():
            vf += [self.cbo_deinterlace.GetValue()]
        if self.txt_vf.GetValue():
            vf += [self.txt_vf.GetValue()]
        vf = ",".join(vf)

        device_id = int(self.cbo_device.GetClientData(self.cbo_device.GetSelection()))
        if device_id == -2:
            # All CUDA
            device_id = list(range(torch.cuda.device_count()))
        else:
            device_id = [device_id]

        depth_model_type = self.cbo_depth_model.GetValue()
        if (self.depth_model is None or (self.depth_model_type != depth_model_type or
                                         self.depth_model_device_id != device_id or
                                         self.depth_model_height != resolution)):
            self.depth_model = None
            self.depth_model_type = None
            self.depth_model_device_id = None
            gc_collect()

        remove_bg = self.chk_rembg.GetValue()
        bg_model_type = self.cbo_bg_model.GetValue()

        max_output_width = max_output_height = None
        max_output_size = self.cbo_max_output_size.GetValue()
        if max_output_size:
            max_output_width, max_output_height = [int(s) for s in max_output_size.split("x")]

        input_path = self.pnl_file.input_path
        resume = self.chk_resume.IsEnabled() and self.chk_resume.GetValue()
        recursive = path.isdir(input_path) and self.chk_recursive.GetValue()
        start_time = self.txt_start_time.GetValue() if self.chk_start_time.GetValue() else None
        end_time = self.txt_end_time.GetValue() if self.chk_end_time.GetValue() else None
        edge_dilation = int(self.cbo_edge_dilation.GetValue()) if self.chk_edge_dilation.IsChecked() else 0
        metadata = "filename" if self.chk_metadata.GetValue() else None

        parser.set_defaults(
            input=input_path,
            output=self.pnl_file.output_path,
            yes=True,  # TODO: remove this

            divergence=float(self.cbo_divergence.GetValue()),
            convergence=float(self.cbo_convergence.GetValue()),
            ipd_offset=float(self.sld_ipd_offset.GetValue()),
            synthetic_view=self.cbo_synthetic_view.GetValue(),
            method=self.cbo_method.GetValue(),
            depth_model=depth_model_type,
            foreground_scale=float(self.cbo_foreground_scale.GetValue()),
            edge_dilation=edge_dilation,
            vr180=vr180,
            half_sbs=half_sbs,
            tb=tb,
            half_tb=half_tb,
            cross_eyed=cross_eyed,
            anaglyph=anaglyph,

            export=export,
            export_disparity=export_disparity,
            export_depth_only=export_depth_only,
            export_depth_fit=export_depth_fit,

            debug_depth=debug_depth,
            ema_normalize=self.chk_ema_normalize.GetValue(),
            ema_decay=float(self.cbo_ema_decay.GetValue()),

            format=self.cbo_image_format.GetValue(),

            max_fps=self.grp_video.max_fps,
            pix_fmt=self.grp_video.pix_fmt,
            colorspace=self.grp_video.colorspace,
            video_format=self.grp_video.video_format,
            video_codec=self.grp_video.video_codec,
            crf=self.grp_video.crf,
            video_bitrate=self.grp_video.bitrate,
            profile_level=self.grp_video.profile_level,
            preset=self.grp_video.preset,
            tune=self.grp_video.tune,

            remove_bg=remove_bg,
            bg_model=bg_model_type,

            pad=pad,
            rotate_right=rotate_right,
            rotate_left=rotate_left,
            disable_exif_transpose=not self.chk_exif_transpose.GetValue(),
            vf=vf,
            max_output_width=max_output_width,
            max_output_height=max_output_height,
            keep_aspect_ratio=self.chk_keep_aspect_ratio.GetValue(),

            gpu=device_id,
            batch_size=int(self.cbo_batch_size.GetValue()),
            resolution=resolution,
            stereo_width=stereo_width,
            max_workers=int(self.cbo_max_workers.GetValue()),
            tta=self.chk_tta.GetValue(),
            disable_amp=not self.chk_fp16.GetValue(),
            low_vram=self.chk_low_vram.GetValue(),
            cuda_stream=self.chk_cuda_stream.GetValue(),

            resume=resume,
            recursive=recursive,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )
        args = parser.parse_args()
        set_state_args(
            args,
            stop_event=self.stop_event,
            suspend_event=self.suspend_event,
            tqdm_fn=functools.partial(TQDMGUI, self),
            depth_model=self.depth_model)
        return args

    def on_click_btn_start(self, event):
        args = self.parse_args()
        if args is None:
            return
        if not self.confirm_overwrite(args):
            return

        self.btn_start.Disable()
        self.btn_cancel.Enable()
        self.btn_suspend.Enable()
        self.stop_event.clear()
        self.suspend_event.set()
        self.prg_tqdm.SetValue(0)
        self.SetStatusText("...")

        if args.state["depth_model"].has_checkpoint_file(args.depth_model):
            # Realod depth model
            self.SetStatusText(f"Loading {args.depth_model}...")
            if args.remove_bg and not has_rembg_model(args.bg_model):
                self.SetStatusText(f"Downloading {args.bg_model}...")
        else:
            # Need to download the model
            self.SetStatusText(f"Downloading {args.depth_model}...")

        startWorker(self.on_exit_worker, iw3_main, wargs=(args,))
        self.processing = True

    def on_exit_worker(self, result):
        try:
            args = result.get()
            self.depth_model = args.state["depth_model"]
            self.depth_model_type = args.depth_model
            self.depth_model_device_id = args.gpu
            self.depth_model_height = args.resolution

            if not self.stop_event.is_set():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetRange())
                self.SetStatusText(T("Finished"))
            else:
                self.SetStatusText(T("Cancelled"))
        except: # noqa
            self.SetStatusText(T("Error"))
            e_type, e, tb = sys.exc_info()
            message = getattr(e, "message", str(e))
            traceback.print_tb(tb)
            wx.MessageBox(message, f"{T('Error')}: {e.__class__.__name__}", wx.OK | wx.ICON_ERROR)

        self.processing = False
        self.btn_cancel.Disable()
        self.btn_suspend.Disable()
        self.btn_suspend.SetLabel(T("Suspend"))
        self.update_start_button_state()

        # free vram
        gc_collect()

    def on_click_btn_cancel(self, event):
        self.suspend_event.set()
        self.stop_event.set()

    def on_click_btn_suspend(self, event):
        if self.suspend_event.is_set():
            self.suspend_event.clear()
            self.btn_suspend.SetLabel(T("Resume"))
        else:
            self.start_time = time()
            self.suspend_pos = self.prg_tqdm.GetValue()
            self.suspend_event.set()
            self.btn_suspend.SetLabel(T("Suspend"))

    def on_tqdm(self, event):
        type, value, desc = event.GetValue()
        desc = desc if desc else ""
        if type == 0:
            # initialize
            if 0 < value:
                self.prg_tqdm.SetRange(value)
            else:
                self.prg_tqdm.SetRange(1)
            self.prg_tqdm.SetValue(0)
            self.start_time = time()
            self.suspend_pos = 0
            self.SetStatusText(f"{0}/{value} {desc}")
        elif type == 1:
            # update
            if self.prg_tqdm.GetValue() + value <= self.prg_tqdm.GetRange():
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            else:
                self.prg_tqdm.SetRange(self.prg_tqdm.GetValue() + value)
                self.prg_tqdm.SetValue(self.prg_tqdm.GetValue() + value)
            now = time()
            pos = self.prg_tqdm.GetValue()
            end_pos = self.prg_tqdm.GetRange()
            fps = (pos - self.suspend_pos) / (now - self.start_time + 1e-6)
            remaining_time = int((end_pos - pos) / fps)
            h = remaining_time // 3600
            m = (remaining_time - h * 3600) // 60
            s = (remaining_time - h * 3600 - m * 60)
            t = f"{m:02d}:{s:02d}" if h == 0 else f"{h:02d}:{m:02d}:{s:02d}"
            self.SetStatusText(f"{pos}/{end_pos} [ {t}, {fps:.2f}FPS ] {desc}")
        elif type == 2:
            # close
            pass

    def save_preset(self, name=None):
        if not name:
            restore_path = True
            name = ""
            config_file = CONFIG_PATH
        else:
            restore_path = False
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")
            if path.exists(config_file):
                with wx.MessageDialog(None,
                                      message=name + "\n" + T("already exists. Overwrite?"),
                                      caption=T("Confirm"), style=wx.YES_NO) as dlg:
                    if dlg.ShowModal() != wx.ID_YES:
                        return

        input_path = self.pnl_file.input_path
        output_path = self.pnl_file.output_path
        preset = name
        try:
            if not restore_path:
                self.pnl_file.set_input_path("")
                self.pnl_file.set_output_path("")
            self.cbo_app_preset.SetValue("")
            manager = persist.PersistenceManager.Get()
            manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
            manager.SetPersistenceFile(config_file)
            persistent_manager_register_all(manager, self)
            for control in self.get_editable_comboboxes():
                persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
            manager.SaveAndUnregister()
            self.reload_preset()
        finally:
            if not restore_path:
                self.pnl_file.set_input_path(input_path)
                self.pnl_file.set_output_path(output_path)
            self.cbo_app_preset.SetValue(preset)

    def list_preset(self):
        presets = [""]
        for fn in os.listdir(PRESET_DIR):
            name = path.splitext(fn)[0]
            presets.append(name)
        return presets

    def reload_preset(self):
        selected = self.cbo_app_preset.GetValue()
        choices = self.list_preset()
        self.cbo_app_preset.SetItems(choices)
        if selected in choices:
            self.cbo_app_preset.SetSelection(choices.index(selected))

    def load_preset(self, name=None):
        if not name:
            restore_path = True
            name = ""
            config_file = CONFIG_PATH
        else:
            restore_path = False
            name = sanitize_filename(name)
            config_file = path.join(PRESET_DIR, f"{name}.cfg")

        input_path = self.pnl_file.input_path
        output_path = self.pnl_file.output_path
        preset = name
        try:
            manager = persist.PersistenceManager.Get()
            manager.SetManagerStyle(persist.PM_DEFAULT_STYLE)
            manager.SetPersistenceFile(config_file)
            persistent_manager_register_all(manager, self)
            for control in self.get_editable_comboboxes():
                persistent_manager_register(manager, control, EditableComboBoxPersistentHandler)
            persistent_manager_restore_all(manager)
            persistent_manager_unregister_all(manager)
        finally:
            if not restore_path:
                self.pnl_file.set_input_path(input_path)
                self.pnl_file.set_output_path(output_path)
            self.cbo_app_preset.SetValue(preset)

    def delete_preset(self, name=None):
        if not name:
            return
        config_file = path.join(PRESET_DIR, f"{name}.cfg")
        if path.exists(config_file):
            with wx.MessageDialog(None,
                                  message=name + "\n" + T("Delete?"),
                                  caption=T("Confirm"), style=wx.YES_NO) as dlg:
                if dlg.ShowModal() != wx.ID_YES:
                    return
            os.unlink(config_file)
        self.reload_preset()

    def on_click_btn_load_preset(self, event):
        self.load_preset(self.cbo_app_preset.GetValue())

    def on_click_btn_save_preset(self, event):
        self.save_preset(self.cbo_app_preset.GetValue())

    def on_click_btn_delete_preset(self, event):
        self.delete_preset(self.cbo_app_preset.GetValue())
        event.Skip()

    def on_click_divergence_warning(self, event):
        self.lbl_divergence_warning.Hide()
        self.GetSizer().Layout()

    def update_divergence_warning(self, *args, **kwargs):
        try:
            divergence = float(self.cbo_divergence.GetValue())
            method = self.cbo_method.GetValue()
            synthetic_view = self.cbo_synthetic_view.GetValue()
            max_divergence = float("inf")

            if method in {"row_flow_v3", "row_flow_v3_sym"}:
                if synthetic_view == "both":
                    max_divergence = 5.0
                else:
                    max_divergence = 5.0 * 0.5
            elif method == "row_flow_v2":
                if synthetic_view == "both":
                    max_divergence = 2.5
                else:
                    max_divergence = 2.5 * 0.5

            if divergence > max_divergence:
                self.lbl_divergence_warning.SetLabel(
                    f"{divergence}: " + T("Out of range of training data") + f": {method}, {synthetic_view}"
                )
                self.lbl_divergence_warning.SetToolTip(
                    T("This result could be unstable"),
                )
                self.lbl_divergence_warning.Show()
            else:
                self.lbl_divergence_warning.SetLabel("")
                self.lbl_divergence_warning.SetToolTip("")
                self.lbl_divergence_warning.Hide()

            self.GetSizer().Layout()
        except ValueError:
            pass


LOCALE_DICT = LOCALES.get(locale.getlocale()[0], {})


def T(s):
    return LOCALE_DICT.get(s, s)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lang", type=str, help="lang, ja_JP, en_US")
    args = parser.parse_args()
    if args.lang:
        global LOCALE_DICT
        LOCALE_DICT = LOCALES.get(args.lang, {})
    sys.argv = [sys.argv[0]]  # clear command arguments

    app = IW3App()
    app.MainLoop()


if __name__ == "__main__":
    main()
