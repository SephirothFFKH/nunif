# mapper function to convert model output to disparity
# see also iw3/training/find_mapper.py
import torch
import math


def softplus01(x, bias, scale):
    # x: 0-1 normalized
    min_v = math.log(1 + math.exp((0 - bias) * scale))
    max_v = math.log(1 + math.exp((1 - bias) * scale))
    v = torch.log(1. + torch.exp((x - bias) * scale))
    return (v - min_v) / (max_v - min_v)


def inv_softplus01(x, bias, scale):
    min_v = ((torch.zeros(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    max_v = ((torch.ones(1, dtype=x.dtype, device=x.device) - bias) * scale).expm1().clamp(min=1e-6).log()
    v = ((x - bias) * scale).expm1().clamp(min=1e-6).log()
    return (v - min_v) / (max_v - min_v)


def distance_to_disparity(x, c):
    c1 = 1.0 + c
    min_v = c / c1
    return ((c / (c1 - x)) - min_v) / (1.0 - min_v)


def inv_distance_to_disparity(x, c):
    return ((c + 1) * x) / (x + c)


def shift(x, value):
    return (1.0 - value) * x + value


def div_shift(x, value, c=0.6):
    x = inv_distance_to_disparity(x, c)
    x = (1.0 - value) * x + value
    x = distance_to_disparity(x, c)
    return x


def resolve_mapper_function(name):
    # https://github.com/nagadomi/nunif/assets/287255/0071a65a-62ff-4928-850c-0ad22bceba41
    if name == "pow2":
        return lambda x: x ** 2
    elif name == "none":
        return lambda x: x
    elif name == "softplus":
        return softplus01
    elif name == "softplus2":
        return lambda x: softplus01(x) ** 2
    elif name in {"mul_1", "mul_2", "mul_3"}:
        # for DepthAnything
        # https://github.com/nagadomi/nunif/assets/287255/2be5c0de-cb72-4c9c-9e95-4855c0730e5c
        param = {
            # none 1x
            "mul_1": {"bias": 0.343, "scale": 12},  # smooth 1.5x
            "mul_2": {"bias": 0.515, "scale": 12},  # smooth 2x
            "mul_3": {"bias": 0.687, "scale": 12},  # smooth 3x
        }[name]
        return lambda x: softplus01(x, **param)
    elif name in {"inv_mul_1", "inv_mul_2", "inv_mul_3"}:
        # for DepthAnything
        # https://github.com/nagadomi/nunif/assets/287255/f580b405-b0bf-4c6a-8362-66372b2ed930
        param = {
            # none 1x
            "inv_mul_1": {"bias": -0.002102, "scale": 7.8788},  # inverse smooth 1.5x
            "inv_mul_2": {"bias": -0.0003, "scale": 6.2626},    # inverse smooth 2x
            "inv_mul_3": {"bias": -0.0001, "scale": 3.4343},    # inverse smooth 3x
        }[name]
        return lambda x: inv_softplus01(x, **param)
    elif name in {"div_25", "div_10", "div_6", "div_4", "div_2", "div_1"}:
        # for ZoeDepth
        # TODO: There is no good reason for this parameter step
        # https://github.com/nagadomi/nunif/assets/287255/46c6b292-040f-4820-93fc-9e001cd53375
        param = {
            "div_25": 2.5,
            "div_10": 1,
            "div_6": 0.6,
            "div_4": 0.4,
            "div_2": 0.2,
            "div_1": 0.1,
        }[name]
        return lambda x: distance_to_disparity(x, param)
    elif name in {"shift_25", "shift_50", "shift_75"}:
        param = {
            "shift_25": 0.25,
            "shift_50": 0.5,
            "shift_75": 0.75,
        }[name]
        return lambda x: shift(x, param)
    elif name in {"div_shift_25", "div_shift_50", "div_shift_75"}:
        param = {
            "div_shift_25": 0.25,
            "div_shift_50": 0.5,
            "div_shift_75": 0.75,
        }[name]
        return lambda x: div_shift(x, param, 0.6)
    else:
        raise NotImplementedError(f"mapper={name}")


def chain(x, functions):
    for f in functions:
        x = f(x)
    return x


def get_mapper(name):
    if ":" in name:
        names = name.split(":")
    else:
        names = [name]
    functions = []
    for name in names:
        if "+" in name:
            # weighted average (interpolation)
            name, weight = name.split("=")
            if not weight:
                weight = 0.5
            else:
                weight = float(weight)
                assert 0.0 <= weight <= 1.0
            mapper_a, mapper_b = name.split("+")
            mapper_a = resolve_mapper_function(mapper_a)
            mapper_b = resolve_mapper_function(mapper_b)
            functions.append(lambda x: mapper_a(x) * (1 - weight) + mapper_b(x) * weight)
        else:
            functions.append(resolve_mapper_function(name))

    return lambda x: chain(x, functions)


METRIC_DIV_MAPPER = [
    "none", "div_25", "div_10",
    "div_6",
    "div_4", "div_2", "div_1",
]
RELATIVE_MUL_MAPPER = [
    "inv_mul_3", "inv_mul_2", "inv_mul_1",
    "none",
    "mul_1", "mul_2", "mul_3",
]
LEGACY_MAPPER = ["pow2", "softplus", "softplus2"]
MAPPER_ALL = ["auto"] + list(dict.fromkeys(LEGACY_MAPPER + RELATIVE_MUL_MAPPER + METRIC_DIV_MAPPER))


def get_mapper_levels(metric_depth, mapper_type=None):
    if metric_depth:
        return METRIC_DIV_MAPPER
    else:
        return RELATIVE_MUL_MAPPER


def resolve_mapper_name(mapper, foreground_scale, metric_depth, mapper_type=None):
    disparity_mapper = None
    if mapper is not None:
        if mapper == "auto":
            if not metric_depth:
                disparity_mapper = "none"
            else:
                disparity_mapper = "div_6"
        else:
            disparity_mapper = mapper
    else:
        if float(foreground_scale).is_integer():
            foreground_scale = int(foreground_scale)
            mapper_levels = get_mapper_levels(metric_depth=metric_depth, mapper_type=mapper_type)
            assert len(mapper_levels) == 7
            disparity_mapper = mapper_levels[foreground_scale + 3]
        else:
            # float value, interpolate two mappers
            if foreground_scale > 0:
                foreground_scale_a = math.floor(foreground_scale)
                foreground_scale_b = math.ceil(foreground_scale)
                weight = foreground_scale - foreground_scale_a
            else:
                foreground_scale = -foreground_scale
                foreground_scale_a = math.floor(foreground_scale)
                foreground_scale_b = math.ceil(foreground_scale)
                weight = foreground_scale - foreground_scale_a
                foreground_scale_a, foreground_scale_b = -foreground_scale_a, -foreground_scale_b

            mapper_levels = get_mapper_levels(metric_depth=metric_depth, mapper_type=mapper_type)
            assert len(mapper_levels) == 7
            mapper_a = mapper_levels[foreground_scale_a + 3]
            mapper_b = mapper_levels[foreground_scale_b + 3]

            disparity_mapper = f"{mapper_a}+{mapper_b}={round(weight, 2)}"

    assert disparity_mapper is not None
    return disparity_mapper
