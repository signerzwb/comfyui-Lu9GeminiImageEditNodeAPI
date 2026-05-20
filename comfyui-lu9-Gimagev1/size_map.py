import math


SIZE_MAP = {
    "1k": {
        "1:1": "1024x1024",
        "3:4": "1152x1536",
        "4:3": "1536x1152",
        "9:16": "1024x1536",
        "16:9": "1536x1024",
    },
    "2k": {
        "1:1": "2048x2048",
        "3:4": "1536x2048",
        "4:3": "2048x1536",
        "4:5": "1632x2048",
        "5:4": "2048x1632",
        "9:16": "1152x2048",
        "16:9": "2048x1152",
        "9:21": "864x2016",
        "21:9": "2016x864",
    },
    "4k": {
        "1:1": "2880x2880",
        "3:4": "2432x3264",
        "4:3": "3264x2432",
        "4:5": "2560x3200",
        "5:4": "3200x2560",
        "9:16": "2160x3840",
        "16:9": "3840x2160",
        "9:21": "1584x3696",
        "21:9": "3696x1584",
    },
}

ASPECT_OPTIONS = {
    "1k": list(SIZE_MAP["1k"].keys()),
    "2k": list(SIZE_MAP["2k"].keys()),
    "4k": list(SIZE_MAP["4k"].keys()),
}

CUSTOM_SIZE_STEP = 64
CUSTOM_MAX_EDGE = 3840
CUSTOM_MIN_PIXELS = 655360
CUSTOM_MAX_PIXELS = 8294400
CUSTOM_MAX_RATIO = 3.0


def parse_size_string(size: str) -> tuple[int, int]:
    width, height = size.split("x")
    return int(width), int(height)


RESOLUTION_LIMITS = {}
for resolution_name, resolution_sizes in SIZE_MAP.items():
    parsed_sizes = [parse_size_string(size) for size in resolution_sizes.values()]
    RESOLUTION_LIMITS[resolution_name] = {
        "max_edge": max(max(width, height) for width, height in parsed_sizes),
        "max_pixels": max(width * height for width, height in parsed_sizes),
    }


def validate_custom_size(width: int, height: int) -> None:
    if width is None or height is None:
        raise ValueError("Custom size requires both width and height.")
    if width < CUSTOM_SIZE_STEP or height < CUSTOM_SIZE_STEP:
        raise ValueError(f"Custom width/height must be at least {CUSTOM_SIZE_STEP}.")
    if width % CUSTOM_SIZE_STEP != 0 or height % CUSTOM_SIZE_STEP != 0:
        raise ValueError(f"Custom width/height must be multiples of {CUSTOM_SIZE_STEP}.")
    if width > CUSTOM_MAX_EDGE or height > CUSTOM_MAX_EDGE:
        raise ValueError(f"Custom width/height must not exceed {CUSTOM_MAX_EDGE}.")

    pixels = width * height
    if pixels < CUSTOM_MIN_PIXELS:
        raise ValueError(
            f"Custom total pixels must be at least {CUSTOM_MIN_PIXELS}, got {pixels}."
        )
    if pixels > CUSTOM_MAX_PIXELS:
        raise ValueError(
            f"Custom total pixels must not exceed {CUSTOM_MAX_PIXELS}, got {pixels}."
        )

    ratio = max(width / height, height / width)
    if ratio > CUSTOM_MAX_RATIO:
        raise ValueError(
            f"Custom aspect ratio must not exceed {CUSTOM_MAX_RATIO}:1, got {width}:{height}."
        )


def parse_ratio_string(aspect_ratio: str) -> float:
    left, right = aspect_ratio.split(":")
    return float(left) / float(right)


def pick_closest_aspect_ratio(resolution: str, width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for auto ratio detection: {width}x{height}")

    available = ASPECT_OPTIONS.get(resolution)
    if not available:
        raise ValueError(f"No aspect ratios configured for resolution {resolution}.")

    target = width / height
    best_ratio = None
    best_delta = None
    for candidate in available:
        delta = abs(parse_ratio_string(candidate) - target)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_ratio = candidate

    if best_ratio is None:
        raise ValueError(f"Unable to find a matching aspect ratio for resolution {resolution}.")
    return best_ratio


def derive_smartauto_size(resolution: str, width: int, height: int) -> tuple[int, int] | None:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size for smartauto detection: {width}x{height}")

    limits = RESOLUTION_LIMITS.get(resolution)
    if not limits:
        raise ValueError(f"No smartauto limits configured for resolution {resolution}.")

    target_ratio = width / height
    if max(target_ratio, 1.0 / target_ratio) > CUSTOM_MAX_RATIO:
        return None

    source_pixels = width * height
    lower_scale = max(CUSTOM_SIZE_STEP / width, CUSTOM_SIZE_STEP / height)
    if source_pixels < CUSTOM_MIN_PIXELS:
        lower_scale = max(lower_scale, math.sqrt(CUSTOM_MIN_PIXELS / source_pixels))

    upper_scale = min(
        limits["max_edge"] / width,
        limits["max_edge"] / height,
        math.sqrt(limits["max_pixels"] / source_pixels),
    )

    if lower_scale > upper_scale:
        return None

    if lower_scale <= 1.0 <= upper_scale:
        scale = 1.0
    elif 1.0 < lower_scale:
        scale = lower_scale
    else:
        scale = upper_scale

    ideal_width = width * scale
    ideal_height = height * scale
    ideal_pixels = ideal_width * ideal_height

    best_candidate = None
    best_score = None
    for candidate_width in range(CUSTOM_SIZE_STEP, limits["max_edge"] + 1, CUSTOM_SIZE_STEP):
        for candidate_height in range(CUSTOM_SIZE_STEP, limits["max_edge"] + 1, CUSTOM_SIZE_STEP):
            pixels = candidate_width * candidate_height
            if pixels < CUSTOM_MIN_PIXELS or pixels > limits["max_pixels"]:
                continue
            if max(candidate_width / candidate_height, candidate_height / candidate_width) > CUSTOM_MAX_RATIO:
                continue

            size_error = (
                abs(candidate_width - ideal_width) / max(ideal_width, CUSTOM_SIZE_STEP)
                + abs(candidate_height - ideal_height) / max(ideal_height, CUSTOM_SIZE_STEP)
            )
            ratio_error = abs((candidate_width / candidate_height) - target_ratio) / max(
                target_ratio, 1e-9
            )
            pixel_error = abs(pixels - ideal_pixels) / max(ideal_pixels, CUSTOM_MIN_PIXELS)
            score = (
                round(size_error, 12),
                round(ratio_error, 12),
                round(pixel_error, 12),
                abs(candidate_width - round(ideal_width)),
                abs(candidate_height - round(ideal_height)),
            )

            if best_score is None or score < best_score:
                best_score = score
                best_candidate = (candidate_width, candidate_height)

    if best_candidate is None:
        return None

    validate_custom_size(best_candidate[0], best_candidate[1])
    return best_candidate


def resolve_smartauto_request(resolution: str, width: int, height: int) -> dict:
    derived_size = derive_smartauto_size(resolution, width, height)
    if derived_size is not None:
        return {
            "resolved_aspect_ratio": "custom",
            "custom_width": derived_size[0],
            "custom_height": derived_size[1],
            "resolved_size_strategy": "smart_custom",
        }

    return {
        "resolved_aspect_ratio": pick_closest_aspect_ratio(resolution, width, height),
        "custom_width": None,
        "custom_height": None,
        "resolved_size_strategy": "smart_fallback_preset",
    }


def resolve_size(
    resolution: str,
    aspect_ratio: str,
    custom_width: int | None = None,
    custom_height: int | None = None,
) -> str:
    if aspect_ratio == "custom":
        width = int(custom_width) if custom_width is not None else None
        height = int(custom_height) if custom_height is not None else None
        validate_custom_size(width, height)
        return f"{width}x{height}"

    try:
        return SIZE_MAP[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported resolution/aspect_ratio combination: {resolution} + {aspect_ratio}"
        ) from exc
