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


def resolve_size(resolution: str, aspect_ratio: str) -> str:
    try:
        return SIZE_MAP[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported resolution/aspect_ratio combination: {resolution} + {aspect_ratio}"
        ) from exc
