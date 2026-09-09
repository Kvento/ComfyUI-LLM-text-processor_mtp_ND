from __future__ import annotations

import fnmatch
import json
import os
import platform
import re
import shlex
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import folder_paths
import comfy.model_management


# =============================================================================
# Folder / model registry
# =============================================================================

LLM_FOLDER = "llm_text_processor_models"
PROMPT_FOLDER = "llm_text_processor_prompts"

NO_SYSTEM_PROMPT = "none"
NO_MMPROJ = "none"
NO_MTP = "none"
NO_MODELS_FOUND = "No GGUF models found"


def llm_root() -> Path:
    return Path(folder_paths.models_dir) / "LLM"


def prompt_root() -> Path:
    return llm_root() / "prompts"


def register_folders() -> None:
    llm_dir = llm_root()
    prompts_dir = prompt_root()

    llm_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Same tuple shape that ComfyUI uses internally:
    # ([search paths], {allowed extensions})
    folder_paths.folder_names_and_paths[LLM_FOLDER] = ([str(llm_dir)], {".gguf"})
    folder_paths.folder_names_and_paths[PROMPT_FOLDER] = ([str(prompts_dir)], {".txt"})


def _basename_lower(name: str) -> str:
    return Path(str(name).replace("\\", "/")).name.lower()


def _is_mmproj_file(name: str) -> bool:
    return "mmproj" in _basename_lower(name)


def _is_mtp_file(name: str) -> bool:
    """
    MTP/draft heads are kept out of the main model list.

    The reference Gemma 4 file is named:
        mtp-gemma-4-31B-it.gguf

    We deliberately use a prefix-oriented check instead of treating every filename
    containing the letters 'mtp' as a draft head. This avoids hiding a normal model
    whose release/model name itself happens to contain '-MTP-'.
    """
    base = _basename_lower(name)
    stem = Path(base).stem
    return (
        stem.startswith("mtp-")
        or stem.startswith("mtp_")
        or stem == "mtp"
        or stem.startswith("draft-")
        or stem.startswith("draft_")
    )


def model_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    models = [
        name
        for name in files
        if not _is_mmproj_file(name) and not _is_mtp_file(name)
    ]
    return models or [NO_MODELS_FOUND]


def mmproj_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    projectors = [name for name in files if _is_mmproj_file(name)]
    return [NO_MMPROJ] + projectors


def mtp_options() -> list[str]:
    files = folder_paths.get_filename_list(LLM_FOLDER)
    mtp_models = [
        name
        for name in files
        if _is_mtp_file(name) and not _is_mmproj_file(name)
    ]
    return [NO_MTP] + mtp_models


def system_prompt_options() -> list[str]:
    files = folder_paths.get_filename_list(PROMPT_FOLDER)
    top_level_files = [
        name
        for name in files
        if os.sep not in name and "/" not in name and "\\" not in name
    ]
    return [NO_SYSTEM_PROMPT] + top_level_files


def full_model_path(name: str) -> Path:
    if name == NO_MODELS_FOUND:
        raise FileNotFoundError(
            f"No GGUF model files were found in {llm_root()}. "
            "Place a .gguf model there and refresh/restart ComfyUI."
        )
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"GGUF model not found: {name}")
    return Path(path)


def full_mmproj_path(name: str) -> Path | None:
    if name == NO_MMPROJ:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"mmproj GGUF file not found: {name}")
    return Path(path)


def full_mtp_path(name: str) -> Path | None:
    if name == NO_MTP:
        return None
    path = folder_paths.get_full_path(LLM_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"MTP GGUF file not found: {name}")
    return Path(path)


def full_system_prompt_path(name: str) -> Path | None:
    if name == NO_SYSTEM_PROMPT:
        return None
    path = folder_paths.get_full_path(PROMPT_FOLDER, name)
    if path is None:
        raise FileNotFoundError(f"System prompt preset not found: {name}")
    return Path(path)


register_folders()


# =============================================================================
# llama.cpp binary manager
# =============================================================================

# Kept aligned with the upstream node at the time this MTP edition was made.
LLAMA_CPP_RELEASE_TAG = "b10472"
RELEASE_API_URL = (
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/"
    f"{LLAMA_CPP_RELEASE_TAG}"
)

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "llama.cpp"


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    cli_executable: str
    asset_patterns: tuple[str, ...]
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class LlamaCliPaths:
    cli: Path


WINDOWS_CUDA_13 = PlatformSpec(
    key="win-x64-cuda13",
    cli_executable="llama-cli.exe",
    asset_patterns=(
        "llama-*-bin-win-cuda-13*-x64.zip",
        "cudart-llama-bin-win-cuda-13*-x64.zip",
    ),
    required_files=(
        "llama-cli.exe",
        "ggml-cuda.dll",
        "cudart64_13.dll",
    ),
)


def _platform_spec() -> PlatformSpec:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows" and machine in {"amd64", "x86_64"}:
        return WINDOWS_CUDA_13

    raise RuntimeError(
        "Automatic llama.cpp binary download currently supports Windows x64 CUDA 13 only. "
        "Other platforms require a manually adapted llama.cpp binary setup."
    )


def _json_get(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-LLM-text-processor-MTP"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_size(num_bytes: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ComfyUI-LLM-text-processor-MTP"},
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size is not None else None
        downloaded = 0
        chunk_size = 1024 * 256
        started_at = time.monotonic()
        last_reported_at = started_at

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                handle.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_reported_at < 1.0:
                    continue

                elapsed = max(now - started_at, 0.001)
                speed = downloaded / elapsed

                if total_size:
                    percent = (downloaded / total_size) * 100
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} / {_format_size(total_size)} "
                        f"({percent:.1f}%) at {_format_size(speed)}/s"
                    )
                else:
                    print(
                        "[LLM Text Processor] "
                        f"Downloaded {_format_size(downloaded)} at {_format_size(speed)}/s"
                    )

                last_reported_at = now

        elapsed = max(time.monotonic() - started_at, 0.001)
        speed = downloaded / elapsed

        if total_size:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} / {_format_size(total_size)} "
                f"(100.0%) at {_format_size(speed)}/s"
            )
        else:
            print(
                "[LLM Text Processor] "
                f"Finished download: {_format_size(downloaded)} at {_format_size(speed)}/s"
            )


def _select_assets(release: dict, spec: PlatformSpec) -> list[dict]:
    assets = release.get("assets", [])
    selected = []
    used_names = set()

    for pattern in spec.asset_patterns:
        matches = [
            asset
            for asset in assets
            if fnmatch.fnmatch(asset.get("name", "").lower(), pattern.lower())
        ]
        if not matches:
            raise RuntimeError(
                f"Could not find llama.cpp release asset matching: {pattern}"
            )

        asset = sorted(matches, key=lambda item: item.get("name", ""))[0]
        if asset["name"] not in used_names:
            selected.append(asset)
            used_names.add(asset["name"])

    return selected


def _find_file(install_dir: Path, name: str) -> Path | None:
    for path in install_dir.rglob(name):
        if path.is_file():
            return path
    return None


def _find_cli_paths(
    install_dir: Path,
    spec: PlatformSpec,
) -> LlamaCliPaths | None:
    cli = _find_file(install_dir, spec.cli_executable)
    if cli is None:
        return None
    return LlamaCliPaths(cli=cli)


def _has_required_files(install_dir: Path, spec: PlatformSpec) -> bool:
    for name in spec.required_files:
        if not any(path.is_file() for path in install_dir.rglob(name)):
            return False
    return True


def _is_complete_install(install_dir: Path, spec: PlatformSpec) -> bool:
    return (
        _find_cli_paths(install_dir, spec) is not None
        and _has_required_files(install_dir, spec)
    )


def _existing_install(spec: PlatformSpec) -> LlamaCliPaths | None:
    install_dir = VENDOR_ROOT / LLAMA_CPP_RELEASE_TAG / spec.key
    if not _is_complete_install(install_dir, spec):
        return None
    return _find_cli_paths(install_dir, spec)


def _extract_assets(assets: list[dict], install_dir: Path) -> None:
    with TemporaryDirectory(prefix="llm-text-processor-llama-download-") as temp:
        temp_dir = Path(temp)

        for asset in assets:
            archive_path = temp_dir / asset["name"]
            print(f"[LLM Text Processor] Downloading {asset['name']}...")
            _download(asset["browser_download_url"], archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(install_dir)


def ensure_llama_cli_paths() -> LlamaCliPaths:
    spec = _platform_spec()

    existing = _existing_install(spec)
    if existing is not None:
        return existing

    release = _json_get(RELEASE_API_URL)
    tag = release.get("tag_name") or LLAMA_CPP_RELEASE_TAG
    install_dir = VENDOR_ROOT / tag / spec.key

    if _is_complete_install(install_dir, spec):
        paths = _find_cli_paths(install_dir, spec)
        if paths is None:
            raise RuntimeError(
                f"Completed install has incomplete CLI executables: {install_dir}"
            )
        return paths

    assets = _select_assets(release, spec)
    install_dir.mkdir(parents=True, exist_ok=True)
    _extract_assets(assets, install_dir)

    paths = _find_cli_paths(install_dir, spec)
    if paths is None:
        raise RuntimeError(
            "Downloaded llama.cpp assets but could not find CLI executables in "
            f"{install_dir}"
        )

    if not _has_required_files(install_dir, spec):
        missing = [
            name
            for name in spec.required_files
            if not any(path.is_file() for path in install_dir.rglob(name))
        ]
        raise RuntimeError(
            "Downloaded llama.cpp assets are incomplete; missing: "
            + ", ".join(missing)
        )

    return paths


# =============================================================================
# llama-cli integration
# =============================================================================

PROMPT_ECHO_END = "... (truncated)"
PROMPT_PADDING = " " * 501

PERF_RE = re.compile(
    r"\[\s*Prompt:\s*[^|\]]+\|\s*Generation:\s*[^\]]+\]"
)

MMPROJ_EMBEDDING_MISMATCH_RE = re.compile(
    r"mismatch between text model \(n_embd = (?P<model>\d+)\) "
    r"and mmproj \(n_embd = (?P<mmproj>\d+)\)",
    flags=re.IGNORECASE,
)

START_THINKING = "[Start thinking]"
END_THINKING = "[End thinking]"

LLAMA_RANDOM_SEED = -1
LLAMA_SEED_MODULUS = 2**32
MAX_LLAMA_SEED = LLAMA_SEED_MODULUS - 1

_MTP_HELP_CHECKED: set[str] = set()


def _tensor_to_temp_png(tensor) -> Path:
    import numpy as np
    from PIL import Image

    array = (
        tensor.detach().cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    pil_image = Image.fromarray(array)

    fd, path = tempfile.mkstemp(
        prefix="llm-text-processor-",
        suffix=".png",
    )
    os.close(fd)

    pil_image.save(path, format="PNG")
    return Path(path)


def tensor_to_temp_pngs(image) -> list[Path]:
    # ComfyUI IMAGE is normally B,H,W,C.
    if hasattr(image, "dim") and image.dim() == 4:
        return [_tensor_to_temp_png(tensor) for tensor in image]
    return [_tensor_to_temp_png(image)]


def _write_temp_text_file(prefix: str, text: str) -> Path:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    os.close(fd)

    text_path = Path(path)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    return text_path


def _write_prompt_file(prompt: str) -> Path:
    return _write_temp_text_file(
        "llm-text-processor-prompt-",
        prompt.strip() + PROMPT_PADDING,
    )


def split_extra_args(extra_args: str) -> list[str]:
    if not extra_args or not extra_args.strip():
        return []

    parts = shlex.split(extra_args, posix=(os.name != "nt"))
    return [part.strip("\"'") for part in parts]


def normalize_llama_seed(seed: int) -> int:
    seed = int(seed)

    if seed == LLAMA_RANDOM_SEED:
        return LLAMA_RANDOM_SEED

    if 0 <= seed <= MAX_LLAMA_SEED:
        return seed

    return seed % LLAMA_SEED_MODULUS


def _ensure_mtp_supported(cli_path: Path) -> None:
    """
    Fail early with a useful message if an old/custom llama.cpp binary does not
    expose draft-mtp support.
    """
    key = str(cli_path.resolve())
    if key in _MTP_HELP_CHECKED:
        return

    try:
        result = subprocess.run(
            [str(cli_path), "--help"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            shell=False,
        )
        help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    except Exception as exc:
        raise RuntimeError(
            f"Could not verify MTP support in llama.cpp: {exc}"
        ) from exc

    if "draft-mtp" not in help_text:
        raise RuntimeError(
            "The installed llama.cpp binary does not advertise 'draft-mtp'. "
            "Update llama.cpp to a build that supports "
            "--model-draft/-md and --spec-type draft-mtp."
        )

    _MTP_HELP_CHECKED.add(key)


def build_command(
    model_path: Path,
    mmproj_path: Path | None,
    mtp_path: Path | None,
    system_prompt_path: Path | None,
    image,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    ctx_size: int,
    memory_mode: str,
    n_gpu_layers: int,
    n_cpu_moe_layers: int,
    seed: int,
    reasoning: str,
    extra_args: list[str] | None = None,
) -> tuple[list[str], tuple[Path | None, ...]]:
    cleanup_paths: list[Path | None] = []

    cli_paths = ensure_llama_cli_paths()

    if mtp_path is not None:
        _ensure_mtp_supported(cli_paths.cli)

    image_paths: list[Path] = []

    if image is not None:
        if mmproj_path is None:
            raise ValueError(
                "Image input requires a selected mmproj GGUF file."
            )

        image_paths = tensor_to_temp_pngs(image)
        cleanup_paths.extend(image_paths)

    prompt_path = _write_prompt_file(prompt)
    cleanup_paths.append(prompt_path)

    command = [
        str(cli_paths.cli),
        "-m", str(model_path),
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-c", str(ctx_size),
        "--seed", str(normalize_llama_seed(seed)),
        "--single-turn",
        "--reasoning", reasoning,
    ]

    # Keep the original node's memory-placement behavior.
    if memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["-ngl", str(n_gpu_layers)])

    if memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
        command.extend(["--n-cpu-moe", str(n_cpu_moe_layers)])

    if system_prompt_path is not None:
        command.extend(["-sysf", str(system_prompt_path)])

    command.extend(["-f", str(prompt_path)])

    if image_paths:
        command.extend(["--mmproj", str(mmproj_path)])
        command.extend([
            "--image",
            ",".join(str(path) for path in image_paths),
        ])

    # User advanced args stay available for parameters such as:
    #   --min-p 0.05
    #   --spec-draft-n-max 2
    #   --spec-draft-p-min 0.8
    if extra_args:
        command.extend(extra_args)

    # MTP selector is authoritative: when an MTP model is selected these flags
    # are appended last, so an accidental '--spec-type none' in extra_args
    # cannot silently disable MTP.
    if mtp_path is not None:
        command.extend([
            "--model-draft", str(mtp_path),
            "--spec-type", "draft-mtp",
        ])

    return command, tuple(cleanup_paths)


def run_llama_cli(
    command: list[str],
    timeout_seconds: int,
    cleanup_paths: tuple[Path | None, ...] = (),
) -> tuple[str, str, str]:
    process = None

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
        )

        stdout, stderr = _communicate_with_interrupt(
            process,
            timeout_seconds,
        )

        result = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout,
            stderr,
        )

    except BaseException:
        if process is not None:
            _stop_process(process)
        raise

    finally:
        for path in cleanup_paths:
            if path and path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass

    if result.returncode != 0:
        stderr = result.stderr.strip()
        message = _parse_llama_error(stderr)

        if message:
            raise RuntimeError(message)

        raise RuntimeError(
            "llama.cpp inference failed with exit code "
            f"{result.returncode}:\n{stderr}"
        )

    return _parse_response(result.stdout + "\n" + result.stderr)


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _communicate_with_interrupt(
    process: subprocess.Popen,
    timeout_seconds: int,
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds

    while True:
        if comfy.model_management.processing_interrupted():
            _stop_process(process)
            comfy.model_management.throw_exception_if_processing_interrupted()

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            _stop_process(process)
            raise TimeoutError(
                f"llama.cpp timed out after {timeout_seconds}s"
            )

        try:
            return process.communicate(timeout=min(0.1, remaining))
        except subprocess.TimeoutExpired:
            continue


def _parse_response(text: str) -> tuple[str, str, str]:
    text = str(text or "")

    if PROMPT_ECHO_END in text:
        text = text.split(PROMPT_ECHO_END, 1)[1]

    perf_match = PERF_RE.search(text)
    perf = perf_match.group(0).strip() if perf_match else ""

    content = text[:perf_match.start()] if perf_match else text
    content = content.strip()

    if not content.startswith(START_THINKING):
        return content, "", perf

    thinking_text = content[len(START_THINKING):]

    if END_THINKING not in thinking_text:
        return "", thinking_text.strip(), perf

    thinking, response = thinking_text.split(END_THINKING, 1)
    return response.strip(), thinking.strip(), perf


def _parse_llama_error(stderr: str) -> str:
    stderr = str(stderr or "")

    match = MMPROJ_EMBEDDING_MISMATCH_RE.search(stderr)
    if match:
        return (
            "Selected mmproj does not match the text model "
            f"(model n_embd={match.group('model')}, "
            f"mmproj n_embd={match.group('mmproj')}). "
            "Choose the mmproj file that belongs to the selected GGUF model."
        )

    # Helpful MTP-oriented errors without making assumptions about one exact
    # llama.cpp wording/version.
    low = stderr.lower()

    if (
        "draft" in low
        and ("vocab" in low or "token" in low)
        and ("mismatch" in low or "incompatible" in low)
    ):
        return (
            "The selected MTP/draft GGUF appears incompatible with the main "
            "model. Choose the MTP head made for this exact model family/base."
        )

    return ""


# =============================================================================
# ComfyUI node
# =============================================================================

class LLMTextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    model_options(),
                    {
                        "tooltip": (
                            "Main GGUF model from ComfyUI/models/LLM. "
                            "mmproj and MTP/draft-head files are hidden."
                        ),
                    },
                ),

                "mmproj": (
                    mmproj_options(),
                    {
                        "default": NO_MMPROJ,
                        "tooltip": (
                            "Optional vision projector GGUF. "
                            "Required only when image input is connected."
                        ),
                    },
                ),

                "mtp": (
                    mtp_options(),
                    {
                        "default": NO_MTP,
                        "tooltip": (
                            "Optional MTP speculative draft-head GGUF. "
                            "Choose none to disable MTP. Selecting a file "
                            "automatically enables llama.cpp draft-mtp."
                        ),
                    },
                ),

                "system_prompt": (
                    system_prompt_options(),
                    {
                        "tooltip": (
                            "System prompt preset from "
                            "ComfyUI/models/LLM/prompts, or none."
                        ),
                    },
                ),

                "prompt": (
                    "STRING",
                    {
                        "default": "Describe this image in detail.",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "User prompt sent to the selected model.",
                    },
                ),

                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 1,
                        "max": 32768,
                        "tooltip": "Maximum number of tokens to generate.",
                    },
                ),

                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature. "
                            "Lower is more deterministic."
                        ),
                    },
                ),

                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling threshold.",
                    },
                ),

                "top_k": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 1000,
                        "tooltip": "Top-K sampling cutoff.",
                    },
                ),

                "repeat_penalty": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.01,
                        "tooltip": "Penalty applied to repeated tokens.",
                    },
                ),

                "ctx_size": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 512,
                        "max": 1048576,
                        "step": 512,
                        "tooltip": (
                            "Context window size in tokens. "
                            "Larger context uses more VRAM/RAM."
                        ),
                    },
                ),

                "memory_mode": (
                    [
                        "auto",
                        "gpu_layers",
                        "cpu_moe_layers",
                        "gpu_and_cpu_moe_layers",
                    ],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Advanced memory placement mode."
                        ),
                        "advanced": True,
                    },
                ),

                "n_gpu_layers": (
                    "INT",
                    {
                        "default": 99,
                        "min": -1,
                        "max": 999,
                        "tooltip": (
                            "Used in gpu_layers and "
                            "gpu_and_cpu_moe_layers modes."
                        ),
                        "advanced": True,
                    },
                ),

                "n_cpu_moe_layers": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 999,
                        "tooltip": (
                            "Used in cpu_moe_layers and "
                            "gpu_and_cpu_moe_layers modes."
                        ),
                        "advanced": True,
                    },
                ),

                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": -1,
                        "max": MAX_LLAMA_SEED,
                        "tooltip": (
                            "Random seed. Use -1 for a random seed."
                        ),
                    },
                ),

                "timeout_seconds": (
                    "INT",
                    {
                        "default": 300,
                        "min": 10,
                        "max": 3600,
                        "tooltip": (
                            "Maximum time before generation is stopped."
                        ),
                    },
                ),

                "reasoning": (
                    ["auto", "on", "off"],
                    {
                        "default": "off",
                        "tooltip": "Reasoning output mode.",
                    },
                ),
            },

            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional image or ComfyUI image batch. "
                            "Requires a matching mmproj."
                        ),
                    },
                ),

                "enable_processing": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Disable to forward prompt directly as RESPONSE "
                            "without running the LLM."
                        ),
                        "advanced": True,
                    },
                ),

                "extra_args": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional llama.cpp CLI parameters. "
                            "For MTP tuning you can use e.g. "
                            "--spec-draft-n-max 2 --spec-draft-p-min 0.8"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("RESPONSE", "REASONING", "PERF")

    OUTPUT_TOOLTIPS = (
        "Final model response with reasoning blocks removed.",
        "Extracted reasoning when present in model output.",
        "llama.cpp prompt and generation speed.",
    )

    FUNCTION = "generate"
    CATEGORY = "LLM Text Processor"
    TITLE = "LLM Text Processor + MTP"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model,
        mmproj,
        mtp,
        system_prompt,
    ):
        # Lists are dynamic because files can be added while ComfyUI is running.
        return True

    def generate(
        self,
        model: str,
        mmproj: str,
        mtp: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        ctx_size: int,
        memory_mode: str,
        n_gpu_layers: int,
        n_cpu_moe_layers: int,
        seed: int,
        timeout_seconds: int,
        reasoning: str,
        image=None,
        enable_processing: bool = True,
        extra_args: str = "",
    ):
        if not enable_processing:
            return (prompt, "", "")

        current_models = model_options()
        current_mmprojs = mmproj_options()
        current_mtps = mtp_options()
        current_system_prompts = system_prompt_options()

        if model not in current_models:
            raise ValueError(f"Model not found: {model}")

        if mmproj not in current_mmprojs:
            raise ValueError(f"mmproj not found: {mmproj}")

        if mtp not in current_mtps:
            raise ValueError(f"MTP model not found: {mtp}")

        if system_prompt not in current_system_prompts:
            raise ValueError(
                f"System prompt preset not found: {system_prompt}"
            )

        model_path = full_model_path(model)
        mmproj_path = full_mmproj_path(mmproj)
        mtp_path = full_mtp_path(mtp)
        system_prompt_path = full_system_prompt_path(system_prompt)

        parsed_extra_args = split_extra_args(extra_args)

        command, cleanup_paths = build_command(
            model_path=model_path,
            mmproj_path=mmproj_path,
            mtp_path=mtp_path,
            system_prompt_path=system_prompt_path,
            image=image,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            ctx_size=ctx_size,
            memory_mode=memory_mode,
            n_gpu_layers=n_gpu_layers,
            n_cpu_moe_layers=n_cpu_moe_layers,
            seed=seed,
            reasoning=reasoning,
            extra_args=parsed_extra_args,
        )

        response, reasoning_text, perf = run_llama_cli(
            command=command,
            timeout_seconds=timeout_seconds,
            cleanup_paths=cleanup_paths,
        )

        return (response, reasoning_text, perf)


NODE_CLASS_MAPPINGS = {
    "LLMTextProcessor": LLMTextProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMTextProcessor": "LLM Text Processor + MTP",
}
