#!/usr/bin/env bash
#
# activate_pyenv.sh — find and activate a pyenv_inferbench_cuda<X>_torch<Y> venv
#
# USAGE (must be SOURCED, not executed, so it can modify your shell env):
#
#   source activate_pyenv.sh                 # auto-detect CUDA, pick best torch match
#   source activate_pyenv.sh 130              # pin cuda130, pick best torch match
#   source activate_pyenv.sh 130 213          # pin cuda130 + torch213 exactly
#   source activate_pyenv.sh --cuda 130 --torch 213
#   source activate_pyenv.sh --list           # just list available envs, don't activate
#
# Requires $BALI_REPO to be set and to contain dirs named:
#   pyenv_inferbench_cuda<CUDAVER>_torch<TORCHVER>
#
# e.g. $BALI_REPO/pyenv_inferbench_cuda130_torch213

# --- guard against being executed instead of sourced ---------------------
(return 0 2>/dev/null)
if [ "$?" -ne 0 ]; then
    echo "ERROR: this script must be sourced, not executed." >&2
    echo "  Run:  source ${BASH_SOURCE[0]:-$0}" >&2
    exit 1
fi

declare -A _PYENV_MODULE_MAP=(
    [cuda130_torch213]="release/25.06 GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.0"
    [cuda124_torch260]="release/24.04 GCCcore/13.3.0 Python/3.12.3 CUDA/12.4.0"
    [cuda121_torch212]="release/24.04 GCCcore/11.3.0 Python/3.10.4 CUDA/12.1.1"
)


_pyenv_echo_err() {
    # respects --quiet: only prints to stderr if not quiet
    [ "$_pyenv_quiet" -eq 1 ] && return 0
    echo "$@" >&2
}
 
_pyenv_echo() {
    # respects --quiet: only prints to stdout if not quiet
    [ "$_pyenv_quiet" -eq 1 ] && return 0
    echo "$@"
}



_pyenv_find_and_activate() {
    local cuda_want="" torch_want="" list_only=0
    local pattern="pyenv_inferbench_cuda"
    _pyenv_quiet=0
    current_path=$(pwd)

    # --- arg parsing (supports flags or bare positional cuda/torch) -----
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --cuda) cuda_want="$2"; shift 2 ;;
            --torch) torch_want="$2"; shift 2 ;;
            --list) list_only=1; shift ;;
	    --quiet|-q) _pyenv_quiet=1; shift ;;
            --help|-h)
                echo "Usage: source activate_pyenv.sh [--cuda VER] [--torch VER] [--list]"
                echo "       source activate_pyenv.sh [CUDA_VER] [TORCH_VER]"
                return 0
                ;;
            *)
                if [ -z "$cuda_want" ]; then
                    cuda_want="$1"
                elif [ -z "$torch_want" ]; then
                    torch_want="$1"
                fi
                shift
                ;;
        esac
    done

    if [[ "$current_path" != *"/BALI"* ]]; then
    	echo "Error: not inside a BALI directory. BALI_REPO could not be set!" >&2
    	exit 1
    fi

    export BALI_REPO="${current_path%%/BALI*}/BALI"
    
    if [ ! -d "$BALI_REPO" ]; then
        echo "ERROR: \$BALI_REPO ($BALI_REPO) does not exist." >&2
        return 1
    fi

    # --- discover candidate envs ----------------------------------------
    local dirs=()
    while IFS= read -r -d '' d; do
        dirs+=("$d")
    done < <(find "$BALI_REPO" -maxdepth 1 -type d -name "${pattern}*_torch*" -print0 2>/dev/null)

    if [ "${#dirs[@]}" -eq 0 ]; then
        echo "ERROR: no envs matching '${pattern}<ver>_torch<ver>' found under $BALI_REPO" >&2
        return 1
    fi

    # --- parse cuda/torch version out of each dir name -------------------
    # name shape: pyenv_inferbench_cuda130_torch213
    local names=() cudas=() torches=()
    local d base
    for d in "${dirs[@]}"; do
        base="$(basename "$d")"
        if [[ "$base" =~ cuda([0-9]+)_torch([0-9]+) ]]; then
            names+=("$d")
            cudas+=("${BASH_REMATCH[1]}")
            torches+=("${BASH_REMATCH[2]}")
        fi
    done

    if [ "$list_only" -eq 1 ]; then
        echo "Available envs in $BALI_REPO:"
        local i
        for i in "${!names[@]}"; do
            printf "  cuda%s_torch%s  ->  %s\n" "${cudas[$i]}" "${torches[$i]}" "${names[$i]}"
        done
        return 0
    fi

    # --- auto-detect installed CUDA version if not given ------------------
    if [ -z "$cuda_want" ]; then
        local detected=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            # e.g. "CUDA Version: 12.4" -> 124
            detected="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -n1 | tr -d '.')"
        fi
        if [ -z "$detected" ] && command -v nvcc >/dev/null 2>&1; then
            detected="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -n1 | tr -d '.')"
        fi
        if [ -n "$detected" ]; then
            echo "Detected CUDA version: $detected (from nvidia-smi/nvcc)"
            cuda_want="$detected"
        else
            echo "Could not auto-detect CUDA version (no nvidia-smi/nvcc found)." >&2
        fi
    fi

    # --- filter candidates -------------------------------------------------
    local match_idx=() i
    for i in "${!names[@]}"; do
        if [ -n "$cuda_want" ] && [ "${cudas[$i]}" != "$cuda_want" ]; then
            continue
        fi
        if [ -n "$torch_want" ] && [ "${torches[$i]}" != "$torch_want" ]; then
            continue
        fi
        match_idx+=("$i")
    done

    if [ "${#match_idx[@]}" -eq 0 ]; then
        echo "ERROR: no env matched cuda='${cuda_want:-any}' torch='${torch_want:-any}'." >&2
        if [ "$_pyenv_quiet" -eq 0 ]; then
		echo "Available envs:" >&2
        	for i in "${!names[@]}"; do
            		printf "  cuda%s_torch%s\n" "${cudas[$i]}" "${torches[$i]}" >&2
        	done
	fi
        return 1
    fi

    local chosen_idx
    if [ "${#match_idx[@]}" -eq 1 ]; then
        chosen_idx="${match_idx[0]}"
    else
        # multiple matches (e.g. cuda pinned, torch not) -> pick highest torch version
        local best="" best_i=""
        for i in "${match_idx[@]}"; do
            if [ -z "$best" ] || [ "${torches[$i]}" -gt "$best" ]; then
                best="${torches[$i]}"
                best_i="$i"
            fi
        done
        chosen_idx="$best_i"
        echo "Multiple envs matched; picking highest torch version: torch${best}"
    fi

    # --- set cache directories -------------------------------------------------
    local cache_dir="$BALI_REPO/.cache_cuda${cudas[$chosen_idx]}_torch${torches[$chosen_idx]}"
    mkdir -p "$cache_dir"

    export XDG_CACHE_HOME="$cache_dir"
    export PIP_CACHE_DIR="$cache_dir/pip"
    export TORCH_HOME="$cache_dir/torch"
    export TRITON_CACHE_DIR="$cache_dir/triton"
    export BENTOML_HOME="$cache_dir/bentoml"
    export HF_HOME="$cache_dir/huggingface"
    export TRANSFORMERS_CACHE="$cache_dir/huggingface"
     
    echo "Cache dir set: $cache_dir"

    # --- module load (if a modules system is available) --------------------
    local key="cuda${cudas[$chosen_idx]}_torch${torches[$chosen_idx]}"
    if command -v module >/dev/null 2>&1 || declare -f module >/dev/null 2>&1; then
        local mod_args="${_PYENV_MODULE_MAP[$key]}"
        if [ -n "$mod_args" ]; then
            echo "Loading modules for $key: module load $mod_args"
            # shellcheck disable=SC2086
            module load $mod_args
            if [ $? -ne 0 ]; then
                echo "ERROR: 'module load $mod_args' failed." >&2
                return 1
            fi
        else
            echo "WARNING: 'module' command is available but no module mapping" >&2
            echo "         found for '$key' in _PYENV_MODULE_MAP. Skipping module load." >&2
        fi
    else
        echo "Note: 'module' command not found, skipping module load step."
    fi
   

    # ---- acivate env----------------------------------------------------------------
    local env_dir="${names[$chosen_idx]}"
    local activate_script="$env_dir/bin/activate"

    if [ ! -f "$activate_script" ]; then
        echo "ERROR: found env dir but no activate script at $activate_script" >&2
        return 1
    fi

    echo "Activating: $env_dir"
    # shellcheck disable=SC1090
    source "$activate_script"
}

_pyenv_find_and_activate "$@"
_PYENV_RC=$?

if [ "$_PYENV_RC" -eq 0 ] && [ -n "$VIRTUAL_ENV" ]; then
    export _PYENV_ACTIVATE_OK=1
else
    export _PYENV_ACTIVATE_OK=0
fi

unset -f _pyenv_find_and_activate _pyenv_echo _pyenv_echo_err
unset _pyenv_quiet
return "$_PYENV_RC" 2>/dev/null || exit "$_PYENV_RC"
