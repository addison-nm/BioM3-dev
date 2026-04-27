#!/usr/bin/env bash
#=============================================================================
#
# FILE: _wandb_resolve.sh
#
# DESCRIPTION: Shared helper sourced by stage{1,2,3}_train_{single,multi}node.sh
#   to resolve whether wandb logging should be enabled. Combines two signals:
#
#     1. User intent     — explicit --wandb True|False in the caller args.
#     2. API capability  — WANDB_API_KEY env var.
#
#   Precedence:
#     - User passes --wandb False → wandb OFF (always honored).
#     - User passes --wandb True  → requires WANDB_API_KEY; errors if unset.
#     - User omits --wandb        → defaults to True if WANDB_API_KEY is set,
#                                   else False (with a warning).
#
#   Sets the shell variable `wandb_resolved`:
#     - The string "--wandb True" or "--wandb False" to append to the
#       entrypoint invocation when the user did NOT pass --wandb.
#     - Empty when the user passed --wandb explicitly (their flag travels
#       through "$@" unchanged; appending again would duplicate).
#
#   USAGE: source _wandb_resolve.sh "$@"
#
#=============================================================================

wandb_resolved=""
_user_wandb=""
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--wandb" ]; then
        _user_wandb="$_arg"
    fi
    _prev="$_arg"
done
unset _prev _arg

if [ -n "${_user_wandb}" ]; then
    case "${_user_wandb}" in
        True|true|1)
            if [ -z "${WANDB_API_KEY:-}" ]; then
                echo "ERROR: --wandb True requested but WANDB_API_KEY is not set." >&2
                echo "       Export WANDB_API_KEY before launching, or pass --wandb False." >&2
                exit 1
            fi
            ;;
        False|false|0)
            : # explicit opt-out, nothing to do
            ;;
        *)
            echo "ERROR: invalid --wandb value '${_user_wandb}' (expected True or False)" >&2
            exit 1
            ;;
    esac
else
    if [ -z "${WANDB_API_KEY:-}" ]; then
        echo "WARNING: WANDB_API_KEY is empty — defaulting to --wandb False"
        wandb_resolved="--wandb False"
    else
        wandb_resolved="--wandb True"
    fi
fi
unset _user_wandb
