#!/bin/bash
# Deprecated: FP8 calibration moved to the vllm-gaudi-calibrate command.
# This stub only prints the equivalent command and will be removed in a future release.
# Option values are shell-quoted, so the printed command can be copied as is.

MODEL="<model>"
OUTPUT="<output_dir>"
EXTRA=""
HELP=false
while getopts "m:d:o:b:l:t:r:euh" opt; do
    case $opt in
        m) printf -v MODEL "%q" "$OPTARG" ;;
        o) printf -v OUTPUT "%q" "$OPTARG" ;;
        l) printf -v EXTRA "%s --limit %q" "$EXTRA" "$OPTARG" ;;
        b) printf -v EXTRA "%s --batch-size %q" "$EXTRA" "$OPTARG" ;;
        t) printf -v EXTRA "%s --tp %q" "$EXTRA" "$OPTARG" ;;
        r) printf -v EXTRA "%s --unify-to-tp %q" "$EXTRA" "$OPTARG" ;;
        e) EXTRA+=" --enforce-eager" ;;
        u) EXTRA+=" --expert-parallel" ;;
        h) HELP=true ;;
        *) ;;
    esac
done

message() {
    cat <<MSG
$(basename "$0") is deprecated and no longer calibrates models.
Text and multimodal models are now calibrated with one command:

    pip install -r calibration/requirements.txt
    vllm-gaudi-calibrate run ${MODEL} -o ${OUTPUT}${EXTRA}

Calibration datasets (-d) are replaced by lm-eval tasks (--tasks). See
docs/configuration/calibration/migration.md for every old option.
MSG
}

# -h is a successful help request: message on stdout, exit 0.
if $HELP; then
    message
    exit 0
fi
message >&2
exit 2
