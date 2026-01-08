#!/bin/bash
set -euo pipefail

Xvfb "$DISPLAY" -screen 0 1280x720x24 +extension GLX +render -noreset &> /tmp/xvfb.log &
sleep 0.2

cd /Linux
cmd=( ./gym_citynav.sh )
if [ -n "${UE_MAP:-}" ]; then
  cmd+=( "$UE_MAP" )
fi
cmd+=( -UnrealCVPort "$UE_PORT" -RenderOffscreen )

"${cmd[@]}" > "$WORKDIR/unreal.log" 2>&1 &
wait