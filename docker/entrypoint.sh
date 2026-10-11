#!/bin/sh
# usage: entrypoint.sh [server|addon|all|<any colorstream command>]
set -e
case "$1" in
  server) exec python3 -m colorstream serve ;;
  addon)  cd addon && exec node index.js ;;
  all)
    python3 -m colorstream serve &
    cd addon && node index.js &
    wait -n
    ;;
  *) exec python3 -m colorstream "$@" ;;
esac
