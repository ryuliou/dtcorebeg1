#!/bin/bash
# Run the YOLO weights download script
/download_yolo_weights.sh
# Now execute the CMD from the Dockerfile or the command passed to docker run
exec "$@"