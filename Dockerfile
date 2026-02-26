# Use Python 3.12 slim for a small, secure base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pixel-patrol from PyPI
# We also install 'uv' as it is recommended by the authors for dependency management,
# though standard pip works too. Here we stick to standard pip for simplicity.
RUN pip install --no-cache-dir pixel-patrol

# Create a volume mount point for data
VOLUME /data

# Expose dashboard ports (8050=report, 8051=launch/processing)
EXPOSE 8050 8051

# Bind to 0.0.0.0 so the server is accessible from outside the container
ENV PIXEL_PATROL_HOST=0.0.0.0

# Set the entrypoint to the pixel-patrol CLI
ENTRYPOINT ["pixel-patrol"]

# Default command (shows help if no arguments provided)
CMD ["--help"]