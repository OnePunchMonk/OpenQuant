# Use a lightweight Python image
FROM python:3.10-slim

# Prevent python from buffering stdout/stderr (better logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system basics (needed for some math libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev tools)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the actual app code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck to keep the cloud service happy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The command to run your website
CMD ["streamlit", "run", "run_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
