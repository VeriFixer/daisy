# Dockerfile - Fully self-contained Ubuntu 22.04 image
FROM ubuntu:24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget sudo make build-essential unzip zip python3 python3-pip python3-venv \
    libicu-dev tzdata ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# --- Install z3 
# Build and install Z3 from the tagged source to ensure exact version
ARG Z3_VERSION="4.15.2"
ENV Z3_VERSION=${Z3_VERSION}

RUN set -eux; \
    url="https://github.com/Z3Prover/z3/releases/download/z3-${Z3_VERSION}/z3-${Z3_VERSION}-x64-glibc-2.39.zip"; \
    echo "Downloading Z3 from $url"; \
    curl -L -o z3-${Z3_VERSION}.zip "$url"; \
    unzip z3-${Z3_VERSION}.zip; \
    # Determine the extracted folder (likely something like z3-${Z3_VERSION}-x64-glibc-2.39) \
    dir="$(unzip -Z -1 z3-${Z3_VERSION}.zip | head -n1 | cut -d/ -f1)"; \
    echo "Installing Z3 from folder: $dir"; \
    cp -a "$dir"/bin/* /usr/local/bin/; \
    cp -a "$dir"/include/* /usr/local/include/ 2>/dev/null || true; \
    ldconfig; \
    # If python bindings present, install them
    rm -rf /app/z3-${Z3_VERSION}.zip /app/"$dir"

# --- Install .NET SDK 8.0 ---
RUN wget https://dot.net/v1/dotnet-install.sh -O /tmp/dotnet-install.sh && \
    chmod +x /tmp/dotnet-install.sh && \
    /tmp/dotnet-install.sh --channel 8.0 --install-dir /usr/share/dotnet && \
    ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet

# --- Create python Env 
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Install Python dependencies ---
COPY src/requirements.txt ./src/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r ./src/requirements.txt

# Install OpenJDK for Java/Gradle builds
RUN apt-get update && apt-get install -y openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# --- Copy entire repository (including vendored submodules) ---
COPY . /app

# --- Build Dafny fork ---
RUN cd external/dafny_fork && make

# --- Build Laurel placeholder finder ---
RUN cd external/dafny_laurel_repair/laurel/placeholder_finder \
    && dotnet build placeholder_finder.csproj

# --- Build Laurel+ placeholder finder ---
RUN cd external/dafny_laurel_repair/laurel/placeholder_finder_better \
    && dotnet build placeholder_finder_laurel_better.csproj
# --- Create a non-root user ---
RUN useradd --create-home researcher

# --- Create temp and ensure permissions ---
RUN mkdir -p /app/temp && mkdir -p /home/researcher \
    && chown -R researcher:researcher /app/temp /home/researcher /home/researcher

# --- Switch to non-root user ---
USER researcher
ENV HOME=/home/researcher
ENV PATH="/app/external/dafny_fork:/app/external/dafny_laurel_repair:$PATH"
# Default command
CMD ["bash"]
