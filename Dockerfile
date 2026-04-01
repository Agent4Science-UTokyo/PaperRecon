ARG UBUNTU_VERSION=24.04
ARG CLAUDECODE_VERSION=2.1.77
ARG CODEX_VERSION=0.115.0

FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
    ca-certificates \
    curl \
    vim \
    sudo \
    bat \
    make \
    perl \
    texlive-full \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $HOME

# Update root user's .bashrc
RUN cp /home/ubuntu/.bashrc $HOME/.bashrc

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies via pixi
COPY ./pixi.toml $HOME/
RUN $HOME/.pixi/bin/pixi install
ENV PATH="$HOME/.pixi/envs/default/bin:$PATH"

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash -s ${CLAUDECODE_VERSION}
ENV PATH="$HOME/.local/bin:$PATH"

# Install Codex (x86_64)
RUN curl -L https://github.com/openai/codex/releases/download/rust-v0.115.0/codex-x86_64-unknown-linux-musl.tar.gz \
    -o /tmp/codex.tar.gz \
    && tar -xzf /tmp/codex.tar.gz -C /tmp \
    && install /tmp/codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
    && rm -rf /tmp/codex.tar.gz /tmp/codex-x86_64-unknown-linux-musl

# Setup bat alias
RUN echo "alias bat='batcat'" >> $HOME/.bashrc

# Install latexpand
RUN curl -fsSL https://gitlab.com/latexpand/latexpand/-/raw/master/latexpand \
    -o /usr/local/bin/latexpand \
    && chmod +x /usr/local/bin/latexpand
