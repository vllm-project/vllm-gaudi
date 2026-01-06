#!/bin/bash

set -e

UCX_DIR=${UCX_DIR:-"/tmp/ucx_source"}
NIXL_DIR=${NIXL_DIR:-"/tmp/nixl_source"}
UCX_INSTALL_DIR=${UCX_INSTALL_DIR:-"/tmp/ucx_install"}

UCX_REPO_URL="https://github.com/openucx/ucx.git"
UCX_COMMIT="1df7b045d36c1e84f2fe9f251de83fb9103fc80e"
NIXL_REPO_URL="https://github.com/ai-dynamo/nixl.git"
NIXL_BRANCH="0.7.0"

# Device specific configuration
if command -v nvidia-smi >/dev/null 2>&1; then
	DEVICE="cuda"
elif command -v hl-smi >/dev/null 2>&1; then
	DEVICE="hpu"
else
	echo "Unknown device, aborting install."
	exit 1
fi

echo "UCX_DIR: $UCX_DIR"
echo "NIXL_DIR: $NIXL_DIR"

echo "Installing prerequisites"
apt-get update
apt install -y build-essential cmake libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev rdma-core \
	pkg-config meson ninja-build autoconf libtool libcjson-dev libaio-dev pybind11-dev

echo "Installing UCX ($UCX_COMMIT) to $UCX_INSTALL_DIR"
ucx_root=$(dirname "$UCX_DIR")
mkdir -p "$ucx_root"
[[ -d $UCX_DIR ]] || git clone "$UCX_REPO_URL" "$UCX_DIR"
cd "$UCX_DIR" && git checkout "$UCX_COMMIT"
./autogen.sh
if [ "$DEVICE" == "hpu" ]; then
	./configure --prefix="$UCX_INSTALL_DIR" --with-mlx5=no --with-gaudi=yes --enable-examples --enable-mt
else
	./configure --prefix="$UCX_INSTALL_DIR" --with-mlx5=no --with-gaudi=no --enable-examples --enable-mt --with-cuda=/usr/local/cuda
fi
make -j 8 && make -j install-strip && ldconfig

echo "Installing NIXL ($NIXL_BRANCH) to $NIXL_DIR"
nixl_root=$(dirname "$NIXL_DIR")
mkdir -p "$nixl_root"
[[ -d $NIXL_DIR ]] || git clone -b "$NIXL_BRANCH" "$NIXL_REPO_URL" "$NIXL_DIR"
cd "$NIXL_DIR"
meson setup --reconfigure build -Ducx_path="$UCX_INSTALL_DIR" -Dinstall_headers=true -Ddisable_gds_backend=false
sed -i "s|\(option('ucx_path', type: 'string', value: \)'[^']*|\1'$UCX_INSTALL_DIR|" "$NIXL_DIR/meson_options.txt"
cd build
ninja && ninja install

pip install "$NIXL_DIR"

echo "Completed nixl install"
echo ""
echo "Set these env vars after installing: "
echo 'export UCX_MEMTYPE_CACHE=0'
echo 'export LD_LIBRARY_PATH="/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"'
echo 'export LD_LIBRARY_PATH="${UCX_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"'
echo '  e.g. export LD_LIBRARY_PATH="/tmp/ucx_install/lib:${LD_LIBRARY_PATH}"'
