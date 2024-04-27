# Directory Content

This directory contains adaptations of the [River](https://github.com/online-ml/river) programs but for online machine learning on embedded systems (ESP32...).

For instance, the k_means directory contains an adaptation of k_means available with River. And so on.

It also contains instructions to build [micropython-ulab](https://micropython-ulab.readthedocs.io/en/latest/). [Micropython](https://micropython.org/) is Python for embedded systems and [ulab](https://github.com/v923z/micropython-ulab) is a numpy-like array manipulation library for Micropython. Moreover, the [Espressif](https://github.com/espressif/) repository contains many open-source projects, including SDKs, components, libraries, solutions, and tools, which aim to help developers bring their projects to life.

## Configure and Run Micropython-ulab on ESP32-based boards in the Linux Operating System

### Prerequisite/Requirement of Micropython-ulab

You must have installed clang or GCC, cmake and ESP-idf on your PC. You can download the ESP-idf from https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html and cmake from https://cmake.org/download/

It's recommended to download the latest stable version of clang/GCC, and cmake. 

Firmware for `Espressif` hardware can be built in two different ways, and you can use any of them.

### Compiling with cmake

First, clone the `ulab`, the `micropython`, as well as the `espressif` repositories:

```bash
export BUILD_DIR=$(pwd)

git clone https://github.com/v923z/micropython-ulab.git ulab
git clone https://github.com/micropython/micropython.git

cd $BUILD_DIR/micropython/

git clone -b v5.1 --recursive https://github.com/espressif/esp-idf.git

```
Also later releases of `esp-idf` are possible (e.g. `v5.2`).

Then install the `ESP-IDF` tools:

```bash
cd esp-idf
./install.sh
. ./export.sh
```

Next, build the `micropython` cross-compiler, and the `ESP` sub-modules:

```bash
cd $BUILD_DIR/micropython/mpy-cross
make
cd $BUILD_DIR/micropython/ports/esp32
make submodules
```
After successfully installing all the modules, You have two options. You can run the build command inside your project or create a `makefile` and make it.

#### Option 1
Go to `$BUILD_DIR/micropython/ports/esp32` and run 
```bash
idf.py build
```
This will generate a micropython image inside `$BUILD_DIR/micropython/ports/esp32/build-ESP32_GENERIC`. You can use that image to flash ESP-32

#### Option 2
You can also compile the firmware with `ulab` according to this second way. In `$BUILD_DIR/micropython/ports/esp32` create a `makefile` with the following content:
```bash
BOARD = GENERIC
USER_C_MODULES = $(BUILD_DIR)/ulab/code/micropython.cmake

include Makefile
```
You specify with the `BOARD` variable, what you want to compile for, a generic board, or `TINYPICO` (for `micropython` version >1.1.5, use `UM_TINYPICO`), etc. Still in `$BUILD_DIR/micropython/ports/esp32`, you can now run `make`.

### Flash ESP-32 with Micropython Image
You can flash your ESP_32 board by running the following command in your root directory.
```bash

$BUILD_DIR/micropython/esp-idf/components/esptool_py/esptool/esptool.py -p /dev/ttyUSB0 -b 460800 --before default_reset --after hard_reset --chip esp32  write_flash --flash_mode dio --flash_size 4MB --flash_freq 40m 0x1000 $BUILD_DIR/micropython/ports/esp32/build-ESP32_GENERIC/bootloader/bootloader.bin 0x8000 $BUILD_DIR/micropython/ports/esp32/build-ESP32_GENERIC/partition_table/partition-table.bin 0x10000 $BUILD_DIR/micropython/ports/esp32/build-ESP32_GENERIC/micropython.bin
```

### Run Micropython-ulab on ESP-32
Just open the terminal and run this command. /dev/ttyUSB0 is the most common port used in Linux for ESP_32, you can change the port accordingly.
```bash
picocom -b 115200 /dev/ttyUSB0
```

