#/bin/bash

# check environment:
pytorch_version=$(python -c "import torch; print(torch.__version__)")
if [[ "$pytorch_version" != 1.0.0* ]] && [[ "$pytorch_version" != 1.4.0* ]]; then
  echo "[Warning] Current pytorch version is not validated. Please use pytorch 1.0.0 or 1.4.0 if you encouter some problems later."
fi
if [ -z "$SNPE_ROOT" ]; then
  echo "[Warning]: SNPE_ROOT not set as an environment variable"
elif [[ "$SNPE_ROOT" != *1.66.0* ]]; then
  echo "[Warning] Current snpe version is not validated. Please use snpe 1.66.0 if you encouter some problems later."
fi

# convert pt to onnx
if [ ! -f "dlc/pt/RCAN_BIX2.pt" ]; then
  python prepare_pt_for_snpe.py
fi

# convert onnx to dlc
if [ ! -f "dlc/RCAN_BIX2.dlc" ]; then
  snpe-onnx-to-dlc -i dlc/pt/RCAN_BIX2.onnx
  cp dlc/pt/RCAN_BIX2.dlc dlc/pt/model.dlc
fi

# we do not call snpe-dlc-quantize, since the output dlc outputs very strange results when we exucute it on Xiaomi 12S
# here we directly use non-quantized dlc for DSP, as "SNPE will automatically quantize the network parameters in order to run on the DSP" (see https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html)

# TODO: package images and dlc

