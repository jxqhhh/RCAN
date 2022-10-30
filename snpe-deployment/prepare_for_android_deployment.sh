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
  snpe-pytorch-to-dlc --input_network dlc/pt/RCAN_BIX2.pt --input_dim input "1,3,256,256" --output_path dlc/RCAN_BIX2.dlc
fi

# TODO: support quantize
#if [ ! -d "images/Set5/LR/LRBI/x2/size256" ]; then
#  mkdir -p images/Set5/LR/LRBI/x2/size256
#  python create_RCAN_raws.py -d images/Set5/LR/LRBI/x2/size256 -s 256 -i ../RCAN_TestCode/LR/LRBI/Set5/x2
#fi

#rm -rf build
#mkdir -p build/images
#cp dlc/RCAN_BIX2.dlc build/model.dlc
#cp images/Set5/LR/LRBI/x2/size256/*.jpg build/images
#cd build
zip -r rcan_bix2.zip *
cd ..
mv build/rcan_bix2.zip .
rm -rf build

