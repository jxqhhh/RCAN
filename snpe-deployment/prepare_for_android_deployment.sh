if [ ! -f "dlc/pt/RCAN_BIX2.pt" ]; then
  python prepare_pt_for_snpe.py
fi
if [ ! -f "dlc/RCAN_BIX2.dlc" ]; then
  #conda deactivate
  #conda activate snpe
  snpe-pytorch-to-dlc --input_network dlc/pt/RCAN_BIX2.pt --input_dim input "1,3,256,256" --output_path dlc/RCAN_BIX2.dlc
fi
if [ ! -d "images/Set5/LR/LRBI/x2/size256" ]; then
  mkdir -p images/Set5/LR/LRBI/x2/size256
  python create_RCAN_raws.py -d images/Set5/LR/LRBI/x2/size256 -s 256 -i ../RCAN_TestCode/LR/LRBI/Set5/x2
fi

rm -rf build
mkdir -p build/images
cp dlc/RCAN_BIX2.dlc build/model.dlc
cp images/Set5/LR/LRBI/x2/size256/*.jpg build/images
cd build
zip -r rcan_bix2.zip *
cd ..
mv build/rcan_bix2.zip .
rm -rf build

