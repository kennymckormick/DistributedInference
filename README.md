# DistributedInference
  A minimal framework for distributed inference, run on PC, CDC and Cluster.

  Inference workload will be split into n parts, and uniformly distributed to cuda devices.

  ## Support
  ### Image Classification Inference with Test Time Augmentation
  Some demo data is provided in demo_data/image_reg, to inference these images, run command:

  `./tools/dist_test.sh 2 --imglist demo_data/image_reg/img_list.txt --imgroot demo_data/image_reg/frames --out test.pkl --out_pred test.txt`

  You can also use ten crop augmentation during test time:

  `./tools/dist_test.sh 2 --imglist demo_data/image_reg/img_list.txt --imgroot demo_data/image_reg/frames --flip_aug --crop_aug five --out test.pkl --out_pred test.txt `

  ### Flow estimation

  Datasets: 

   - [x] FlowFrameDataset
   - [ ] FlowVideoDataset

  Algorithms:

  - [x] FlowNet2
  - [ ] PWCNet
  - [ ] VCN

  First, switch to flow branch, compile needed packages with `bash compile.sh`

  To run inference on demo data, run command:

  `./tools/dist_test.sh 2 --imglist demo_data/flow_est/img_list.txt --imgroot demo_data/flow_est/ --pad_base 64`

  Since FlowNet2 only run with images who's width and height are devided by 64, pad_base needs to set to 64.

  Add `--vis`, you can get colorful visualization instead of gray scale flow images.
