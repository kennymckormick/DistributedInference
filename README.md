# DistributedInference
A minimal framework for distributed inference, run on PC, CDC and Cluster.

Inference workload will be split into n parts, and uniformly distributed to cuda devices.

## Support
### Image Classification Inference with Test Time Augmentation
Some demo data is provided in demo_data/image_reg, to inference these images, run command:

`./tools/dist_test.sh 2 --imglist demo_data/image_reg/img_list.txt --imgroot demo_data/image_reg/frames --out test.pkl --out_pred test.txt`

You can also use ten crop augmentation during test time:

`./tools/dist_test.sh 2 --imglist demo_data/image_reg/img_list.txt --imgroot demo_data/image_reg/frames --flip_aug --crop_aug five --out test.pkl --out_pred test.txt `

## Todo
- [ ] Support Optical Flow Inference
