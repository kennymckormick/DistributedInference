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
   - [x] FlowVideoDataset

  Algorithms:

  - [x] FlowNet2
  - [x] PWCNet
  - [x] VCN

First, switch to flow branch, compile needed packages with `bash compile.sh`

To run inference on demo data (frames), run command:

  `./tools/dist_test.sh 2 --list demo_data/flow_est/img_list.txt --root demo_data/flow_est/ --pad_base 64 --se 512 --out_se 256 --algo flownet2 --input img`

To run inference in demo data (videos), run command:

`./tools/dist_test.sh 2 --list demo_data/flow_est/vid_list.txt --root demo_data/flow_est/ --pad_base 64 --se 512 --out_se 256 --algo flownet2 --input vid`

Add `--vis`, you can get colorful visualization instead of gray scale flow images.

Here are two demos for videos in UCF101, the layout is:

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-nrix{text-align:center;vertical-align:middle}
.tg .tg-0lax{text-align:left;vertical-align:top}
</style>
<table class="tg" style="width:350px">
  <tr>
    <th class="tg-nrix"></th>
    <th class="tg-nrix" colspan="2">RGB</th>
    <th class="tg-baqh" colspan="2">TVL1</th>
    <th class="tg-0lax"></th>
  </tr>
  <tr>
    <td class="tg-nrix" colspan="2">Flownet2</td>
    <td class="tg-baqh" colspan="2">PWCNet</td>
    <td class="tg-baqh" colspan="2">VCN</td>
  </tr>
</table>

Billiards: 

<img src="images/billiards.gif" >

IceDancing: 

<img src="images/icedancing.gif">