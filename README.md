[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1kaiser/yolov5_segment_mask/blob/master/yolov5_masked_Segment.ipynb)

<h2>Important Updates</h2>
<ul>
<li><strong>Segmentation Models <g-emoji class="g-emoji" alias="star" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2b50.png">⭐</g-emoji> NEW</strong>: SOTA YOLOv5-seg COCO-pretrained segmentation models are now available for the first time (<a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1345216006" data-permission-text="Title is private" data-url="https://github.com/ultralytics/yolov5/issues/9052" data-hovercard-type="pull_request" data-hovercard-url="/ultralytics/yolov5/pull/9052/hovercard" href="https://github.com/ultralytics/yolov5/pull/9052">#9052</a> by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/glenn-jocher/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/glenn-jocher">@glenn-jocher</a>, <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/AyushExel/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/AyushExel">@AyushExel</a> and <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Laughing-q/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Laughing-q">@Laughing-q</a>)</li>
<li><strong>Paddle Paddle Export</strong>: Export any YOLOv5 model (cls, seg, det) to Paddle format with python export.py --include paddle (<a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1376874918" data-permission-text="Title is private" data-url="https://github.com/ultralytics/yolov5/issues/9459" data-hovercard-type="pull_request" data-hovercard-url="/ultralytics/yolov5/pull/9459/hovercard" href="https://github.com/ultralytics/yolov5/pull/9459">#9459</a> by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/glenn-jocher/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/glenn-jocher">@glenn-jocher</a>)</li>
<li><strong>YOLOv5 AutoCache</strong>: Use <code>python train.py --cache ram</code> will now scan available memory and compare against predicted dataset RAM usage. This reduces risk in caching and should help improve adoption of the dataset caching feature, which can significantly speed up training. (<a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1435085802" data-permission-text="Title is private" data-url="https://github.com/ultralytics/yolov5/issues/10027" data-hovercard-type="pull_request" data-hovercard-url="/ultralytics/yolov5/pull/10027/hovercard" href="https://github.com/ultralytics/yolov5/pull/10027">#10027</a> by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/glenn-jocher/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/glenn-jocher">@glenn-jocher</a>)</li>
<li><strong>Comet Logging and Visualization Integration:</strong> Free forever, <a href="https://bit.ly/yolov5-readme-comet" rel="nofollow">Comet</a> lets you save YOLOv5 models, resume training, and interactively visualise and debug predictions. (<a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1356875326" data-permission-text="Title is private" data-url="https://github.com/ultralytics/yolov5/issues/9232" data-hovercard-type="pull_request" data-hovercard-url="/ultralytics/yolov5/pull/9232/hovercard" href="https://github.com/ultralytics/yolov5/pull/9232">#9232</a> by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/DN6/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/DN6">@DN6</a>)</li>
</ul>
<h3>New Segmentation Checkpoints</h3>
<p>We trained YOLOv5 segmentations models on COCO for 300 epochs at image size 640 using A100 GPUs. We exported all models to ONNX FP32 for CPU speed tests and to TensorRT FP16 for GPU speed tests. We ran all speed tests on Google <a href="https://colab.research.google.com/signup" rel="nofollow">Colab Pro</a> notebooks for easy reproducibility.</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>size<br><sup>(pixels)</sup></th>
<th>mAP<sup>box<br>50-95</sup></th>
<th>mAP<sup>mask<br>50-95</sup></th>
<th>Train time<br><sup>300 epochs<br>A100 (hours)</sup></th>
<th>Speed<br><sup>ONNX CPU<br>(ms)</sup></th>
<th>Speed<br><sup>TRT A100<br>(ms)</sup></th>
<th>params<br><sup>(M)</sup></th>
<th>FLOPs<br><sup><a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/640/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/640">@640</a> (B)</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n-seg.pt">YOLOv5n-seg</a></td>
<td>640</td>
<td>27.6</td>
<td>23.4</td>
<td>80:17</td>
<td><strong>62.7</strong></td>
<td><strong>1.2</strong></td>
<td><strong>2.0</strong></td>
<td><strong>7.1</strong></td>
</tr>
<tr>
<td><a href="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt">YOLOv5s-seg</a></td>
<td>640</td>
<td>37.6</td>
<td>31.7</td>
<td>88:16</td>
<td>173.3</td>
<td>1.4</td>
<td>7.6</td>
<td>26.4</td>
</tr>
<tr>
<td><a href="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m-seg.pt">YOLOv5m-seg</a></td>
<td>640</td>
<td>45.0</td>
<td>37.1</td>
<td>108:36</td>
<td>427.0</td>
<td>2.2</td>
<td>22.0</td>
<td>70.8</td>
</tr>
<tr>
<td><a href="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l-seg.pt">YOLOv5l-seg</a></td>
<td>640</td>
<td>49.0</td>
<td>39.9</td>
<td>66:43 (2x)</td>
<td>857.4</td>
<td>2.9</td>
<td>47.9</td>
<td>147.7</td>
</tr>
<tr>
<td><a href="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x-seg.pt">YOLOv5x-seg</a></td>
<td>640</td>
<td><strong>50.7</strong></td>
<td><strong>41.4</strong></td>
<td>62:56 (3x)</td>
<td>1579.2</td>
<td>4.5</td>
<td>88.8</td>
<td>265.7</td>
</tr>
</tbody>
</table>
<ul>
<li>All checkpoints are trained to 300 epochs with SGD optimizer with <code>lr0=0.01</code> and <code>weight_decay=5e-5</code> at image size 640 and all default settings.<br>Runs logged to <a href="https://wandb.ai/glenn-jocher/YOLOv5_v70_official" rel="nofollow">https://wandb.ai/glenn-jocher/YOLOv5_v70_official</a></li>
<li><strong>Accuracy</strong> values are for single-model single-scale on COCO dataset.<br>Reproduce by <code>python segment/val.py --data coco.yaml --weights yolov5s-seg.pt</code></li>
<li><strong>Speed</strong> averaged over 100 inference images using a <a href="https://colab.research.google.com/signup" rel="nofollow">Colab Pro</a> A100 High-RAM instance. Values indicate inference speed only (NMS adds about 1ms per image). <br>Reproduce by <code>python segment/val.py --data coco.yaml --weights yolov5s-seg.pt --batch 1</code></li>
<li><strong>Export</strong> to ONNX at FP32 and TensorRT at FP16 done with <code>export.py</code>. <br>Reproduce by <code>python export.py --weights yolov5s-seg.pt --include engine --device 0 --half</code></li>
</ul>
<h2>New Segmentation Usage Examples</h2>
<h3>Train</h3>
<p>YOLOv5 segmentation training supports auto-download COCO128-seg segmentation dataset with <code>--data coco128-seg.yaml</code> argument and manual download of COCO-segments dataset with <code>bash data/scripts/get_coco.sh --train --val --segments</code> and then <code>python train.py --data coco.yaml</code>.</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="# Single-GPU
python segment/train.py --model yolov5s-seg.pt --data coco128-seg.yaml --epochs 5 --img 640

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --model yolov5s-seg.pt --data coco128-seg.yaml --epochs 5 --img 640 --device 0,1,2,3"><pre><span class="pl-c"><span class="pl-c">#</span> Single-GPU</span>
python segment/train.py --model yolov5s-seg.pt --data coco128-seg.yaml --epochs 5 --img 640

<span class="pl-c"><span class="pl-c">#</span> Multi-GPU DDP</span>
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 segment/train.py --model yolov5s-seg.pt --data coco128-seg.yaml --epochs 5 --img 640 --device 0,1,2,3</pre></div>
<h3>Val</h3>
<p>Validate YOLOv5m-seg accuracy on ImageNet-1k dataset:</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="bash data/scripts/get_coco.sh --val --segments  # download COCO val segments split (780MB, 5000 images)
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate"><pre>bash data/scripts/get_coco.sh --val --segments  <span class="pl-c"><span class="pl-c">#</span> download COCO val segments split (780MB, 5000 images)</span>
python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  <span class="pl-c"><span class="pl-c">#</span> validate</span></pre></div>
<h3>Predict</h3>
<p>Use pretrained YOLOv5m-seg to predict bus.jpg:</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="python segment/predict.py --weights yolov5m-seg.pt --data data/images/bus.jpg"><pre>python segment/predict.py --weights yolov5m-seg.pt --data data/images/bus.jpg</pre></div>
<div class="highlight highlight-source-python notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m-seg.pt')  # load from PyTorch Hub (WARNING: inference not yet supported)"><pre><span class="pl-s1">model</span> <span class="pl-c1">=</span> <span class="pl-s1">torch</span>.<span class="pl-s1">hub</span>.<span class="pl-en">load</span>(<span class="pl-s">'ultralytics/yolov5'</span>, <span class="pl-s">'custom'</span>, <span class="pl-s">'yolov5m-seg.pt'</span>)  <span class="pl-c"># load from PyTorch Hub (WARNING: inference not yet supported)</span></pre></div>
<table>
<thead>
<tr>
<th><a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg"><img src="https://user-images.githubusercontent.com/26833433/203113421-decef4c4-183d-4a0a-a6c2-6435b33bc5d3.jpg" alt="zidane" style="max-width: 100%;"></a></th>
<th><a target="_blank" rel="noopener noreferrer nofollow" href="https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg"><img src="https://user-images.githubusercontent.com/26833433/203113416-11fe0025-69f7-4874-a0a6-65d0bfe2999a.jpg" alt="bus" style="max-width: 100%;"></a></th>
</tr>
</thead>
</table>
<h3>Export</h3>
<p>Export YOLOv5s-seg model to ONNX and TensorRT:</p>
<div class="highlight highlight-source-shell notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0"><pre>python export.py --weights yolov5s-seg.pt --include onnx engine --img 640 --device 0</pre></div>
<h2>Changelog</h2>
<ul>
<li>Changes between <strong>previous release and this release</strong>: <a class="commit-link" href="https://github.com/ultralytics/yolov5/compare/v6.2...v7.0"><tt>v6.2...v7.0</tt></a></li>
<li>Changes <strong>since this release</strong>: <a class="commit-link" href="https://github.com/ultralytics/yolov5/compare/v7.0...HEAD"><tt>v7.0...HEAD</tt></a></li>
</ul>
