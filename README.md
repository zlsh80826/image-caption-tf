# Image Caption Tensorflow

* Image caption model base on [
Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) with some modifications.
* The dataset come from [Microsoft COCO 2014](http://cocodataset.org/#home) train and valid, and we do some redistribution.
* This model is trained for [NTHU CS565600](http://www.cs.nthu.edu.tw/~shwu/) image caption competition.
* Our model achieved 0.944 CIDEr-D score on single model, which is the 1st place of the [Image Caption Kaggle Competition](https://www.kaggle.com/c/datalabcup-image-caption-2018fall/leaderboard).
* We provide end to end scripts and pretrained weight for reproduction.
* [This slides](https://docs.google.com/presentation/d/1LU6CEDQIag6S8wtL4aTgqb9HuYRfsg2jOsGEOAhkN5g/present?slide=id.p) briefly describe the implementation
* If you meet any problem, feel free to contact (zlsh80826@gmail.com).

## Requirements

Here are some required libraries.

### General 
* python >= 3.6
* cuda >= 10.0 (or base on your tensorflow version)

### Python
* please refer requirements.txt

## Reproduce from scratch

### Download the data

```Bash
cd data
sh download.sh
```

### Redistribute the data (Competition required)

```Bash
python split.py
```

### Generate the image features

We use the [NASNet](https://arxiv.org/abs/1707.07012) model [pretrained by Keras](https://keras.io/applications/) 
to get the image features. This step may took over one hour.

```Bash
python nasnet.py
```

### Create tensorflow records

```Bash
cd ../script
python create_tfrecord.py
```

### Train

```Bash
python train.py
```

### Evaluate on validation set

```Bash
python inference.py
```

## Performance

||CIDEr-D|
|---|---|
|Single Model|0.944|
|Ensemble Model|0.955|
