<br />
<p align="center">
  <h1 align="center">CLIP Visual Spatial Reasoning</h1>
  <h3 align="center">Benchmark CLIP models using Visual Spatial Reasoning.</h3>
  
  <p align="center">  
    <a href="https://github.com/cambridgeltl/visual-spatial-reasoning">Original Visual Spatial Reasoning repo</a>
  </p>
</p>

Note: Currently this is true zero shot (so no fine tuning). I benchmark the following CLIP models:

* OpenClip laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
* OpenClip laion/CLIP-ViT-H-14-laion2B-s32B-b79K
* OpenAI Clip openai/clip-vit-large-patch14-336

Findings:

* Using the (True) / (False) modifiers proposed in the paper results gives no better than random results.
* After experimenting with many stratagies for modifying the propts I was able to get results at 55% (so slightly better than average)

Open questions:

* Will fine tuning the modle show same/better results as the model types in the VSR paper
* How do the different relationship score (does CLIP nativly understand any relationships resonable well)


### v-002 results

uses the modified prompts ie:
  The horse is left of
  The horse is left of the person.

```
python src\eval002.py --model_url laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
```
Score: 55.23%

```
python src\eval002.py --model_url laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```
Score: 55.44%

```
python src\eval002.py --model_url openai/clip-vit-large-patch14-336
```
Score: 54.39%

### v-001 results

```
python src\eval001.py --model_url laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
```
Score: 55.23%

```
python src\eval001.py --model_url laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```
Score: 53.83%

```
python src\eval001.py --model_url openai/clip-vit-large-patch14-336
```
Score: 53.86%

### v-000 results

uses the prompts from the VSR paper (but without retraining); ie:
  The horse is left of the person. (False)
  The horse is left of the person. (True)

```
python src\eval000.py --model_url laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
```
Score: 49.24%

```
python src\eval000.py --model_url laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```
Score: 49.51%

```
python src\eval000.py --model_url openai/clip-vit-large-patch14-336
```
Score: 48.85%

### install
```
conda env create
conda activate clip-vsr
```

### run
```
python src\eval.py
```

#### Download images
See [`data/`](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data) folder's readme. Images should be saved under `data/images/`.


### Citation
If you use the VSR dataset please site the orginal authors:
```bibtex
@article{Liu2022VisualSR,
  title={Visual Spatial Reasoning},
  author={Fangyu Liu and Guy Edward Toh Emerson and Nigel Collier},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.00363}
}
```

### License
This project is licensed under the [Apache-2.0 License](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/master/LICENSE).
