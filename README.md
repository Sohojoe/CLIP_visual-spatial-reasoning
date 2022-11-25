<br />
<p align="center">
  <h1 align="center">CLIP Visual Spatial Reasoning</h1>
  <h3 align="center">Benchmark CLIP models using Visual Spatial Reasoning.</h3>
  
  <p align="center">  
    <a href="https://github.com/cambridgeltl/visual-spatial-reasoning">Original Visual Spatial Reasoning repo</a>
  </p>
</p>


### results
(basically random; looking into why)
```
openai/clip-vit-base-patch32  52.47035573122529%
clip-vit-large-patch14        50.74110671936759
clip-vit-large-patch14-336    50.79051383399209
```

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
