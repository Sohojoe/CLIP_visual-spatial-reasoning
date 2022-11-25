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
only FALSE examples from all_vsr_validated_data 
clip-vit-large-patch14-336         = 59.27898402294142
clip-vit-large-patch14             = 61.90086030315445
openai/clip-vit-base-patch32       = 41.765669807455963
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
