# Consistency<sup>2</sup>: Consistent and Fast 3D Painting with Latent Consistency Models

This repository complements our CVPR 2024 [AI for 3D Generation](https://ai3dg.github.io/) Workshop Paper titled "Consistency<sup>2</sup>: Consistent and Fast 3D Painting with Latent Consistency Models" by [Tianfu Wang](https://https://tianfwang.github.io/), 
[Anton Obukhov](https://www.obukhov.ai/).
and [Konrad Schindler](https://igp.ethz.ch/personen/person-detail.html?persid=143986),

## Installation

This codebase uses ```conda``` + ```pip``` for environment setup 

Clone the repository. Inside the repo directory run the command:

```source ./scripts/setup_env.sh```

We use the ```nvdiffrast``` renderer and therefore an Nvidia GPU is required to run the program.

## Run

Try paint our default horse mesh setting simply with 

```python run.py```

The process takes less than 2 minutes on an Nvidia GPU.

Results will appear in the ```output``` directory

To try more settings and meshes adjust the config settings in as shown in the ```config``` directory.

## License

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

[<img src="doc/badges/badge-license.svg" height="32"/>](http://creativecommons.org/licenses/by-nc-sa/4.0/)


 

