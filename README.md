# Deep-Audio-Visualization

Deep Learning based Music Visualizer based on the work of Michael R Lin. Please checkout his [Blog Post](https://m-lin-dm.github.io/Deep_audio_embedding/).
This repository is a rewrite of his approach in Pytorch.
0.5-4s windows of a music track are encoded in 3D with a 1D CNN autoencoder. The track is then played and the latent space points are visualized as path in the 3D space. In addition 10 frequency bands alternate around the path based on their current intensity. E.g. deep bass tones can be observed as spikes in the blue line.


Check out a visualization of the Kodaline song Brother:

[![Brother (Kodaline) - Visualized with Machine Learning](https://img.youtube.com/vi/-rkOwTGEmGM/0.jpg)](https://www.youtube.com/watch?v=-rkOwTGEmGM)

