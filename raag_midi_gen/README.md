# Overview
This directory is meant to function as a python package. When you setup this cookiecutter project, it will install this package in editable mode right after it initializes your conda environment ('editable mode' means that you can edit the code in this package, and it will reflect immediately wherever you import this package).

The idea is that you can use this directory to place custom classes/functions that you want to be able to import in other parts of the project. 

For example, let's say you create a new kind of transform for your image data after a bunch of notebook-based experimentation. You can take that transform, encapsulate it in a class (`CustomTransform`), and place it in a python module (`custom_transforms`) in this directory. Then, you can simply import it in other notebooks by doing `from raag_midi_gen.custom_transforms import CustomTransform`.

