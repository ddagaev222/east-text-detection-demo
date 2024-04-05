# EAST Text Detection DEMO

This is a demo application demonstrates how EAST text detection model works in the real environment.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Dependencies](#2-install-dependencies)
  - [3. Environment Setup](#3-environment-setup)
- [Usage](#usage)
- [License](#license)

## Overview

EAST model for text detection from natural scenes.
The algorithm is called “EAST” because it’s an: Efficient and Accurate Scene Text detection pipeline.

The EAST pipeline is capable of predicting words and lines of text at arbitrary orientations on 720p images, and furthermore, can run at 13 FPS, according to the authors.

## Prerequisites

List any prerequisites or system requirements needed to set up the development environment. This could include:

- Python version: >=3.9, <4.0
- Required poetry or pip

## Setup

Install the project directly from git repository (add venv-path/Lib/site-packages to the PATH or go to venv-path/Lib/site-packages before run east_demo executable):

```bash
pip install git+https://github.com/ddagaev222/east-text-detection-demo.git
.east_demo
```


### 1. Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/ddagaev222/east-text-detection-demo.git
```

### 2. Install Dependencies

Navigate to the project directory and install the required dependencies using Poetry:

**using Poetry**
```bash
cd east-text-detection-demo
poetry install
```

**using Pip (on Ubuntu or linux-based OS)**
```bash
cd east-text-detection-demo
pip install -r requirements.txt
```

**using Pip (on Windows)**
```bash
cd east-text-detection-demo
pip install -r requirements_win.txt
```

**For developers: this project uses pre-commit hooks, to install them mannualy run after installation completed**
```bash
pre-commit install
```

### 3. Environment Setup

Install poetry
```bash
pip install poetry
```

Add venv-path/Lib/site-packages to the PATH
```bash
PATH = $PATH + "<path-to-venv-site-package-dir>"
```

## Usage

To run the project locally, use the following command:

**Poetry**
```bash
poetry run east_demo
```

**Or just**
```bash
cd ~/your-venv-path/Lib/site-packages
east_demo
```

## License

Copyright 2017 The TensorFlow Authors.  All rights reserved.
