# Third-party HGT implementation

`models/conv.py` and `models/hgt.py` are adapted from **pyHGT**, the PyTorch
Geometric implementation of *Heterogeneous Graph Transformer* (WWW 2020).

- Repository: https://github.com/acbull/pyHGT
- Authors of the HGT paper: Ziniu Hu, Yuxiao Dong, Kuansan Wang, Yizhou Sun.
- License: MIT; upstream copyright holder: `acbull`.
- Upstream sources: `pyHGT/conv.py` and `pyHGT/model.py`.
- Retained upstream components: `DenseHGTConv`, `RelTemporalEncoding`, the
  Dense HGT branch of `GeneralConv`, and `GNN`.
- Local adaptations include passing layer attention to the dynamical model,
  removing unused graph-convolution/model alternatives and updating imports.
  These components are not claimed as original work of this release.

The upstream license is reproduced below. It applies to the upstream-derived
code independently of the license for this release's original contributions.

```text
The MIT License

Copyright (c) 2019 acbull

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
