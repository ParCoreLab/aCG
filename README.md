# aCG: GPU-Accelerated Iterative Linear Solvers

**aCG** is a suite of GPU-accelerated iterative linear solvers based on the conjugate gradient (CG) method. It supports NVIDIA and AMD GPUs, as well as multi-GPU systems utilizing GPU-aware MPI, NCCL, RCCL, or NVSHMEM.

## Implementation Details

### NVIDIA GPUs
* **Core:** CUDA implementations of CG and pipelined CG.
* **Communication:**
    * **Host-initiated:** GPU-aware MPI, NCCL, or NVSHMEM.
    * **Device-initiated:** NVSHMEM (performed directly by the GPU).

### AMD GPUs
* **Core:** HIP implementations of CG and pipelined CG.
* **Communication:**
    * **Host-initiated:** GPU-aware MPI or RCCL.

## Installation and Usage
Please see the file `INSTALL` for instructions on how to build and install the software.

## License
aCG is free software, available under a permissive software license. See the file `LICENSE` for copying conditions.

## Reference
If you use aCG in your research, please cite the following paper:

> James D. Trotter, Sinan Ekmekçibaşı, Doğan Sağbili, Johannes Langguth, Xing Cai, and Didem Unat. 2025. **CPU- and GPU-initiated Communication Strategies for Conjugate Gradient Methods on Large GPU Clusters.** In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)*. Association for Computing Machinery, New York, NY, USA, 298–315. https://doi.org/10.1145/3712285.3759774

### BibTeX
```bibtex
@inproceedings{Trotter2025,
  author    = {Trotter, James D. and Ekmek\c{c}iba\c{s}\i, Sinan and Sa\u{g}bili, Do\u{g}an and Langguth, Johannes and Cai, Xing and Unat, Didem},
  title     = {CPU- and GPU-initiated Communication Strategies for Conjugate Gradient Methods on Large GPU Clusters},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  pages     = {298--315},
  doi       = {10.1145/3712285.3759774}
}