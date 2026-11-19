# aCG: GPU-Accelerated Iterative Linear Solvers

**aCG** is a suite of GPU-accelerated iterative linear solvers based on Krylov subspace methods, including the conjugate gradient (CG) method and the stabilized biconjugate gradient (BiCGStab) method. It supports NVIDIA and AMD GPUs, as well as multi-GPU systems utilizing GPU-aware MPI, NCCL, RCCL, or NVSHMEM.

## Implementation Details

### NVIDIA GPUs
* **Core:** CUDA implementations of CG, pipelined CG, and pipelined BiCGStab.
* **Communication:**
    * **Host-initiated:** GPU-aware MPI, NCCL, or NVSHMEM.
    * **Device-initiated:** NVSHMEM (performed directly by the GPU).

### AMD GPUs
* **Core:** HIP implementations of CG, pipelined CG, and pipelined BiCGStab.
* **Communication:**
    * **Host-initiated:** GPU-aware MPI or RCCL.

## Installation and Usage
Please see the file `INSTALL` for instructions on how to build and install the software.

## License
aCG is free software, available under a permissive software license. See the file `LICENSE` for copying conditions.

## References
If you use aCG in your research, please cite the following papers.

The pipelined preconditioned CG (Pipe-PCG) and pipelined BiCGStab (Pipe-BiCGStab) solvers with triple overlap of collective communication, point-to-point communication, and computation are described in:

> Sinan Ekmekçibaşı, James D. Trotter, Johannes Langguth, Xing Cai, and Didem Unat. 2026. **Triple Overlap in Pipelined Krylov Methods on Multi-GPU Systems.** Accepted to the *SC26 Workshops (IA^3 Workshop)*. arXiv:2610.25058. https://doi.org/10.48550/arXiv.2610.25058

The CPU- and GPU-initiated communication strategies for CG are described in:

> James D. Trotter, Sinan Ekmekçibaşı, Doğan Sağbili, Johannes Langguth, Xing Cai, and Didem Unat. 2025. **CPU- and GPU-initiated Communication Strategies for Conjugate Gradient Methods on Large GPU Clusters.** In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25)*. Association for Computing Machinery, New York, NY, USA, 298–315. https://doi.org/10.1145/3712285.3759774

### BibTeX
```bibtex

@INPROCEEDINGS {,
author = { Ekmekçibaşi, Sinan and Trotter, James D. and Langguth, Johannes and Cai, Xing and Unat, Didem },
booktitle = { SC26-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis },
title = {{ Triple Overlap in Pipelined Krylov Methods on Multi-GPU Systems }},
year = {2026},
volume = {},
ISSN = {},
pages = {2171-2183},
abstract = { We introduce a novel triple-overlap implementation of the pipelined preconditioned conjugate gradient (Pipe-PCG) and pipelined stabilized biconjugate gradient (Pipe-BiCGStab) solvers, methods widely used in scientific applications. By leveraging a collective communication library, our approach extends conventional computation-communication overlap to concurrently execute collective operations, irregular P2P communication, and compute kernels, enabling computation-communication-communication overlap across multiple GPUs. We evaluate the design on the MareNostrum 5 (H100) and Leonardo (A100) supercomputers, comparing against PETSc and the state-of-the-art two-stream NCCL implementation using SuiteSparse matrices and three massive instances scaled up to 512 GPUs. The three-stream design outperforms the two-stream baseline on every instance, with the margin widening from 2-6% at 4 GPUs to 18-47% at 64 GPUs. At 64 GPUs, the geometric-mean speedup over the two-stream baseline is 1.33× on MareNostrum 5 and 1.23× on Leonardo for Pipe-PCG, and reaches 1.30× for Pipe-BiCGStab. In large scale experiments, we observed a speedup up to 35% in Pipe-PCG and up to 33% in Pipe-BiCGStab compared to two-stream design. For PETSc, the margin is even wider, up to 3.27x speed-up in both Pipe-PCG and Pipe-BiCGStab. Related code for the proposed method is available online: https://github.com/ParCoreLab/aCG },
keywords = {Parallel processing;pipeline processing;iterative methods;sparse matrices;graphics processing units;parallel algorithms},
doi = {10.1109/SCW72768.2026.00252},
url = {https://doi.ieeecomputersociety.org/10.1109/SCW72768.2026.00252},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month =Nov}


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
```
