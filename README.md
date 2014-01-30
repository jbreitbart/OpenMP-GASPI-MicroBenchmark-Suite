OpenMP-GASPI-MicroBenchmark-Suite
=================================
This is the Mixed-mode OpenMP/MPI/GASPI MicroBenchmark Suite. It is based on
the Mixed-mode OpenMP/MPI MicroBenchmark Suite by Mark Bull, Jim Enright
and Fiona Reid.

This version requires an installed GASPI implementation. You can grab one from

http://www.gpi-site.com/gpi2/download/

By default the makefile expects the installtion to be in the GPI2 subdirectory
of the benchmark directory, however the path can be changed.

The GASPI benchmark supports both one- and two-sided communication. By default
the two sided communication benchmark is run, however you can enable the one-
sided communication benchmark by compiling with the option -DONE_SIDED (see
makefile). The number of threads used by the Benchmark can be controlled by
using the -DTHREADS=XX option.

Every other option should be the same as the original benchmark.

OpenMP/MPI MicroBenchmark Suite Readme below
=================================

/*****************************************************************************
 *                                                                           *
 *             Mixed-mode OpenMP/MPI MicroBenchmark Suite - Version 1.0      *
 *                                                                           *
 *                            produced by                                    *
 *                                                                           *
 *                Mark Bull, Jim Enright and Fiona Reid                      *
 *                                                                           *
 *                                at                                         *
 *                                                                           *
 *                Edinburgh Parallel Computing Centre                        *
 *                                                                           *
 *   email: markb@epcc.ed.ac.uk, fiona@epcc.ed.ac.uk                         *
 *                                                                           *
 *                                                                           *
 *              Copyright 2012, The University of Edinburgh                  *
 *                                                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 ****************************************************************************/

===============
 Licence
===============

This software is released under the licence in Licence.txt

===============
 Introduction
===============
This document describes how to install and run the
OpenMP/MPI mixed-mode microbenchmark suite.
It also gives a description of the source files and
modules of the code.

For more information see:
http://www2.epcc.ed.ac.uk/~markb/mpiopenmpbench/intro.html
**** Link needs to be updated to point to new website when it is available ****

======================================================
Compiling
======================================================

A makefile is supplied with the code containing
build rules for each source file.

To compile for a particular system set the FC and
FFLAGS variables in the makefile.

Set CC to the Fortran compiler you want to use.

Add compiler flags you want to use to the CFLAGS
variable.
These should include flags to process OpenMP directive
and MPI calls and also some standard optimisations.

e.g.
FC=	mpicc
FFLAGS=	-mp -O3

The code can then be compiled using the "make"
command.


======================================================
Running
======================================================

An executable, mixedModeBenchmark, is produced after
successful compilation.

The executable takes an input file as an argument:
e.g. mpiexec -np 4 ./mixedModeBenchmark inputFile

This contains benchmark setup information (min/max data
size and benchmark target time) and a list
of the benchmarks to be run.

A sample input file is shown below:

1 # min data size
4194304 # max data size
1.00 # target time
masteronlyPingpong
0 1
funnelledPingpong
3 1
multiplePingpong
-1 0
funnelledHaloExchange
alltoall
gather
allReduce
barrier
scatter

For all pingpong and pingpong benchmarks the ranks of
participating MPI processes must be specified in the
line following the benchmark name.
The number may be negative, in which case it is the
offset from the last MPI process.
An offest of -1 is the last rank, numMPIprocs -1.
