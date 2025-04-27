/* This file is part of acg.
 *
 * Copyright 2025 Koç University and Simula Research Laboratory
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *  James D. Trotter <james@simula.no>
 *  Sinan Ekmekçibaşı <sekmekcibasi23@ku.edu.tr>
 *
 * Last modified: 2025-04-26
 *
 * Example application for multi-GPU solvers using the conjugate
 * gradient (CG) method.
 *
 */

#define _GNU_SOURCE

#include "acg/config.h"
#include "acg/cghip.h"
#include "acg/cgpetsc.h"
#include "acg/comm.h"
#include "acg/error.h"
#include "acg/fmtspec.h"
#include "acg/graph.h"
#include "acg/halo.h"
#include "acg/mtxfile.h"
#include "acg/rocshmem.h"
#include "acg/symcsrmatrix.h"
#include "acg/time.h"
#include "acg/vector.h"

#ifdef ACG_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef ACG_HAVE_PETSC
#include <petsc.h>
#endif
#ifdef ACG_HAVE_MPI
#include <mpi.h>
#endif
#ifdef ACG_HAVE_METIS
#include <metis.h>
#endif

#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

#ifdef ACG_HAVE_RCCL
#include <rccl/rccl.h>
#endif

#include <float.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <sched.h>

const char * program_name = "acg-hip";
const char * program_version = "0.9.4";
const char * program_copyright =
    "Copyright (C) 2025 Simula Research Laboratory, Koç University";
const char * program_license =
    "Copyright 2025 Koç University and Simula Research Laboratory\n"
    "\n"
    "Permission is hereby granted, free of charge, to any person\n"
    "obtaining a copy of this software and associated documentation\n"
    "files (the “Software”), to deal in the Software without\n"
    "restriction, including without limitation the rights to use, copy,\n"
    "modify, merge, publish, distribute, sublicense, and/or sell copies\n"
    "of the Software, and to permit persons to whom the Software is\n"
    "furnished to do so, subject to the following conditions:\n"
    "\n"
    "The above copyright notice and this permission notice shall be\n"
    "included in all copies or substantial portions of the Software.\n"
    "\n"
    "THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,\n"
    "EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\n"
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND\n"
    "NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS\n"
    "BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN\n"
    "ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN\n"
    "CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
    "SOFTWARE.\n";
#ifndef _GNU_SOURCE
const char * program_invocation_name;
const char * program_invocation_short_name;
#endif

/*
 * solver types
 */

/**
 * ‘acgsolvertype’ is a type for enumerating different linear solvers.
 */
enum acgsolvertype
{
    acgsolver_acg,             /* native CG */
    acgsolver_acg_pipelined,   /* native CG (pipelined) */
    acgsolver_acg_device,      /* native device-side CG */
    acgsolver_petsc,           /* PETSc CG */
    acgsolver_petsc_pipelined, /* PETSc CG (pipelined) */
};

/**
 * ‘acgsolvertypestr()’ returns a string for a solverunicator type.
 */
const char * acgsolvertypestr(enum acgsolvertype solvertype)
{
    if (solvertype == acgsolver_acg) { return "acg"; }
    else if (solvertype == acgsolver_acg_pipelined) { return "acg-pipelined"; }
    else if (solvertype == acgsolver_acg_device) { return "acg-device"; }
    else if (solvertype == acgsolver_petsc) { return "petsc"; }
    else if (solvertype == acgsolver_petsc_pipelined) { return "petsc-pipelined"; }
    else { return "unknown"; }
}

/*
 * parsing numbers
 */

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    const char * s,
    char ** outendptr,
    int base,
    long long int * out_number,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *out_number = number;
    return 0;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int(
    int * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int32_t()’ parses a string to produce a number that may be
 * represented as a signed, 32-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int32_t(
    int32_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT32_MIN || y > INT32_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_int64_t()’ parses a string to produce a number that may be
 * represented as a signed, 64-bit integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a signed integer, ‘ERANGE’ is returned.
 */
int parse_int64_t(
    int64_t * x,
    const char * s,
    char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(s, endptr, 10, &y, bytes_read);
    if (err) return err;
    if (y < INT64_MIN || y > INT64_MAX) return ERANGE;
    *x = y;
    return 0;
}

/**
 * ‘parse_double()’ parses a string to produce a number that may be
 * represented as ‘double’.
 *
 * The number is parsed using ‘strtod()’, following the conventions
 * documented in the man page for that function.  In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly.  The parsed number is stored in ‘number’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned. If the resulting number
 * cannot be represented as a double, ‘ERANGE’ is returned.
 */
int parse_double(
    double * x,
    const char * s,
    char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    *x = strtod(s, &endptr);
    if ((errno == ERANGE && (*x == HUGE_VAL || *x == -HUGE_VAL)) ||
        (errno != 0 && x == 0)) { return errno; }
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    return 0;
}

#ifndef ACG_IDX_SIZE
#define parse_acgidx_t parse_int
#elif ACG_IDX_SIZE == 32
#define parse_acgidx_t parse_int32_t
#elif ACG_IDX_SIZE == 64
#define parse_acgidx_t parse_int64_t
#endif

/*
 * program options and help text
 */

/**
 * ‘program_options_print_usage()’ prints a usage text.
 */
static void program_options_print_usage(
    FILE * f)
{
    fprintf(f, "Usage: %s [OPTION..] A [b] [x0]\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Solve a linear system of equations ‘Ax=b’ using the conjugate gradient (CG)\n");
    fprintf(f, " method for a matrix ‘A’ and right-hand side vector ‘b’.\n");
    fprintf(f, "\n");
    fprintf(f, " Positional arguments:\n");
    fprintf(f, "  A    path to Matrix Market file for a matrix A\n");
    fprintf(f, "  b    optional path to Matrix Market file for a right-hand side vector b\n");
    fprintf(f, "  x0   optional path to Matrix Market file for an initial guess x0\n");
#ifdef ACG_HAVE_LIBZ
    fprintf(f, "\n");
    fprintf(f, " Input options:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    fprintf(f, "  --binary              read Matrix Market files in binary format\n");
    fprintf(f, "\n");
    fprintf(f, " Partitioning options:\n");
    fprintf(f, "  --partition=FILE      read partition vector from Matrix Market file.\n");
    fprintf(f, "  --binary-partition    read partition vector in binary format\n");
    fprintf(f, "  --seed=N              random number seed. [0]\n");
    fprintf(f, "\n");
    fprintf(f, " Solver options:\n");
    fprintf(f, "  --solver TYPE         acg, acg-pipelined, acg-device or petsc. [acg]\n");
    fprintf(f, "  --max-iterations N    maximum number of iterations. [100]\n");
    fprintf(f, "  --diff-atol TOL       stopping criterion for difference in solution iterates, ‖xₖ₊₁-xₖ‖ < TOL. [0]\n");
    fprintf(f, "  --diff-rtol TOL       stopping criterion for relative difference in solution iterates, ‖xₖ₊₁-xₖ‖/‖x₀‖ < TOL. [0]\n");
    fprintf(f, "  --residual-atol TOL   stopping criterion for residual norm, ‖b-Ax‖ < TOL. [0]\n");
    fprintf(f, "  --residual-rtol TOL   stopping criterion for relative residual norm, ‖b-Ax‖/‖b‖ < TOL. [1e-9]\n");
    fprintf(f, "  --epsilon TOL         add TOL to the diagonal of A. [0]\n");
    fprintf(f, "  --warmup N            perform N warmup iterations. [10]\n");
    fprintf(f, "\n");
    fprintf(f, " Communication library options:\n");
    fprintf(f, "  --comm TYPE           none, mpi, rccl or rocshmem. [mpi]\n");
    fprintf(f, "\n");
    fprintf(f, " Solver verification options:\n");
    fprintf(f, "  --manufactured-solution  Use a manufactured solution and right-hand side.\n");
    fprintf(f, "\n");
    fprintf(f, " Output options:\n");
    /* fprintf(f, "  --repeat=N           repeat solver N times\n"); */
    fprintf(f, "  --numfmt FMT         Format string for outputting numerical values.\n");
    fprintf(f, "                       The format specifiers '%%e', '%%E', '%%f', '%%F',\n");
    fprintf(f, "                       '%%g' or '%%G' may be used. Flags, field width and\n");
    fprintf(f, "                       precision may also be specified, e.g., \"%%+3.1f\".\n");
    fprintf(f, "  --output-comm-matrix print communication matrix to standard output\n");
    fprintf(f, "\n");
    fprintf(f, "  -v, --verbose        be more verbose\n");
    fprintf(f, "  -q, --quiet          suppress output\n");
    fprintf(f, "\n");
    fprintf(f, " Other options:\n");
    fprintf(f, "  -h, --help           display this help and exit\n");
    fprintf(f, "  --version            display version information and exit\n");
    fprintf(f, "\n");
    fprintf(f, "Report bugs to: <james@simula.no>\n");
}

/**
 * ‘program_options_print_version()’ prints version information.
 */
static void program_options_print_version(
    FILE * f)
{
    fprintf(f, "%s %s\n", program_name, program_version);
    fprintf(f, "32/64-bit integers: %ld-bit\n", sizeof(acgidx_t)*CHAR_BIT);
#ifdef ACG_ENABLE_PROFILING
    fprintf(f, "profiling: enabled\n");
#else
    fprintf(f, "profiling: disabled\n");
#endif
#ifdef ACG_HAVE_MPI
    char mpistr[MPI_MAX_LIBRARY_VERSION_STRING] = ""; int len;
    MPI_Get_library_version(mpistr, &len);
    fprintf(f, "MPI: %d.%d (%s)\n", MPI_VERSION, MPI_SUBVERSION, mpistr);
#else
    fprintf(f, "MPI: no\n");
#endif
#ifdef ACG_HAVE_LIBZ
    fprintf(f, "zlib: "ZLIB_VERSION"\n");
#else
    fprintf(f, "zlib: no\n");
#endif
#ifdef ACG_HAVE_METIS
    fprintf(f, "metis: %d.%d.%d (%d-bit index, %d-bit real)\n",
            METIS_VER_MAJOR, METIS_VER_MINOR, METIS_VER_SUBMINOR,
            IDXTYPEWIDTH, REALTYPEWIDTH);
#else
    fprintf(f, "metis: no\n");
#endif
#ifdef ACG_HAVE_PETSC
    fprintf(f, "PETSc: %d.%d.%d\n", PETSC_VERSION_MAJOR, PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR);
#else
    fprintf(f, "PETSc: no\n");
#endif
    fprintf(f, "\n");
    fprintf(f, "%s\n", program_copyright);
    fprintf(f, "%s\n", program_license);
}

/**
 * ‘program_options’ contains data to related program options.
 */
struct program_options
{
    /* input options */
    char * Apath;
    char * bpath;
    char * x0path;
    int gzip;
    int binary;
    int rowpartbinary;

    /* partitioning options */
    char * rowpartspath;
    acgidx_t seed;

    /* linear solver options */
    /* int repeat; */
    enum acgsolvertype solvertype;
    double diffatol, diffrtol;
    double residualatol, residualrtol;
    int maxits;
    double epsilon;
    int warmup;

    /* communication library options */
    enum acgcommtype commtype;

    /* solver verification options */
    int manufactured_solution;

    /* output options */
    char * numfmt;
    int output_comm_matrix;
    int verbose;
    bool quiet;

    /* other options */
    bool help;
    bool version;
};

/**
 * ‘program_options_init()’ configures the default program options.
 */
static int program_options_init(
    struct program_options * args)
{
    args->Apath = NULL;
    args->bpath = NULL;
    args->x0path = NULL;
    args->gzip = 0;
    args->binary = 0;
    args->rowpartbinary = 0;

    /* partitioning options */
    args->rowpartspath = NULL;
    args->seed = 0;

    /* solver options */
    /* args->repeat = 1; */
    args->solvertype = acgsolver_acg;
    args->diffatol = args->diffrtol = 0;
    args->residualatol = 0;
    args->residualrtol = 1e-9;
    args->maxits = 100;
    args->epsilon = 0;
    args->warmup = 10;

    /* communication library options */
    args->commtype = acgcomm_mpi;

    /* solver verification options */
    args->manufactured_solution = 0;

    /* output options */
    args->numfmt = NULL;
    args->output_comm_matrix = 0;
    args->verbose = 0;
    args->quiet = false;

    /* other options */
    args->help = false;
    args->version = false;
    return 0;
}

/**
 * ‘program_options_free()’ frees memory and other resources
 * associated with parsing program options.
 */
static void program_options_free(
    struct program_options * args)
{
    if (args->Apath) free(args->Apath);
    if (args->bpath) free(args->bpath);
    if (args->x0path) free(args->x0path);
    if (args->rowpartspath) free(args->rowpartspath);
    if (args->numfmt) free(args->numfmt);
}

/**
 * ‘parse_program_options()’ parses program options.
 */
static int parse_program_options(
    int argc,
    char ** argv,
    struct program_options * args,
    int * nargs)
{
    *nargs = 0;
    (*nargs)++; argv++;

    /* parse program options */
    int num_positional_arguments_consumed = 0;
    while (*nargs < argc) {

#ifdef ACG_HAVE_LIBZ
        if (strcmp(argv[0], "-z") == 0 ||
            strcmp(argv[0], "--gzip") == 0 ||
            strcmp(argv[0], "--gunzip") == 0 ||
            strcmp(argv[0], "--ungzip") == 0)
        {
            args->gzip = 1;
            (*nargs)++; argv++; continue;
        }
#endif
        if (strcmp(argv[0], "--binary") == 0) {
            args->binary = 1;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--binary-partition") == 0) {
            args->rowpartbinary = 1;
            (*nargs)++; argv++; continue;
        }

        /* partitioning options */
        if (strstr(argv[0], "--partition") == argv[0]) {
            int n = strlen("--partition");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            args->rowpartspath = strdup(s);
            if (!args->rowpartspath) return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--seed") == argv[0]) {
            int n = strlen("--seed");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_acgidx_t(&args->seed, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }

        /* linear solver options */
        if (strstr(argv[0], "--solver") == argv[0]) {
            int n = strlen("--solver");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            if (strcasecmp(s, "acg") == 0) {
                args->solvertype = acgsolver_acg;
            } else if (strcasecmp(s, "acg-pipelined") == 0) {
                args->solvertype = acgsolver_acg_pipelined;
            } else if (strcasecmp(s, "acg-device") == 0) {
                args->solvertype = acgsolver_acg_device;
            } else if (strcasecmp(s, "petsc") == 0) {
                args->solvertype = acgsolver_petsc;
            } else if (strcasecmp(s, "petsc-pipelined") == 0) {
                args->solvertype = acgsolver_petsc_pipelined;
            } else { return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--diff-atol") == argv[0]) {
            int n = strlen("--diff-atol");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_double(&args->diffatol, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--diff-rtol") == argv[0]) {
            int n = strlen("--diff-rtol");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_double(&args->diffrtol, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--residual-atol") == argv[0]) {
            int n = strlen("--residual-atol");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_double(&args->residualatol, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--residual-rtol") == argv[0]) {
            int n = strlen("--residual-rtol");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_double(&args->residualrtol, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--max-iterations") == argv[0]) {
            int n = strlen("--max-iterations");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_int(&args->maxits, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        /* if (strstr(argv[0], "--repeat") == argv[0]) { */
        /*     int n = strlen("--repeat"); */
        /*     const char * s = &argv[0][n]; */
        /*     if (*s == '=') { s++; } */
        /*     else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; } */
        /*     else { return EINVAL; } */
        /*     char * endptr; */
        /*     if (parse_int(&args->repeat, s, &endptr, NULL)) return EINVAL; */
        /*     if (*endptr != '\0') return EINVAL; */
        /*     (*nargs)++; argv++; continue; */
        /* } */
        if (strstr(argv[0], "--epsilon") == argv[0]) {
            int n = strlen("--epsilon");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_double(&args->epsilon, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }
        if (strstr(argv[0], "--warmup") == argv[0]) {
            int n = strlen("--warmup");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_int(&args->warmup, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
            (*nargs)++; argv++; continue;
        }

        /* communication library options */
        if (strstr(argv[0], "--comm") == argv[0]) {
            int n = strlen("--comm");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            if (strcasecmp(s, "none") == 0) {
                args->commtype = acgcomm_null;
            } else if (strcasecmp(s, "mpi") == 0) {
                args->commtype = acgcomm_mpi;
            } else if (strcasecmp(s, "rccl") == 0) {
                args->commtype = acgcomm_rccl;
            } else if (strcasecmp(s, "rocshmem") == 0) {
                args->commtype = acgcomm_rocshmem;
            } else { return EINVAL; }
            (*nargs)++; argv++; continue;
        }

        /* solver verification options */
        if (strcmp(argv[0], "--manufactured-solution") == 0) {
            args->manufactured_solution = 1;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--no-manufactured-solution") == 0) {
            args->manufactured_solution = 0;
            (*nargs)++; argv++; continue;
        }

        /* output options */
        if (strstr(argv[0], "--numfmt") == argv[0]) {
            int n = strlen("--numfmt");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            struct fmtspec spec;
            if (fmtspec_parse(&spec, s, NULL)) { return EINVAL; }
            args->numfmt = strdup(s);
            if (!args->numfmt) { free(args->numfmt); return EINVAL; }
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--output-comm-matrix") == 0) {
            args->output_comm_matrix = 1;
            (*nargs)++; argv++; continue;
        } else if (strcmp(argv[0], "--no-output-comm-matrix") == 0) {
            args->output_comm_matrix = 0;
            (*nargs)++; argv++; continue;
        }

        /* other options */
        if (strcmp(argv[0], "-q") == 0 || strcmp(argv[0], "--quiet") == 0) {
            args->quiet = true;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "-v") == 0 || strcmp(argv[0], "--verbose") == 0) {
            args->verbose++;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "-h") == 0 || strcmp(argv[0], "--help") == 0) {
            args->help = true;
            (*nargs)++; argv++; return 0;
        }
        if (strcmp(argv[0], "--version") == 0) {
            args->version = true;
            (*nargs)++; argv++; return 0;
        }

        /* stop parsing options after '--'  */
        if (strcmp(argv[0], "--") == 0) {
            (*nargs)++; argv++;
            break;
        }

        /* unrecognised option */
        if (strlen(argv[0]) > 1 && argv[0][0] == '-' &&
            ((argv[0][1] < '0' || argv[0][1] > '9') && argv[0][1] != '.'))
            return EINVAL;

        /*
         * positional arguments
         */
        if (num_positional_arguments_consumed == 0) {
            args->Apath = strdup(argv[0]);
            if (!args->Apath) return errno;
        } else if (num_positional_arguments_consumed == 1) {
            args->bpath = strdup(argv[0]);
            if (!args->bpath) return errno;
        } else if (num_positional_arguments_consumed == 2) {
            args->x0path = strdup(argv[0]);
            if (!args->x0path) return errno;
        } else { return EINVAL; }
        num_positional_arguments_consumed++;
        (*nargs)++; argv++;
    }
    return 0;
}

/*
 * Matrix Market output
 */

/**
 * ‘printf_mtxfilecomment()’ formats comment lines using a printf-like
 * syntax.
 *
 * Note that because ‘fmt’ is a printf-style format string, where '%'
 * is used to denote a format specifier, then ‘fmt’ must begin with
 * "%%" to produce the initial '%' character that is required for a
 * comment line. The ‘fmt’ string must also end with a newline
 * character, '\n'.
 *
 * The caller must call ‘free’ with the returned pointer to free the
 * allocated storage.
 */
static char * printf_mtxfilecomment(const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    int len = vsnprintf(NULL, 0, fmt, va);
    va_end(va);
    if (len < 0) return NULL;

    char * s = (char *) malloc(len+1);
    if (!s) return NULL;

    va_start(va, fmt);
    int newlen = vsnprintf(s, len+1, fmt, va);
    va_end(va);
    if (newlen < 0 || len != newlen) { free(s); return NULL; }
    s[newlen] = '\0';
    return s;
}

/*
 * main
 */

int main(int argc, char *argv[])
{
#ifndef _GNU_SOURCE
    /* set program invocation name */
    program_invocation_name = argv[0];
    program_invocation_short_name = (
        strrchr(program_invocation_name, '/')
        ? strrchr(program_invocation_name, '/') + 1
        : program_invocation_name);
#endif

    int err = ACG_SUCCESS, errcode = 0;
    bool errexit = false;
    acgtime_t t0, t1;

    /* 1a) initialise MPI */
    const MPI_Comm mpicomm = MPI_COMM_WORLD;
    int commsize, rank;
    const int root = 0;
    int mpierrcode = 0;
    char mpierrstr[MPI_MAX_ERROR_STRING];
    int mpierrstrlen;
    int threadlevel;
    mpierrcode = MPI_Init_thread(
        &argc, &argv, MPI_THREAD_FUNNELED, &threadlevel);
    if (mpierrcode) {
        MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Init_thread failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(mpicomm, EXIT_FAILURE);
    }
    mpierrcode = MPI_Comm_size(mpicomm, &commsize);
    if (mpierrcode) {
        MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_size failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(mpicomm, EXIT_FAILURE);
    }
    mpierrcode = MPI_Comm_rank(mpicomm, &rank);
    if (mpierrcode) {
        MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: MPI_Comm_rank failed with %s\n",
                program_invocation_short_name, mpierrstr);
        MPI_Abort(mpicomm, EXIT_FAILURE);
    }
    if (rank == root) {
        char mpiversionstr[MPI_MAX_LIBRARY_VERSION_STRING];
        int len;
        mpierrcode = MPI_Get_library_version(mpiversionstr, &len);
        if (mpierrcode) {
            MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
            fprintf(stderr, "%s: MPI_Query_thread failed with %s\n",
                    program_invocation_short_name, mpierrstr);
            MPI_Abort(mpicomm, EXIT_FAILURE);
        }
        int threadlevel;
        mpierrcode = MPI_Query_thread(&threadlevel);
        if (mpierrcode) {
            MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
            fprintf(stderr, "%s: MPI_Query_thread failed with %s\n",
                    program_invocation_short_name, mpierrstr);
            MPI_Abort(mpicomm, EXIT_FAILURE);
        }
        fprintf(stderr, "MPI version %s (thread level: ", mpiversionstr);
        if (threadlevel == MPI_THREAD_SINGLE) fprintf(stderr, "MPI_THREAD_SINGLE");
        else if (threadlevel == MPI_THREAD_FUNNELED) fprintf(stderr, "MPI_THREAD_FUNNELED");
        else if (threadlevel == MPI_THREAD_SERIALIZED) fprintf(stderr, "MPI_THREAD_SERIALIZED");
        else if (threadlevel == MPI_THREAD_MULTIPLE) fprintf(stderr, "MPI_THREAD_MULTIPLE");
        else fprintf(stderr, "unknown");
        fprintf(stderr, ")\n");
    }

    /* 1b) parse program options */
    struct program_options args;
    err = program_options_init(&args);
    errexit = err;
    MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (errexit) {
        if (err) fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    errexit = err;
    MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (err) {
        if (rank == root) {
            fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                    strerror(err), argv[nargs]);
        }
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    MPI_Allreduce(MPI_IN_PLACE, &args.help, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (args.help) {
        if (rank == root) program_options_print_help(stdout);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    MPI_Allreduce(MPI_IN_PLACE, &args.version, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (args.version) {
        if (rank == root) program_options_print_version(stdout);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_SUCCESS;
    }
    errexit = !args.Apath;
    MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (errexit) {
        if (rank == root) program_options_print_usage(stdout);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    errexit = args.bpath && args.manufactured_solution;
    MPI_Allreduce(MPI_IN_PLACE, &errexit, 1, MPI_C_BOOL, MPI_LOR, mpicomm);
    if (errexit) {
        if (rank == root) program_options_print_usage(stdout);
        program_options_free(&args);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const char * Apath = args.Apath;
    const char * bpath = args.bpath;
    int quiet = args.quiet;
    int verbose = args.verbose;
    const char * numfmt = args.numfmt;
    acgidx_t seed = args.seed;
    double diffatol = args.diffatol;
    double diffrtol = args.diffrtol;
    double residualatol = args.residualatol;
    double residualrtol = args.residualrtol;
    int maxits = args.maxits;
    int output_comm_matrix = args.output_comm_matrix;
    int use_rccl = args.commtype == acgcomm_rccl;
    int use_rocshmem = args.commtype == acgcomm_rocshmem;
    int use_petsc = (args.solvertype == acgsolver_petsc || args.solvertype == acgsolver_petsc_pipelined);

    /* select a HIP device */
    int sharedrank = -1;
    int sharedcommsize = 1;
    MPI_Comm sharedcomm;
    err = MPI_Comm_split_type(
        mpicomm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &sharedcomm);
    if (err) {
        char mpierrstr[MPI_MAX_ERROR_STRING];
        int mpierrstrlen = MPI_MAX_ERROR_STRING;
        MPI_Error_string(mpierrcode, mpierrstr, &mpierrstrlen);
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, mpierrstr);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(sharedcomm, &sharedrank);
    MPI_Comm_size(sharedcomm, &sharedcommsize);
    MPI_Comm_free(&sharedcomm);
    int ndevices;
    err = hipGetDeviceCount(&ndevices);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, hipGetErrorString(err));
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    int device = sharedrank % ndevices;
    err = hipSetDevice(device);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, hipGetErrorString(err));
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    err = sched_getaffinity(0, sizeof(mask), &mask);
    int cpuslen = 0;
    for (int i = 0, n = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) { cpuslen += snprintf(NULL, 0, n==0 ? "%d" : ",%d", i); n++; }
    }
    char * cpus = malloc(cpuslen+1);
    memset(cpus, 0, cpuslen);
    for (int i = 0, j = 0, n = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) { j += snprintf(&cpus[j], cpuslen-j+1, n==0 ? "%d" : ",%d", i); n++; }
    }

    if (rank == root) {
        fprintf(stderr, "%'d MPI processes\n", commsize);
#ifdef _OPENMP
        #pragma omp parallel
        if (args.verbose > 0 && rank == root) {
            #pragma omp master
            fprintf(stderr, "%'d OpenMP threads\n", omp_get_num_threads());
        }
#else
        fprintf(stderr, "OpenMP is disabled\n", omp_get_num_threads());
#endif
        fprintf(stderr, "Mapping of MPI processes to CPU cores and HIP devices:\n");
        for (int p = 0; p < commsize; p++) {
            if (rank == p) {
                int len = 0;
                char processorname[MPI_MAX_PROCESSOR_NAME+1];
                MPI_Get_processor_name(processorname, &len);
                processorname[MPI_MAX_PROCESSOR_NAME] = '\0';
                fprintf(stderr, " rank %'d -> CPU cores %s and device %'d on %s\n", rank, cpus, device, processorname);
            } else {
                int cpuslen;
                MPI_Recv(&cpuslen, 1, MPI_INT, p, 0, mpicomm, MPI_STATUS_IGNORE);
                char * cpus = malloc(cpuslen+1);
                MPI_Recv(cpus, cpuslen, MPI_CHAR, p, 0, mpicomm, MPI_STATUS_IGNORE);
                cpus[cpuslen] = '\0';
                int device;
                MPI_Recv(&device, 1, MPI_INT, p, 0, mpicomm, MPI_STATUS_IGNORE);
                int len = 0;
                char processorname[MPI_MAX_PROCESSOR_NAME+1];
                MPI_Recv(&len, 1, MPI_INT, p, 0, mpicomm, MPI_STATUS_IGNORE);
                if (len > MPI_MAX_PROCESSOR_NAME) len = MPI_MAX_PROCESSOR_NAME;
                MPI_Recv(processorname, len, MPI_CHAR, p, 0, mpicomm, MPI_STATUS_IGNORE);
                processorname[len] = processorname[MPI_MAX_PROCESSOR_NAME] = '\0';
                fprintf(stderr, " rank %'d -> CPU cores %s and device %'d on %s\n", p, cpus, device, processorname);
                free(cpus);
            }
        }
    } else {
        MPI_Send(&cpuslen, 1, MPI_INT, root, 0, mpicomm);
        MPI_Send(cpus, cpuslen, MPI_CHAR, root, 0, mpicomm);
        MPI_Send(&device, 1, MPI_INT, root, 0, mpicomm);
        int len = 0;
        char processorname[MPI_MAX_PROCESSOR_NAME+1];
        MPI_Get_processor_name(processorname, &len);
        if (len > MPI_MAX_PROCESSOR_NAME) len = MPI_MAX_PROCESSOR_NAME;
        processorname[len] = processorname[MPI_MAX_PROCESSOR_NAME] = '\0';
        MPI_Send(&len, 1, MPI_INT, root, 0, mpicomm);
        MPI_Send(processorname, len, MPI_CHAR, root, 0, mpicomm);
    }
    free(cpus);

    /* initialise RCCL */
#ifdef ACG_HAVE_RCCL
    if (rank == root) {
        int ncclversion = 0;
        ncclGetVersion(&ncclversion);
        fprintf(stderr, "RCCL version %d\n", ncclversion);
    }
    ncclComm_t ncclcomm;
    if (use_rccl) {
        ncclUniqueId nccluid;
        if (rank == root) ncclGetUniqueId(&nccluid);
        MPI_Bcast(&nccluid, sizeof(nccluid), MPI_BYTE, root, mpicomm);
        err = ncclCommInitRank(&ncclcomm, commsize, nccluid, rank);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, ncclGetErrorString(err));
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }
#endif

    /* initialise ROCSHMEM */
#if defined(ACG_HAVE_ROCSHMEM)
    if (rank == root) {
        int major, minor, patch;
        acg_rocshmemx_vendor_get_version_info(&major, &minor, &patch);
        fprintf(stderr, "ROCSHMEM version %d.%d.%d\n", major, minor, patch);
    }
    if (use_rocshmem) {
        int rocshmemerrcode;
        err = acgcomm_rocshmem_init(mpicomm, root, &rocshmemerrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err,rocshmemerrcode));
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }
#endif

    if (rank == root) {
        if (args.solvertype == acgsolver_acg) {
            fprintf(stderr, "using ACG solver\n");
        } else if (args.solvertype == acgsolver_acg_pipelined) {
            fprintf(stderr, "using aCG solver (pipelined)\n");
        } else if (args.solvertype == acgsolver_acg_device) {
            fprintf(stderr, "using aCG solver (device-side)\n");
        } else if (args.solvertype == acgsolver_petsc) {
            fprintf(stderr, "using PETSc solver\n");
        } else if (args.solvertype == acgsolver_petsc_pipelined) {
            fprintf(stderr, "using PETSc solver (pipelined)\n");
        } else {
            fprintf(stderr, "invalid solver type\n");
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }

    /* create communicator for acg solver */
    struct acgcomm comm;
    if (use_rccl) {
#if defined(ACG_HAVE_RCCL)
        if (rank == root) fprintf(stderr, "Using RCCL for communication\n");
        int rcclerrcode;
        err = acgcomm_init_rccl(&comm, ncclcomm, &rcclerrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err,rcclerrcode));
            ncclCommDestroy(ncclcomm);
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
#else
        if (rank == root) fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(ACG_ERR_RCCL_NOT_SUPPORTED,0));
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
#endif
    } else if (use_rocshmem) {
#if defined(ACG_HAVE_ROCSHMEM)
        if (rank == root) fprintf(stderr, "Using ROCSHMEM for communication\n");
        int rocshmemerrcode;
        err = acgcomm_init_rocshmem(&comm, mpicomm, &rocshmemerrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err,rocshmemerrcode));
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
#else
        if (rank == root) fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(ACG_ERR_ROCSHMEM_NOT_SUPPORTED,0));
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
#endif
    } else {
        if (rank == root) fprintf(stderr, "Using MPI for communication\n");
        err = acgcomm_init_mpi(&comm, mpicomm, &mpierrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err,mpierrcode));
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }

    /* initialise hipblas and hipsparse */
    hipblasHandle_t hipblas;
    err = hipblasCreate(&hipblas);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, errcode));
        acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
        if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
    }
    hipsparseHandle_t hipsparse;
    err = hipsparseCreate(&hipsparse);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, errcode));
        hipblasDestroy(&hipblas);
        hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
        acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
        if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
    }

    /* initialise PETSc */
#ifdef ACG_HAVE_PETSC
    if (use_petsc) {
        int petscargc = argc;
        char ** petscargv = argv;
        int petscerrcode = 0;
        const char * petscerrstr;
        char * petscerrspecific;
        PetscOptionsSetValue(NULL, "-no_signal_handler", "true");
        petscerrcode = PetscInitialize(&petscargc, &petscargv, NULL, NULL);
        if (petscerrcode) {
            PetscErrorMessage(petscerrcode, &petscerrstr, &petscerrspecific);
            if (rank == root) {
                fprintf(stderr, "%s: PETSCInitialize: %s (%s)\n",
                        program_invocation_short_name, petscerrstr, petscerrspecific);
            }
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        petscerrcode = PetscDeviceInitialize(PETSC_DEVICE_HIP);
        if (petscerrcode) {
            PetscErrorMessage(petscerrcode, &petscerrstr, &petscerrspecific);
            if (rank == root) {
                fprintf(stderr, "%s: PETSCInitialize: %s (%s)\n",
                        program_invocation_short_name, petscerrstr, petscerrspecific);
            }
            PetscFinalize();
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }
#endif

    if (verbose > 0) {
        if (rank == root) fprintf(stderr, "reading matrix: ");
        MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
    }

    /* 1. read matrix on the root process */
    struct acgmtxfile mtxfile;
    int64_t lines_read = 0, bytes_read = 0;
    if (rank == root) {
        int idxbase = 0;
        enum mtxlayout layout = mtxrowmajor;
        err = acgmtxfile_read(
            &mtxfile, layout, args.binary, idxbase, mtxdouble,
            Apath, args.gzip, &lines_read, &bytes_read);
        if (err) {
            if (lines_read < 0) {
                fprintf(stderr, "%s: %s: %s\n",
                        program_invocation_short_name,
                        Apath, acgerrcodestr(err, 0));
            } else {
                fprintf(stderr, "%s: %s:%" PRId64 ": %s\n",
                        program_invocation_short_name,
                        Apath, lines_read+1, acgerrcodestr(err, 0));
            }
            errexit = true;
            MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        if (mtxfile.object != mtxmatrix) {
            fprintf(stderr, "%s: %s: expected matrix; object is %s\n",
                    program_invocation_short_name, Apath, mtxobjectstr(mtxfile.object));
            errexit = true;
            MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        if (mtxfile.format != mtxcoordinate) {
            fprintf(stderr, "%s: %s: expected coordinate; format is %s\n",
                    program_invocation_short_name, Apath, mtxformatstr(mtxfile.format));
            errexit = true;
            MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        if (mtxfile.symmetry != mtxsymmetric) {
            fprintf(stderr, "%s: %s: expected symmetric; symmetry is %s\n",
                    program_invocation_short_name, Apath, mtxsymmetrystr(mtxfile.symmetry));
            errexit = true;
            MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        errexit = false;
        MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
    } else {
        MPI_Bcast(&errexit, 1, MPI_C_BOOL, root, mpicomm);
        if (errexit) {
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    }

    if (verbose > 0) {
        MPI_Barrier(mpicomm); gettime(&t1);
        if (rank == root) {
            int64_t mtxsz =
                (mtxfile.rowidx ? mtxfile.nnzs*sizeof(mtxfile.rowidx) : 0)
                + (mtxfile.colidx ? mtxfile.nnzs*sizeof(mtxfile.colidx) : 0)
                + (mtxfile.data && mtxfile.datatype == mtxint ? mtxfile.nnzs*sizeof(int) : 0)
                + (mtxfile.data && mtxfile.datatype == mtxdouble ? mtxfile.nnzs*sizeof(double) : 0);
            fprintf(stderr, "%'.6f seconds (%'.1f MB/s, %'.1f MiB, %"PRIdx" rows, %"PRId64" nonzeros)\n",
                    elapsed(t0,t1), 1.0e-6*bytes_read/elapsed(t0,t1),
                    (double) mtxsz/1024.0/1024.0,
                    mtxfile.nrows, mtxfile.nnzs);
        }
    }

    /* 2. partition the matrix on the root process */
    struct acgsymcsrmatrix Aroot;
    int nparts = commsize;
    struct acgsymcsrmatrix * Ap = NULL;
    if (rank == root) {

        if (verbose > 0) {
            fprintf(stderr, "converting to symcsrmatrix: ");
            gettime(&t0);
        }

        /* 2a) initialise matrix */
        acgidx_t N = mtxfile.nrows;
        int64_t nnzs = mtxfile.nnzs;
        int idxbase = mtxfile.idxbase;
        const acgidx_t * rowidx = mtxfile.rowidx;
        const acgidx_t * colidx = mtxfile.colidx;
        const double * a = (const double *) mtxfile.data;
        int err = acgsymcsrmatrix_init_real_double(
            &Aroot, N, nnzs, idxbase, rowidx, colidx, a);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        acgmtxfile_free(&mtxfile);

        if (verbose > 0) {
            gettime(&t1);
            int64_t rownnzsmin = INT64_MAX, rownnzsmax = 0;
            for (acgidx_t i = 0; i < Aroot.nprows; i++) {
                rownnzsmin = rownnzsmin < Aroot.rownnzs[i] ? rownnzsmin : Aroot.rownnzs[i];
                rownnzsmax = rownnzsmax > Aroot.rownnzs[i] ? rownnzsmax : Aroot.rownnzs[i];
            }
            int64_t sz =
                (Aroot.nzrows ? Aroot.nprows*sizeof(Aroot.nzrows) : 0)
                + (Aroot.rownnzs ? Aroot.nprows*sizeof(Aroot.rownnzs) : 0)
                + (Aroot.rowptr ? (Aroot.nprows+1)*sizeof(Aroot.rowptr) : 0)
                + (Aroot.rowidx ? Aroot.npnzs*sizeof(Aroot.rowidx) : 0)
                + (Aroot.colidx ? Aroot.npnzs*sizeof(Aroot.colidx) : 0)
                + (Aroot.a ? Aroot.npnzs*sizeof(Aroot.a) : 0)
                + (Aroot.frowptr ? (Aroot.nprows+1)*sizeof(Aroot.frowptr) : 0)
                + (Aroot.fcolidx ? Aroot.fnpnzs*sizeof(Aroot.fcolidx) : 0)
                + (Aroot.fa ? Aroot.fnpnzs*sizeof(Aroot.fa) : 0);
            fprintf(stderr, "%'.6f seconds (%'.1f MiB, %"PRId64" to %"PRId64" (upper triangular) nonzeros per row)\n",
                    elapsed(t0,t1), (double) sz/1024.0/1024.0,
                    rownnzsmin, rownnzsmax);
        }

        /* 2b) partition matrix rows, or read a partitioning from file */
        const char * rowpartspath = args.rowpartspath;
        int * rowparts = NULL;
        if (!rowpartspath && nparts > 1) {

            if (verbose > 0) {
                fprintf(stderr, "partitioning matrix rows into %'d parts:", nparts);
                if (verbose > 1) fprintf(stderr, "\n");
                gettime(&t0);
            }

            enum metis_partitioner partitioner = metis_partgraphrecursive;
            rowparts = malloc(N*sizeof(*rowparts));
            if (!rowparts) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            acgidx_t objval;
            err = acgsymcsrmatrix_partition_rows(
                &Aroot, nparts, partitioner, rowparts, &objval, seed, verbose > 0 ? verbose-1 : 0);
            if (err) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }

            if (verbose > 0) {
                gettime(&t1);
                if (verbose > 1) fprintf(stderr, "done in %'.6f seconds (objective value: %"PRIdx")\n", elapsed(t0,t1), objval);
                else fprintf(stderr, " %'.6f seconds (objective value: %"PRIdx")\n", elapsed(t0,t1), objval);
            }

        } else if (rowpartspath) {

            if (verbose > 0) {
                fprintf(stderr, "reading row partitioning from file: ");
                gettime(&t0);
            }

            /* read the row partition vector from a file */
            acgidx_t N = mtxfile.nrows;
            int idxbase = 0;
            enum mtxlayout layout = mtxrowmajor;
            int64_t lines_read = 0, bytes_read = 0;
            struct acgmtxfile mtxfile;
            err = acgmtxfile_read(
                &mtxfile, layout, args.rowpartbinary, idxbase, mtxint,
                rowpartspath, args.gzip, &lines_read, &bytes_read);
            if (err) {
                if (verbose > 0) fprintf(stderr, "\n");
                if (lines_read < 0) {
                    fprintf(stderr, "%s: %s: %s\n",
                            program_invocation_short_name,
                            rowpartspath, acgerrcodestr(err, 0));
                } else {
                    fprintf(stderr, "%s: %s:%" PRId64 ": %s\n",
                            program_invocation_short_name,
                            rowpartspath, lines_read+1, acgerrcodestr(err, 0));
                }
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }

            if (!(mtxfile.object == mtxvector || (mtxfile.object == mtxmatrix && mtxfile.ncols == 1))) {
                if (verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: expected vector; object is %s\n",
                        program_invocation_short_name, rowpartspath, mtxobjectstr(mtxfile.object));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.format != mtxarray) {
                if (verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: expected array; format is %s\n",
                        program_invocation_short_name, rowpartspath, mtxformatstr(mtxfile.format));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.symmetry != mtxgeneral) {
                if (verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: expected general; symmetry is %s\n",
                        program_invocation_short_name, rowpartspath, mtxsymmetrystr(mtxfile.symmetry));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.nrows != N) {
                if (verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s: expected %" PRIdx " rows; number of rows %" PRIdx "\n",
                        program_invocation_short_name, rowpartspath, N, mtxfile.nrows);
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            rowparts = mtxfile.data;
            mtxfile.data = NULL;
            acgmtxfile_free(&mtxfile);
            for (int64_t i = 0; i < mtxfile.nnzs; i++) rowparts[i] = rowparts[i]-1;

            if (verbose > 0) {
                gettime(&t1);
                fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
            }
        }

        if (nparts > 1) {
            if (verbose > 0) {
                fprintf(stderr, "partitioning matrix:");
                if (verbose > 1) fputc('\n', stderr);
                gettime(&t0);
            }

            /* 2c) partition into submatrices */
            Ap = malloc(nparts*sizeof(*Ap));
            err = acgsymcsrmatrix_partition(&Aroot, nparts, rowparts, Ap, verbose > 0 ? verbose-1 : 0);
            if (err) {
                if (verbose > 0) fprintf(stderr, "\n");
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            acgsymcsrmatrix_free(&Aroot);
            free(rowparts);

            if (verbose > 0) {
                gettime(&t1);
                if (verbose > 1) fprintf(stderr, "done partitioning matrix in");
                fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
            }
        } else { Ap = &Aroot; }
    }

    /* 2d) if requested, output the communication matrix */
    if (output_comm_matrix) {
        acgtime_t t0, t1;
        int64_t bytes_written = 0;
        if (verbose > 0 && rank == root) {
            fprintf(stderr, "writing communication matrix to standard output: ");
            gettime(&t0);
        }

        if (rank == root) {
            char * comment = printf_mtxfilecomment(
                "%% this file was generated by %s %s\n",
                program_name, program_version);

            acgidx_t nrows = nparts;
            acgidx_t ncols = nparts;
            int64_t nnzs = 0;
            for (int p = 0; p < nparts; p++) {
                const struct acgsymcsrmatrix * A = &Ap[p];
                const struct acggraph * g = A->graph;
                nnzs += g->nneighbours;
            }
            int idxbase = 0;
            acgidx_t * rowidx = malloc(nnzs*sizeof(*rowidx));
            acgidx_t * colidx = malloc(nnzs*sizeof(*colidx));
            double * sendcounts = malloc(nnzs*sizeof(*sendcounts));
            acgidx_t l = 0;
            for (int p = 0; p < nparts; p++) {
                const struct acgsymcsrmatrix * A = &Ap[p];
                const struct acggraph * g = A->graph;
                for (int r = 0; r < g->nneighbours; r++, l++) {
                    struct acggraphneighbour * neighbour = &g->neighbours[r];
                    rowidx[l] = p;
                    colidx[l] = neighbour->neighbourpart;
                    sendcounts[l] = neighbour->nbordernodes;
                }
            }
            err = mtxfile_fwrite_double(
                stdout, 0, mtxmatrix, mtxcoordinate, mtxinteger, mtxgeneral, comment,
                nrows, ncols, nnzs, 1, idxbase, rowidx, colidx, sendcounts,
                NULL, &bytes_written);
            free(sendcounts); free(colidx); free(rowidx); free(comment);
            if (err) {
                if (rank == root) {
                    if (verbose > 0) fprintf(stderr, "\n");
                    fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                            acgerrcodestr(err, 0));
                }
                if (rank == root && nparts > 1) {
                    for (int p = 0; p < nparts; p++) acgsymcsrmatrix_free(&Ap[p]);
                    free(Ap);
                }
                return EXIT_FAILURE;
            }
        }

        if (verbose > 0 && rank == root) {
            gettime(&t1);
            fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }
    }

    struct acgsymcsrmatrix A;
    if (nparts > 1) {
        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "scattering submatrices: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        /* 3. scatter submatrices from root to all processes */
        err = acgsymcsrmatrix_scatter(
            Ap, 1, &A, 1, root, mpicomm, &mpierrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }
    } else { A = Aroot; }

    if (verbose > 0) {
        if (rank == root) fprintf(stderr, "setting up right-hand side: ");
        MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
    }

    /* 4. initialise right-hand side vector on the root process */
    struct acgvector broot;
    struct acgvector * bp = NULL;
    if (rank == root) {
        acgidx_t N = A.nrows;

        /* 4a) read the vector from a file, if requested */
        if (bpath) {
            int idxbase = 0;
            enum mtxlayout layout = mtxrowmajor;
            int64_t lines_read = 0, bytes_read = 0;
            struct acgmtxfile mtxfile;
            err = acgmtxfile_read(
                &mtxfile, layout, args.binary, idxbase, mtxdouble,
                bpath, args.gzip, &lines_read, &bytes_read);
            if (err) {
                if (lines_read < 0) {
                    fprintf(stderr, "%s: %s: %s\n",
                            program_invocation_short_name,
                            bpath, acgerrcodestr(err, 0));
                } else {
                    fprintf(stderr, "%s: %s:%" PRId64 ": %s\n",
                            program_invocation_short_name,
                            bpath, lines_read+1, acgerrcodestr(err, 0));
                }
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }

            if (!(mtxfile.object == mtxvector || (mtxfile.object == mtxmatrix && mtxfile.ncols == 1))) {
                fprintf(stderr, "%s: %s: expected vector; object is %s\n",
                        program_invocation_short_name, bpath, mtxobjectstr(mtxfile.object));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.format != mtxarray) {
                fprintf(stderr, "%s: %s: expected array; format is %s\n",
                        program_invocation_short_name, bpath, mtxformatstr(mtxfile.format));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.symmetry != mtxgeneral) {
                fprintf(stderr, "%s: %s: expected general; symmetry is %s\n",
                        program_invocation_short_name, bpath, mtxsymmetrystr(mtxfile.symmetry));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            if (mtxfile.nrows != N) {
                fprintf(stderr, "%s: %s: expected %" PRIdx " rows; number of rows %" PRIdx "\n",
                        program_invocation_short_name, bpath, N, mtxfile.nrows);
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            err = acgvector_init_real_double(
                &broot, N, (const double *) mtxfile.data);
            if (err) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }
            acgmtxfile_free(&mtxfile);

        } else {
            /* 4b) or, initialise the right-hand side values to one */
            err = acgvector_alloc(&broot, N);
            if (err) {
                fprintf(stderr, "%s: %s\n",
                        program_invocation_short_name,
                        acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
                if (use_petsc) PetscFinalize();
#endif
                hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                MPI_Finalize();
                hipDeviceReset();
                return EXIT_FAILURE;
            }

            if (args.manufactured_solution) {
                double bnrm2 = 0.0;
                /* generate random floating-point numbers in the range
                 * [-1,1], then normalise. */
                if (args.seed != 0) srand(args.seed);
                for (acgidx_t i = 0; i < N; i++) {
                    broot.x[i] = (2.0*(rand()/(double)RAND_MAX)-1.0);
                    bnrm2 += broot.x[i]*broot.x[i];
                }
                bnrm2 = sqrt(bnrm2);
                #pragma omp parallel for
                for (acgidx_t i = 0; i < N; i++) broot.x[i] /= bnrm2;
            } else {
                /* by default, set every value to one */
                for (acgidx_t i = 0; i < N; i++) broot.x[i] = 1.0;
            }
        }

        /* 4c) scatter right-hand side values to partitioned subvectors */
        if (nparts > 1) {
            bp = malloc(nparts*sizeof(*bp));
#ifdef ACG_HAVE_OPENMP
            #pragma omp parallel for
            for (int p = 0; p < nparts; p++) {
                err = acgsymcsrmatrix_vector(&Ap[p], &bp[p]);
                if (err) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, acgerrcodestr(err,0));
                    MPI_Abort(mpicomm, EXIT_FAILURE);
                }
                err = acgvector_usga(&bp[p], &broot);
                if (err) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, acgerrcodestr(err,0));
                    MPI_Abort(mpicomm, EXIT_FAILURE);
                }
            }
#else
            for (int p = 0; p < nparts; p++) {
                err = acgsymcsrmatrix_vector(&Ap[p], &bp[p]);
                if (err) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, acgerrcodestr(err,0));
#ifdef ACG_HAVE_PETSC
                    if (use_petsc) PetscFinalize();
#endif
                    hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                    acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                    if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                    if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                    MPI_Finalize();
                    hipDeviceReset();
                    return EXIT_FAILURE;
                }
                err = acgvector_usga(&bp[p], &broot);
                if (err) {
                    fprintf(stderr, "%s: %s\n",
                            program_invocation_short_name, acgerrcodestr(err,0));
#ifdef ACG_HAVE_PETSC
                    if (use_petsc) PetscFinalize();
#endif
                    hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
                    acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
                    if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
                    if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
                    MPI_Finalize();
                    hipDeviceReset();
                    return EXIT_FAILURE;
                }
            }
#endif
            acgvector_free(&broot);
        }
    }
    if (rank == root && nparts > 1) {
        for (int p = 0; p < nparts; p++) acgsymcsrmatrix_free(&Ap[p]);
        free(Ap);
    }

    /* 5. scatter partial right-hand side vectors from root to all processes */
    struct acgvector b;
    if (nparts > 1) {
        err = acgvector_scatter(bp, 1, &b, 1, root, mpicomm, &mpierrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
    } else { b = broot; }

    if (rank == root && nparts > 1) {
        for (int p = 0; p < nparts; p++) acgvector_free(&bp[p]);
        free(bp);
    }

    if (verbose > 0) {
        MPI_Barrier(mpicomm); gettime(&t1);
        if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
    }

    struct acgvector xsol;
    if (args.manufactured_solution) {
        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "computing right-hand side for manufactured solution: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        err = acgvector_init_copy(&xsol, &b);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        acgvector_setzero(&b);
        err = acgsymcsrmatrix_dsymvmpi(
            1.0, &A, &xsol, &b, NULL, NULL, NULL, mpicomm, 0, &mpierrcode);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, mpierrcode));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }
    }

    if (verbose > 0) {
        if (rank == root) fprintf(stderr, "setting up initial guess: ");
        MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
    }

    /* 6. allocate storage for solution vector on each process */
    struct acgvector x;
    err = acgsymcsrmatrix_vector(&A, &x);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
        if (use_petsc) PetscFinalize();
#endif
        hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
        acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
        if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
    }
    acgvector_setzero(&x);

    if (verbose > 0) {
        MPI_Barrier(mpicomm); gettime(&t1);
        if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
    }

    /* convert matrix to full storage format */
    if (verbose > 0) {
        if (rank == root) fprintf(stderr, "converting matrix to full storage format: ");
        MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
    }
    err = acgsymcsrmatrix_dsymv_init(&A, args.epsilon);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
        if (use_petsc) PetscFinalize();
#endif
        hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
        acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
        if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
        MPI_Finalize();
        hipDeviceReset();
        return EXIT_FAILURE;
    }
    if (verbose > 0) {
        MPI_Barrier(mpicomm); gettime(&t1);
        int64_t prownnzsmin = INT64_MAX, prownnzsmax = 0;
        for (acgidx_t i = 0; i < A.nprows; i++) {
            int64_t rownnzs = A.frowptr[i+1]-A.frowptr[i];
            if (i > A.borderrowoffset) rownnzs += A.orowptr[i-A.borderrowoffset+1]-A.orowptr[i-A.borderrowoffset];
            prownnzsmin = prownnzsmin < rownnzs ? prownnzsmin : rownnzs;
            prownnzsmax = prownnzsmax > rownnzs ? prownnzsmax : rownnzs;
        }
        int64_t rownnzsmin, rownnzsmax;
        MPI_Reduce(&prownnzsmin, &rownnzsmin, 1, MPI_INT64_T, MPI_MIN, root, mpicomm);
        MPI_Reduce(&prownnzsmax, &rownnzsmax, 1, MPI_INT64_T, MPI_MAX, root, mpicomm);
        if (rank == root) {
            fprintf(stderr, "%'.6f seconds, %"PRId64" to %"PRId64" nonzeros per row\n", elapsed(t0,t1), rownnzsmin, rownnzsmax);
        }
    }

    if (!use_petsc) {
        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "preparing solver: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        /* 7. prepare acg solver */
        struct acgsolverhip cg;
        err = acgsolverhip_init(&cg, &A, hipblas, hipsparse, &comm);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }

        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "running solver: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        /* 8. solve Ax=b using the conjugate gradient algorithm */
        int tag = 99;

        if (args.solvertype == acgsolver_acg) {
            err = acgsolverhip_solvempi(
                &cg, &A, &b, &x, maxits,
                diffatol, diffrtol, residualatol, residualrtol, args.warmup,
                &comm, tag, &mpierrcode, hipblas, hipsparse);
        } else if (args.solvertype == acgsolver_acg_pipelined) {
            err = acgsolverhip_solve_pipelined(
                &cg, &A, &b, &x, maxits,
                diffatol, diffrtol, residualatol, residualrtol, args.warmup,
                &comm, tag, &mpierrcode, hipblas, hipsparse);
        } else if (args.solvertype == acgsolver_acg_device) {
            err = acgsolverhip_solve_device(
                &cg, &A, &b, &x, maxits,
                diffatol, diffrtol, residualatol, residualrtol, args.warmup,
                &comm, &mpierrcode);
        }

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }

        /* 9. output solver information */
        acgsolverhip_fwritempi(stderr, &cg, 0, verbose > 1 ? verbose-2 : 0, mpicomm, 0);
        if (err) {
            if (rank == root)
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, mpierrcode));
            acgsolverhip_free(&cg);
            acgvector_free(&x);
            acgvector_free(&b);
            acgsymcsrmatrix_free(&A);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
            if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        acgsolverhip_free(&cg);
    } else {
        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "preparing solver: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        enum acgpetscksptype ksptype = PETSC_KSPCG;
        if (args.solvertype == acgsolver_petsc_pipelined)
            ksptype = PETSC_KSPPIPECG;

        /* 7. prepare PETSc solver */
        struct acgsolverpetsc cg;
        err = acgsolverpetsc_init(&cg, &A, ACG_DEVICE_HIP, ksptype, &comm);
        if (err) {
            fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }

        if (verbose > 0) {
            if (rank == root) fprintf(stderr, "running solver: ");
            MPI_Barrier(mpicomm); gettime(&t0); MPI_Barrier(mpicomm);
        }

        /* 8. solve Ax=b using the conjugate gradient algorithm */
        int tag = 99;
        err = acgsolverpetsc_solvempi(
            &cg, &A, &b, &x, maxits,
            diffatol, diffrtol, residualatol, residualrtol, args.warmup,
            mpicomm, tag, &mpierrcode);

        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }

        /* 9. output solver information */
        acgsolverpetsc_fwritempi(stderr, &cg, 0, verbose > 1 ? verbose-2 : 0, mpicomm, 0);
        if (err) {
            if (rank == root)
                fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, mpierrcode));
            acgsolverpetsc_free(&cg);
            acgvector_free(&x);
            acgvector_free(&b);
            acgsymcsrmatrix_free(&A);
#ifdef ACG_HAVE_PETSC
            if (use_petsc) PetscFinalize();
#endif
            hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
            acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
        if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
            if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
            MPI_Finalize();
            hipDeviceReset();
            return EXIT_FAILURE;
        }
        acgsolverpetsc_free(&cg);
    }
    hipDeviceSynchronize();
    MPI_Barrier(mpicomm);

    /* print manufactured solution error */
    if (args.manufactured_solution) {
        double x0err = 0.0;
        acgvector_dnrm2mpi(&xsol, &x0err, NULL, mpicomm, &mpierrcode);
        if (rank == root) fprintf(stderr, "initial error 2-norm: %.*g\n", DBL_DIG, x0err);
        acgvector_daxpy(-1.0, &x, &xsol, NULL, NULL);
        double xerr = 0.0;
        acgvector_dnrm2mpi(&xsol, &xerr, NULL, mpicomm, &mpierrcode);
        if (rank == root) fprintf(stderr, "error 2-norm: %.*g\n", DBL_DIG, xerr);
    }

    /* 10. output solution vector */
    if (!quiet) {
        if (verbose > 0 && rank == root) {
            fprintf(stderr, "writing solution to standard output:\n");
            gettime(&t0);
        }

        char * comment = NULL;
        if (rank == root) {
            comment = printf_mtxfilecomment(
                "%% this file was generated by %s %s\n",
                program_name, program_version);
        }

        int64_t nnz;
        err = mtxfile_fwrite_mpi_double(
            stdout, mtxvector, mtxarray, mtxreal, mtxgeneral, comment,
            x.size, 0, &nnz, 1,
            x.num_nonzeros-x.num_ghost_nonzeros,
            x.idxbase, NULL, NULL, x.x,
            x.num_nonzeros, x.idx, 0, NULL,
            numfmt, NULL,
            root, mpicomm, &mpierrcode);
        err = acgerrmpi(mpicomm, err, NULL, &errno, &mpierrcode);
        if (err) {
            if (rank == root) {
                fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                        acgerrcodestr(err, mpierrcode));
            }
            if (rank == root) free(comment);
            return EXIT_FAILURE;
        }
        if (rank == root) free(comment);

        fflush(stdout);
        if (verbose > 0) {
            MPI_Barrier(mpicomm); gettime(&t1);
            if (rank == root) fprintf(stderr, "done in %'.6f seconds\n", elapsed(t0,t1));
        }
    }

    /* 11. clean up and exit */
    acgvector_free(&x);
    acgvector_free(&b);
    acgsymcsrmatrix_free(&A);
#ifdef ACG_HAVE_PETSC
    if (use_petsc) PetscFinalize();
#endif
    hipsparseDestroy(hipsparse); hipblasDestroy(hipblas);
    acgcomm_free(&comm);
#ifdef ACG_HAVE_ROCSHMEM
    if (use_rocshmem) acg_rocshmem_finalize();
#endif
#ifdef ACG_HAVE_RCCL
    if (use_rccl) ncclCommDestroy(ncclcomm);
#endif
    MPI_Finalize();
    hipDeviceReset();
    return EXIT_SUCCESS;
}
