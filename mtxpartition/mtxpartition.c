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
 * Authors: James D. Trotter <james@simula.no>
 *
 * Last modified: 2025-04-26
 *
 * Partition matrices in Matrix Market format.
 *
 */

#define _GNU_SOURCE

#include "acg/config.h"
#include "acg/comm.h"
#include "acg/error.h"
#include "acg/fmtspec.h"
#include "acg/mtxfile.h"
#include "acg/symcsrmatrix.h"
#include "acg/time.h"

#ifdef ACG_HAVE_METIS
#include <metis.h>
#endif

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <sched.h>

const char * program_name = "mtxpartition";
const char * program_version = "0.1.0";
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
    fprintf(f, "Usage: %s [OPTION..] FILE\n", program_name);
}

/**
 * ‘program_options_print_help()’ prints a help text.
 */
static void program_options_print_help(
    FILE * f)
{
    program_options_print_usage(f);
    fprintf(f, "\n");
    fprintf(f, " Partition a matrix using METIS.\n");
#ifdef ACG_HAVE_LIBZ
    fprintf(f, "\n");
    fprintf(f, " Input options:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    fprintf(f, "  --binary             read Matrix Market files in binary format\n");
    fprintf(f, "\n");
    fprintf(f, " Partitioning options:\n");
    fprintf(f, "  --parts=N            number of parts to use for partitioning. [2]\n");
    fprintf(f, "  --seed=N             random number seed. [0]\n");
    fprintf(f, "\n");
    fprintf(f, " Output options:\n");
    fprintf(f, "  --numfmt FMT         Format string for outputting numerical values.\n");
    fprintf(f, "                       The format specifiers '%%e', '%%E', '%%f', '%%F',\n");
    fprintf(f, "                       '%%g' or '%%G' may be used. Flags, field width and\n");
    fprintf(f, "                       precision may also be specified, e.g., \"%%+3.1f\".\n");
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
    int gzip;
    int binary;

    /* partitioning options */
    int nparts;
    acgidx_t seed;

    /* output options */
    char * numfmt;
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
    args->gzip = 0;
    args->binary = 0;

    /* partitioning options */
    args->nparts = 2;
    args->seed = 0;

    /* output options */
    args->numfmt = NULL;
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

        /* partitioning options */
        if (strstr(argv[0], "--parts") == argv[0]) {
            int n = strlen("--parts");
            const char * s = &argv[0][n];
            if (*s == '=') { s++; }
            else if (*s == '\0' && argc-*nargs > 1) { (*nargs)++; argv++; s=argv[0]; }
            else { return EINVAL; }
            char * endptr;
            if (parse_int(&args->nparts, s, &endptr, NULL)) return EINVAL;
            if (*endptr != '\0') return EINVAL;
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
    acgtime_t t0, t1;

    /* 1. parse program options */
    struct program_options args;
    err = program_options_init(&args);
    if (err) {
        if (err) fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(err));
        return EXIT_FAILURE;
    }

    int nargs;
    err = parse_program_options(argc, argv, &args, &nargs);
    if (err) {
        fprintf(stderr, "%s: %s %s\n", program_invocation_short_name,
                strerror(err), argv[nargs]);
        program_options_free(&args);
        return EXIT_FAILURE;
    }
    if (args.help) {
        program_options_print_help(stdout);
        program_options_free(&args);
        return EXIT_SUCCESS;
    }
    if (args.version) {
        program_options_print_version(stdout);
        program_options_free(&args);
        return EXIT_SUCCESS;
    }
    if (!args.Apath) {
        program_options_print_usage(stdout);
        program_options_free(&args);
        return EXIT_FAILURE;
    }

    const char * Apath = args.Apath;
    int quiet = args.quiet;
    int verbose = args.verbose;
    const char * numfmt = args.numfmt;
    acgidx_t seed = args.seed;

    if (verbose > 0) {
        fprintf(stderr, "reading matrix: ");
        gettime(&t0);
    }

    /* 1. read matrix on the root process */
    struct acgmtxfile mtxfile;
    int64_t lines_read = 0, bytes_read = 0;
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
        return EXIT_FAILURE;
    }

    if (mtxfile.object != mtxmatrix) {
        fprintf(stderr, "%s: %s: expected matrix; object is %s\n",
                program_invocation_short_name, Apath, mtxobjectstr(mtxfile.object));
        return EXIT_FAILURE;
    }
    if (mtxfile.format != mtxcoordinate) {
        fprintf(stderr, "%s: %s: expected coordinate; format is %s\n",
                program_invocation_short_name, Apath, mtxformatstr(mtxfile.format));
        return EXIT_FAILURE;
    }
    if (mtxfile.symmetry != mtxsymmetric) {
        fprintf(stderr, "%s: %s: expected symmetric; symmetry is %s\n",
                program_invocation_short_name, Apath, mtxsymmetrystr(mtxfile.symmetry));
        return EXIT_FAILURE;
    }

    if (verbose > 0) {
        gettime(&t1);
        int64_t mtxsz =
            (mtxfile.rowidx ? mtxfile.nnzs*sizeof(mtxfile.rowidx) : 0)
            + (mtxfile.colidx ? mtxfile.nnzs*sizeof(mtxfile.colidx) : 0)
            + (mtxfile.data && mtxfile.datatype == mtxint ? mtxfile.nnzs*sizeof(int) : 0)
            + (mtxfile.data && mtxfile.datatype == mtxdouble ? mtxfile.nnzs*sizeof(double) : 0);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s, %'.1f MiB)\n",
                elapsed(t0,t1), 1.0e-6*bytes_read/elapsed(t0,t1),
                (double) mtxsz/1024.0/1024.0);
    }

    if (verbose > 0) {
        fprintf(stderr, "converting to symcsrmatrix:");
        if (verbose > 1) fprintf(stderr, "\n");
        gettime(&t0);
    }

    /* 2a) initialise matrix */
    acgidx_t N = mtxfile.nrows;
    int64_t nnzs = mtxfile.nnzs;
    const acgidx_t * rowidx = mtxfile.rowidx;
    const acgidx_t * colidx = mtxfile.colidx;
    const double * a = (const double *) mtxfile.data;
    struct acgsymcsrmatrix A;
    err = acgsymcsrmatrix_init_real_double(
        &A, N, nnzs, idxbase, rowidx, colidx, a);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
        return EXIT_FAILURE;
    }
    acgmtxfile_free(&mtxfile);

    if (verbose > 0) {
        gettime(&t1);
        int64_t sz =
            (A.nzrows ? A.nprows*sizeof(A.nzrows) : 0)
            + (A.rownnzs ? A.nprows*sizeof(A.rownnzs) : 0)
            + (A.rowptr ? (A.nprows+1)*sizeof(A.rowptr) : 0)
            + (A.rowidx ? A.npnzs*sizeof(A.rowidx) : 0)
            + (A.colidx ? A.npnzs*sizeof(A.colidx) : 0)
            + (A.a ? A.npnzs*sizeof(A.a) : 0)
            + (A.frowptr ? (A.nprows+1)*sizeof(A.frowptr) : 0)
            + (A.fcolidx ? A.fnpnzs*sizeof(A.fcolidx) : 0)
            + (A.fa ? A.fnpnzs*sizeof(A.fa) : 0);
        fprintf(stderr, "%'.6f seconds (%'.1f MiB)\n",
                elapsed(t0,t1), (double) sz/1024.0/1024.0);
    }

    int nparts = args.nparts;
    if (verbose > 0) {
        fprintf(stderr, "partitioning matrix into %'d parts:", nparts);
        if (verbose > 1) fprintf(stderr, "\n");
        gettime(&t0);
    }

    /* 2b) partition matrix rows */
    enum metis_partitioner partitioner = metis_partgraphrecursive;
    int * rowparts = malloc(N*sizeof(*rowparts));
    if (!rowparts) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, strerror(errno));
        return EXIT_FAILURE;
    }
    acgidx_t objval;
    err = acgsymcsrmatrix_partition_rows(
        &A, nparts, partitioner, rowparts, &objval, seed, verbose > 0 ? verbose-1 : 0);
    if (err) {
        fprintf(stderr, "%s: %s\n", program_invocation_short_name, acgerrcodestr(err, 0));
        return EXIT_FAILURE;
    }

    if (verbose > 0) {
        gettime(&t1);
        if (verbose > 1) fprintf(stderr, "done in %'.6f seconds objective value: %"PRIdx"\n", elapsed(t0,t1), objval);
        else fprintf(stderr, " %'.6f seconds\n", elapsed(t0,t1));
    }

    /* 3. output the row partition */
    if (!args.quiet) {
        acgtime_t t0, t1;
        int64_t bytes_written = 0;
        if (verbose > 0) {
            fprintf(stderr, "writing partition vector to standard output: ");
            gettime(&t0);
        }

        char * comment = printf_mtxfilecomment(
            "%% this file was generated by %s %s\n",
            program_name, program_version);

        acgidx_t nrows = A.nrows;
        acgidx_t ncols = 1;
        int64_t nnzs = A.nrows;
        for (acgidx_t i = 0; i < A.nrows; i++) rowparts[i] = rowparts[i]+1;
        err = mtxfile_fwrite_int(
            stdout, 0, mtxvector, mtxarray, mtxinteger, mtxgeneral, comment,
            nrows, ncols, nnzs, 1, 0, NULL, NULL, rowparts,
            NULL, &bytes_written);
        free(comment);
        if (err) {
            if (verbose > 0) fprintf(stderr, "\n");
            fprintf(stderr, "%s: %s\n", program_invocation_short_name,
                    acgerrcodestr(err, 0));
            free(rowparts);
            acgsymcsrmatrix_free(&A);
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            gettime(&t1);
            fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }
    }

    /* 4. clean up and exit */
    free(rowparts);
    acgsymcsrmatrix_free(&A);
    return EXIT_SUCCESS;
}
