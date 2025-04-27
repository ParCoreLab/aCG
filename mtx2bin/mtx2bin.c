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
 * Convert mtx to binary format
 *
 */

#define _GNU_SOURCE

#include "acg/config.h"
#include "acg/error.h"
#include "acg/mtxfile.h"
#include "acg/time.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <sched.h>

const char * program_name = "mtx2bin";
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
    fprintf(f, " Convert mtx files to binary format\n");
#ifdef ACG_HAVE_LIBZ
    fprintf(f, "\n");
    fprintf(f, " Input options:\n");
    fprintf(f, "  -z, --gzip, --gunzip, --ungzip    filter files through gzip\n");
#endif
    fprintf(f, "\n");
    fprintf(f, " Output options:\n");
    fprintf(f, "  --double             convert values to double-precision floating point [default]\n");
    fprintf(f, "  --integer            convert values to integers\n");
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

    /* output options */
    enum mtxdatatype datatype;
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

    /* output options */
    args->datatype = mtxdouble;
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

        if (strcmp(argv[0], "--double") == 0) {
            args->datatype = mtxdouble;
            (*nargs)++; argv++; continue;
        }
        if (strcmp(argv[0], "--integer") == 0) {
            args->datatype = mtxint;
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

    if (verbose > 0) {
        fprintf(stderr, "reading matrix: ");
        gettime(&t0);
    }

    /* 1. read matrix market file */
    struct acgmtxfile mtxfile;
    int64_t lines_read = 0, bytes_read = 0;
    int idxbase = 0;
    enum mtxlayout layout = mtxrowmajor;
    err = acgmtxfile_read(
        &mtxfile, layout, 0, idxbase, args.datatype,
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

    /* if (mtxfile.object != mtxmatrix) { */
    /*     fprintf(stderr, "%s: %s: expected matrix; object is %s\n", */
    /*             program_invocation_short_name, Apath, mtxobjectstr(mtxfile.object)); */
    /*     return EXIT_FAILURE; */
    /* } */
    /* if (mtxfile.format != mtxcoordinate) { */
    /*     fprintf(stderr, "%s: %s: expected coordinate; format is %s\n", */
    /*             program_invocation_short_name, Apath, mtxformatstr(mtxfile.format)); */
    /*     return EXIT_FAILURE; */
    /* } */
    /* if (mtxfile.symmetry != mtxsymmetric) { */
    /*     fprintf(stderr, "%s: %s: expected symmetric; symmetry is %s\n", */
    /*             program_invocation_short_name, Apath, mtxsymmetrystr(mtxfile.symmetry)); */
    /*     return EXIT_FAILURE; */
    /* } */

    if (verbose > 0) {
        gettime(&t1);
        fprintf(stderr, "%'.6f seconds (%'.1f MB/s)\n",
                elapsed(t0,t1), 1.0e-6*bytes_read/elapsed(t0,t1));
    }

    /* 2. output matrix in binary format */
    if (!args.quiet) {
        acgtime_t t0, t1;
        int64_t bytes_written = 0;
        if (verbose > 0) {
            fprintf(stderr, "writing binary matrix to standard output: ");
            gettime(&t0);
        }

        int binary = 1;
        if (args.datatype == mtxdouble) {
            err = mtxfile_fwrite_double(
                stdout, binary, mtxfile.object, mtxfile.format, mtxfile.field, mtxfile.symmetry, NULL,
                mtxfile.nrows, mtxfile.ncols, mtxfile.nnzs, mtxfile.nvalspernz, mtxfile.idxbase,
                mtxfile.rowidx, mtxfile.colidx, mtxfile.data, NULL, &bytes_written);
        } else if (args.datatype == mtxint) {
            err = mtxfile_fwrite_int(
                stdout, binary, mtxfile.object, mtxfile.format, mtxfile.field, mtxfile.symmetry, NULL,
                mtxfile.nrows, mtxfile.ncols, mtxfile.nnzs, mtxfile.nvalspernz, mtxfile.idxbase,
                mtxfile.rowidx, mtxfile.colidx, mtxfile.data, NULL, &bytes_written);
        }
        if (err) {
            errno = err;
            fprintf(stderr, "%s: %s\n",
                    program_invocation_short_name, acgerrcodestr(ACG_ERR_ERRNO, err));
            return EXIT_FAILURE;
        }

        if (verbose > 0) {
            gettime(&t1);
            fprintf(stderr, "%'.6f seconds\n", elapsed(t0,t1));
        }
    }

    /* 3. clean up and exit */
    acgmtxfile_free(&mtxfile);
    return EXIT_SUCCESS;
}
