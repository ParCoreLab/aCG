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
 * formatted output
 */

#include "acg/fmtspec.h"

#include <errno.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * ‘fmtspec_init()’ creates a format specifier.
 */
struct fmtspec fmtspec_init(
    enum fmtspec_flags flags,
    enum fmtspec_width width,
    enum fmtspec_precision precision,
    enum fmtspec_length length,
    enum fmtspec_type specifier)
{
    struct fmtspec f = {
        .flags = flags,
        .width = width,
        .precision = precision,
        .length = length,
        .specifier = specifier};
    return f;
}

static const char * length_str(
    enum fmtspec_length length)
{
    switch (length) {
    case fmtspec_length_none: return "";
    case fmtspec_length_hh: return "hh";
    case fmtspec_length_h: return "h";
    case fmtspec_length_l: return "l";
    case fmtspec_length_ll: return "ll";
    case fmtspec_length_j: return "j";
    case fmtspec_length_z: return "z";
    case fmtspec_length_t: return "t";
    case fmtspec_length_L: return "L";
    default: return strerror(EINVAL);
    }
}

static const char * specifier_str(
    enum fmtspec_type specifier)
{
    switch (specifier) {
    case fmtspec_d: return "d";
    case fmtspec_u: return "u";
    case fmtspec_o: return "o";
    case fmtspec_x: return "x";
    case fmtspec_X: return "X";
    case fmtspec_f: return "f";
    case fmtspec_F: return "F";
    case fmtspec_e: return "e";
    case fmtspec_E: return "E";
    case fmtspec_g: return "g";
    case fmtspec_G: return "G";
    case fmtspec_a: return "a";
    case fmtspec_A: return "A";
    case fmtspec_c: return "c";
    case fmtspec_s: return "s";
    case fmtspec_p: return "p";
    case fmtspec_n: return "n";
    case fmtspec_percent: return "%";
    default: return strerror(EINVAL);
    }
}

/**
 * ‘fmtspecstr()’ is a string representing the given format specifier.
 */
char * fmtspecstr(
    struct fmtspec format)
{
    if ((format.width == fmtspec_width_none ||
         format.width == fmtspec_width_star) &&
        (format.precision == fmtspec_precision_none ||
         format.precision == fmtspec_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if (format.width >= 0 &&
               (format.precision == fmtspec_precision_none ||
                format.precision == fmtspec_precision_star))
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%d%s%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision == fmtspec_precision_none ? "" : ".*",
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if ((format.width == fmtspec_width_none ||
                format.width == fmtspec_width_star) &&
               format.precision >= 0)
    {
        const char * format_str = "%%" "%s%s%s%s%s" "%s.%d%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width == fmtspec_width_none ? "" : "*",
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else if (format.width >= 0 && format.precision >= 0) {
        const char * format_str = "%%" "%s%s%s%s%s" "%d.%d%s%s";
        size_t len = snprintf(
            NULL, 0, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        char * s = malloc(len+1);
        size_t newlen = snprintf(
            s, len+1, format_str,
            format.flags & fmtspec_flags_minus ? "-" : "",
            format.flags & fmtspec_flags_plus ? "+" : "",
            format.flags & fmtspec_flags_space ? " " : "",
            format.flags & fmtspec_flags_number_sign ? "#" : "",
            format.flags & fmtspec_flags_zero ? "0" : "",
            format.width,
            format.precision,
            length_str(format.length),
            specifier_str(format.specifier));
        if (len != newlen) {
            free(s);
            return NULL;
        }
        return s;
    } else {
        size_t len = 1024;
        char * s = malloc(len+1);
        strncpy(s, strerror(EINVAL), len);
        return s;
    }
}

static int fmtspec_parse_flags(
    const char ** s,
    enum fmtspec_flags * flags)
{
    *flags = 0;
    while (**s == '-' || **s == '+' || **s == ' ' || **s == '#' || **s == '0')
    {
        switch (**s) {
        case '-': *flags |= fmtspec_flags_minus; break;
        case '+': *flags |= fmtspec_flags_plus; break;
        case ' ': *flags |= fmtspec_flags_space; break;
        case '#': *flags |= fmtspec_flags_number_sign; break;
        case '0': *flags |= fmtspec_flags_zero; break;
        default: return EINVAL;
        }
        (*s)++;
    }
    return 0;
}

/**
 * ‘parse_long_long_int()’ parses a string to produce a number that
 * may be represented with the type ‘long long int’.
 */
static int parse_long_long_int(
    long long int * outnumber,
    int base,
    const char * s,
    const char ** outendptr,
    int64_t * bytes_read)
{
    errno = 0;
    char * endptr;
    long long int number = strtoll(s, &endptr, base);
    if ((errno == ERANGE && (number == LLONG_MAX || number == LLONG_MIN)) ||
        (errno != 0 && number == 0))
        return errno;
    if (s == endptr) return EINVAL;
    if (outendptr) *outendptr = endptr;
    if (bytes_read) *bytes_read += endptr - s;
    *outnumber = number;
    return 0;
}

/**
 * ‘parse_int()’ parses a string to produce a number that may be
 * represented as an integer.
 *
 * The number is parsed using ‘strtoll()’, following the conventions
 * documented in the man page for that function. In addition, some
 * further error checking is performed to ensure that the number is
 * parsed correctly. The parsed number is stored in ‘x’.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘0’ is returned. Otherwise, if the input contained
 * invalid characters, ‘EINVAL’ is returned, or if the resulting
 * number cannot be represented as a signed integer, ‘ERANGE’ is
 * returned.
 */
static int parse_int(
    int * x,
    const char * s,
    const char ** endptr,
    int64_t * bytes_read)
{
    long long int y;
    int err = parse_long_long_int(&y, 10, s, endptr, bytes_read);
    if (err) return err;
    if (y < INT_MIN || y > INT_MAX) return ERANGE;
    *x = y;
    return 0;
}

static int fmtspec_parse_width(
    const char ** s,
    enum fmtspec_width * width)
{
    if (**s == '*') {
        *width = fmtspec_width_star;
        (*s)++;
        return 0;
    }

    int n;
    const char * t;
    int err = parse_int(&n, *s, &t, NULL);
    if (err) {
        *width = fmtspec_width_none;
        return 0;
    }
    if (n < 0) return EINVAL;
    *width = n;
    *s = t;
    return 0;
}

static int fmtspec_parse_precision(
    const char ** s,
    enum fmtspec_precision * precision)
{
    if (**s != '.') {
        *precision = fmtspec_precision_none;
        return 0;
    }
    (*s)++;

    if (**s == '*') {
        *precision = fmtspec_precision_star;
        (*s)++;
        return 0;
    }

    int n;
    const char * t;
    int err = parse_int(&n, *s, &t, NULL);
    if (err) {
        *precision = fmtspec_precision_none;
        return 0;
    }
    if (n < 0) return EINVAL;
    *precision = n;
    *s = t;
    return 0;
}

static int fmtspec_parse_length(
    const char ** s,
    enum fmtspec_length * length)
{
    if (**s == 'h' && *((*s)+1) == 'h') {
        *length = fmtspec_length_hh;
        *s += 2;
    } else if (**s == 'h' && *((*s)+1) != 'h') {
        *length = fmtspec_length_h;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) != 'l') {
        *length = fmtspec_length_l;
        (*s)++;
    } else if (**s == 'l' && *((*s)+1) == 'l') {
        *length = fmtspec_length_ll;
        *s += 2;
    } else if (**s == 'j') {
        *length = fmtspec_length_j;
        (*s)++;
    } else if (**s == 'z') {
        *length = fmtspec_length_z;
        (*s)++;
    } else if (**s == 't') {
        *length = fmtspec_length_t;
        (*s)++;
    } else if (**s == 'L') {
        *length = fmtspec_length_L;
        (*s)++;
    } else {
        *length = fmtspec_length_none;
    }
    return 0;
}

static int fmtspec_parse_type(
    const char ** s,
    enum fmtspec_type * specifier)
{
    switch (**s) {
    case 'd': *specifier = fmtspec_d; break;
    case 'i': *specifier = fmtspec_i; break;
    case 'u': *specifier = fmtspec_u; break;
    case 'o': *specifier = fmtspec_o; break;
    case 'x': *specifier = fmtspec_x; break;
    case 'X': *specifier = fmtspec_X; break;
    case 'f': *specifier = fmtspec_f; break;
    case 'F': *specifier = fmtspec_F; break;
    case 'e': *specifier = fmtspec_e; break;
    case 'E': *specifier = fmtspec_E; break;
    case 'g': *specifier = fmtspec_g; break;
    case 'G': *specifier = fmtspec_G; break;
    case 'a': *specifier = fmtspec_a; break;
    case 'A': *specifier = fmtspec_A; break;
    case 'c': *specifier = fmtspec_c; break;
    case 's': *specifier = fmtspec_s; break;
    case 'p': *specifier = fmtspec_p; break;
    case 'n': *specifier = fmtspec_n; break;
    case '%': *specifier = fmtspec_percent; break;
    default: return EINVAL;
    }
    (*s)++;
    return 0;
}

/**
 * ‘fmtspec_parse()’ parses a string containing a format specifier.
 *
 * If ‘endptr’ is not ‘NULL’, the address stored in ‘endptr’ points to
 * the first character beyond the characters that were consumed during
 * parsing.
 *
 * On success, ‘fmtspec_parse()’ returns ‘0’.  Otherwise, if the input
 * contained invalid characters, ‘EINVAL’ is returned.
 */
int fmtspec_parse(
    struct fmtspec * format,
    const char * s,
    const char ** endptr)
{
    if (!s) return EINVAL;
    const char * t = s;
    if (*t != '%') return EINVAL;
    t++;

    int err = fmtspec_parse_flags(&t, &format->flags);
    if (err) return err;
    err = fmtspec_parse_width(&t, &format->width);
    if (err) return err;
    err = fmtspec_parse_precision(&t, &format->precision);
    if (err) return err;
    err = fmtspec_parse_length(&t, &format->length);
    if (err) return err;
    err = fmtspec_parse_type(&t, &format->specifier);
    if (err) return err;
    if (endptr) *endptr = t;
    return 0;
}
