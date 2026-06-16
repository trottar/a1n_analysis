#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2024-09-29 14:41:58 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trottar.iii@gmail.com>
#
# Copyright (c) trottar
#
import signal
from collections import defaultdict
import sys, os, subprocess

try:
    from tabulate import tabulate as _tabulate
except ImportError:
    _tabulate = None

################################################################################################################################################

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)


def src_path(*parts):
    return os.path.join(SRC_DIR, *parts)


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


################################################################################################################################################

def safe_tabulate(rows, headers=(), tablefmt=None):
    if _tabulate is not None:
        return _tabulate(rows, headers=headers, tablefmt=tablefmt)

    rows = [list(map(str, row)) for row in rows]
    headers = list(map(str, headers)) if headers else []

    if not rows and not headers:
        return ""

    n_cols = max([len(headers)] + [len(row) for row in rows])

    def pad(row):
        return row + [""] * (n_cols - len(row))

    padded_rows = [pad(row) for row in rows]
    padded_headers = pad(headers) if headers else []
    width_source = padded_rows + ([padded_headers] if headers else [])
    widths = [max(len(row[col]) for row in width_source) for col in range(n_cols)]

    def format_row(row):
        padded = pad(row)
        return " | ".join(padded[col].ljust(widths[col]) for col in range(n_cols))

    lines = []
    if headers:
        lines.append(format_row(headers))
        lines.append("-+-".join("-" * width for width in widths))
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)

################################################################################################################################################

def show_pdf_with_evince(file_path):

    def signal_handler(sig, frame):
        print("\n\n\tCTRL+C detected...Exiting analysis!")
        sys.exit(2)
        
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:            
            user_input = input("\nDo you want to open {}? (y/n): ".format(file_path))
            
            if user_input.lower() == 'y':
                process = subprocess.Popen(['evince', file_path])
                print("\n\n\nPress CTRL+C to exit or close window to continue...")
                process.wait()  # Pauses the script until Evince is closed

                break
            elif user_input.lower() == 'c':
                print("File closed...")
                break
            elif user_input.lower() == 'n':
                break
            elif user_input.lower() == 'q':
                print("Quitting...")
                sys.exit(2)
            else:
                print("Invalid input. Please enter 'y' to open or 'n'/'c' to continue or 'q' to quit.")
                continue
    
    except FileNotFoundError:
        print("Evince not found. Please make sure it is installed.")
    except Exception as e:
        print("An error occurred: {}".format(e))
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\n\tCTRL+C detected...Exiting analysis!")
        process.terminate()
        sys.exit(2)        

################################################################################################################################################        

def most_common_combination(tuples_list):
    count_dict = defaultdict(int)

    # Count occurrences of each combination of (i, j, k)
    for tup in tuples_list:
        combination = tup[:3]  # Extract the first three elements
        count_dict[combination] += 1

    # Find the combination with the maximum count
    most_common = max(count_dict.items(), key=lambda item: item[1])[0]

    return most_common

################################################################################################################################################        
