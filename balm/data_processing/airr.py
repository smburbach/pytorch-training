#!/usr/bin/python
# filename: airr.py

#
# Copyright (c) 2023 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


import os
import subprocess as sp
import tempfile
from typing import Optional, List

import abutils


def transform_airr(
    airr_data: str,
    output_dir: str,
    keep_paired_csv: bool = True,
    keep_sorted_airr: bool = False,
    temp_dir: Optional[str] = None,
    missing_chain_token: str = "<unk>",
    id_key: str = "sequence_id",
    sequence_key: str = "sequence_aa",
    locus_key: str = "locus",
    id_delim: str = "_",
    id_delim_occurence: int = 1,
    debug: bool = False,
) -> str:
    # process data input
    if os.path.isfile(airr_data):
        airr_data = [airr_data]
    # set up directories
    if temp_dir is None:
        temp_dir = output_dir
    sort_dir = os.path.join(temp_dir, "sorted")
    csv_dir = os.path.join(temp_dir, "csv")
    abutils.io.makedir(sort_dir)
    abutils.io.makedir(csv_dir)

    for airr_file in airr_data:
        positions = get_column_positions(airr_file, id_key, sequence_key, locus_key)
        sorted_file = sort_airr_file(
            airr_file=airr_file,
            sort_dir=sort_dir,
            id_pos=positions["id_pos"],
            debug=debug,
        )
        paired_csv = make_paired_csv(
            sorted_file,
            csv_dir=csv_dir,
            delim=id_delim,
            delim_occurence=id_delim_occurence,
            **positions,
        )
        roberta_txt = build_roberta_txt(
            paired_csv=paired_csv,
            output_dir=output_dir,
            missing_chain_token=missing_chain_token,
        )


def sort_airr_file(
    airr_file: str, sort_dir: str, id_pos: int = 0, debug: bool = False
) -> str:
    sorted_file = os.path.join(sort_dir, os.path.basename(airr_file))
    sort_cmd = f"tail -n +2 {airr_file} | "
    sort_cmd += f'sort -t"\t" -k {id_pos + 1},{id_pos + 1} >> {sorted_file}'
    p = sp.Popen(sort_cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)
    stdout, stderr = p.communicate()
    if debug:
        print(stdout)
        print(stderr)
    return sorted_file


def make_paired_csv(
    sorted_file: str,
    csv_dir: str,
    id_pos: int = 0,
    seq_pos: int = 3,
    locus_pos: int = 61,
    delim: str = "_",
    delim_occurence: int = 1,
):
    csv_file = os.path.join(csv_dir, os.path.basename(sorted_file))
    params = {
        "id_pos": id_pos,
        "seq_pos": seq_pos,
        "locus_pos": locus_pos,
        "delim": delim,
        "delim_occurance": delim_occurence,
    }
    prev = None
    with open(csv_file, "w") as csv:
        with open(sorted_file, "r") as f:
            for line in f:
                if not line.strip:
                    continue
                curr = CSVLine(line, **params)
                if prev is None:
                    pair = [curr]
                    prev = curr
                elif curr.name == prev.name:
                    pair.append(curr)
                    prev = curr
                else:
                    csv_line = build_csv_line(pair)
                    csv.write(csv_line + "\n")
                    pair = [curr]
                    prev = curr
            # process the last line(s)
            csv_line = build_csv_line(pair)
            csv.write(csv_line + "\n")
    return csv_file


def build_roberta_txt(
    paired_csv: str, output_dir: str, missing_chain_token: str = "<unk>"
):
    pass


def get_column_positions(
    airr_file: str,
    id_key: str = "sequence_id",
    sequence_key: str = "sequence_aa",
    locus_key: str = "locus",
) -> List[int]:
    head_cmd = f"head -n 1 {airr_file}"
    p = sp.Popen(head_cmd, stdout=sp.PIPE, shell=True)
    stdout = p.communicate()
    header = stdout.decode("utf-8").strip().split("\t")
    id_pos = header.index(id_key)
    seq_pos = header.index(sequence_key)
    locus_pos = header.index(locus_key)
    return {"id_pos": id_pos, "seq_pos": seq_pos, "locus_pos": locus_pos}


def build_csv_line(lines) -> str:
    line_data = [lines[0].name]
    for locus in ["IGH", "IGK", "IGL"]:
        seqs = [l for l in lines if l.locus == locus]
        if seqs:
            seq = seqs[0]
            line_data.append(seq.id)
            line_data.append(seq.seq)
        else:
            line_data.append("")
            line_data.append("")
    return ",".join(line_data)


class CSVLine:
    def __init__(
        self,
        line: str,
        id_pos: int = 0,
        seq_pos: int = 3,
        locus_pos: int = 61,
        delim: str = "_",
        delim_occurrence: int = 1,
    ):
        self.raw_line = line
        self.line = line.strip().split("\t")
        self.id_pos = id_pos
        self.seq_pos = seq_pos
        self.locus_pos = locus_pos
        self.delim = delim
        self.delim_occurrence = delim_occurrence

    @property
    def id(self) -> str:
        return self.line[self.id_pos]

    @property
    def name(self) -> str:
        return self.delim.join(self.line)[self.id_pos].split(self.delim)[
            : self.delim_occurrence
        ]

    @property
    def seq(self) -> str:
        return self.line[self.seq_pos]

    @property
    def locus(self) -> str:
        return self.line[self.locus_pos]
