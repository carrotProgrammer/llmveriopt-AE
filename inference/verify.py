#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile

# -----------------------------
# Extract IR from prompt
# -----------------------------
# model_latency / qwen_SFT
USER_PATTERN_IM = re.compile(
    r"<\|im_start\|>user\s*(.*?)<\|im_end\|>",
    re.DOTALL
)

# model_correctness
USER_PATTERN_CORR = re.compile(
    r"User:\s*(.*?)(?=\s*Assistant:)",
    re.DOTALL
)

# llama model
USER_PATTERN_LLAMA3 = re.compile(
    r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)(?=<\|eot_id\|>)",
    re.DOTALL
)


def select_user_pattern(input_path: str):
    p = input_path.lower()

    # 1.（model_latency / sft_qwen）
    if "model_latency" in p or "sft_qwen" in p:
        return USER_PATTERN_IM

    # 2. correctness CSV
    if "model_correctness" in p:
        return USER_PATTERN_CORR

    # 3. LLAMA3 SFT
    if "sft_llama3" in p:
        return USER_PATTERN_LLAMA3

    # default
    return USER_PATTERN_IM


def extract_ir_from_prompt(prompt_text: str, pattern) -> str:
    m = pattern.search(prompt_text)
    if not m:
        return ""
    return m.group(1).strip()

# -----------------------------
# Clean output IR (your rule)
# -----------------------------
MODULE_START_PATTERN = re.compile(
    r";\s*ModuleID\s*=.*",
    re.IGNORECASE
)

def clean_output_ir(out_text: str) -> str:
    text = out_text.strip()

    if text.lower().endswith("</answer>"):
        text = text[:-len("</answer>")].rstrip()

    mm = MODULE_START_PATTERN.search(text)
    if not mm:
        return text

    return text[mm.start():].strip()


# -----------------------------
# Measure latency using aarch64_tti_latency
# -----------------------------
def measure_latency(ir_text: str, latency_bin: str):
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp:
        tmp.write(ir_text.encode())
        tmp.flush()
        ir_path = tmp.name

    try:
        output = subprocess.check_output(
            [latency_bin, ir_path],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        os.remove(ir_path)
        return None  # latency fail = None

    os.remove(ir_path)

    m = re.search(r"==== Module total latency:\s+(\d+)", output)
    if m:
        return int(m.group(1))
    return None

# -----------------------------
# Size (llc + llvm-size)
# -----------------------------
# ============================================================
# Fixed configuration parameters (always in English comments)
# ============================================================
MTRIPLE = ""                    # Use target triple from IR (leave empty to auto-detect)
MCPU = ""                       # Do not specify CPU (use default)
EXTRA_LLC_FLAGS = ["-mattr=+crc"]   # Extra llc flags for AArch64 testing
LLC_TIMEOUT = 15                # Timeout (seconds) for llc / llvm-size
# ============================================================

def measure_size(ir_text: str, llvm_bin: str):
    llc_bin = os.path.join(llvm_bin, "llc")
    llvm_size_bin = os.path.join(llvm_bin, "llvm-size")

    with tempfile.TemporaryDirectory() as td:
        ll_path = os.path.join(td, "tmp.ll")
        obj_path = os.path.join(td, "tmp.o")

        with open(ll_path, "w", encoding="utf-8") as f:
            f.write(ir_text)

        cmd = [
            llc_bin,
            "-O0",
            "-filetype=obj",
            "-o", obj_path,
            ll_path
        ]

        if MTRIPLE:
            cmd += ["-mtriple", MTRIPLE]

        if MCPU:
            cmd += ["-mcpu", MCPU]

        if EXTRA_LLC_FLAGS:
            cmd += EXTRA_LLC_FLAGS

        # Compile IR
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=LLC_TIMEOUT,
                check=True,
                text=True
            )
        except Exception:
            return None

        # Call llvm-size
        try:
            out = subprocess.run(
                [llvm_size_bin, obj_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=LLC_TIMEOUT,
                check=True,
                text=True
            ).stdout
        except Exception:
            return None

        # Parse text + data (sum_mode = "text_data")
        lines = out.strip().splitlines()
        if len(lines) < 2:
            return None

        cols = lines[-1].split()
        # text + data
        if len(cols) >= 2 and cols[0].isdigit() and cols[1].isdigit():
            return int(cols[0]) + int(cols[1])

        return None

# -----------------------------
# ICount
# -----------------------------
def count_instructions(ir_text: str, llvm_bin: str, instcount_path: str) -> int | None:
    """
    Count the total number of instructions in the given LLVM IR text.

    This uses the LLVM InstCount pass:
      opt -load-pass-plugin=InstCount.so -passes=function(instcount)

    Returns:
        int  -> number of instructions
        None -> counting failed or no instructions found
    """
    opt_path = os.path.join(llvm_bin, "opt")

    # Write IR into a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=True) as ir_file:
        ir_file.write(ir_text)
        ir_file.flush()

        # Run opt with InstCount pass
        result = subprocess.run(
            [
                opt_path,
                "-load-pass-plugin", instcount_path,
                "-passes=function(instcount)",
                "-debug-pass-manager",
                "-disable-output",
                ir_file.name
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Example stderr pattern:
        # Running pass: InstCountPass on foo (42 instructions)
        pattern = re.compile(
            r"Running pass: InstCountPass on .+? \((\d+) instruction(?:s)?\)",
            re.MULTILINE
        )

        total = 0
        for line in result.stderr.splitlines():
            m = pattern.search(line)
            if m:
                total += int(m.group(1))

        return total if total > 0 else None



# -----------------------------
# Alive2 verification
# -----------------------------
def verify_single_ir(src_ir, tgt_ir, alive_tv_path, timeout=15, debug=False):

    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as src, \
         tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tgt:

        src.write(src_ir.encode())
        tgt.write(tgt_ir.encode())
        src.flush()
        tgt.flush()

        cmd = f"{alive_tv_path} {src.name} {tgt.name} --smt-to={timeout * 1000}"

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 5,
                check=False
            )
            output = (res.stdout.decode(errors="ignore") +
                      res.stderr.decode(errors="ignore")).strip()
        except subprocess.TimeoutExpired:
            os.remove(src.name); os.remove(tgt.name)
            return "timeout", "Alive2 verification timed out."

    os.remove(src.name); os.remove(tgt.name)

    # Status logic unchanged
    if "Transformation seems to be correct!" in output:
        return "correct", output

    def _count(pat):
        m = re.search(pat, output)
        return int(m.group(1)) if m else 0

    if _count(r"(\d+)\s+Alive2 errors") > 0:
        return "alive2 can't prove", output
    if _count(r"(\d+)\s+incorrect transformations") > 0:
        return "semantic error", output
    if _count(r"(\d+)\s+failed-to-prove transformations") > 0:
        return "alive2 can't prove", output
    if _count(r"(\d+)\s+correct transformations") > 0:
        return "correct", output

    if re.search(r"\berror:", output, re.IGNORECASE):
        return "syntactic error", output

    return "syntactic error", output


# -----------------------------
# Main driver
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alive", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--llvm-bin", required=True)
    parser.add_argument("--instcount", required=True, help="Path to InstCount.so plugin")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for idx, row in enumerate(reader):
            if len(row) < 2:
                continue

            prompt_raw = row[0]
            output_raw = row[1]

            user_pattern = select_user_pattern(args.input)

            src_ir = extract_ir_from_prompt(prompt_raw, user_pattern)
            tgt_ir = clean_output_ir(output_raw)

            # ---- Alive2 ----
            status, alive_summary = verify_single_ir(
                src_ir, tgt_ir, args.alive, debug=args.debug
            )

            # ---- Latency for src ----
            src_latency = measure_latency(src_ir, args.latency)

            # ---- Latency for tgt ----
            tgt_latency = measure_latency(tgt_ir, args.latency)
            
            # ---- Size ----
            src_size = measure_size(src_ir, args.llvm_bin)
            tgt_size = measure_size(tgt_ir, args.llvm_bin)

            # ---- Icount ----
            src_inst = count_instructions(src_ir, args.llvm_bin, args.instcount)
            tgt_inst = count_instructions(tgt_ir, args.llvm_bin, args.instcount)

            # If incorrect → FALLBACK to original IR
            if status == "correct":
                final_tgt_latency = tgt_latency
                final_tgt_size = tgt_size
                final_tgt_inst = tgt_inst
            else:
                final_tgt_latency = src_latency
                final_tgt_size = src_size
                final_tgt_inst = src_inst
            
            results[idx] = {
                "status": status,
                "alive_summary": alive_summary,
                "src_latency": src_latency,
                "tgt_latency": final_tgt_latency,
                "src_size": src_size,
                "tgt_size": final_tgt_size,
                "src_inst": src_inst,
                "tgt_inst": final_tgt_inst,
            }
            
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[VERIFY] Saved metrics to:", args.output)


if __name__ == "__main__":
    main()
